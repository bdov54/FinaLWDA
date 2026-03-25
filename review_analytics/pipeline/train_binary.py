import argparse
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException


DEFAULT_BINARY_COLLECTION = "processed_data_binary_123vs45_v1"

DEFAULT_MODELS = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
]

MODEL_ALIASES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
}

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# =========================================================
# BASIC UTILS
# =========================================================
def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


# =========================================================
# METRICS
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
    }


# =========================================================
# TRAINER
# =========================================================
class WeightedLabelSmoothingTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.label_smoothing = float(label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits

        weight = None if self.class_weights is None else self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# =========================================================
# DATA PREP
# =========================================================
def to_hf_dataset(df_: pd.DataFrame) -> Dataset:
    cols = ["review_id", "aspect", "aspect_text", "sentence", "label_id"]
    keep_cols = [c for c in cols if c in df_.columns]
    return Dataset.from_pandas(df_[keep_cols].copy(), preserve_index=False)


def tokenize_pair(batch, tokenizer, max_length=128):
    aspect_input = batch["aspect_text"] if "aspect_text" in batch else batch["aspect"]
    return tokenizer(
        aspect_input,
        batch["sentence"],
        truncation=True,
        max_length=max_length,
    )


def build_training_args(**kwargs):
    """
    Compatible wrapper:
    - sửa eval_strategy -> evaluation_strategy nếu version cũ
    - tự bỏ các arg mới mà version transformers hiện tại không support
    """
    work = kwargs.copy()

    while True:
        try:
            return TrainingArguments(**work)
        except TypeError as e:
            msg = str(e)

            if "eval_strategy" in msg and "unexpected keyword" in msg:
                work["evaluation_strategy"] = work.pop("eval_strategy")
                continue

            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad_key = m.group(1)
                if bad_key in work:
                    logger.warning("TrainingArguments không support '%s' -> tự bỏ arg này.", bad_key)
                    work.pop(bad_key, None)
                    continue

            raise


def compute_class_weights(label_ids: pd.Series, mode: str = "sqrt_inv"):
    counts = label_ids.value_counts().sort_index()
    for i in [0, 1]:
        if i not in counts.index:
            counts.loc[i] = 0
    counts = counts.sort_index()

    total = counts.sum()

    if mode == "none":
        return counts, None

    if mode == "inv":
        weights = (total / (counts + 1e-9)).values
    elif mode == "sqrt_inv":
        weights = np.sqrt(total / (counts + 1e-9)).values
    else:
        raise ValueError("class_weight_mode must be one of: none, inv, sqrt_inv")

    weights = weights / weights.mean()
    return counts, weights.astype(np.float32)


def get_num_proc():
    cpu = os.cpu_count() or 1
    if platform.system().lower().startswith("win"):
        return 1
    return max(1, min(4, cpu // 2 if cpu > 1 else 1))


def get_dataloader_workers():
    if platform.system().lower().startswith("win"):
        return 0
    cpu = os.cpu_count() or 1
    return min(2, max(0, cpu - 1))


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def prepare_dataframe_from_mongo(input_col: str) -> pd.DataFrame:
    s = load_settings()
    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    store.ensure_indexes(
        s.col_raw,
        s.col_processed,
        s.col_raw_web,
        processed_binary_col=input_col,
    )

    proc_col = store.collection(input_col)
    docs = list(proc_col.find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError(f"Mongo collection '{input_col}' đang rỗng. Chạy preprocess_binary trước.")

    df = pd.DataFrame(docs)

    required = ["review_id", "sentence", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"processed_data thiếu cột: {missing}. Hiện có: {df.columns.tolist()}")

    if "aspect_text" not in df.columns and "aspect" not in df.columns:
        raise ValueError("Cần có ít nhất aspect_text hoặc aspect trong collection.")

    if "aspect_text" not in df.columns:
        df["aspect_text"] = df["aspect"].astype(str).str.replace("_", " ", regex=False)
    if "aspect" not in df.columns:
        df["aspect"] = df["aspect_text"]

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(LABEL2ID)].copy()
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["aspect_text"] = df["aspect_text"].astype(str).str.strip()
    df["aspect"] = df["aspect"].astype(str).str.strip()
    df["review_id"] = df["review_id"].astype(str)

    df = df[(df["sentence"] != "") & (df["aspect_text"] != "")].copy()
    return df.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["review_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, df["label_id"], groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["label_id"], groups=temp_df["review_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# =========================================================
# SPEED / HARDWARE HELPERS
# =========================================================
def init_torch_speedups():
    # // CHANGED: bật TF32 ở mức PyTorch nếu có CUDA để tăng tốc matmul trên GPU phù hợp
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def get_speed_config(disable_torch_compile: bool = False) -> Dict:
    # // CHANGED: auto chọn bf16/fp16/tf32/torch_compile/optimizer fused nếu môi trường support
    use_cuda = torch.cuda.is_available()
    bf16_supported = bool(use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    use_bf16 = bf16_supported
    use_fp16 = bool(use_cuda and not use_bf16)
    use_tf32 = bool(use_cuda)
    use_torch_compile = bool(use_cuda and hasattr(torch, "compile") and not disable_torch_compile)

    optim_name = "adamw_torch_fused" if use_cuda else "adamw_torch"

    return {
        "use_cuda": use_cuda,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "tf32": use_tf32,
        "torch_compile": use_torch_compile,
        "optim": optim_name,
    }


def inspect_token_lengths(df: pd.DataFrame, tokenizer, sample_size: int = 5000) -> Dict:
    # // CHANGED: đo phân bố độ dài token để biết max_len có đang quá lớn không
    tmp = df[["aspect_text", "sentence"]].dropna().copy()
    if tmp.empty:
        return {}

    if len(tmp) > sample_size:
        tmp = tmp.sample(sample_size, random_state=42)

    lengths = []
    for _, row in tmp.iterrows():
        enc = tokenizer(
            row["aspect_text"],
            row["sentence"],
            truncation=False,
            add_special_tokens=True,
        )
        lengths.append(len(enc["input_ids"]))

    s = pd.Series(lengths)
    p95 = int(s.quantile(0.95))
    suggestion = min(max(64, p95), 256)

    return {
        "n_sampled": int(len(s)),
        "mean": float(round(s.mean(), 2)),
        "p50": int(s.quantile(0.50)),
        "p90": int(s.quantile(0.90)),
        "p95": p95,
        "p99": int(s.quantile(0.99)),
        "max": int(s.max()),
        "suggested_max_len": suggestion,
    }


# =========================================================
# HISTORY / CHART HELPERS
# =========================================================
def history_to_dataframe(log_history):
    rows = []
    for x in log_history:
        rows.append(
            {
                "epoch": x.get("epoch"),
                "step": x.get("step"),
                "loss": x.get("loss"),
                "eval_loss": x.get("eval_loss"),
                "eval_accuracy": x.get("eval_accuracy"),
                "eval_precision_macro": x.get("eval_precision_macro"),
                "eval_recall_macro": x.get("eval_recall_macro"),
                "eval_f1_macro": x.get("eval_f1_macro"),
                "learning_rate": x.get("learning_rate"),
                "grad_norm": x.get("grad_norm"),
            }
        )
    return pd.DataFrame(rows)


def plot_loss_curve(history_df: pd.DataFrame, out_path: Path, title: str):
    train_loss = history_df[history_df["loss"].notna()][["epoch", "loss"]].drop_duplicates(subset=["epoch"])
    eval_loss = history_df[history_df["eval_loss"].notna()][["epoch", "eval_loss"]].drop_duplicates(subset=["epoch"])

    if train_loss.empty and eval_loss.empty:
        return

    plt.figure(figsize=(8, 5))
    if not train_loss.empty:
        plt.plot(train_loss["epoch"], train_loss["loss"], marker="o", label="train_loss")
    if not eval_loss.empty:
        plt.plot(eval_loss["epoch"], eval_loss["eval_loss"], marker="o", label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_val_metrics_curve(history_df: pd.DataFrame, out_path: Path, title: str):
    metric_df = history_df[
        history_df["eval_accuracy"].notna() | history_df["eval_f1_macro"].notna()
    ][["epoch", "eval_accuracy", "eval_f1_macro"]].drop_duplicates(subset=["epoch"])

    if metric_df.empty:
        return

    plt.figure(figsize=(8, 5))
    if metric_df["eval_accuracy"].notna().any():
        plt.plot(metric_df["epoch"], metric_df["eval_accuracy"], marker="o", label="eval_accuracy")
    if metric_df["eval_f1_macro"].notna().any():
        plt.plot(metric_df["epoch"], metric_df["eval_f1_macro"], marker="o", label="eval_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grad_norm(history_df: pd.DataFrame, out_path: Path, title: str):
    grad_df = history_df[history_df["grad_norm"].notna()][["epoch", "grad_norm"]].drop_duplicates(subset=["epoch"])
    if grad_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(grad_df["epoch"], grad_df["grad_norm"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Grad Norm")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix_png(cm, labels, out_path: Path, title: str):
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_metrics(report_dict: dict, labels: list, out_path: Path, title: str):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(9, 5))
    for idx, metric in enumerate(metrics):
        vals = [report_dict[label][metric] for label in labels]
        plt.bar(x + (idx - 1) * width, vals, width=width, label=metric)

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overall_metrics(metrics: dict, out_path: Path, title: str):
    keys = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro", "test_f1_weighted"]
    labels = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    vals = [metrics.get(k, np.nan) for k in keys]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# COMPARISON CHARTS
# =========================================================
def plot_compare_overall(run_infos: List[dict], out_path: Path, title: str):
    keys = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro", "test_f1_weighted"]
    labels = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]

    x = np.arange(len(labels))
    n = len(run_infos)
    width = 0.8 / max(n, 1)

    plt.figure(figsize=(11, 5))
    for idx, info in enumerate(run_infos):
        values = [info["metrics"].get(k, np.nan) for k in keys]
        offset = (idx - (n - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=info["display_name"])

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_class_metric(run_infos: List[dict], cls_name: str, out_path: Path, title: str):
    metrics_list = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics_list))
    n = len(run_infos)
    width = 0.8 / max(n, 1)

    plt.figure(figsize=(9, 5))
    for idx, info in enumerate(run_infos):
        report = info["report"]
        values = [report[cls_name][m] for m in metrics_list]
        offset = (idx - (n - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=info["display_name"])

    plt.xticks(x, metrics_list)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_best_val_f1(run_infos: List[dict], out_path: Path, title: str):
    labels = [info["display_name"] for info in run_infos]
    vals = [info["trainer_state"].get("best_metric", np.nan) for info in run_infos]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Best val F1_macro")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_runtime(run_infos: List[dict], out_path: Path, title: str):
    # // CHANGED: so sánh tốc độ train giữa các model
    labels = [info["display_name"] for info in run_infos]
    vals = [info["train_metrics"].get("train_runtime", np.nan) for info in run_infos]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylabel("Seconds")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_throughput(run_infos: List[dict], out_path: Path, title: str):
    # // CHANGED: so sánh throughput train
    labels = [info["display_name"] for info in run_infos]
    vals = [info["train_metrics"].get("train_samples_per_second", np.nan) for info in run_infos]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylabel("Samples / second")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_val_f1_curves(run_infos: List[dict], out_path: Path, title: str):
    # // CHANGED: overlay learning curves để đưa vào dashboard
    plt.figure(figsize=(9, 5))
    for info in run_infos:
        hist = info["history_df"]
        metric_df = hist[hist["eval_f1_macro"].notna()][["epoch", "eval_f1_macro"]].drop_duplicates(subset=["epoch"])
        if not metric_df.empty:
            plt.plot(metric_df["epoch"], metric_df["eval_f1_macro"], marker="o", label=info["display_name"])

    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 Macro")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_eval_loss_curves(run_infos: List[dict], out_path: Path, title: str):
    # // CHANGED: overlay eval loss giữa các model
    plt.figure(figsize=(9, 5))
    for info in run_infos:
        hist = info["history_df"]
        metric_df = hist[hist["eval_loss"].notna()][["epoch", "eval_loss"]].drop_duplicates(subset=["epoch"])
        if not metric_df.empty:
            plt.plot(metric_df["epoch"], metric_df["eval_loss"], marker="o", label=info["display_name"])

    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_aspect_error_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    # // CHANGED: thêm summary lỗi theo aspect để dashboard có thể so model theo khía cạnh
    if pred_df.empty or "aspect_text" not in pred_df.columns:
        return pd.DataFrame()

    out = (
        pred_df.groupby(["aspect", "aspect_text"], dropna=False)
        .agg(
            n_samples=("aspect", "size"),
            n_errors=("is_error", "sum"),
            error_rate=("is_error", "mean"),
            avg_confidence=("confidence", "mean"),
            avg_prob_negative=("prob_negative", "mean"),
            avg_prob_positive=("prob_positive", "mean"),
        )
        .reset_index()
    )

    float_cols = ["error_rate", "avg_confidence", "avg_prob_negative", "avg_prob_positive"]
    out[float_cols] = out[float_cols].round(4)
    out = out.sort_values(["error_rate", "n_samples"], ascending=[False, False]).reset_index(drop=True)
    return out


def plot_compare_aspect_error(run_infos: List[dict], out_path: Path, title: str, top_n: int = 10):
    # // CHANGED: biểu đồ compare error rate theo aspect
    rows = []
    for info in run_infos:
        if info["aspect_error_df"].empty:
            continue
        tmp = info["aspect_error_df"].copy()
        tmp["display_name"] = info["display_name"]
        rows.append(tmp)

    if not rows:
        return

    all_df = pd.concat(rows, ignore_index=True)
    top_aspects = (
        all_df.groupby("aspect_text", as_index=False)["n_samples"]
        .sum()
        .sort_values("n_samples", ascending=False)
        .head(top_n)["aspect_text"]
        .tolist()
    )

    plot_df = all_df[all_df["aspect_text"].isin(top_aspects)].copy()
    if plot_df.empty:
        return

    model_order = plot_df["display_name"].drop_duplicates().tolist()
    aspect_order = (
        plot_df.groupby("aspect_text", as_index=False)["n_samples"]
        .sum()
        .sort_values("n_samples", ascending=True)["aspect_text"]
        .tolist()
    )

    x = np.arange(len(aspect_order))
    n = len(model_order)
    width = 0.8 / max(n, 1)

    plt.figure(figsize=(12, 6))
    for idx, model_name in enumerate(model_order):
        sub = plot_df[plot_df["display_name"] == model_name].set_index("aspect_text").reindex(aspect_order)
        vals = sub["error_rate"].values
        offset = (idx - (n - 1) / 2) * width
        plt.bar(x + offset, vals, width=width, label=model_name)

    plt.xticks(x, aspect_order, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Error rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# TRAIN / EVAL
# =========================================================
def build_batch_name(models: List[str], max_len: int, epochs: int, lr: float) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"b123v45_{ts}"


def run_single_model(
    batch_dir: Path,
    model_name: str,
    display_name: str,
    input_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_len: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_bs: int,
    eval_bs: int,
    grad_accum: int,
    warmup_ratio: float,
    early_stopping_patience: int,
    class_weight_mode: str,
    label_smoothing: float,
    disable_torch_compile: bool,
):
    num_proc = get_num_proc()
    speed_cfg = get_speed_config(disable_torch_compile=disable_torch_compile)

    logger.info("============================================================")
    logger.info("🚀 Training %s (%s)", display_name, model_name)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            logger.info("GPU name: %s", torch.cuda.get_device_name(0))
            logger.info("GPU capability: %s", torch.cuda.get_device_capability(0))
        except Exception:
            pass

    logger.info("Speed config: %s", speed_cfg)
    logger.info("Train label dist: %s", train_df["label"].value_counts(normalize=True).round(4).to_dict())
    logger.info("Val label dist: %s", val_df["label"].value_counts(normalize=True).round(4).to_dict())
    logger.info("Test label dist: %s", test_df["label"].value_counts(normalize=True).round(4).to_dict())

    counts, class_weights = compute_class_weights(train_df["label_id"], mode=class_weight_mode)
    logger.info("⚖️ Train counts: %s", counts.to_dict())
    logger.info(
        "⚖️ class_weights[%s]: %s",
        class_weight_mode,
        None if class_weights is None else {i: float(round(w, 4)) for i, w in enumerate(class_weights)},
    )

    train_ds_raw = to_hf_dataset(train_df)
    val_ds_raw = to_hf_dataset(val_df)
    test_ds_raw = to_hf_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # // CHANGED: log phân bố token để biết max_len đã hợp lý chưa
    token_stats = inspect_token_lengths(pd.concat([train_df, val_df, test_df], ignore_index=True), tokenizer=tokenizer)
    logger.info("Token length stats: %s", token_stats)
    if token_stats:
        logger.info(
            "Current max_len=%d | suggested_max_len≈%d",
            max_len,
            token_stats.get("suggested_max_len", max_len),
        )

    train_ds = train_ds_raw.map(lambda b: tokenize_pair(b, tokenizer, max_len), batched=True, num_proc=num_proc)
    val_ds = val_ds_raw.map(lambda b: tokenize_pair(b, tokenizer, max_len), batched=True, num_proc=num_proc)
    test_ds = test_ds_raw.map(lambda b: tokenize_pair(b, tokenizer, max_len), batched=True, num_proc=num_proc)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    tensor_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        tensor_cols.append("token_type_ids")

    train_ds.set_format(type="torch", columns=tensor_cols)
    val_ds.set_format(type="torch", columns=tensor_cols)
    test_ds.set_format(type="torch", columns=tensor_cols)

    # // CHANGED: pad_to_multiple_of=8 giúp Tensor Core hiệu quả hơn khi dùng mixed precision
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    total_params, trainable_params = count_trainable_params(model)
    logger.info("Model params: total=%d | trainable=%d", total_params, trainable_params)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_name = "mbert" if model_name == "bert-base-multilingual-cased" else "xlmr"
    run_name = f"{short_name}_{run_id}"

    run_dir = batch_dir / "runs" / run_name
    trainer_out = run_dir / "trainer_out"
    charts_dir = run_dir / "charts"
    split_dir = run_dir / "splits"
    model_dir = run_dir / "model"
    tokenizer_dir = run_dir / "tokenizer"
    reports_dir = run_dir / "reports"

    trainer_out.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    args = build_training_args(
        output_dir=str(trainer_out),
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        save_total_limit=2,
        bf16=speed_cfg["bf16"],
        fp16=speed_cfg["fp16"],
        tf32=speed_cfg["tf32"],
        torch_compile=speed_cfg["torch_compile"],
        optim=speed_cfg["optim"],
        report_to="none",
        dataloader_num_workers=get_dataloader_workers(),
        dataloader_pin_memory=torch.cuda.is_available(),
        group_by_length=True,
        remove_unused_columns=True,
        save_safetensors=True,
    )

    trainer = WeightedLabelSmoothingTrainer(
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    meta = {
        "run_id": run_id,
        "run_name": run_name,
        "display_name": display_name,
        "label_mode": "binary_123_vs_45",
        "input_collection": input_col,
        "model_name": model_name,
        "max_len": max_len,
        "token_stats": token_stats,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "train_bs": train_bs,
        "eval_bs": eval_bs,
        "grad_accum": grad_accum,
        "warmup_ratio": warmup_ratio,
        "early_stopping_patience": early_stopping_patience,
        "class_weight_mode": class_weight_mode,
        "class_weights": None if class_weights is None else [float(x) for x in class_weights],
        "label_smoothing": label_smoothing,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tokenizer_num_proc": num_proc,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "speed_config": speed_cfg,
        "created_at": now_utc_iso(),
    }

    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:
        pass

    save_json(meta, run_dir / "training_config.json")

    train_df.to_csv(split_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(split_dir / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(split_dir / "test.csv", index=False, encoding="utf-8-sig")

    # // CHANGED: lưu train metrics riêng để so sánh tốc độ giữa 2 model
    train_result = trainer.train()
    train_metrics = train_result.metrics
    save_json(train_metrics, run_dir / "train_metrics.json")

    history_df = history_to_dataframe(trainer.state.log_history)
    history_df.to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    trainer_state_summary = {
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "epoch": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
    }
    save_json(trainer_state_summary, run_dir / "trainer_state_summary.json")

    plot_loss_curve(history_df, charts_dir / f"{run_name}_loss_curve.png", f"Loss Curve - {display_name}")
    plot_val_metrics_curve(history_df, charts_dir / f"{run_name}_val_metrics_curve.png", f"Validation Metrics - {display_name}")
    plot_grad_norm(history_df, charts_dir / f"{run_name}_grad_norm.png", f"Grad Norm - {display_name}")

    pred_out = trainer.predict(test_ds, metric_key_prefix="test")
    y_true = pred_out.label_ids
    y_pred = np.argmax(pred_out.predictions, axis=-1)

    probs = torch.softmax(torch.tensor(pred_out.predictions), dim=-1).numpy()
    confidence = probs.max(axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(2)],
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(2)],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics = pred_out.metrics

    pred_df = test_df.copy()
    pred_df["y_true_id"] = y_true
    pred_df["y_pred_id"] = y_pred
    pred_df["y_true"] = pred_df["y_true_id"].map(ID2LABEL)
    pred_df["y_pred"] = pred_df["y_pred_id"].map(ID2LABEL)
    pred_df["confidence"] = confidence
    pred_df["prob_negative"] = probs[:, 0]
    pred_df["prob_positive"] = probs[:, 1]
    pred_df["is_error"] = (pred_df["y_true_id"] != pred_df["y_pred_id"]).astype(int)
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # // CHANGED: aspect error summary để dashboard/model comparison dùng
    aspect_error_df = build_aspect_error_summary(pred_df)
    aspect_error_df.to_csv(reports_dir / "aspect_error_summary.csv", index=False, encoding="utf-8-sig")
    save_json(aspect_error_df.to_dict(orient="records"), reports_dir / "aspect_error_summary.json")

    # // CHANGED: extra confidence summary
    confidence_summary = {
        "avg_confidence_all": float(round(pred_df["confidence"].mean(), 4)),
        "avg_confidence_correct": float(round(pred_df.loc[pred_df["is_error"] == 0, "confidence"].mean(), 4))
        if (pred_df["is_error"] == 0).any()
        else None,
        "avg_confidence_error": float(round(pred_df.loc[pred_df["is_error"] == 1, "confidence"].mean(), 4))
        if (pred_df["is_error"] == 1).any()
        else None,
    }
    save_json(confidence_summary, reports_dir / "confidence_summary.json")

    save_json(metrics, run_dir / "metrics.json")
    save_json(report_dict, run_dir / "classification_report.json")
    save_json(cm, run_dir / "confusion_matrix.json")
    save_json(LABEL2ID, run_dir / "label2id.json")
    save_json(ID2LABEL, run_dir / "id2label.json")
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    plot_confusion_matrix_png(
        cm=cm,
        labels=[ID2LABEL[0], ID2LABEL[1]],
        out_path=charts_dir / f"{run_name}_confusion_matrix.png",
        title=f"Confusion Matrix - {display_name}",
    )
    plot_per_class_metrics(
        report_dict=report_dict,
        labels=[ID2LABEL[0], ID2LABEL[1]],
        out_path=charts_dir / f"{run_name}_per_class_metrics.png",
        title=f"Per-class Metrics - {display_name}",
    )
    plot_overall_metrics(
        metrics=metrics,
        out_path=charts_dir / f"{run_name}_overall_test_metrics.png",
        title=f"Overall Test Metrics - {display_name}",
    )

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    logger.info(
        "✅ Finished %s | test_f1_macro=%.4f | train_runtime=%s sec",
        display_name,
        metrics.get("test_f1_macro", float("nan")),
        train_metrics.get("train_runtime"),
    )
    logger.info("✅ Saved run at: %s", str(run_dir))

    return {
        "display_name": display_name,
        "model_name": model_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "report": report_dict,
        "trainer_state": trainer_state_summary,
        "run_name": run_name,
        "history_df": history_df,
        "pred_df": pred_df,
        "aspect_error_df": aspect_error_df,
        "confidence_summary": confidence_summary,
    }


# =========================================================
# BATCH COMPARISON
# =========================================================
def compare_all_runs(batch_dir: Path, run_infos: List[dict]):
    if len(run_infos) < 2:
        return

    comparison_dir = batch_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    plot_compare_overall(
        run_infos,
        comparison_dir / "compare_overall_metrics.png",
        "Model Comparison - Overall Metrics",
    )
    plot_compare_best_val_f1(
        run_infos,
        comparison_dir / "compare_best_val_f1.png",
        "Model Comparison - Best Validation F1",
    )
    plot_compare_class_metric(
        run_infos,
        "negative",
        comparison_dir / "compare_class_negative.png",
        "Model Comparison - Class: negative",
    )
    plot_compare_class_metric(
        run_infos,
        "positive",
        comparison_dir / "compare_class_positive.png",
        "Model Comparison - Class: positive",
    )

    # // CHANGED: thêm comparison chart về tốc độ và learning curves
    plot_compare_runtime(
        run_infos,
        comparison_dir / "compare_train_runtime.png",
        "Model Comparison - Train Runtime",
    )
    plot_compare_throughput(
        run_infos,
        comparison_dir / "compare_train_throughput.png",
        "Model Comparison - Train Throughput",
    )
    plot_compare_val_f1_curves(
        run_infos,
        comparison_dir / "compare_val_f1_curve.png",
        "Model Comparison - Validation F1 Curve",
    )
    plot_compare_eval_loss_curves(
        run_infos,
        comparison_dir / "compare_eval_loss_curve.png",
        "Model Comparison - Eval Loss Curve",
    )
    plot_compare_aspect_error(
        run_infos,
        comparison_dir / "compare_aspect_error_rate.png",
        "Model Comparison - Aspect Error Rate",
        top_n=10,
    )

    summary_rows = []
    for info in run_infos:
        summary_rows.append(
            {
                "display_name": info["display_name"],
                "model_name": info["model_name"],
                "run_dir": str(info["run_dir"]),
                "best_val_f1": info["trainer_state"].get("best_metric"),
                "test_accuracy": info["metrics"].get("test_accuracy"),
                "test_precision_macro": info["metrics"].get("test_precision_macro"),
                "test_recall_macro": info["metrics"].get("test_recall_macro"),
                "test_f1_macro": info["metrics"].get("test_f1_macro"),
                "test_f1_weighted": info["metrics"].get("test_f1_weighted"),
                "train_runtime": info["train_metrics"].get("train_runtime"),
                "train_samples_per_second": info["train_metrics"].get("train_samples_per_second"),
                "train_steps_per_second": info["train_metrics"].get("train_steps_per_second"),
                "test_runtime": info["metrics"].get("test_runtime"),
                "test_samples_per_second": info["metrics"].get("test_samples_per_second"),
                "avg_confidence_all": info["confidence_summary"].get("avg_confidence_all"),
                "avg_confidence_correct": info["confidence_summary"].get("avg_confidence_correct"),
                "avg_confidence_error": info["confidence_summary"].get("avg_confidence_error"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(comparison_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    save_json(summary_rows, comparison_dir / "model_comparison.json")

    # // CHANGED: file dashboard-ready gọn hơn để app đọc thẳng
    dashboard_ready = {
        "created_at": now_utc_iso(),
        "models": summary_rows,
        "comparison_files": {
            "compare_overall_metrics": "compare_overall_metrics.png",
            "compare_best_val_f1": "compare_best_val_f1.png",
            "compare_class_negative": "compare_class_negative.png",
            "compare_class_positive": "compare_class_positive.png",
            "compare_train_runtime": "compare_train_runtime.png",
            "compare_train_throughput": "compare_train_throughput.png",
            "compare_val_f1_curve": "compare_val_f1_curve.png",
            "compare_eval_loss_curve": "compare_eval_loss_curve.png",
            "compare_aspect_error_rate": "compare_aspect_error_rate.png",
        },
    }
    save_json(dashboard_ready, comparison_dir / "model_comparison_dashboard_ready.json")

    if not summary_df.empty:
        best_row = summary_df.sort_values("test_f1_macro", ascending=False).iloc[0].to_dict()
        save_json(best_row, comparison_dir / "best_model_by_test_f1_macro.json")

    # // CHANGED: aspect comparison csv/json
    aspect_frames = []
    for info in run_infos:
        tmp = info["aspect_error_df"].copy()
        if tmp.empty:
            continue
        tmp["display_name"] = info["display_name"]
        aspect_frames.append(tmp)

    if aspect_frames:
        aspect_all_df = pd.concat(aspect_frames, ignore_index=True)
        aspect_all_df.to_csv(comparison_dir / "aspect_comparison_long.csv", index=False, encoding="utf-8-sig")
        save_json(aspect_all_df.to_dict(orient="records"), comparison_dir / "aspect_comparison_long.json")


# =========================================================
# MAIN BATCH RUN
# =========================================================
def run_train_batch(
    models: List[str],
    input_col: str = DEFAULT_BINARY_COLLECTION,
    max_len: int = 128,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    train_bs: int = 8,
    eval_bs: int = 16,
    grad_accum: int = 1,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 1,
    class_weight_mode: str = "sqrt_inv",
    label_smoothing: float = 0.05,
    disable_torch_compile: bool = False,
):
    s = load_settings()
    set_seed(seed)
    init_torch_speedups()

    df = prepare_dataframe_from_mongo(input_col)
    train_df, val_df, test_df = split_dataframe(df, seed=seed)

    logger.info("✅ Shared split sizes: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))

    batch_name = build_batch_name(models, max_len=max_len, epochs=epochs, lr=lr)
    batch_dir = Path(s.artifact_dir) / "model_batches" / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_config = {
        "batch_name": batch_name,
        "input_collection": input_col,
        "models": models,
        "max_len": max_len,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "train_bs": train_bs,
        "eval_bs": eval_bs,
        "grad_accum": grad_accum,
        "warmup_ratio": warmup_ratio,
        "early_stopping_patience": early_stopping_patience,
        "class_weight_mode": class_weight_mode,
        "label_smoothing": label_smoothing,
        "disable_torch_compile": disable_torch_compile,
        "created_at": now_utc_iso(),
        "shared_split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    save_json(batch_config, batch_dir / "batch_config.json")

    run_infos = []
    for model_name in models:
        display_name = MODEL_ALIASES.get(model_name, model_name)
        info = run_single_model(
            batch_dir=batch_dir,
            model_name=model_name,
            display_name=display_name,
            input_col=input_col,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            max_len=max_len,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            train_bs=train_bs,
            eval_bs=eval_bs,
            grad_accum=grad_accum,
            warmup_ratio=warmup_ratio,
            early_stopping_patience=early_stopping_patience,
            class_weight_mode=class_weight_mode,
            label_smoothing=label_smoothing,
            disable_torch_compile=disable_torch_compile,
        )
        run_infos.append(info)

    compare_all_runs(batch_dir, run_infos)

    batch_summary = []
    for info in run_infos:
        batch_summary.append(
            {
                "display_name": info["display_name"],
                "model_name": info["model_name"],
                "run_dir": str(info["run_dir"]),
                "best_val_f1": info["trainer_state"].get("best_metric"),
                "test_accuracy": info["metrics"].get("test_accuracy"),
                "test_f1_macro": info["metrics"].get("test_f1_macro"),
                "test_f1_weighted": info["metrics"].get("test_f1_weighted"),
                "train_runtime": info["train_metrics"].get("train_runtime"),
                "train_samples_per_second": info["train_metrics"].get("train_samples_per_second"),
                "avg_confidence_all": info["confidence_summary"].get("avg_confidence_all"),
            }
        )

    save_json(batch_summary, batch_dir / "batch_summary.json")

    logger.info("============================================================")
    logger.info("✅ Finished batch training.")
    logger.info("✅ Batch dir: %s", str(batch_dir))
    logger.info("============================================================")


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Danh sách model sẽ chạy nối tiếp. Mặc định: mBERT baseline + XLM-R",
    )
    ap.add_argument("--input-col", type=str, default=DEFAULT_BINARY_COLLECTION)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--early-stopping-patience", type=int, default=1)
    ap.add_argument("--class-weight-mode", type=str, default="sqrt_inv", choices=["none", "inv", "sqrt_inv"])
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--disable-torch-compile", action="store_true")  # // CHANGED: cho phép tắt compile nếu máy lỗi
    args = ap.parse_args()

    if not args.run:
        print("Use: python -m review_analytics.pipeline.train_binary --run")
        return

    try:
        run_train_batch(
            models=args.models,
            input_col=args.input_col,
            max_len=args.max_len,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            seed=args.seed,
            train_bs=args.train_batch_size,
            eval_bs=args.eval_batch_size,
            grad_accum=args.grad_accum,
            warmup_ratio=args.warmup_ratio,
            early_stopping_patience=args.early_stopping_patience,
            class_weight_mode=args.class_weight_mode,
            label_smoothing=args.label_smoothing,
            disable_torch_compile=args.disable_torch_compile,
        )
    except Exception as e:
        logger.error("Train binary batch failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()