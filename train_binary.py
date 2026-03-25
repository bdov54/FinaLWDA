import argparse
import itertools
import json
import math
import os
import platform
import random
import re
import sys
from copy import deepcopy
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
from review_analytics.exception.exception import ReviewAnalyticsException
from review_analytics.logging import logger


# // CHANGED: giảm warning tokenizer fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# // CHANGED: search space mặc định vừa phải, tránh nổ số trial
DEFAULT_SEARCH_SPACE = {
    "max_len": [64],
    "lr": [1e-5, 2e-5, 3e-5],
    "epochs": [3, 5],
    "train_bs": [8],
    "eval_bs": [16],
    "grad_accum": [1],
    "warmup_ratio": [0.06, 0.10],
    "early_stopping_patience": [2],
    "class_weight_mode": ["none", "sqrt_inv"],
    "label_smoothing": [0.0, 0.03],
}

# // CHANGED: registry để lưu best config dùng lại lần sau
BEST_REGISTRY_REL_PATH = Path("model_registry") / "best_hparams_registry.json"


# =========================================================
# BASIC UTILS
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def slugify(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x.strip("_")


def safe_float(x, ndigits=6):
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except Exception:
        return None


def safe_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


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


def metrics_from_predictions(y_true, y_pred, prefix: str = "") -> Dict:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    pre = f"{prefix}_" if prefix else ""
    return {
        f"{pre}accuracy": safe_float(acc),
        f"{pre}precision_macro": safe_float(p_macro),
        f"{pre}recall_macro": safe_float(r_macro),
        f"{pre}f1_macro": safe_float(f1_macro),
        f"{pre}precision_weighted": safe_float(p_weighted),
        f"{pre}recall_weighted": safe_float(r_weighted),
        f"{pre}f1_weighted": safe_float(f1_weighted),
    }


def choose_best_threshold(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    threshold_grid: List[float],
) -> Dict:
    # // CHANGED: tune threshold trên validation để tối ưu macro F1
    rows = []
    for thr in threshold_grid:
        y_pred = (prob_positive >= thr).astype(int)
        m = metrics_from_predictions(y_true, y_pred, prefix="val")
        rows.append(
            {
                "threshold": float(thr),
                "val_accuracy": m["val_accuracy"],
                "val_precision_macro": m["val_precision_macro"],
                "val_recall_macro": m["val_recall_macro"],
                "val_f1_macro": m["val_f1_macro"],
                "val_f1_weighted": m["val_f1_weighted"],
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "val_f1_weighted", "val_accuracy", "threshold"],
        ascending=[False, False, False, True],
    )
    best = df.iloc[0].to_dict()
    best["threshold_grid"] = [float(x) for x in threshold_grid]
    return best


def build_report_and_cm(y_true: np.ndarray, y_pred: np.ndarray):
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
    return report_text, report_dict, cm


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
    - ưu tiên eval_strategy mới
    - nếu version cũ không support thì tự fallback
    - tự bỏ các arg mới nếu environment chưa support
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
# SEARCH SPACE / REGISTRY
# =========================================================
def parse_list_arg(arg_value: Optional[str], cast_func):
    if not arg_value:
        return None
    items = [x.strip() for x in str(arg_value).split(",") if x.strip()]
    return [cast_func(x) for x in items]


def build_search_space_from_args(args) -> Dict:
    # // CHANGED: search space lấy từ CLI, nếu không có dùng mặc định
    space = deepcopy(DEFAULT_SEARCH_SPACE)

    mapping = {
        "max_len": ("search_max_lens", int),
        "lr": ("search_lrs", float),
        "epochs": ("search_epochs", int),
        "train_bs": ("search_train_batch_sizes", int),
        "eval_bs": ("search_eval_batch_sizes", int),
        "grad_accum": ("search_grad_accums", int),
        "warmup_ratio": ("search_warmup_ratios", float),
        "early_stopping_patience": ("search_early_stopping_patiences", int),
        "class_weight_mode": ("search_class_weight_modes", str),
        "label_smoothing": ("search_label_smoothings", float),
    }

    for key, (arg_name, caster) in mapping.items():
        v = getattr(args, arg_name, None)
        parsed = parse_list_arg(v, caster)
        if parsed:
            space[key] = parsed

    return space


def expand_search_space(search_space: Dict, seed: int, max_trials_per_model: Optional[int] = None) -> List[Dict]:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    all_trials = []
    for combo in itertools.product(*values):
        d = dict(zip(keys, combo))
        all_trials.append(d)

    # // CHANGED: tránh số lượng trial quá lớn
    if max_trials_per_model is not None and len(all_trials) > max_trials_per_model:
        rnd = random.Random(seed)
        rnd.shuffle(all_trials)
        all_trials = all_trials[:max_trials_per_model]

    return all_trials


def get_registry_path() -> Path:
    s = load_settings()
    return Path(s.artifact_dir) / BEST_REGISTRY_REL_PATH


def load_best_registry() -> Dict:
    return load_json(get_registry_path(), default={})


def save_best_registry(obj: Dict):
    save_json(obj, get_registry_path())


def registry_key(input_col: str, model_name: str) -> str:
    return f"{input_col}__{model_name}"


# =========================================================
# SPEED / HARDWARE HELPERS
# =========================================================
def init_torch_speedups():
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
            verbose=False,
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


def plot_tuning_results_bar(df: pd.DataFrame, out_path: Path, title: str, score_col: str):
    # // CHANGED: leaderboard trial cho từng model
    if df.empty or score_col not in df.columns:
        return

    tmp = df.copy().sort_values(score_col, ascending=False).head(15)
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(tmp)), tmp[score_col].values)
    plt.xticks(range(len(tmp)), tmp["trial_name"].tolist(), rotation=35, ha="right")
    plt.ylabel(score_col)
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
    vals = [info["selection_metric"] for info in run_infos]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Best tuned val F1_macro")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare_runtime(run_infos: List[dict], out_path: Path, title: str):
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
def build_batch_name(models: List[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"b123v45_tuned_{ts}"


def build_trial_name(model_name: str, trial_cfg: Dict, trial_idx: int) -> str:
    short_name = "mbert" if model_name == "bert-base-multilingual-cased" else "xlmr"
    return (
        f"{short_name}_t{trial_idx:03d}"
        f"_len{trial_cfg['max_len']}"
        f"_ep{trial_cfg['epochs']}"
        f"_lr{trial_cfg['lr']}"
        f"_bs{trial_cfg['train_bs']}"
        f"_ga{trial_cfg['grad_accum']}"
        f"_cw{trial_cfg['class_weight_mode']}"
        f"_ls{trial_cfg['label_smoothing']}"
    )


def evaluate_with_threshold(prob_pos: np.ndarray, y_true: np.ndarray, threshold: float, prefix: str) -> Tuple[np.ndarray, Dict]:
    y_pred = (prob_pos >= threshold).astype(int)
    m = metrics_from_predictions(y_true, y_pred, prefix=prefix)
    return y_pred, m


def run_single_trial(
    trials_root: Path,
    model_name: str,
    display_name: str,
    input_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trial_cfg: Dict,
    trial_idx: int,
    seed: int,
    disable_torch_compile: bool,
    threshold_grid: List[float],
):
    # // CHANGED: 1 trial = 1 cấu hình hyperparameter
    num_proc = get_num_proc()
    speed_cfg = get_speed_config(disable_torch_compile=disable_torch_compile)

    counts, class_weights = compute_class_weights(train_df["label_id"], mode=trial_cfg["class_weight_mode"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    token_stats = inspect_token_lengths(pd.concat([train_df, val_df, test_df], ignore_index=True), tokenizer=tokenizer)

    logger.info("------------------------------------------------------------")
    logger.info("🚀 Trial %d | %s (%s)", trial_idx, display_name, model_name)
    logger.info("Trial config: %s", trial_cfg)
    logger.info("Token stats: %s", token_stats)
    logger.info("Speed config: %s", speed_cfg)
    logger.info("Train counts: %s", counts.to_dict())

    train_ds_raw = to_hf_dataset(train_df)
    val_ds_raw = to_hf_dataset(val_df)
    test_ds_raw = to_hf_dataset(test_df)

    train_ds = train_ds_raw.map(
        lambda b: tokenize_pair(b, tokenizer, trial_cfg["max_len"]),
        batched=True,
        num_proc=num_proc,
    )
    val_ds = val_ds_raw.map(
        lambda b: tokenize_pair(b, tokenizer, trial_cfg["max_len"]),
        batched=True,
        num_proc=num_proc,
    )
    test_ds = test_ds_raw.map(
        lambda b: tokenize_pair(b, tokenizer, trial_cfg["max_len"]),
        batched=True,
        num_proc=num_proc,
    )

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    tensor_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        tensor_cols.append("token_type_ids")

    train_ds.set_format(type="torch", columns=tensor_cols)
    val_ds.set_format(type="torch", columns=tensor_cols)
    test_ds.set_format(type="torch", columns=tensor_cols)

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

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trial_name = build_trial_name(model_name, trial_cfg, trial_idx)
    run_dir = trials_root / trial_name
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
        learning_rate=trial_cfg["lr"],
        per_device_train_batch_size=trial_cfg["train_bs"],
        per_device_eval_batch_size=trial_cfg["eval_bs"],
        gradient_accumulation_steps=trial_cfg["grad_accum"],
        num_train_epochs=trial_cfg["epochs"],
        eval_strategy="epoch",  # // CHANGED: dùng tên mới, wrapper tự fallback nếu cần
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        weight_decay=0.01,
        warmup_ratio=trial_cfg["warmup_ratio"],
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
        label_smoothing=trial_cfg["label_smoothing"],
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=trial_cfg["early_stopping_patience"])],
    )

    meta = {
        "run_id": run_id,
        "trial_name": trial_name,
        "display_name": display_name,
        "input_collection": input_col,
        "model_name": model_name,
        "trial_config": trial_cfg,
        "token_stats": token_stats,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tokenizer_num_proc": num_proc,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "speed_config": speed_cfg,
        "threshold_grid": [float(x) for x in threshold_grid],
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

    plot_loss_curve(history_df, charts_dir / f"{trial_name}_loss_curve.png", f"Loss Curve - {trial_name}")
    plot_val_metrics_curve(history_df, charts_dir / f"{trial_name}_val_metrics_curve.png", f"Validation Metrics - {trial_name}")
    plot_grad_norm(history_df, charts_dir / f"{trial_name}_grad_norm.png", f"Grad Norm - {trial_name}")

    # // CHANGED: predict trên val để tune threshold
    val_pred_out = trainer.predict(val_ds, metric_key_prefix="val_final")
    val_probs = torch.softmax(torch.tensor(val_pred_out.predictions), dim=-1).numpy()
    val_prob_pos = val_probs[:, 1]
    val_confidence = val_probs.max(axis=1)
    y_val_true = val_pred_out.label_ids

    threshold_result = choose_best_threshold(
        y_true=y_val_true,
        prob_positive=val_prob_pos,
        threshold_grid=threshold_grid,
    )
    best_threshold = float(threshold_result["threshold"])
    y_val_pred, val_metrics_tuned = evaluate_with_threshold(
        prob_pos=val_prob_pos,
        y_true=y_val_true,
        threshold=best_threshold,
        prefix="val",
    )

    threshold_df = pd.DataFrame(
        [
            {
                "threshold": r_thr,
                "val_f1_macro": safe_float(metrics_from_predictions(y_val_true, (val_prob_pos >= r_thr).astype(int), prefix="val")["val_f1_macro"]),
                "val_f1_weighted": safe_float(metrics_from_predictions(y_val_true, (val_prob_pos >= r_thr).astype(int), prefix="val")["val_f1_weighted"]),
                "val_accuracy": safe_float(metrics_from_predictions(y_val_true, (val_prob_pos >= r_thr).astype(int), prefix="val")["val_accuracy"]),
            }
            for r_thr in threshold_grid
        ]
    )
    threshold_df.to_csv(reports_dir / "threshold_search_val.csv", index=False, encoding="utf-8-sig")
    save_json(threshold_result, reports_dir / "threshold_search_best.json")

    val_pred_df = val_df.copy()
    val_pred_df["y_true_id"] = y_val_true
    val_pred_df["y_pred_id"] = y_val_pred
    val_pred_df["y_true"] = val_pred_df["y_true_id"].map(ID2LABEL)
    val_pred_df["y_pred"] = val_pred_df["y_pred_id"].map(ID2LABEL)
    val_pred_df["confidence"] = val_confidence
    val_pred_df["prob_negative"] = val_probs[:, 0]
    val_pred_df["prob_positive"] = val_probs[:, 1]
    val_pred_df["threshold_used"] = best_threshold
    val_pred_df["is_error"] = (val_pred_df["y_true_id"] != val_pred_df["y_pred_id"]).astype(int)
    val_pred_df.to_csv(run_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")

    # // CHANGED: test chỉ dùng config + threshold đã chọn từ val
    test_pred_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_probs = torch.softmax(torch.tensor(test_pred_out.predictions), dim=-1).numpy()
    test_prob_pos = test_probs[:, 1]
    test_confidence = test_probs.max(axis=1)
    y_test_true = test_pred_out.label_ids
    y_test_pred, test_metrics_tuned = evaluate_with_threshold(
        prob_pos=test_prob_pos,
        y_true=y_test_true,
        threshold=best_threshold,
        prefix="test",
    )

    final_test_metrics = {
        **test_metrics_tuned,
        "test_runtime": safe_float(test_pred_out.metrics.get("test_runtime")),
        "test_samples_per_second": safe_float(test_pred_out.metrics.get("test_samples_per_second")),
        "test_steps_per_second": safe_float(test_pred_out.metrics.get("test_steps_per_second")),
        "threshold_used": best_threshold,
    }

    report_text, report_dict, cm = build_report_and_cm(y_test_true, y_test_pred)

    pred_df = test_df.copy()
    pred_df["y_true_id"] = y_test_true
    pred_df["y_pred_id"] = y_test_pred
    pred_df["y_true"] = pred_df["y_true_id"].map(ID2LABEL)
    pred_df["y_pred"] = pred_df["y_pred_id"].map(ID2LABEL)
    pred_df["confidence"] = test_confidence
    pred_df["prob_negative"] = test_probs[:, 0]
    pred_df["prob_positive"] = test_probs[:, 1]
    pred_df["threshold_used"] = best_threshold
    pred_df["is_error"] = (pred_df["y_true_id"] != pred_df["y_pred_id"]).astype(int)
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    aspect_error_df = build_aspect_error_summary(pred_df)
    aspect_error_df.to_csv(reports_dir / "aspect_error_summary.csv", index=False, encoding="utf-8-sig")
    save_json(aspect_error_df.to_dict(orient="records"), reports_dir / "aspect_error_summary.json")

    confidence_summary = {
        "avg_confidence_all": safe_float(pred_df["confidence"].mean(), 4),
        "avg_confidence_correct": safe_float(pred_df.loc[pred_df["is_error"] == 0, "confidence"].mean(), 4)
        if (pred_df["is_error"] == 0).any()
        else None,
        "avg_confidence_error": safe_float(pred_df.loc[pred_df["is_error"] == 1, "confidence"].mean(), 4)
        if (pred_df["is_error"] == 1).any()
        else None,
    }
    save_json(confidence_summary, reports_dir / "confidence_summary.json")

    save_json(final_test_metrics, run_dir / "metrics.json")
    save_json(report_dict, run_dir / "classification_report.json")
    save_json(cm, run_dir / "confusion_matrix.json")
    save_json(LABEL2ID, run_dir / "label2id.json")
    save_json(ID2LABEL, run_dir / "id2label.json")
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    plot_confusion_matrix_png(
        cm=cm,
        labels=[ID2LABEL[0], ID2LABEL[1]],
        out_path=charts_dir / f"{trial_name}_confusion_matrix.png",
        title=f"Confusion Matrix - {trial_name}",
    )
    plot_per_class_metrics(
        report_dict=report_dict,
        labels=[ID2LABEL[0], ID2LABEL[1]],
        out_path=charts_dir / f"{trial_name}_per_class_metrics.png",
        title=f"Per-class Metrics - {trial_name}",
    )
    plot_overall_metrics(
        metrics=final_test_metrics,
        out_path=charts_dir / f"{trial_name}_overall_test_metrics.png",
        title=f"Overall Test Metrics - {trial_name}",
    )

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    trial_summary = {
        "trial_name": trial_name,
        "display_name": display_name,
        "model_name": model_name,
        "run_dir": str(run_dir),
        "trial_config": deepcopy(trial_cfg),
        "selection_metric": safe_float(val_metrics_tuned["val_f1_macro"]),
        "best_threshold": best_threshold,
        "val_metrics": val_metrics_tuned,
        "test_metrics": final_test_metrics,
        "train_metrics": {k: safe_float(v) for k, v in train_metrics.items()},
        "best_val_metric_from_trainer": safe_float(trainer_state_summary.get("best_metric")),
        "confidence_summary": confidence_summary,
    }
    save_json(trial_summary, run_dir / "trial_summary.json")

    logger.info(
        "✅ Finished trial %s | val_f1_macro=%.4f | test_f1_macro=%.4f | threshold=%.2f",
        trial_name,
        val_metrics_tuned["val_f1_macro"],
        final_test_metrics["test_f1_macro"],
        best_threshold,
    )

    return {
        "trial_name": trial_name,
        "display_name": display_name,
        "model_name": model_name,
        "run_dir": run_dir,
        "metrics": final_test_metrics,
        "train_metrics": train_metrics,
        "report": report_dict,
        "trainer_state": trainer_state_summary,
        "history_df": history_df,
        "pred_df": pred_df,
        "aspect_error_df": aspect_error_df,
        "confidence_summary": confidence_summary,
        "selection_metric": val_metrics_tuned["val_f1_macro"],
        "best_threshold": best_threshold,
        "trial_config": deepcopy(trial_cfg),
        "val_metrics": val_metrics_tuned,
        "token_stats": token_stats,
    }


def run_model_search(
    batch_dir: Path,
    model_name: str,
    display_name: str,
    input_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trials: List[Dict],
    seed: int,
    disable_torch_compile: bool,
    threshold_grid: List[float],
) -> Tuple[dict, pd.DataFrame]:
    # // CHANGED: chạy search cho từng model, chọn best trial theo val_f1_macro đã tune threshold
    model_root = batch_dir / "models" / slugify(display_name)
    trials_root = model_root / "trials"
    tuning_root = model_root / "tuning"
    trials_root.mkdir(parents=True, exist_ok=True)
    tuning_root.mkdir(parents=True, exist_ok=True)

    run_infos = []
    summary_rows = []

    for trial_idx, trial_cfg in enumerate(trials, start=1):
        info = run_single_trial(
            trials_root=trials_root,
            model_name=model_name,
            display_name=display_name,
            input_col=input_col,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            trial_cfg=trial_cfg,
            trial_idx=trial_idx,
            seed=seed,
            disable_torch_compile=disable_torch_compile,
            threshold_grid=threshold_grid,
        )
        run_infos.append(info)

        summary_rows.append(
            {
                "trial_name": info["trial_name"],
                "display_name": info["display_name"],
                "model_name": info["model_name"],
                "run_dir": str(info["run_dir"]),
                "best_threshold": info["best_threshold"],
                "val_f1_macro": info["val_metrics"]["val_f1_macro"],
                "val_f1_weighted": info["val_metrics"]["val_f1_weighted"],
                "val_accuracy": info["val_metrics"]["val_accuracy"],
                "test_f1_macro": info["metrics"]["test_f1_macro"],
                "test_f1_weighted": info["metrics"]["test_f1_weighted"],
                "test_accuracy": info["metrics"]["test_accuracy"],
                "train_runtime": safe_float(info["train_metrics"].get("train_runtime")),
                "train_samples_per_second": safe_float(info["train_metrics"].get("train_samples_per_second")),
                "config_json": json.dumps(info["trial_config"], ensure_ascii=False),
            }
        )

    tuning_df = pd.DataFrame(summary_rows).sort_values(
        ["val_f1_macro", "val_f1_weighted", "val_accuracy", "train_runtime"],
        ascending=[False, False, False, True],
    )
    tuning_df.to_csv(tuning_root / "tuning_results.csv", index=False, encoding="utf-8-sig")
    save_json(tuning_df.to_dict(orient="records"), tuning_root / "tuning_results.json")

    plot_tuning_results_bar(
        tuning_df,
        tuning_root / "tuning_leaderboard_val_f1.png",
        f"Tuning Leaderboard - {display_name}",
        score_col="val_f1_macro",
    )

    best_info = run_infos[tuning_df.index[0]]
    best_trial_summary = {
        "display_name": display_name,
        "model_name": model_name,
        "best_trial_name": best_info["trial_name"],
        "best_run_dir": str(best_info["run_dir"]),
        "best_config": best_info["trial_config"],
        "best_threshold": best_info["best_threshold"],
        "selection_metric": best_info["selection_metric"],
        "val_metrics": best_info["val_metrics"],
        "test_metrics": best_info["metrics"],
        "train_metrics": {k: safe_float(v) for k, v in best_info["train_metrics"].items()},
        "created_at": now_utc_iso(),
    }
    save_json(best_trial_summary, tuning_root / "best_trial.json")

    return best_info, tuning_df


# =========================================================
# BATCH COMPARISON
# =========================================================
def compare_best_runs(batch_dir: Path, run_infos: List[dict]):
    if len(run_infos) == 0:
        return

    comparison_dir = batch_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    if len(run_infos) >= 2:
        plot_compare_overall(
            run_infos,
            comparison_dir / "compare_overall_metrics.png",
            "Model Comparison - Overall Metrics",
        )
        plot_compare_best_val_f1(
            run_infos,
            comparison_dir / "compare_best_val_f1.png",
            "Model Comparison - Best Tuned Validation F1",
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
                "best_trial_name": info["trial_name"],
                "run_dir": str(info["run_dir"]),
                "best_val_f1": info["selection_metric"],
                "best_threshold": info["best_threshold"],
                "test_accuracy": info["metrics"].get("test_accuracy"),
                "test_precision_macro": info["metrics"].get("test_precision_macro"),
                "test_recall_macro": info["metrics"].get("test_recall_macro"),
                "test_f1_macro": info["metrics"].get("test_f1_macro"),
                "test_f1_weighted": info["metrics"].get("test_f1_weighted"),
                "train_runtime": info["train_metrics"].get("train_runtime"),
                "train_samples_per_second": info["train_metrics"].get("train_samples_per_second"),
                "avg_confidence_all": info["confidence_summary"].get("avg_confidence_all"),
                "config": info["trial_config"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["test_f1_macro", "best_val_f1", "train_runtime"],
        ascending=[False, False, True],
    )
    summary_df.to_csv(comparison_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    save_json(summary_rows, comparison_dir / "model_comparison.json")

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
        best_row = summary_df.iloc[0].to_dict()
        save_json(best_row, comparison_dir / "best_model_overall.json")


def update_best_registry(input_col: str, best_run_infos: List[dict]):
    # // CHANGED: lưu best config per model để dùng lại ở lần sau
    registry = load_best_registry()
    for info in best_run_infos:
        key = registry_key(input_col=input_col, model_name=info["model_name"])
        registry[key] = {
            "input_collection": input_col,
            "display_name": info["display_name"],
            "model_name": info["model_name"],
            "best_trial_name": info["trial_name"],
            "best_run_dir": str(info["run_dir"]),
            "best_config": info["trial_config"],
            "best_threshold": info["best_threshold"],
            "best_val_f1_macro": info["selection_metric"],
            "test_f1_macro": info["metrics"].get("test_f1_macro"),
            "updated_at": now_utc_iso(),
        }
    save_best_registry(registry)


# =========================================================
# MAIN BATCH RUN
# =========================================================
def run_train_batch(
    models: List[str],
    input_col: str = DEFAULT_BINARY_COLLECTION,
    seed: int = 42,
    disable_torch_compile: bool = False,
    search_space: Optional[Dict] = None,
    max_trials_per_model: Optional[int] = None,
    use_registered_best: bool = False,
    threshold_start: float = 0.30,
    threshold_end: float = 0.70,
    threshold_step: float = 0.02,
):
    s = load_settings()
    set_seed(seed)
    init_torch_speedups()

    df = prepare_dataframe_from_mongo(input_col)
    train_df, val_df, test_df = split_dataframe(df, seed=seed)

    logger.info("✅ Shared split sizes: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))

    batch_name = build_batch_name(models)
    batch_dir = Path(s.artifact_dir) / "model_batches" / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    threshold_grid = np.round(np.arange(threshold_start, threshold_end + 1e-9, threshold_step), 4).tolist()

    batch_config = {
        "batch_name": batch_name,
        "input_collection": input_col,
        "models": models,
        "seed": seed,
        "disable_torch_compile": disable_torch_compile,
        "search_space": search_space,
        "max_trials_per_model": max_trials_per_model,
        "use_registered_best": use_registered_best,
        "threshold_grid": threshold_grid,
        "created_at": now_utc_iso(),
        "shared_split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    save_json(batch_config, batch_dir / "batch_config.json")

    registry = load_best_registry()
    best_run_infos = []
    tuning_summary_rows = []

    for model_name in models:
        display_name = MODEL_ALIASES.get(model_name, model_name)

        if use_registered_best:
            key = registry_key(input_col=input_col, model_name=model_name)
            reg = registry.get(key)
            if reg and reg.get("best_config"):
                trials = [reg["best_config"]]
                logger.info("♻️ Using registered best config for %s: %s", display_name, reg["best_config"])
            else:
                trials = expand_search_space(search_space, seed=seed, max_trials_per_model=max_trials_per_model)
                logger.warning("Không tìm thấy registry cho %s -> fallback sang search.", display_name)
        else:
            trials = expand_search_space(search_space, seed=seed, max_trials_per_model=max_trials_per_model)

        best_info, tuning_df = run_model_search(
            batch_dir=batch_dir,
            model_name=model_name,
            display_name=display_name,
            input_col=input_col,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            trials=trials,
            seed=seed,
            disable_torch_compile=disable_torch_compile,
            threshold_grid=threshold_grid,
        )

        best_run_infos.append(best_info)
        tuning_summary_rows.append(
            {
                "display_name": best_info["display_name"],
                "model_name": best_info["model_name"],
                "best_trial_name": best_info["trial_name"],
                "best_val_f1_macro": best_info["selection_metric"],
                "best_threshold": best_info["best_threshold"],
                "test_f1_macro": best_info["metrics"]["test_f1_macro"],
                "n_trials": int(len(tuning_df)),
                "best_config_json": json.dumps(best_info["trial_config"], ensure_ascii=False),
            }
        )

    compare_best_runs(batch_dir, best_run_infos)
    update_best_registry(input_col=input_col, best_run_infos=best_run_infos)

    batch_summary = []
    for info in best_run_infos:
        batch_summary.append(
            {
                "display_name": info["display_name"],
                "model_name": info["model_name"],
                "best_trial_name": info["trial_name"],
                "run_dir": str(info["run_dir"]),
                "best_val_f1_macro": info["selection_metric"],
                "best_threshold": info["best_threshold"],
                "test_accuracy": info["metrics"].get("test_accuracy"),
                "test_f1_macro": info["metrics"].get("test_f1_macro"),
                "test_f1_weighted": info["metrics"].get("test_f1_weighted"),
                "train_runtime": info["train_metrics"].get("train_runtime"),
                "train_samples_per_second": info["train_metrics"].get("train_samples_per_second"),
                "avg_confidence_all": info["confidence_summary"].get("avg_confidence_all"),
                "best_config": info["trial_config"],
            }
        )

    save_json(batch_summary, batch_dir / "batch_summary.json")
    save_json(tuning_summary_rows, batch_dir / "tuning_summary.json")

    logger.info("============================================================")
    logger.info("✅ Finished batch tuning + training.")
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
        help="Danh sách model sẽ chạy nối tiếp.",
    )
    ap.add_argument("--input-col", type=str, default=DEFAULT_BINARY_COLLECTION)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-torch-compile", action="store_true")

    # // CHANGED: search space CLI
    ap.add_argument("--search-max-lens", type=str, default=None, help='VD: "64,80"')
    ap.add_argument("--search-lrs", type=str, default=None, help='VD: "1e-5,2e-5,3e-5"')
    ap.add_argument("--search-epochs", type=str, default=None, help='VD: "3,5"')
    ap.add_argument("--search-train-batch-sizes", type=str, default=None, help='VD: "8,16"')
    ap.add_argument("--search-eval-batch-sizes", type=str, default=None, help='VD: "16,32"')
    ap.add_argument("--search-grad-accums", type=str, default=None, help='VD: "1,2"')
    ap.add_argument("--search-warmup-ratios", type=str, default=None, help='VD: "0.06,0.1"')
    ap.add_argument("--search-early-stopping-patiences", type=str, default=None, help='VD: "2,3"')
    ap.add_argument("--search-class-weight-modes", type=str, default=None, help='VD: "none,sqrt_inv"')
    ap.add_argument("--search-label-smoothings", type=str, default=None, help='VD: "0.0,0.03,0.05"')

    ap.add_argument("--max-trials-per-model", type=int, default=None)
    ap.add_argument("--use-registered-best", action="store_true")

    # // CHANGED: threshold tuning
    ap.add_argument("--threshold-start", type=float, default=0.30)
    ap.add_argument("--threshold-end", type=float, default=0.70)
    ap.add_argument("--threshold-step", type=float, default=0.02)

    args = ap.parse_args()

    if not args.run:
        print("Use: python -m review_analytics.pipeline.train_binary --run")
        return

    try:
        search_space = build_search_space_from_args(args)

        run_train_batch(
            models=args.models,
            input_col=args.input_col,
            seed=args.seed,
            disable_torch_compile=args.disable_torch_compile,
            search_space=search_space,
            max_trials_per_model=args.max_trials_per_model,
            use_registered_best=args.use_registered_best,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.threshold_step,
        )
    except Exception as e:
        logger.error("Train binary tuning failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()