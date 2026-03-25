import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException


label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def to_hf_dataset(df_: pd.DataFrame) -> Dataset:
    # chỉ lấy đúng 3 cột cần cho modeling
    return Dataset.from_pandas(df_[["aspect", "sentence", "label_id"]].copy())


def tokenize_pair(batch, tokenizer, max_length=256):
    # input = (aspect, sentence)
    return tokenizer(
        batch["aspect"],
        batch["sentence"],
        truncation=True,
        max_length=max_length,
        # không padding ở đây (để collator pad động)
    )


def build_training_args(**kwargs):
    """
    Compatible wrapper: một số version transformers dùng evaluation_strategy,
    một số doc bạn viết eval_strategy.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "eval_strategy" in msg and "unexpected keyword" in msg:
            kwargs = kwargs.copy()
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
            return TrainingArguments(**kwargs)
        raise


def run_train(model_name="xlm-roberta-base", max_len=256, epochs=5, lr=2e-5, weight_decay=0.01, seed=42):
    s = load_settings()
    set_seed(seed)

    # ===== load processed_data from Mongo =====
    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    store.ensure_indexes(s.col_raw, s.col_processed, s.col_raw_web)

    proc_col = store.collection(s.col_processed)
    docs = list(proc_col.find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError("Mongo processed_data đang rỗng. Chạy preprocess trước.")

    df = pd.DataFrame(docs)

    # ===== validate columns =====
    required = ["review_id", "sentence", "aspect", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"processed_data thiếu cột: {missing}. Hiện có: {df.columns.tolist()}")

    # ===== normalize =====
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(label2id)].copy()
    df["label_id"] = df["label"].map(label2id).astype(int)

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["aspect"] = df["aspect"].astype(str).str.strip()
    df["review_id"] = df["review_id"].astype(str)

    # drop empty
    df = df[(df["sentence"] != "") & (df["aspect"] != "")].copy()

    # ===== split (no leakage) =====
    groups = df["review_id"]  # ✅ group theo review_id

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, df["label_id"], groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["label_id"], groups=temp_df["review_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    logger.info("✅ Split sizes: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))
    logger.info("Train label dist: %s", train_df["label"].value_counts(normalize=True).round(3).to_dict())

    # ===== class weights from train =====
    counts = train_df["label_id"].value_counts().sort_index()
    for i in [0, 1, 2]:
        if i not in counts.index:
            counts.loc[i] = 0
    counts = counts.sort_index()

    total = counts.sum()
    class_weights = (total / (counts + 1e-9)).values
    class_weights = class_weights / class_weights.mean()

    logger.info("⚖️ counts: %s", counts.to_dict())
    logger.info("⚖️ class_weights: %s", {i: float(round(w, 3)) for i, w in enumerate(class_weights)})

    # ===== HF datasets =====
    train_ds_raw = to_hf_dataset(train_df)
    val_ds_raw = to_hf_dataset(val_df)
    test_ds_raw = to_hf_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = train_ds_raw.map(lambda b: tokenize_pair(b, tokenizer, max_len), batched=True)
    val_ds = val_ds_raw.map(lambda b: tokenize_pair(b, tokenizer, max_len), batched=True)
    test_ds = test_ds_raw.map(lambda b: tokenize_pair(b, tokenizer, max_len), batched=True)

    # rename label col for Trainer
    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # ===== output dir =====
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(s.artifact_dir) / "model" / f"{model_name.replace('/', '_')}_{run_id}"
    (out_dir / "trainer_out").mkdir(parents=True, exist_ok=True)

    # ===== training args =====
    args = build_training_args(
        output_dir=str(out_dir / "trainer_out"),
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        weight_decay=weight_decay,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # ===== save training config (for web later) =====
    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "max_len": max_len,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "class_weights": [float(x) for x in class_weights],
        "label2id": label2id,
        "id2label": id2label,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except Exception:
        pass

    (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("🚀 Training model=%s ...", model_name)
    trainer.train()

    # ===== evaluate =====
    metrics = trainer.evaluate(test_ds)
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    report = classification_report(
        y_true, y_pred, target_names=[id2label[i] for i in range(3)], digits=4
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # ===== save artifacts =====
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "id2label.json").write_text(json.dumps(id2label, indent=2, ensure_ascii=False), encoding="utf-8")

    # save model+tokenizer for web serving
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    logger.info("✅ Saved model artifacts at: %s", str(out_dir))
    logger.info("✅ Test metrics: %s", metrics)
    logger.info("\n%s", report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", type=str, default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.run:
        print("Use: python -m review_analytics.pipeline.train --run --model xlm-roberta-base")
        return

    try:
        run_train(
            model_name=args.model,
            max_len=args.max_len,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            seed=args.seed,
        )
    except Exception as e:
        logger.error("Train failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()