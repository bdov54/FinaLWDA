import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pymongo import UpdateOne
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException


DEFAULT_INPUT_COLLECTION = "processed_data_binary_123vs45_v1"
DEFAULT_OUTPUT_COLLECTION = "sentence_predictions_binary_123vs45_v1"
ACTIVE_SERVING_REL_PATH = Path("model_registry") / "active_binary_model.json"


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc():
    return datetime.now(timezone.utc)


def get_active_manifest_path() -> Path:
    s = load_settings()
    return Path(s.artifact_dir) / ACTIVE_SERVING_REL_PATH


def load_active_manifest() -> Dict:
    path = get_active_manifest_path()
    manifest = load_json(path, {})
    if not manifest:
        raise FileNotFoundError(f"Không tìm thấy active manifest: {path}")
    return manifest


def load_model_and_tokenizer(manifest: Dict):
    model_dir = manifest["model_dir"]
    tokenizer_dir = manifest["tokenizer_dir"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def batched_inference(
    model,
    tokenizer,
    device: str,
    aspect_texts: List[str],
    sentences: List[str],
    max_len: int,
):
    enc = tokenizer(
        aspect_texts,
        sentences,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()

    return probs


def run_materialize(
    input_col: str = DEFAULT_INPUT_COLLECTION,
    output_col: str = DEFAULT_OUTPUT_COLLECTION,
    batch_size: int = 64,
    refresh_existing: bool = False,
):
    s = load_settings()

    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    store.ensure_indexes(
        s.col_raw,
        s.col_processed,
        s.col_raw_web,
        processed_binary_col=input_col,
    )

    in_collection = store.collection(input_col)
    out_collection = store.collection(output_col)

    out_collection.create_index(
        [("review_id", 1), ("sent_id", 1), ("aspect", 1)],
        unique=True,
        name="uq_review_sent_aspect_pred_binary",
    )
    out_collection.create_index([("review_id", 1)], name="idx_review_id_pred_binary")
    out_collection.create_index([("label", 1)], name="idx_label_pred_binary")
    out_collection.create_index([("updated_at", -1)], name="idx_updated_at_pred_binary")

    manifest = load_active_manifest()
    tokenizer, model, device = load_model_and_tokenizer(manifest)

    best_threshold = float(manifest.get("best_threshold", 0.5))
    max_len = int(manifest.get("max_len", 128))

    docs = list(
        in_collection.find(
            {},
            {
                "_id": 0,
                "review_id": 1,
                "lang": 1,
                "sent_id": 1,
                "sentence": 1,
                "aspect": 1,
                "aspect_text": 1,
                "hit_keywords": 1,
                "label_id": 1,
                "label": 1,
                "review_rating": 1,
                "review_created_at": 1,
            },
        )
    )
    if not docs:
        raise RuntimeError(f"Collection input rỗng: {input_col}")

    df = pd.DataFrame(docs)
    required = ["review_id", "sent_id", "sentence", "aspect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong input collection: {missing}")

    if "aspect_text" not in df.columns:
        df["aspect_text"] = df["aspect"].astype(str).str.replace("_", " ", regex=False)

    df["review_id"] = df["review_id"].astype(str)
    df["sent_id"] = df["sent_id"].astype(int)
    df["sentence"] = df["sentence"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    df["aspect_text"] = df["aspect_text"].astype(str)

    if not refresh_existing:
        existing_keys = set(
            (
                str(x["review_id"]),
                int(x["sent_id"]),
                str(x["aspect"]),
            )
            for x in out_collection.find(
                {},
                {"_id": 0, "review_id": 1, "sent_id": 1, "aspect": 1},
            )
        )
        mask = df.apply(
            lambda r: (r["review_id"], int(r["sent_id"]), r["aspect"]) not in existing_keys,
            axis=1,
        )
        df = df[mask].reset_index(drop=True)

    if df.empty:
        logger.info("✅ Không có dòng mới cần materialize.")
        return

    logger.info("🚀 Materializing sentence predictions...")
    logger.info("Input rows to infer: %d", len(df))
    logger.info("Serving model: %s", manifest.get("display_name"))
    logger.info("Threshold: %.4f", best_threshold)
    logger.info("Device: %s", device)

    ops = []
    total_batches = math.ceil(len(df) / batch_size)
    now = now_utc()

    for i in range(total_batches):
        chunk = df.iloc[i * batch_size : (i + 1) * batch_size].copy()

        probs = batched_inference(
            model=model,
            tokenizer=tokenizer,
            device=device,
            aspect_texts=chunk["aspect_text"].tolist(),
            sentences=chunk["sentence"].tolist(),
            max_len=max_len,
        )

        prob_negative = probs[:, 0]
        prob_positive = probs[:, 1]
        confidence = probs.max(axis=1)
        pred_label = np.where(prob_positive >= best_threshold, "positive", "negative")
        pred_label_id = np.where(prob_positive >= best_threshold, 1, 0)

        chunk["prob_negative"] = prob_negative
        chunk["prob_positive"] = prob_positive
        chunk["confidence"] = confidence
        chunk["pred_label"] = pred_label
        chunk["pred_label_id"] = pred_label_id

        for row in chunk.to_dict(orient="records"):
            key = {
                "review_id": str(row["review_id"]),
                "sent_id": int(row["sent_id"]),
                "aspect": str(row["aspect"]),
            }

            doc = {
                "review_id": str(row["review_id"]),
                "lang": row.get("lang"),
                "sent_id": int(row["sent_id"]),
                "sentence": row["sentence"],
                "aspect": row["aspect"],
                "aspect_text": row["aspect_text"],
                "hit_keywords": row.get("hit_keywords"),
                "true_label": row.get("label"),
                "true_label_id": row.get("label_id"),
                "review_rating": row.get("review_rating"),
                "review_created_at": row.get("review_created_at"),
                "label": row["pred_label"],
                "label_id": int(row["pred_label_id"]),
                "prob_negative": float(row["prob_negative"]),
                "prob_positive": float(row["prob_positive"]),
                "confidence": float(row["confidence"]),
                "threshold_used": best_threshold,
                "serving_model_name": manifest.get("model_name"),
                "serving_display_name": manifest.get("display_name"),
                "serving_run_dir": manifest.get("run_dir"),
                "updated_at": now,
            }

            ops.append(
                UpdateOne(
                    key,
                    {
                        "$set": doc,
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )

        if len(ops) >= 1000:
            out_collection.bulk_write(ops, ordered=False)
            ops = []

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            logger.info("Processed batch %d/%d", i + 1, total_batches)

    if ops:
        out_collection.bulk_write(ops, ordered=False)

    logger.info("✅ Materialize sentence predictions done. Output collection: %s", output_col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-col", type=str, default=DEFAULT_INPUT_COLLECTION)
    ap.add_argument("--output-col", type=str, default=DEFAULT_OUTPUT_COLLECTION)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-existing", action="store_true")
    args = ap.parse_args()

    try:
        run_materialize(
            input_col=args.input_col,
            output_col=args.output_col,
            batch_size=args.batch_size,
            refresh_existing=args.refresh_existing,
        )
    except Exception as e:
        logger.error("Materialize sentence predictions binary failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()