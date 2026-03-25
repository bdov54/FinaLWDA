import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pymongo import ASCENDING, UpdateOne
from pymongo.errors import OperationFailure

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException


DEFAULT_PRED_COL = "sentence_predictions_binary_123vs45_v1"
DEFAULT_OUTPUT_COL = "comment_summary_binary_123vs45_v1"


# =========================================================
# BASIC UTILS
# =========================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_mongo_safe(obj: Any) -> Any:
    """
    # // CHANGED:
    Convert toàn bộ numpy/pandas scalar về Python native type
    để pymongo encode được.
    """
    if isinstance(obj, dict):
        return {str(k): to_mongo_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_mongo_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_mongo_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [to_mongo_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()

    # pandas NA / numpy nan
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def pick_first(row: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in row:
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            return v
    return default


def normalize_review_id(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def normalize_label(v: Any) -> Optional[str]:
    if v is None:
        return None

    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    s = str(v).strip().lower()

    if s in {"positive", "pos", "1", "true", "yes"}:
        return "positive"
    if s in {"negative", "neg", "0", "false", "no"}:
        return "negative"

    return None


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return None


# =========================================================
# INDEX HELPERS
# =========================================================
def safe_create_index(collection, keys, unique: bool = False, name: Optional[str] = None):
    """
    # // CHANGED:
    Không tạo index mù quáng nữa.
    Nếu collection đã có index cùng keys nhưng tên khác,
    chỉ log warning rồi bỏ qua để tránh IndexOptionsConflict.
    """
    desired_keys = list(keys)

    try:
        existing = list(collection.list_indexes())
    except Exception as e:
        logger.warning("Không đọc được danh sách index của %s: %s", collection.name, str(e))
        existing = []

    for idx in existing:
        existing_keys = list(idx["key"].items())
        if existing_keys == desired_keys:
            existing_name = idx.get("name")
            existing_unique = bool(idx.get("unique", False))

            if existing_unique == unique:
                logger.info(
                    "Index đã tồn tại trên %s: name=%s keys=%s unique=%s",
                    collection.name,
                    existing_name,
                    desired_keys,
                    existing_unique,
                )
                return existing_name

            logger.warning(
                "Collection %s đã có index cùng keys=%s nhưng unique khác "
                "(existing unique=%s, requested unique=%s). Giữ nguyên index cũ.",
                collection.name,
                desired_keys,
                existing_unique,
                unique,
            )
            return existing_name

    try:
        created_name = collection.create_index(keys, unique=unique, name=name, background=True)
        logger.info(
            "Đã tạo index trên %s: name=%s keys=%s unique=%s",
            collection.name,
            created_name,
            desired_keys,
            unique,
        )
        return created_name
    except OperationFailure as e:
        msg = str(e)
        if e.code == 85 or "already exists with a different name" in msg.lower():
            logger.warning(
                "Bỏ qua tạo index trên %s vì index tương đương có thể đã tồn tại: %s",
                collection.name,
                msg,
            )
            return name or "existing_equivalent_index"
        raise


# =========================================================
# PROB / CONFIDENCE HELPERS
# =========================================================
def infer_prob_positive(row: Dict[str, Any]) -> Optional[float]:
    keys = [
        "prob_positive",
        "positive_prob",
        "score_positive",
        "pred_prob_positive",
        "y_prob_positive",
    ]
    v = pick_first(row, keys, default=None)
    p = safe_float(v)
    if p is None:
        return None
    if p < 0:
        return 0.0
    if p > 1:
        return 1.0
    return p


def infer_prob_negative(row: Dict[str, Any]) -> Optional[float]:
    keys = [
        "prob_negative",
        "negative_prob",
        "score_negative",
        "pred_prob_negative",
        "y_prob_negative",
    ]
    v = pick_first(row, keys, default=None)
    p = safe_float(v)
    if p is None:
        return None
    if p < 0:
        return 0.0
    if p > 1:
        return 1.0
    return p


def infer_confidence(row: Dict[str, Any], prob_pos: Optional[float], prob_neg: Optional[float]) -> Optional[float]:
    v = pick_first(row, ["confidence", "pred_confidence", "score", "max_prob"], default=None)
    c = safe_float(v)
    if c is not None:
        if c < 0:
            return 0.0
        if c > 1:
            return 1.0
        return c

    if prob_pos is not None and prob_neg is not None:
        return max(prob_pos, prob_neg)

    if prob_pos is not None:
        return max(prob_pos, 1.0 - prob_pos)

    if prob_neg is not None:
        return max(prob_neg, 1.0 - prob_neg)

    return None


def infer_label(row: Dict[str, Any], prob_pos: Optional[float], prob_neg: Optional[float]) -> Optional[str]:
    keys = [
        "pred_label",
        "predicted_label",
        "y_pred",
        "label_pred",
        "prediction",
        "sentiment_pred",
    ]
    raw = pick_first(row, keys, default=None)
    norm = normalize_label(raw)
    if norm is not None:
        return norm

    # fallback từ y_pred_id / pred_label_id
    pred_id = pick_first(row, ["pred_label_id", "y_pred_id", "prediction_id"], default=None)
    pred_id = safe_int(pred_id)
    if pred_id == 1:
        return "positive"
    if pred_id == 0:
        return "negative"

    # fallback từ probability
    if prob_pos is not None:
        return "positive" if prob_pos >= 0.5 else "negative"
    if prob_neg is not None:
        return "negative" if prob_neg >= 0.5 else "positive"

    return None


# =========================================================
# SUMMARY BUILDERS
# =========================================================
def build_sentence_items(group_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []

    sort_cols = [c for c in ["sentence_index", "sent_idx", "idx"] if c in group_df.columns]
    if sort_cols:
        group_df = group_df.sort_values(sort_cols, kind="stable")
    else:
        group_df = group_df.reset_index(drop=True)

    for _, r in group_df.iterrows():
        row = r.to_dict()

        prob_pos = infer_prob_positive(row)
        prob_neg = infer_prob_negative(row)

        if prob_pos is None and prob_neg is not None:
            prob_pos = 1.0 - prob_neg
        if prob_neg is None and prob_pos is not None:
            prob_neg = 1.0 - prob_pos

        label = infer_label(row, prob_pos, prob_neg)
        conf = infer_confidence(row, prob_pos, prob_neg)

        aspect = normalize_text(pick_first(row, ["aspect"], default=""))
        aspect_text = normalize_text(pick_first(row, ["aspect_text", "aspect"], default=""))
        sentence = normalize_text(pick_first(row, ["sentence"], default=""))

        item = {
            "sentence": sentence,
            "aspect": aspect,
            "aspect_text": aspect_text,
            "pred_label": label,
            "predicted_label": label,  # alias
            "confidence": conf,
            "prob_positive": prob_pos,
            "prob_negative": prob_neg,
            "sentence_index": safe_int(pick_first(row, ["sentence_index", "sent_idx", "idx"], default=None)),
            "true_label": normalize_label(pick_first(row, ["label", "true_label", "y_true"], default=None)),
            "true_label_id": safe_int(pick_first(row, ["label_id", "y_true_id"], default=None)),
            "pred_label_id": safe_int(pick_first(row, ["pred_label_id", "y_pred_id"], default=None)),
        }

        rows.append(to_mongo_safe(item))

    return rows


def build_aspect_summaries(sentence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[str, List[Dict[str, Any]]] = {}

    for item in sentence_items:
        key = item.get("aspect_text") or item.get("aspect") or "unknown"
        bucket.setdefault(key, []).append(item)

    out = []

    for aspect_text, items in bucket.items():
        pos_count = sum(1 for x in items if x.get("pred_label") == "positive")
        neg_count = sum(1 for x in items if x.get("pred_label") == "negative")

        prob_pos_vals = [x.get("prob_positive") for x in items if x.get("prob_positive") is not None]
        prob_neg_vals = [x.get("prob_negative") for x in items if x.get("prob_negative") is not None]
        conf_vals = [x.get("confidence") for x in items if x.get("confidence") is not None]

        avg_prob_pos = float(np.mean(prob_pos_vals)) if prob_pos_vals else None
        avg_prob_neg = float(np.mean(prob_neg_vals)) if prob_neg_vals else None
        avg_conf = float(np.mean(conf_vals)) if conf_vals else None

        if avg_prob_pos is not None:
            pred_label = "positive" if avg_prob_pos >= 0.5 else "negative"
        else:
            pred_label = "positive" if pos_count >= neg_count else "negative"

        aspect_value = None
        for x in items:
            if x.get("aspect"):
                aspect_value = x.get("aspect")
                break

        out.append(
            to_mongo_safe(
                {
                    "aspect": aspect_value or aspect_text,
                    "aspect_text": aspect_text,
                    "n_sentences": int(len(items)),
                    "positive_count": int(pos_count),
                    "negative_count": int(neg_count),
                    "pred_label": pred_label,
                    "predicted_label": pred_label,  # alias
                    "avg_prob_positive": avg_prob_pos,
                    "avg_prob_negative": avg_prob_neg,
                    "avg_confidence": avg_conf,
                }
            )
        )

    out = sorted(out, key=lambda x: (x.get("aspect_text") or ""))
    return out


def build_comment_summary_doc(review_id: str, group_df: pd.DataFrame, pred_col: str) -> Dict[str, Any]:
    first_row = group_df.iloc[0].to_dict()

    sentence_items = build_sentence_items(group_df)
    aspect_summaries = build_aspect_summaries(sentence_items)

    pos_count = sum(1 for x in sentence_items if x.get("pred_label") == "positive")
    neg_count = sum(1 for x in sentence_items if x.get("pred_label") == "negative")

    prob_pos_vals = [x.get("prob_positive") for x in sentence_items if x.get("prob_positive") is not None]
    prob_neg_vals = [x.get("prob_negative") for x in sentence_items if x.get("prob_negative") is not None]
    conf_vals = [x.get("confidence") for x in sentence_items if x.get("confidence") is not None]

    overall_prob_pos = float(np.mean(prob_pos_vals)) if prob_pos_vals else None
    overall_prob_neg = float(np.mean(prob_neg_vals)) if prob_neg_vals else None
    overall_conf = float(np.mean(conf_vals)) if conf_vals else None

    threshold = safe_float(
        pick_first(
            first_row,
            ["threshold", "best_threshold", "decision_threshold", "serving_threshold"],
            default=0.5,
        )
    )
    if threshold is None:
        threshold = 0.5

    if overall_prob_pos is not None:
        predicted_label = "positive" if overall_prob_pos >= threshold else "negative"
    else:
        predicted_label = "positive" if pos_count >= neg_count else "negative"

    comment_text = pick_first(
        first_row,
        ["comment_text", "review_text", "review", "comment", "text", "content"],
        default="",
    )
    comment_text = normalize_text(comment_text)

    rating = safe_float(pick_first(first_row, ["rating", "stars", "score"], default=None))
    source = pick_first(first_row, ["source"], default=None)
    model_name = pick_first(first_row, ["model_name"], default=None)
    model_display_name = pick_first(first_row, ["display_name", "model_display_name"], default=None)
    input_collection = pick_first(first_row, ["input_collection"], default=None)

    doc = {
        "review_id": review_id,
        "comment_text": comment_text,
        "review_text": comment_text,  # alias
        "rating": rating,
        "source": source,
        "pred_collection": pred_col,
        "input_collection": input_collection,
        "model_name": model_name,
        "model_display_name": model_display_name,
        "n_sentences": int(len(sentence_items)),
        "n_aspects": int(len(aspect_summaries)),
        "positive_count": int(pos_count),
        "negative_count": int(neg_count),
        "pred_label": predicted_label,
        "predicted_label": predicted_label,  # alias
        "overall_pred_label": predicted_label,  # alias
        "prob_positive": overall_prob_pos,
        "prob_negative": overall_prob_neg,
        "confidence": overall_conf,
        "threshold": float(threshold),
        "aspect_summaries": aspect_summaries,
        "sentence_predictions": sentence_items,
        "updated_at": now_utc_iso(),
    }

    return to_mongo_safe(doc)


# =========================================================
# MAIN LOGIC
# =========================================================
def run_build_summary(
    pred_col: str,
    output_col: str,
    refresh_existing: bool = False,
    batch_size: int = 500,
):
    s = load_settings()
    store = MongoStore(s.mongodb_uri, s.mongodb_db)

    pred_collection = store.collection(pred_col)
    out_collection = store.collection(output_col)

    # // CHANGED:
    # Không gọi store.ensure_indexes(... processed_binary_col=...)
    # vì chỗ đó từng gây conflict tên index giữa các collection.
    safe_create_index(
        pred_collection,
        [("review_id", ASCENDING)],
        unique=False,
        name=f"ix_review_id_{pred_col}",
    )
    safe_create_index(
        out_collection,
        [("review_id", ASCENDING)],
        unique=True,
        name=f"uq_review_id_{output_col}",
    )

    docs = list(pred_collection.find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError(f"Collection '{pred_col}' đang rỗng.")

    df = pd.DataFrame(docs)
    if "review_id" not in df.columns:
        raise ValueError(f"Collection '{pred_col}' thiếu cột 'review_id'.")

    df["review_id"] = df["review_id"].apply(normalize_review_id)
    df = df[df["review_id"] != ""].copy()

    if df.empty:
        raise RuntimeError(f"Collection '{pred_col}' không có review_id hợp lệ.")

    if not refresh_existing:
        existing_ids = set(out_collection.distinct("review_id"))
        if existing_ids:
            before = len(df)
            df = df[~df["review_id"].isin(existing_ids)].copy()
            logger.info(
                "Skip %d review_id đã tồn tại trong %s vì refresh_existing=False",
                before - len(df),
                output_col,
            )

    if df.empty:
        logger.info("Không còn dữ liệu mới để build summary.")
        return

    logger.info(
        "Bắt đầu build comment summary: pred_col=%s | output_col=%s | n_rows=%d | n_reviews=%d",
        pred_col,
        output_col,
        len(df),
        df["review_id"].nunique(),
    )

    ops: List[UpdateOne] = []
    n_written = 0

    grouped = df.groupby("review_id", sort=True, dropna=False)

    for i, (review_id, group_df) in enumerate(grouped, start=1):
        summary_doc = build_comment_summary_doc(review_id=review_id, group_df=group_df, pred_col=pred_col)
        summary_doc = to_mongo_safe(summary_doc)

        filter_doc = {"review_id": review_id}
        filter_doc = to_mongo_safe(filter_doc)

        set_doc = dict(summary_doc)
        created_at = now_utc_iso()

        ops.append(
            UpdateOne(
                filter_doc,
                {
                    "$set": set_doc,
                    "$setOnInsert": {"created_at": created_at},
                },
                upsert=True,
            )
        )

        if len(ops) >= batch_size:
            out_collection.bulk_write(ops, ordered=False)
            n_written += len(ops)
            logger.info("Đã ghi %d summary docs vào %s", n_written, output_col)
            ops = []

        if i % 500 == 0:
            logger.info("Đã xử lý %d review_id", i)

    if ops:
        out_collection.bulk_write(ops, ordered=False)
        n_written += len(ops)

    logger.info(
        "✅ Build comment summary hoàn tất | output_col=%s | total_written=%d",
        output_col,
        n_written,
    )


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-col", type=str, default=DEFAULT_PRED_COL)
    ap.add_argument("--output-col", type=str, default=DEFAULT_OUTPUT_COL)
    ap.add_argument("--refresh-existing", action="store_true")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    try:
        run_build_summary(
            pred_col=args.pred_col,
            output_col=args.output_col,
            refresh_existing=args.refresh_existing,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error("Build comment summary binary failed: %s", str(e), exc_info=True)
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()