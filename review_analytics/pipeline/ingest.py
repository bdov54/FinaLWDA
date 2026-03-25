import argparse
from datetime import datetime
import pandas as pd

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logging
from review_analytics.exception.exception import ReviewAnalyticsException

REQUIRED_RAW_COLS = ["title", "url", "stars", "name", "reviewUrl", "text"]

def ingest_local_csv():
    s = load_settings()

    if not s.raw_csv_path.exists():
        raise FileNotFoundError(f"Không thấy raw csv: {s.raw_csv_path}")

    df = pd.read_csv(s.raw_csv_path)
    missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV thiếu cột: {missing}. Hiện có: {df.columns.tolist()}")

    # chuẩn hoá cột
    df = df[REQUIRED_RAW_COLS].copy()
    df = df.rename(columns={"stars": "rating", "name": "author", "reviewUrl": "review_id"})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text"] = df["text"].astype("string").fillna("").str.strip()

    # metadata
    df["source"] = "google_maps_csv"

    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    store.ensure_indexes(s.col_raw, s.col_processed, s.col_raw_web)

    col = store.collection(s.col_raw)

    upserts = 0
    now = datetime.utcnow()

    for d in df.to_dict(orient="records"):
        rid = str(d.get("review_id", "")).strip()
        if not rid:
            continue

        # ✅ created_at chỉ set khi insert lần đầu, updated_at set mỗi lần chạy
        col.update_one(
            {"review_id": rid},
            {
                "$set": {**d, "updated_at": now},
                "$setOnInsert": {"created_at": now}
            },
            upsert=True
        )

        # ✅ backfill: nếu doc cũ đã tồn tại trước đây mà chưa có created_at
        col.update_one(
            {"review_id": rid, "created_at": {"$exists": False}},
            {"$set": {"created_at": now}}
        )

        upserts += 1

    logging.info("✅ Ingested raw docs: %d (upserted)", upserts)
    logging.info("✅ Raw collection: %s.%s", s.mongodb_db, s.col_raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Ingest RAW_CSV_PATH -> Mongo raw_data")
    args = ap.parse_args()

    if not args.run:
        print("Use: python -m review_analytics.pipeline.ingest --run")
        return

    try:
        ingest_local_csv()
    except Exception as e:
        logging.error("Ingest failed: %s", str(e))
        raise ReviewAnalyticsException(e)

if __name__ == "__main__":
    main()