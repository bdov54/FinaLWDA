import os, sys, glob, hashlib
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from review_analytics.logging import logging
from review_analytics.exception.exception import ReviewAnalyticsException


def make_uid(row: dict) -> str:
    # Ưu tiên reviewUrl vì thường là unique
    base = str(row.get("reviewUrl") or row.get("url") or "")
    if base.strip():
        return hashlib.sha1(base.strip().encode("utf-8")).hexdigest()

    # fallback nếu thiếu reviewUrl
    base = f"{row.get('stars')}|{row.get('name')}|{row.get('text')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def latest_csv(raw_dir="TSN_Data/raw_data"):
    files = sorted(glob.glob(os.path.join(raw_dir, "raw_data.csv")))
    if not files:
        raise FileNotFoundError("Không tìm thấy raw_data.csv trong TSN_Data/raw_data/")
    return files[-1]


def main():
    try:
        load_dotenv()
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB")
        col_name = os.getenv("MONGODB_RAW_COLLECTION", "raw_data")

        if not uri or not db_name:
            raise ValueError("Thiếu MONGODB_URI hoặc MONGODB_DB trong .env")

        csv_path = latest_csv()
        logging.logging.info("RAW insert from CSV: %s", csv_path)

        df = pd.read_csv(csv_path)
        records = df.to_dict("records")

        client = MongoClient(uri)
        db = client[db_name]
        col = db[col_name]

        # unique index để chạy lại không bị nhân bản
        col.create_index("review_uid", unique=True)

        upserted = 0
        for r in records:
            r["review_uid"] = make_uid(r)
            r["_ingested_at"] = datetime.utcnow()

            col.update_one(
                {"review_uid": r["review_uid"]},
                {"$set": r},
                upsert=True
            )
            upserted += 1

        logging.logging.info("Upserted %d docs into %s.%s", upserted, db_name, col_name)
        print(f"Upserted {upserted} docs into {db_name}.{col_name}")

    except Exception as e:
        logging.logging.error("push_raw_to_mongo failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()
