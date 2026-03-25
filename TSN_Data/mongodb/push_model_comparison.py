import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from review_analytics.logging import logging
from review_analytics.exception.exception import ReviewAnalyticsException


def get_artifact_model_comparison() -> List[Dict[str, Any]]:
    """Load model comparison data từ artifacts"""
    artifact_dir = BASE_DIR / "artifacts" / "model_batches" / "b123v45_tuned_20260316_082735" / "comparison"
    comparison_file = artifact_dir / "model_comparison.json"

    if not comparison_file.exists():
        raise FileNotFoundError(f"Không tìm thấy {comparison_file}")

    try:
        with open(comparison_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"✓ Loaded model comparison from: {comparison_file}")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        raise ReviewAnalyticsException(f"Lỗi khi load model comparison: {e}")


def push_model_comparison():
    """Push model comparison data lên MongoDB"""
    try:
        load_dotenv()
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB")
        col_name = "model_comparisons"

        if not uri or not db_name:
            raise ValueError("Thiếu MONGODB_URI hoặc MONGODB_DB trong .env")

        # Load dữ liệu từ artifacts
        comparisons = get_artifact_model_comparison()
        logging.info(f"Tải được {len(comparisons)} model comparison từ artifacts")

        # Connect MongoDB
        client = MongoClient(uri)
        db = client[db_name]
        col = db[col_name]

        # Drop collection cũ để update lại
        col.drop()
        logging.info(f"✓ Dropped old collection: {col_name}")

        # Insert dữ liệu mới
        if comparisons:
            result = col.insert_many(comparisons)
            logging.info(f"✓ Inserted {len(result.inserted_ids)} documents vào {col_name}")
        else:
            logging.warning("Không có dữ liệu model comparison để push")

        client.close()
        logging.info("✓ Hoàn thành push model comparison lên MongoDB")

    except Exception as e:
        error_msg = f"Lỗi khi push model comparison: {e}"
        logging.error(error_msg)
        raise ReviewAnalyticsException(error_msg)


if __name__ == "__main__":
    push_model_comparison()
