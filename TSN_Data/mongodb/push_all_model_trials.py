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


def get_all_model_trials() -> List[Dict[str, Any]]:
    """Load tất cả trial results từ artifacts (mBERT + XLM-R) và enrich với metrics"""
    artifact_dir = BASE_DIR / "artifacts" / "model_batches" / "b123v45_tuned_20260316_082735" / "models"
    
    all_trials = []
    
    for family in ["mBERT", "XLM-R"]:
        tuning_results_file = artifact_dir / family / "tuning" / "tuning_results.json"
        
        if not tuning_results_file.exists():
            logging.warning(f"Không tìm thấy {tuning_results_file}")
            continue
        
        try:
            with open(tuning_results_file, "r", encoding="utf-8") as f:
                trials = json.load(f)
            
            if isinstance(trials, list):
                # Parse config_json string to dict + load metrics từ trial directories
                for trial in trials:
                    if "config_json" in trial and isinstance(trial["config_json"], str):
                        try:
                            trial["config"] = json.loads(trial["config_json"])
                        except:
                            trial["config"] = {}
                    
                    # Load precision/recall metrics từ trial's metrics.json
                    if "trial_name" in trial:
                        trial_metrics_path = artifact_dir / family / "trials" / trial["trial_name"] / "metrics.json"
                        if trial_metrics_path.exists():
                            try:
                                with open(trial_metrics_path, "r", encoding="utf-8") as f:
                                    metrics = json.load(f)
                                # Add precision and recall from detailed metrics
                                for key in ["test_precision_macro", "test_recall_macro"]:
                                    if key in metrics:
                                        trial[key] = metrics[key]
                                logging.debug(f"  - Loaded metrics for {trial['trial_name']}")
                            except Exception as e:
                                logging.warning(f"  - Lỗi load metrics cho {trial['trial_name']}: {e}")
                
                all_trials.extend(trials)
                logging.info(f"✓ Loaded {len(trials)} trials từ {family}")
        except Exception as e:
            logging.error(f"Lỗi khi load {tuning_results_file}: {e}")
    
    return all_trials


def push_all_model_trials():
    """Push tất cả model trials lên MongoDB"""
    try:
        load_dotenv()
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB")
        col_name = "model_trials"

        if not uri or not db_name:
            raise ValueError("Thiếu MONGODB_URI hoặc MONGODB_DB trong .env")

        # Load dữ liệu từ artifacts
        trials = get_all_model_trials()
        logging.info(f"Tải được {len(trials)} trials từ artifacts")

        if not trials:
            raise ValueError("Không tìm thấy trials data")

        # Connect MongoDB
        client = MongoClient(uri)
        db = client[db_name]
        col = db[col_name]

        # Drop collection cũ để update lại
        col.drop()
        logging.info(f"✓ Dropped old collection: {col_name}")

        # Insert dữ liệu mới
        result = col.insert_many(trials)
        logging.info(f"✓ Inserted {len(result.inserted_ids)} trial documents vào {col_name}")
        
        # Drop old collection model_comparisons (không còn dùng)
        col_old = db["model_comparisons"]
        col_old.drop()
        logging.info("✓ Dropped old model_comparisons collection")

        client.close()
        logging.info("✓ Hoàn thành push all model trials lên MongoDB")

    except Exception as e:
        error_msg = f"Lỗi khi push model trials: {e}"
        logging.error(error_msg)
        raise ReviewAnalyticsException(error_msg)


if __name__ == "__main__":
    push_all_model_trials()
