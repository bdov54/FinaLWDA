import os
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from apify_client import ApifyClient

from review_analytics.logging import logging
from review_analytics.exception.exception import ReviewAnalyticsException

RAW_DIR = Path("TSN_Data/raw_data")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    try:
        load_dotenv()

        token = os.getenv("APIFY_TOKEN")
        actor_id = os.getenv("APIFY_ACTOR_ID")

        if not token:
            raise ValueError("Missing APIFY_TOKEN in .env")
        if not actor_id:
            raise ValueError("Missing APIFY_ACTOR_ID in .env")

        input_path = RAW_DIR / "apify_input.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing {input_path}. Copy Input JSON từ Apify UI vào đây.")

        run_input = json.loads(input_path.read_text(encoding="utf-8"))

        logging.logging.info("Start Apify run: actor_id=%s", actor_id)

        client = ApifyClient(token)
        run = client.actor(actor_id).call(run_input=run_input)

        logging.logging.info(
            "Run finished: status=%s dataset_id=%s",
            run.get("status"),
            run.get("defaultDatasetId")
        )

        dataset_id = run["defaultDatasetId"]
        items = list(client.dataset(dataset_id).iterate_items())
        logging.logging.info("Fetched %d items from dataset", len(items))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CHỈ SAVE CSV
        df = pd.DataFrame(items)
        csv_path = RAW_DIR / f"reviews_{ts}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        logging.logging.info("Saved csv=%s", csv_path)
        print("Fetched:", len(items))
        print("Saved:", csv_path)

    except Exception as e:
        logging.logging.error("Apify raw export failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)

if __name__ == "__main__":
    main()
