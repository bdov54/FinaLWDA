import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_db: str

    raw_csv_path: Path
    processed_csv_path: Path
    artifact_dir: Path

    col_raw: str = "raw_data"
    col_processed: str = "processed_data"
    col_raw_web: str = "raw_data_from_web"

def load_settings() -> Settings:
    load_dotenv()

    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    mongodb_db  = os.getenv("MONGODB_DB", "review_analytics").strip()

    raw_csv = Path(os.getenv("RAW_CSV_PATH", "TSN_Data/raw_data/raw_data.csv"))
    processed_csv = Path(os.getenv("PROCESSED_CSV_PATH", "TSN_Data/processed_data/processed.csv"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

    if not mongodb_uri:
        raise ValueError("Missing MONGODB_URI in .env")

    return Settings(
        mongodb_uri=mongodb_uri,
        mongodb_db=mongodb_db,
        raw_csv_path=raw_csv,
        processed_csv_path=processed_csv,
        artifact_dir=artifact_dir,
    )