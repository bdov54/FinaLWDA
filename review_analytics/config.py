import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

try:
    import streamlit as st
except Exception:
    st = None


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


def _get_secret(name: str, default: str = "") -> str:
    # Local: đọc từ .env
    load_dotenv()

    # Streamlit Cloud / local secrets.toml
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name]).strip()
        except Exception:
            pass

    # Fallback env var
    return os.getenv(name, default).strip()


def load_settings() -> Settings:
    mongodb_uri = _get_secret("MONGODB_URI")
    mongodb_db = _get_secret("MONGODB_DB", "review_analytics")

    raw_csv = Path(_get_secret("RAW_CSV_PATH", "TSN_Data/raw_data/raw_data.csv"))
    processed_csv = Path(_get_secret("PROCESSED_CSV_PATH", "TSN_Data/processed_data/processed.csv"))
    artifact_dir = Path(_get_secret("ARTIFACT_DIR", "artifacts"))

    if not mongodb_uri:
        raise ValueError("Missing MONGODB_URI in Streamlit secrets or environment variables")

    return Settings(
        mongodb_uri=mongodb_uri,
        mongodb_db=mongodb_db,
        raw_csv_path=raw_csv,
        processed_csv_path=processed_csv,
        artifact_dir=artifact_dir,
    )
