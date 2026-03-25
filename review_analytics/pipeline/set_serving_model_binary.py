import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from review_analytics.config import load_settings
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException


ACTIVE_SERVING_REL_PATH = Path("model_registry") / "active_binary_model.json"


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def get_active_manifest_path() -> Path:
    s = load_settings()
    return Path(s.artifact_dir) / ACTIVE_SERVING_REL_PATH


def resolve_run_dir_from_batch(batch_dir: Path) -> Path:
    best_file = batch_dir / "comparison" / "best_model_overall.json"
    if not best_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {best_file}")

    best_info = load_json(best_file, {})
    run_dir = best_info.get("run_dir")
    if not run_dir:
        raise ValueError(f"best_model_overall.json không có run_dir: {best_file}")

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir không tồn tại: {run_dir}")

    return run_dir


def build_manifest_from_run_dir(run_dir: Path) -> Dict:
    trial_summary = load_json(run_dir / "trial_summary.json", {})
    training_config = load_json(run_dir / "training_config.json", {})

    model_dir = run_dir / "model"
    tokenizer_dir = run_dir / "tokenizer"

    if not model_dir.exists():
        raise FileNotFoundError(f"Không thấy model dir: {model_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Không thấy tokenizer dir: {tokenizer_dir}")

    best_threshold = trial_summary.get("best_threshold", 0.5)
    model_name = trial_summary.get("model_name") or training_config.get("model_name")
    display_name = trial_summary.get("display_name") or training_config.get("display_name") or model_name

    trial_cfg = trial_summary.get("trial_config", {})
    max_len = trial_cfg.get("max_len", 128)

    manifest = {
        "task": "binary_comment_sentence_sentiment",
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "model_name": model_name,
        "display_name": display_name,
        "best_threshold": float(best_threshold),
        "max_len": int(max_len),
        "source_trial_summary": str(run_dir / "trial_summary.json"),
        "source_training_config": str(run_dir / "training_config.json"),
        "activated_at": now_utc_iso(),
    }
    return manifest


def run_set_serving_model(run_dir_str: Optional[str] = None, batch_dir_str: Optional[str] = None):
    if not run_dir_str and not batch_dir_str:
        raise ValueError("Cần truyền --run-dir hoặc --batch-dir")

    if run_dir_str:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir không tồn tại: {run_dir}")
    else:
        batch_dir = Path(batch_dir_str)
        if not batch_dir.exists():
            raise FileNotFoundError(f"batch_dir không tồn tại: {batch_dir}")
        run_dir = resolve_run_dir_from_batch(batch_dir)

    manifest = build_manifest_from_run_dir(run_dir)
    out_path = get_active_manifest_path()
    save_json(manifest, out_path)

    logger.info("✅ Active binary serving model updated.")
    logger.info("Run dir: %s", manifest["run_dir"])
    logger.info("Threshold: %s", manifest["best_threshold"])
    logger.info("Manifest path: %s", str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--batch-dir", type=str, default=None)
    args = ap.parse_args()

    try:
        run_set_serving_model(
            run_dir_str=args.run_dir,
            batch_dir_str=args.batch_dir,
        )
    except Exception as e:
        logger.error("Set serving model binary failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()