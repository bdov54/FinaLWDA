import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

st.write("Streamlit OK")
st.write("Plotly version:", plotly.__version__)

def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "artifacts").exists() and (p / "review_analytics").exists():
            return p
    return start.parent


PROJECT_ROOT = find_project_root(BASE_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.pipeline.infer_runtime_binary import (
    PRED_COMMENT_COL,
    PRED_SENTENCE_COL,
    aggregate_aspect_table,
    build_comment_summary_doc,
    build_comment_text_from_pred_df,
    build_rows_from_custom_comment,
    generate_gemini_commentary,
    generate_model_comparison_commentary,
    predict_binary_pair,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ACTIVE_SERVING_REL_PATH = Path("model_registry") / "active_binary_model.json"
COMPARE_BATCH_NAME = "b123v45_tuned_20260316_082735"
COMPARE_BATCH_DIR = PROJECT_ROOT / "artifacts" / "model_batches" / COMPARE_BATCH_NAME

BEST_MODEL_TRIALS = {
    "mBERT": "mbert_t002_len80_ep5_lr1e-05_bs8_ga1_cwsqrt_inv_ls0.03",
    "XLM-R": "xlmr_t008_len80_ep3_lr2e-05_bs8_ga1_cwnone_ls0.0",
}

# -----------------------------------------------------------------------------
# UI config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Airport Review Intelligence - Mongo",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = {
    "bg": "#08111F",
    "panel": "#101B31",
    "panel2": "#0E1730",
    "text": "#F5F7FB",
    "muted": "#B7C3D7",
    "accent": "#27C2FF",
    "accent2": "#7B61FF",
    "good": "#57D58D",
    "bad": "#FF5E7E",
    "warn": "#FFC857",
    "grid": "rgba(127,127,127,0.18)",
    "axis": "rgba(127,127,127,0.30)",
}

CUSTOM_CSS = f"""
<style>
:root {{
    --app-bg: {THEME['bg']};
    --app-text: {THEME['text']};
    --panel-bg: {THEME['panel']};
    --panel-bg-2: {THEME['panel2']};

    --sidebar-top: rgba(16,27,49,0.98);
    --sidebar-bottom: rgba(9,18,34,0.98);

    --panel-border: rgba(255,255,255,0.12);
    --panel-border-strong: rgba(39,194,255,0.30);
    --muted-text: {THEME['muted']};
    --soft-shadow: 0 10px 28px rgba(0,0,0,0.28);

    --badge-bg: linear-gradient(
        90deg,
        rgba(39,194,255,0.12),
        rgba(123,97,255,0.10)
    );

    --hover-accent: rgba(39,194,255,0.12);
    --selected-accent: rgba(39,194,255,0.18);
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
    background:
        radial-gradient(1100px 580px at 10% 0%, rgba(39,194,255,0.12), transparent 55%),
        radial-gradient(900px 520px at 95% 10%, rgba(123,97,255,0.10), transparent 55%),
        {THEME['bg']} !important;
    color: var(--app-text) !important;
}}

/* Main content cũng phải cùng màu để không bị lệch nền */
[data-testid="stAppViewContainer"] > .main {{
    background:
        radial-gradient(1100px 580px at 10% 0%, rgba(39,194,255,0.12), transparent 55%),
        radial-gradient(900px 520px at 95% 10%, rgba(123,97,255,0.10), transparent 55%),
        {THEME['bg']} !important;
}}

/* Fix thanh trắng phía trên của Streamlit */
[data-testid="stHeader"] {{
    background: rgba(8,17,31,0.96) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}}

header[data-testid="stHeader"] > div {{
    background: transparent !important;
}}

[data-testid="stToolbar"] {{
    right: 1rem;
}}

[data-testid="stDecoration"] {{
    background: linear-gradient(
        90deg,
        {THEME['bad']},
        {THEME['warn']},
        {THEME['accent']}
    ) !important;
}}

.block-container {{
    padding-top: 3.2rem;
    padding-bottom: 2rem;
}}

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            180deg,
            var(--sidebar-top),
            var(--sidebar-bottom)
        ) !important;
    border-right: 1px solid rgba(255,255,255,0.10);
}}

section[data-testid="stSidebar"] * {{
    color: var(--app-text) !important;
}}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: var(--app-text) !important;
    opacity: 1 !important;
}}

.stApp h1 {{
    text-shadow: 0 2px 18px rgba(39,194,255,0.10);
}}

p, li, label, .stMarkdown, .stCaption, .stText {{
    color: var(--app-text) !important;
}}

.small-badge {{
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: var(--badge-bg);
    color: var(--app-text) !important;
    border: 1px solid rgba(39,194,255,0.30);
    font-weight: 700;
    font-size: 0.84rem;
    box-shadow: 0 6px 20px rgba(39,194,255,0.08);
}}

.path-code {{
    font-size: 0.80rem;
    font-family: monospace;
    color: var(--app-text) !important;
}}

div[data-testid="column"] > div {{
    height: 100%;
}}

/* KPI cards */
.kpi {{
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.96),
            rgba(12,22,42,0.96)
        );
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 18px;
    box-shadow:
        0 10px 28px rgba(0,0,0,0.28),
        inset 0 0 0 1px rgba(39,194,255,0.04);
    height: 190px;
    min-height: 190px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
}}

.kpi-title {{
    font-size: 0.82rem;
    color: var(--muted-text) !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    line-height: 1.45;
    min-height: 2.6rem;
    margin-bottom: 8px;
}}

.kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    color: var(--app-text) !important;
    line-height: 1.08;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
    hyphens: auto;
    min-height: 4.2rem;
    display: flex;
    align-items: center;
}}

.kpi-value--long {{
    font-size: 1.55rem;
    line-height: 1.05;
}}

.kpi-value--xlong {{
    font-size: 1.18rem;
    line-height: 1.02;
}}

.kpi-sub {{
    font-size: 0.9rem;
    color: var(--muted-text) !important;
    margin-top: 8px;
    min-height: 2.8rem;
    overflow-wrap: anywhere;
    word-break: break-word;
}}

/* Tabs */
.stTabs {{
    margin-top: 14px !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.55rem;
    margin-top: 10px !important;
    margin-bottom: 20px !important;
    padding-top: 2px !important;
}}

.stTabs [data-baseweb="tab"] {{
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.94),
            rgba(14,23,48,0.94)
        ) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-bottom: none !important;
    border-radius: 16px 16px 0 0 !important;
    color: var(--app-text) !important;
    padding-inline: 1.05rem !important;
    box-shadow:
        0 8px 18px rgba(0,0,0,0.16),
        inset 0 0 0 1px rgba(39,194,255,0.04);
}}

.stTabs [aria-selected="true"] {{
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.99),
            rgba(14,23,48,0.99)
        ) !important;
    border-color: rgba(39,194,255,0.38) !important;
    border-bottom: 2px solid {THEME['accent']} !important;
    box-shadow:
        0 10px 22px rgba(0,0,0,0.18),
        inset 0 0 0 1px rgba(39,194,255,0.08);
}}

/* Inputs / selectbox / dropdown */
[data-baseweb="select"] > div,
[data-baseweb="base-input"] > div,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {{
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.94),
            rgba(14,23,48,0.94)
        ) !important;
    color: var(--app-text) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    box-shadow:
        0 6px 16px rgba(0,0,0,0.14),
        inset 0 0 0 1px rgba(39,194,255,0.04) !important;
}}

[data-baseweb="select"] > div:focus-within,
[data-baseweb="base-input"] > div:focus-within,
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {{
    border-color: rgba(39,194,255,0.38) !important;
    box-shadow:
        0 0 0 1px rgba(39,194,255,0.18),
        0 8px 18px rgba(0,0,0,0.18) !important;
}}

[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] div {{
    color: var(--app-text) !important;
    -webkit-text-fill-color: var(--app-text) !important;
}}

div[role="listbox"] {{
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.98),
            rgba(14,23,48,0.98)
        ) !important;
    color: var(--app-text) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    box-shadow: 0 10px 24px rgba(0,0,0,0.24) !important;
}}

div[role="option"] {{
    background: transparent !important;
    color: var(--app-text) !important;
}}

div[role="option"]:hover {{
    background: var(--hover-accent) !important;
}}

div[role="option"][aria-selected="true"] {{
    background: var(--selected-accent) !important;
    color: var(--app-text) !important;
}}

/* Buttons */
.stButton > button,
.stDownloadButton > button {{
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.94),
            rgba(14,23,48,0.94)
        ) !important;
    color: var(--app-text) !important;
    box-shadow:
        0 8px 18px rgba(0,0,0,0.16),
        inset 0 0 0 1px rgba(39,194,255,0.04);
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    border-color: rgba(39,194,255,0.42) !important;
    box-shadow:
        0 0 0 1px rgba(39,194,255,0.18),
        0 10px 22px rgba(0,0,0,0.20) !important;
}}

/* Toggle / checkbox / radio */
[data-testid="stToggle"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {{
    color: var(--app-text) !important;
}}

/* Expander */
.streamlit-expanderHeader {{
    color: var(--app-text) !important;
    background:
        linear-gradient(
            180deg,
            rgba(16,27,49,0.94),
            rgba(14,23,48,0.94)
        ) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px;
    box-shadow:
        0 8px 18px rgba(0,0,0,0.14),
        inset 0 0 0 1px rgba(39,194,255,0.04);
}}

/* Code blocks */
pre, code, .stCodeBlock {{
    color: var(--app-text) !important;
    background-color: rgba(255,255,255,0.03) !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.12);
}}

/* Plotly text */
.js-plotly-plot svg text {{
    fill: var(--app-text) !important;
}}

.js-plotly-plot .gtitle,
.js-plotly-plot .xtitle,
.js-plotly-plot .ytitle,
.js-plotly-plot .legendtext,
.js-plotly-plot .annotation-text {{
    fill: var(--app-text) !important;
    color: var(--app-text) !important;
}}

/* Plotly hover tooltip */
.js-plotly-plot .hoverlayer .hovertext rect {{
    fill: rgba(8,17,31,0.96) !important;
    stroke: rgba(39,194,255,0.36) !important;
    stroke-width: 1px !important;
}}

.js-plotly-plot .hoverlayer .hovertext text,
.js-plotly-plot .hoverlayer .hovertext tspan {{
    fill: {THEME['text']} !important;
    color: {THEME['text']} !important;
}}

/* Divider */
hr {{
    border-color: rgba(255,255,255,0.10) !important;
}}

/* JSON / alert boxes */
[data-testid="stJson"] *,
.stAlert *,
.element-container * {{
    color: inherit;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def json_load(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def safe_float(v: Any, default: float = np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def deep_find(data: Any, keys: List[str]) -> Any:
    key_set = {k.lower() for k in keys}

    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).lower() in key_set:
                return v
        for _, v in data.items():
            found = deep_find(v, keys)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = deep_find(item, keys)
            if found is not None:
                return found

    return None


def resolve_path_maybe_relative(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    if not p.is_absolute():
        p2 = (base / p).resolve()
        if p2.exists():
            return p2
    return p


def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Settings / stores / manifest
# -----------------------------------------------------------------------------
def get_store() -> MongoStore:
    s = load_settings()
    return MongoStore(s.mongodb_uri, s.mongodb_db)


def get_artifact_root() -> Path:
    s = load_settings()
    artifact_dir = Path(s.artifact_dir)

    if artifact_dir.exists():
        return artifact_dir.resolve()
    if not artifact_dir.is_absolute():
        p2 = (PROJECT_ROOT / artifact_dir).resolve()
        if p2.exists():
            return p2
    return (PROJECT_ROOT / "artifacts").resolve()


def get_active_manifest_path() -> Path:
    return get_artifact_root() / ACTIVE_SERVING_REL_PATH


@st.cache_data(show_spinner=False)
def load_active_manifest() -> Dict[str, Any]:
    path = get_active_manifest_path()
    manifest = json_load(path, {}) or {}
    if not manifest:
        return {}

    for key in ["run_dir", "model_dir", "tokenizer_dir"]:
        if key in manifest and manifest[key]:
            manifest[key] = str(resolve_path_maybe_relative(manifest[key], PROJECT_ROOT))

    return manifest


# -----------------------------------------------------------------------------
# Mongo loaders
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_raw_df(limit: int = 0, order: str = "newest") -> pd.DataFrame:
    s = load_settings()
    store = get_store()
    col = store.collection(s.col_raw)

    sort_field = "created_at"
    sort_dir = -1 if order == "newest" else 1

    cursor = col.find({}, {"_id": 0})
    try:
        cursor = cursor.sort(sort_field, sort_dir)
    except Exception:
        cursor = cursor.sort("_id", sort_dir)

    if limit > 0:
        cursor = cursor.limit(limit)

    return pd.DataFrame(list(cursor))


@st.cache_data(ttl=60)
def load_pred_sentence_df(run_dir: str, model_name: str) -> pd.DataFrame:
    store = get_store()
    col = store.collection(PRED_SENTENCE_COL)

    docs = list(col.find({"serving_run_dir": run_dir}, {"_id": 0}))
    if not docs and model_name:
        docs = list(col.find({"serving_model_name": model_name}, {"_id": 0}))
    if not docs:
        docs = list(col.find({}, {"_id": 0}))

    return pd.DataFrame(docs)


@st.cache_data(ttl=60)
def load_pred_comment_df(display_name: str, model_name: str) -> pd.DataFrame:
    store = get_store()
    col = store.collection(PRED_COMMENT_COL)

    docs = []
    if display_name:
        docs = list(col.find({"model_display_name": display_name}, {"_id": 0}))
    if not docs and model_name:
        docs = list(col.find({"model_name": model_name}, {"_id": 0}))
    if not docs:
        docs = list(col.find({}, {"_id": 0}))

    return pd.DataFrame(docs)


# -----------------------------------------------------------------------------
# Artifact comparison loaders
# -----------------------------------------------------------------------------
def find_trial_dir(family: str, trial_name: str) -> Optional[Path]:
    family_dir = COMPARE_BATCH_DIR / "models" / family
    candidates = [
        family_dir / trial_name,
        family_dir / "trials" / trial_name,
    ]

    for p in candidates:
        if p.exists():
            return p

    for p in family_dir.rglob(trial_name):
        if p.is_dir():
            return p

    return None


@st.cache_data(show_spinner=False)
def build_local_model_registry() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}

    for family, trial_name in BEST_MODEL_TRIALS.items():
        trial_dir = find_trial_dir(family, trial_name)

        if trial_dir is None:
            registry[family] = {
                "available": False,
                "family": family,
                "display_name": family,
                "trial_name": trial_name,
                "error": f"Không tìm thấy trial dir: {trial_name}",
            }
            continue

        model_dir = trial_dir / "model"
        tokenizer_dir = trial_dir / "tokenizer"
        reports_dir = trial_dir / "reports"

        metrics = json_load(trial_dir / "metrics.json", {}) or {}
        trial_summary = json_load(trial_dir / "trial_summary.json", {}) or {}
        train_metrics = json_load(trial_dir / "train_metrics.json", {}) or {}
        training_cfg = json_load(trial_dir / "training_config.json", {}) or {}
        trainer_state_summary = json_load(trial_dir / "trainer_state_summary.json", {}) or {}

        required = [
            model_dir / "config.json",
            model_dir / "model.safetensors",
            tokenizer_dir / "tokenizer.json",
            tokenizer_dir / "tokenizer_config.json",
        ]

        if family == "mBERT":
            required.append(tokenizer_dir / "vocab.txt")
        else:
            required.append(tokenizer_dir / "sentencepiece.bpe.model")

        missing = [str(p.name) for p in required if not p.exists()]

        threshold_info = json_load(reports_dir / "threshold_search_best.json", {}) or {}
        threshold = deep_find(threshold_info, ["best_threshold", "threshold", "optimal_threshold"])
        max_len = deep_find(training_cfg, ["max_len", "max_length"])

        val_f1 = deep_find(trial_summary, ["val_f1_macro"]) or deep_find(
            metrics, ["val_f1_macro", "eval_f1_macro", "best_val_f1_macro"]
        )
        test_f1 = deep_find(trial_summary, ["test_f1_macro"]) or deep_find(
            metrics, ["test_f1_macro", "f1_macro"]
        )
        test_acc = deep_find(metrics, ["test_accuracy", "accuracy", "accuracy_test"])
        test_prec = deep_find(metrics, ["test_precision_macro", "precision_macro"])
        test_rec = deep_find(metrics, ["test_recall_macro", "recall_macro"])
        test_f1_weighted = deep_find(metrics, ["test_f1_weighted", "f1_weighted"])

        train_runtime = deep_find(train_metrics, ["train_runtime"]) or deep_find(
            trainer_state_summary, ["train_runtime"]
        )
        train_sps = deep_find(train_metrics, ["train_samples_per_second"]) or deep_find(
            trainer_state_summary, ["train_samples_per_second"]
        )
        train_tps = deep_find(train_metrics, ["train_steps_per_second"]) or deep_find(
            trainer_state_summary, ["train_steps_per_second"]
        )

        registry[family] = {
            "available": len(missing) == 0,
            "family": family,
            "display_name": family,
            "trial_name": trial_name,
            "trial_dir": str(trial_dir.resolve()),
            "model_dir": str(model_dir.resolve()),
            "tokenizer_dir": str(tokenizer_dir.resolve()),
            "training_history_path": str((trial_dir / "training_history.csv").resolve()),
            "metrics": metrics,
            "trial_summary": trial_summary,
            "train_metrics": train_metrics,
            "training_config": training_cfg,
            "trainer_state_summary": trainer_state_summary,
            "threshold": safe_float(threshold, 0.50),
            "max_len": int(max_len) if max_len is not None else 80,
            "val_f1_macro": safe_float(val_f1),
            "test_f1_macro": safe_float(test_f1),
            "test_accuracy": safe_float(test_acc),
            "test_precision_macro": safe_float(test_prec),
            "test_recall_macro": safe_float(test_rec),
            "test_f1_weighted": safe_float(test_f1_weighted),
            "train_runtime": safe_float(train_runtime),
            "train_samples_per_second": safe_float(train_sps),
            "train_steps_per_second": safe_float(train_tps),
            "missing": missing,
        }

    return registry


@st.cache_data(show_spinner=False)
def build_comparison_df() -> pd.DataFrame:
    registry = build_local_model_registry()
    rows = []

    for family, info in registry.items():
        rows.append(
            {
                "model": family,
                "trial_name": info.get("trial_name"),
                "available": info.get("available"),
                "max_len": info.get("max_len"),
                "threshold": info.get("threshold"),
                "test_f1_macro": info.get("test_f1_macro"),
                "val_f1_macro": info.get("val_f1_macro"),
                "test_accuracy": info.get("test_accuracy"),
                "test_precision_macro": info.get("test_precision_macro"),
                "test_recall_macro": info.get("test_recall_macro"),
                "test_f1_weighted": info.get("test_f1_weighted"),
                "train_runtime": info.get("train_runtime"),
                "train_samples_per_second": info.get("train_samples_per_second"),
                "train_steps_per_second": info.get("train_steps_per_second"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["available", "test_f1_macro"], ascending=[False, False]).reset_index(
            drop=True
        )
    return df


@st.cache_data(show_spinner=False)
def load_training_history(model_family: str) -> pd.DataFrame:
    registry = build_local_model_registry()
    info = registry[model_family]

    try:
        return pd.read_csv(info["training_history_path"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_model_comparison_from_mongo(top_n: int = 2) -> pd.DataFrame:
    """Load all model trials từ MongoDB và filter top 1 best model từ mỗi family"""
    try:
        store = get_store()
        col = store.collection("model_trials")
        docs = list(col.find({}, {"_id": 0}))
        
        if not docs:
            return pd.DataFrame()
        
        df = pd.DataFrame(docs)
        
        # Parse config if it's a string
        if "config" in df.columns:
            df["config"] = df["config"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        # Standardize column names for compatibility with UI
        if "display_name" in df.columns:
            df["model"] = df["display_name"]
        if "trial_name" in df.columns:
            df["trial_name"] = df["trial_name"]
        if "best_threshold" in df.columns:
            df["threshold"] = df["best_threshold"]
        
        # Extract max_len from config
        if "config" in df.columns:
            df["max_len"] = df["config"].apply(lambda x: x.get("max_len", 80) if isinstance(x, dict) else 80)
        else:
            df["max_len"] = 80
        
        # Add missing columns with defaults
        if "available" not in df.columns:
            df["available"] = True
        if "test_f1_weighted" not in df.columns:
            df["test_f1_weighted"] = df["test_f1_macro"]
        
        # Load precision/recall from trial artifacts if missing in MongoDB
        if "test_precision_macro" not in df.columns or df["test_precision_macro"].isna().any():
            df = _enrich_metrics_from_artifacts(df)
        
        # Get top 1 from each model family based on test_f1_macro
        if "test_f1_macro" in df.columns and "model" in df.columns:
            df = df.sort_values("test_f1_macro", ascending=False)
            df = df.groupby("model", sort=False).head(1).reset_index(drop=True)
            df = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
        
        return df
    except Exception as e:
        st.warning(f"Không load được model trials từ MongoDB: {e}")
        return pd.DataFrame()


def _enrich_metrics_from_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Load precision/recall metrics from trial artifacts"""
    # Get absolute path to artifacts directory
    project_root = Path(__file__).parent.parent
    artifact_base = project_root / "artifacts" / "model_batches" / "b123v45_tuned_20260316_082735" / "models"
    
    for idx, row in df.iterrows():
        trial_name = row.get("trial_name")
        model_family = row.get("model", "").split()[0]  # "mBERT" or "XLM-R"
        
        if not trial_name or not model_family:
            continue
        
        # Try two possible paths:
        # 1. For XLM-R: artifact_base / "XLM-R" / "trials" / trial_name / "metrics.json"
        # 2. For mBERT: artifact_base / "mBERT" / trial_name / "metrics.json"
        
        metrics_path = artifact_base / model_family / "trials" / trial_name / "metrics.json"
        if not metrics_path.exists():
            # Try alternate path without "trials" directory
            metrics_path = artifact_base / model_family / trial_name / "metrics.json"
        
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                
                # Update row with precision/recall if available
                if "test_precision_macro" in metrics:
                    df.at[idx, "test_precision_macro"] = metrics["test_precision_macro"]
                if "test_recall_macro" in metrics:
                    df.at[idx, "test_recall_macro"] = metrics["test_recall_macro"]
            except Exception as e:
                logging.warning(f"Could not load metrics for {trial_name}: {e}")
    
    return df



# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------
def run_manual_comment_inference(comment_text: str, active_manifest: Dict[str, Any]) -> pd.DataFrame:
    rows_df = build_rows_from_custom_comment(comment_text)
    if rows_df.empty:
        return rows_df

    outs = []
    for _, row in rows_df.iterrows():
        pred = predict_binary_pair(
            sentence=row["sentence"],
            aspect_text=row["aspect_text"],
            model_dir=active_manifest["model_dir"],
            tokenizer_dir=active_manifest["tokenizer_dir"],
            max_length=int(active_manifest.get("max_len", 128)),
        )
        item = row.to_dict()
        item.update(pred)
        outs.append(item)

    return pd.DataFrame(outs)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def render_kpi(title: str, value: str, sub: str = ""):
    value_str = "" if value is None else str(value)
    longest_token = max((len(x) for x in value_str.split()), default=len(value_str))

    value_class = "kpi-value"
    if len(value_str) >= 16 or longest_token >= 14:
        value_class += " kpi-value--long"
    if len(value_str) >= 24 or longest_token >= 18:
        value_class += " kpi-value--xlong"

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">{title}</div>
            <div class="{value_class}" title="{value_str}">{value_str}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,27,49,0.55)",
        font=dict(color=THEME["text"]),
        margin=dict(l=18, r=18, t=40, b=18),
        legend=dict(
            font=dict(color=THEME["text"]),
            bgcolor="rgba(16,27,49,0.35)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(8,17,31,0.96)",
            bordercolor="rgba(39,194,255,0.36)",
            font=dict(color=THEME["text"], size=15),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        linecolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.12)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        linecolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.12)",
    )
    return fig

def build_overall_kpis(comment_df: pd.DataFrame, pred_df: pd.DataFrame, aspect_df: pd.DataFrame):
    total_comments = len(comment_df)
    total_rows = len(pred_df)
    unique_aspects = (
        aspect_df["aspect"].nunique()
        if (not aspect_df.empty and "aspect" in aspect_df.columns)
        else 0
    )

    neg_comment_rate = 0.0
    pos_comment_rate = 0.0
    avg_aspects_per_comment = 0.0
    avg_conf = 0.0
    top_priority = "-"
    top_strength = "-"

    if not comment_df.empty and "overall_label" in comment_df.columns:
        neg_comment_rate = (comment_df["overall_label"] == "negative").mean()
        pos_comment_rate = (comment_df["overall_label"] == "positive").mean()

    if not comment_df.empty and "n_matched_rows" in comment_df.columns:
        avg_aspects_per_comment = float(comment_df["n_matched_rows"].mean())

    if not comment_df.empty and "overall_confidence" in comment_df.columns:
        avg_conf = float(comment_df["overall_confidence"].mean())

    if not aspect_df.empty:
        if "priority_score" in aspect_df.columns:
            top_priority = aspect_df.sort_values(
                ["priority_score", "mentions"], ascending=[False, False]
            ).iloc[0]["aspect_text"]
        if "strength_score" in aspect_df.columns:
            top_strength = aspect_df.sort_values(
                ["strength_score", "mentions"], ascending=[False, False]
            ).iloc[0]["aspect_text"]

    return {
        "total_comments": total_comments,
        "total_rows": total_rows,
        "unique_aspects": unique_aspects,
        "neg_comment_rate": neg_comment_rate,
        "pos_comment_rate": pos_comment_rate,
        "avg_aspects_per_comment": avg_aspects_per_comment,
        "avg_conf": avg_conf,
        "top_priority": top_priority,
        "top_strength": top_strength,
    }


def prepare_selected_comment_pred_df(
    selected_review_id: str, all_pred_df: pd.DataFrame
) -> pd.DataFrame:
    if all_pred_df.empty or "review_id" not in all_pred_df.columns:
        return pd.DataFrame()

    return (
        all_pred_df[all_pred_df["review_id"].astype(str) == str(selected_review_id)]
        .copy()
        .reset_index(drop=True)
    )


def show_aspect_block(aspect_df: pd.DataFrame, top_n: int, min_mentions: int, title_prefix: str):
    if aspect_df.empty:
        st.info("Không có aspect nào để hiển thị.")
        return

    aspect_df = aspect_df[aspect_df["mentions"] >= min_mentions].copy()
    if aspect_df.empty:
        st.info("Không có aspect nào đạt ngưỡng min_mentions.")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"{title_prefix} • Top aspect được nhắc đến")
        fig1 = px.bar(
            aspect_df.sort_values("mentions", ascending=False).head(top_n),
            x="mentions",
            y="aspect_text",
            orientation="h",
            color="mentions",
            color_continuous_scale=["#2BB3FF", "#7B61FF"],
        )
        fig1.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig1, 420), use_container_width=True)

    with c2:
        st.subheader(f"{title_prefix} • Tỷ lệ 2 lớp theo aspect")
        rate_df = aspect_df.sort_values("mentions", ascending=False).head(top_n).copy()
        rate_long = rate_df.melt(
            id_vars=["aspect_text"],
            value_vars=["negative_rate", "positive_rate"],
            var_name="sentiment",
            value_name="rate",
        )
        rate_long["sentiment"] = rate_long["sentiment"].map(
            {"negative_rate": "negative", "positive_rate": "positive"}
        )
        fig2 = px.bar(
            rate_long,
            x="rate",
            y="aspect_text",
            orientation="h",
            color="sentiment",
            barmode="stack",
            color_discrete_map={"negative": THEME["bad"], "positive": THEME["good"]},
        )
        fig2.update_xaxes(tickformat=".0%")
        fig2.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig2, 420), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader(f"{title_prefix} • Aspect tiêu cực nhất")
        fig3 = px.bar(
            aspect_df.sort_values(["negative_rate", "mentions"], ascending=[False, False]).head(top_n),
            x="negative_rate",
            y="aspect_text",
            orientation="h",
            color="negative_rate",
            color_continuous_scale=["#FF8A8A", "#FF5E7E"],
        )
        fig3.update_xaxes(tickformat=".0%")
        fig3.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig3, 420), use_container_width=True)

    with c4:
        st.subheader(f"{title_prefix} • Aspect tích cực nhất")
        fig4 = px.bar(
            aspect_df.sort_values(["positive_rate", "mentions"], ascending=[False, False]).head(top_n),
            x="positive_rate",
            y="aspect_text",
            orientation="h",
            color="positive_rate",
            color_continuous_scale=["#99F2B0", "#57D58D"],
        )
        fig4.update_xaxes(tickformat=".0%")
        fig4.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig4, 420), use_container_width=True)

    st.subheader(f"{title_prefix} • Ưu tiên xử lý vs điểm mạnh")
    fig5 = px.scatter(
        aspect_df,
        x="mentions",
        y="negative_rate",
        size="priority_score",
        color="net_score",
        hover_data=["aspect_text", "positive_rate", "avg_confidence"],
        color_continuous_scale=["#FF5E7E", "#FFC857", "#57D58D"],
    )
    fig5.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig5, 460), use_container_width=True)

    st.dataframe(
        aspect_df.sort_values(["mentions", "negative_rate"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Header + workflow guidance
# -----------------------------------------------------------------------------
st.markdown("## ✈️ Airport Review Intelligence Dashboard")
st.markdown(
    "<span class='small-badge'>MongoDB • Binary ABSA • Sentence-level aggregation • Local artifact comparison • Gemini AI Summary</span>",
    unsafe_allow_html=True,
)

active_manifest = load_active_manifest()
compare_df = load_model_comparison_from_mongo(top_n=2)
local_registry = build_local_model_registry()

st.sidebar.markdown("## ⚙️ Controls")
top_n = st.sidebar.slider("Top N aspect", 5, 25, 10)
min_mentions = st.sidebar.slider("Min mentions", 1, 20, 1)
demo_n = st.sidebar.slider("Số comment cho Demo N", 5, 200, 30)
show_debug = st.sidebar.toggle("Hiện diagnostics", value=False)

if active_manifest:
    st.sidebar.write(f"**Serving model:** {active_manifest.get('display_name', '-')}")
    st.sidebar.write(f"**Model name:** {active_manifest.get('model_name', '-')}")
    st.sidebar.write(f"**Threshold:** {active_manifest.get('best_threshold', '-')}")
else:
    st.sidebar.error("Chưa thấy active_binary_model.json")

with st.sidebar.expander("Quy trình chạy đúng", expanded=False):
    st.code(
        """
python -m review_analytics.pipeline.set_serving_model_binary --run-dir <RUN_DIR>
python -m review_analytics.pipeline.materialize_sentence_predictions_binary --refresh-existing
python -m review_analytics.pipeline.build_comment_summary_binary --refresh-existing
streamlit run My_client/app_mongo.py
"""
    )

with st.sidebar.expander("Active manifest path", expanded=False):
    st.code(str(get_active_manifest_path()))

if show_debug:
    st.subheader("Diagnostics")
    st.write("PROJECT_ROOT:", str(PROJECT_ROOT))
    st.write("ARTIFACT_ROOT:", str(get_artifact_root()))
    st.write("ACTIVE_MANIFEST:", active_manifest)
    st.dataframe(pd.DataFrame(local_registry).T.reset_index(drop=False), use_container_width=True)

# Load Mongo data only when manifest exists
pred_sentence_df = pd.DataFrame()
pred_comment_df = pd.DataFrame()

if active_manifest:
    pred_sentence_df = load_pred_sentence_df(
        str(active_manifest.get("run_dir", "")),
        str(active_manifest.get("model_name", "")),
    )
    pred_comment_df = load_pred_comment_df(
        str(active_manifest.get("display_name", "")),
        str(active_manifest.get("model_name", "")),
    )

# Top KPIs
if not compare_df.empty:
    best_row = compare_df.sort_values("test_f1_macro", ascending=False).iloc[0]
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        render_kpi("Best overall model", str(best_row["model"]), str(best_row["trial_name"]))
    with k2:
        render_kpi(
            "Best test F1 macro",
            f"{safe_float(best_row['test_f1_macro']):.4f}",
            "hiệu suất tổng quát",
        )
    with k3:
        val_f1 = safe_float(best_row.get("val_f1_macro"))
        render_kpi(
            "Best val F1 macro",
            f"{val_f1:.4f}" if not np.isnan(val_f1) else "-",
            "trên validation",
        )
    with k4:
        render_kpi(
            "Threshold",
            f"{safe_float(best_row['threshold']):.2f}",
            "ngưỡng tối ưu từ artifacts",
        )

tab_overall, tab_compare, tab_one, tab_many = st.tabs(
    ["📊 Overall", "📈 Model Comparison", "📝 Demo 1 Comment", "📚 Demo N comments"]
)

# -----------------------------------------------------------------------------
# TAB 1: Overall (must be DB-driven)
# -----------------------------------------------------------------------------
with tab_overall:
    st.subheader("Thực trạng sân bay từ dữ liệu đã materialize trong MongoDB")

    if active_manifest:
        st.caption(f"Active serving model: {active_manifest.get('display_name', '-')}")

    if pred_sentence_df.empty or pred_comment_df.empty:
        st.warning(
            """
Dashboard Overall cần dữ liệu trong database. Hãy chạy đúng quy trình trước:
1) set_serving_model_binary.py
2) materialize_sentence_predictions_binary.py --refresh-existing
3) build_comment_summary_binary.py --refresh-existing
"""
        )
    else:
        overall_aspect_df = aggregate_aspect_table(pred_sentence_df)
        kpis = build_overall_kpis(pred_comment_df, pred_sentence_df, overall_aspect_df)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi("Total comments", f"{kpis['total_comments']:,}", "số review đã tổng hợp")
        with k2:
            render_kpi("Matched rows", f"{kpis['total_rows']:,}", "sentence-aspect predictions")
        with k3:
            render_kpi("Unique aspects", f"{kpis['unique_aspects']}", "khía cạnh được nhắc đến")
        with k4:
            render_kpi("Avg confidence", f"{kpis['avg_conf']:.3f}", "trung bình mức tin cậy")

        k5, k6, k7, k8 = st.columns(4)
        with k5:
            render_kpi(
                "Negative comments",
                f"{kpis['neg_comment_rate']*100:.1f}%",
                "tỷ lệ overall negative",
            )
        with k6:
            render_kpi(
                "Positive comments",
                f"{kpis['pos_comment_rate']*100:.1f}%",
                "tỷ lệ overall positive",
            )
        with k7:
            render_kpi(
                "Avg matched rows / comment",
                f"{kpis['avg_aspects_per_comment']:.2f}",
                "mức độ phủ aspect",
            )
        with k8:
            render_kpi("Top priority aspect", kpis["top_priority"], "ưu tiên kiểm tra")

        c1, c2 = st.columns(2)

        with c1:
            overall_label_df = (
                pred_comment_df["overall_label"]
                .value_counts(dropna=False)
                .rename_axis("label")
                .reset_index(name="count")
            )
            fig = px.pie(
                overall_label_df,
                names="label",
                values="count",
                hole=0.55,
                color="label",
                color_discrete_map={"negative": THEME["bad"], "positive": THEME["good"]},
            )
            st.subheader("Comment-level sentiment distribution")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)

        with c2:
            st.subheader("Priority score by aspect")
            figp = px.bar(
                overall_aspect_df.sort_values(["priority_score", "mentions"], ascending=[False, False]).head(top_n),
                x="priority_score",
                y="aspect_text",
                orientation="h",
                color="priority_score",
                color_continuous_scale=["#FFD166", "#FF5E7E"],
            )
            figp.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(style_fig(figp, 360), use_container_width=True)

        show_aspect_block(overall_aspect_df, top_n=top_n, min_mentions=min_mentions, title_prefix="Overall")

        st.subheader("🤖 AI Executive Summary")
        if st.button("Generate overall AI summary", key="btn_overall_ai"):
            ai_text = generate_gemini_commentary(
                summary_doc=build_comment_summary_doc(pred_sentence_df, review_id="overall"),
                aspect_df=overall_aspect_df,
                sample_rows=pred_sentence_df,
                scope_name="toàn bộ dữ liệu đánh giá",
            )
            st.info(ai_text)


# -----------------------------------------------------------------------------
# TAB 2: Best Model Comparison (MongoDB-driven)
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Best Models Comparison")
    
    # Add unique labels for each model trial
    compare_df_labeled = compare_df.copy()
    compare_df_labeled["model_label"] = compare_df_labeled.groupby("model", sort=False).cumcount() + 1
    compare_df_labeled["model_label"] = compare_df_labeled["model"] + " #" + compare_df_labeled["model_label"].astype(str)
    
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # Define distinct colors for each model family
    model_colors = {
        "mBERT #1": "#7B61FF",    # Purple
        "XLM-R #1": "#36A2EB",    # Cyan Blue
    }

    if not compare_df.empty:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                compare_df_labeled,
                x="model_label",
                y="test_f1_macro",
                color="model_label",
                text="test_f1_macro",
                color_discrete_map=model_colors,
                title="Test F1 macro",
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=100))
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)

        with c2:
            metric_cols = [
                c
                for c in ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
                if c in compare_df.columns
            ]
            melt_df = compare_df_labeled.melt(
                id_vars=["model_label"],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            )
            fig2 = px.bar(
                melt_df,
                x="metric",
                y="value",
                color="model_label",
                barmode="group",
                color_discrete_map=model_colors,
                title="Overall test metrics",
            )
            fig2.update_layout(showlegend=True)
            st.plotly_chart(style_fig(fig2, 360), use_container_width=True)

        rt_cols = [
            c
            for c in ["train_runtime", "train_samples_per_second", "train_steps_per_second"]
            if c in compare_df.columns
        ]
        if rt_cols:
            rt_df = compare_df_labeled[["model_label"] + rt_cols].melt(
                id_vars=["model_label"], var_name="metric", value_name="value"
            )
            fig3 = px.bar(
                rt_df,
                x="metric",
                y="value",
                color="model_label",
                barmode="group",
                color_discrete_map=model_colors,
                title="Training runtime / throughput",
            )
            fig3.update_layout(showlegend=True)
            st.plotly_chart(style_fig(fig3, 360), use_container_width=True)

    st.subheader("Training curves")
    for family in [m for m in ["mBERT", "XLM-R"] if m in local_registry]:
        with st.expander(f"{family} curves", expanded=(family == "XLM-R")):
            hist = load_training_history(family)
            if hist.empty:
                st.info("Không đọc được training history.")
            else:
                x_col = (
                    "epoch"
                    if "epoch" in hist.columns
                    else ("step" if "step" in hist.columns else hist.columns[0])
                )

                cc1, cc2, cc3 = st.columns(3)

                with cc1:
                    if "eval_f1_macro" in hist.columns:
                        st.plotly_chart(
                            style_fig(
                                px.line(
                                    hist,
                                    x=x_col,
                                    y="eval_f1_macro",
                                    markers=True,
                                    title=f"{family} • Validation F1 macro",
                                ),
                                300,
                            ),
                            use_container_width=True,
                        )

                with cc2:
                    if "eval_loss" in hist.columns:
                        st.plotly_chart(
                            style_fig(
                                px.line(
                                    hist,
                                    x=x_col,
                                    y="eval_loss",
                                    markers=True,
                                    title=f"{family} • Evaluation loss",
                                ),
                                300,
                            ),
                            use_container_width=True,
                        )

                with cc3:
                    if "loss" in hist.columns:
                        st.plotly_chart(
                            style_fig(
                                px.line(
                                    hist,
                                    x=x_col,
                                    y="loss",
                                    markers=True,
                                    title=f"{family} • Train loss",
                                ),
                                300,
                            ),
                            use_container_width=True,
                        )

    # Detailed Metrics Comparison
    st.subheader("Detailed Performance Metrics")
    
    # Create metrics comparison dataframe
    if not compare_df.empty:
        # Only select columns that exist
        available_metric_cols = [
            col for col in ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
            if col in compare_df.columns
        ]
        
        if available_metric_cols:
            metrics_for_radar = compare_df[["model"] + available_metric_cols].copy()
            
            # Normalize for better visualization
            metrics_long = metrics_for_radar.melt(
                id_vars=["model"],
                value_vars=available_metric_cols,
                var_name="metric",
                value_name="score"
            )
            
            metrics_long["metric"] = metrics_long["metric"].str.replace("test_", "").str.replace("_macro", "").str.upper()
            
            # Bar chart for metrics comparison
            fig_metrics = px.bar(
                metrics_long,
                x="metric",
                y="score",
                color="model",
                barmode="group",
                text="score",
                title="Detailed Performance Metrics Comparison",
                color_discrete_map={
                    "mBERT #1": "#7B61FF",
                    "XLM-R #1": "#36A2EB",
                }
            )
            fig_metrics.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_metrics.update_layout(
                yaxis_range=[0, 1],
                hovermode="x unified",
                height=400,
                showlegend=True
            )
            st.plotly_chart(style_fig(fig_metrics, 400), use_container_width=True)
        else:
            st.info("No metric columns available for comparison")
    
    # Confusion Matrix
    st.subheader("Confusion Matrices")
    
    if not compare_df.empty:
        cm_cols = st.columns(len(compare_df))
        
        for idx, (_, row) in enumerate(compare_df.iterrows()):
            model_name = row["model"]
            trial_name = row["trial_name"]
            
            with cm_cols[idx]:
                st.write(f"**{model_name}**")
                
                try:
                    trial_dir = find_trial_dir(model_name, trial_name)
                    if trial_dir:
                        cm_path = trial_dir / "confusion_matrix.json"
                        if cm_path.exists():
                            cm_data = json_load(cm_path, {})
                            if cm_data:
                                cm_array = np.array(cm_data)
                                
                                # Create heatmap
                                cm_df = pd.DataFrame(
                                    cm_array,
                                    index=["Predicted Negative", "Predicted Positive"],
                                    columns=["Actual Negative", "Actual Positive"]
                                )
                                
                                fig_cm = px.imshow(
                                    cm_df,
                                    text_auto=True,
                                    color_continuous_scale="RdYlGn",
                                    title=f"Confusion Matrix",
                                    labels=dict(x="Actual", y="Predicted", color="Count"),
                                    aspect="auto"
                                )
                                fig_cm.update_layout(
                                    height=350,
                                    coloraxis_showscale=False,
                                    font=dict(size=12, color="white")
                                )
                                fig_cm.update_xaxes(side="bottom")
                                st.plotly_chart(style_fig(fig_cm, 350), use_container_width=True)
                            else:
                                st.warning("Confusion matrix data is empty")
                        else:
                            st.info("Confusion matrix file not found")
                    else:
                        st.warning("Trial directory not found")
                except Exception as e:
                    st.error(f"Error loading confusion matrix: {str(e)}")

    # Gemini AI Model Comparison Analysis
    st.subheader("🤖 AI Model Comparison Analysis")
    if st.button("Generate model comparison insights", key="btn_model_comparison_ai"):
        if not compare_df.empty:
            ai_text = generate_model_comparison_commentary(compare_df)
            st.info(ai_text)
        else:
            st.warning("Không có model data để so sánh.")


# -----------------------------------------------------------------------------
# TAB 3: Demo 1 comment (manual inference or DB review)
# -----------------------------------------------------------------------------
with tab_one:
    st.subheader("Demo theo 1 comment")

    mode = st.radio("Chế độ", ["Chọn từ MongoDB", "Nhập thủ công"], horizontal=True)

    selected_pred_df = pd.DataFrame()
    comment_text = ""
    comment_meta: Dict[str, Any] = {}

    if mode == "Chọn từ MongoDB":
        order_mode = st.selectbox("Chọn thứ tự", ["newest", "oldest"])
        raw_df = load_raw_df(limit=200, order=order_mode)

        if raw_df.empty:
            st.warning("Không load được raw data từ Mongo.")
        else:
            raw_df["display"] = raw_df.apply(
                lambda r: f"{str(r.get('author', 'unknown'))} | rating={r.get('rating', '-')} | review_id={str(r.get('review_id', ''))[:60]}",
                axis=1,
            )
            selected_display = st.selectbox("Chọn review", raw_df["display"].tolist())
            selected_row = raw_df[raw_df["display"] == selected_display].iloc[0]
            selected_review_id = str(selected_row["review_id"])

            selected_pred_df = prepare_selected_comment_pred_df(selected_review_id, pred_sentence_df)
            comment_text = str(selected_row.get("text", "") or "").strip()

            if not comment_text:
                comment_text = build_comment_text_from_pred_df(selected_pred_df)

            comment_meta = {
                "review_id": selected_review_id,
                "author": selected_row.get("author"),
                "rating": selected_row.get("rating"),
                "created_at": selected_row.get("created_at"),
                "title": selected_row.get("title"),
                "url": selected_row.get("url"),
            }

    else:
        comment_text = st.text_area(
            "Nhập comment",
            height=140,
            value="Nhân viên khá lịch sự nhưng khu soi chiếu quá chậm và nhà vệ sinh chưa sạch.",
        )

        if st.button("Analyze manual comment", type="primary"):
            if not active_manifest:
                st.error("Chưa có active_binary_model.json. Hãy chạy set_serving_model_binary.py trước.")
            else:
                selected_pred_df = run_manual_comment_inference(comment_text, active_manifest)
                comment_meta = {
                    "review_id": "manual_input",
                    "model": active_manifest.get("display_name"),
                }

    if comment_meta:
        st.markdown("### Comment metadata")
        st.json(comment_meta, expanded=False)

    if comment_text:
        st.markdown("### Comment text")
        st.write(comment_text)

    if not selected_pred_df.empty:
        one_aspect_df = aggregate_aspect_table(selected_pred_df)
        one_summary = build_comment_summary_doc(
            selected_pred_df,
            review_id=str(comment_meta.get("review_id", "manual_input")),
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi("Overall label", str(one_summary["overall_label"]), "kết luận ở mức comment")
        with c2:
            render_kpi(
                "Overall confidence",
                f"{one_summary['overall_confidence']:.3f}",
                "độ mạnh của kết luận",
            )
        with c3:
            render_kpi("Sentences", str(one_summary["n_sentences"]), "số câu sau khi split")
        with c4:
            render_kpi(
                "Matched rows",
                str(one_summary["n_matched_rows"]),
                "câu-aspect được nhận diện",
            )

        st.subheader("Sentence-level predictions")
        st.dataframe(
            selected_pred_df[
                [
                    c
                    for c in [
                        "sent_id",
                        "sentence",
                        "aspect_text",
                        "hit_keywords",
                        "pred_label",
                        "prob_negative",
                        "prob_positive",
                        "confidence",
                    ]
                    if c in selected_pred_df.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        show_aspect_block(one_aspect_df, top_n=top_n, min_mentions=1, title_prefix="1 Comment")

        st.subheader("🤖 AI Detailed Commentary")
        if st.button("Generate 1-comment AI summary", key="btn_one_ai"):
            ai_text = generate_gemini_commentary(
                summary_doc=one_summary,
                aspect_df=one_aspect_df,
                sample_rows=selected_pred_df,
                scope_name="một comment cụ thể",
            )
            st.info(ai_text)


# -----------------------------------------------------------------------------
# TAB 4: Demo N comments (DB-driven)
# -----------------------------------------------------------------------------
with tab_many:
    st.subheader("Demo theo N comment")

    mode_n = st.selectbox("Cách lấy comment", ["newest", "oldest", "all"])
    raw_df = load_raw_df(
        limit=0 if mode_n == "all" else demo_n,
        order=mode_n if mode_n != "all" else "newest",
    )

    if raw_df.empty:
        st.warning("Không lấy được raw reviews từ Mongo.")
    elif pred_sentence_df.empty:
        st.warning(
            "Tab này cần dữ liệu đã materialize trong Mongo. Hãy chạy materialize + build summary trước."
        )
    else:
        if mode_n == "all":
            review_ids = raw_df["review_id"].astype(str).tolist()
        else:
            review_ids = raw_df["review_id"].astype(str).head(demo_n).tolist()

        subset_pred_df = pred_sentence_df[
            pred_sentence_df["review_id"].astype(str).isin(review_ids)
        ].copy()

        subset_comment_df = (
            pred_comment_df[pred_comment_df["review_id"].astype(str).isin(review_ids)].copy()
            if not pred_comment_df.empty
            else pd.DataFrame()
        )

        if subset_pred_df.empty:
            st.warning("Không có prediction cho tập comment này.")
        else:
            subset_aspect_df = aggregate_aspect_table(subset_pred_df)
            subset_summary = build_comment_summary_doc(
                subset_pred_df, review_id=f"{mode_n}_{len(review_ids)}_comments"
            )

            kk1, kk2, kk3, kk4 = st.columns(4)

            with kk1:
                render_kpi("Selected comments", f"{len(review_ids):,}", "quy mô mẫu đang xem")
            with kk2:
                render_kpi("Matched rows", f"{len(subset_pred_df):,}", "sentence-aspect predictions")
            with kk3:
                neg_rate = (
                    (subset_comment_df["overall_label"] == "negative").mean() * 100
                    if (not subset_comment_df.empty and "overall_label" in subset_comment_df.columns)
                    else 0.0
                )
                render_kpi("Overall negative", f"{neg_rate:.1f}%", "comment-level negative")
            with kk4:
                top_strength = (
                    subset_aspect_df.sort_values(["strength_score", "mentions"], ascending=[False, False]).iloc[0]["aspect_text"]
                    if not subset_aspect_df.empty
                    else "-"
                )
                render_kpi("Top strength", top_strength, "điểm mạnh nổi bật")

            show_aspect_block(
                subset_aspect_df,
                top_n=top_n,
                min_mentions=min_mentions,
                title_prefix=f"Demo N = {len(review_ids)}",
            )

            st.subheader("Comment-level table")
            if not subset_comment_df.empty:
                view_cols = [
                    c
                    for c in [
                        "review_id",
                        "author",
                        "raw_rating",
                        "overall_label",
                        "overall_confidence",
                        "n_sentences",
                        "n_matched_rows",
                    ]
                    if c in subset_comment_df.columns
                ]
                st.dataframe(
                    subset_comment_df[view_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("🤖 AI Management Summary")
            if st.button("Generate N-comments AI summary", key="btn_many_ai"):
                ai_text = generate_gemini_commentary(
                    summary_doc=subset_summary,
                    aspect_df=subset_aspect_df,
                    sample_rows=subset_pred_df,
                    scope_name=f"nhóm {len(review_ids)} comment",
                )
                st.info(ai_text)
