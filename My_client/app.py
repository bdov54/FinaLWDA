import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent              # My_client/
PROJECT_ROOT = BASE_DIR.parent                          # project root
DEFAULT_DATA_FILE = BASE_DIR / "absa_output.csv"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "model_batches"
DEFAULT_TRANSFORM_ROOT = PROJECT_ROOT / "artifacts" / "transform"

# =========================================================
# OPTIONAL DEPENDENCIES (lazy-safe)
# =========================================================
try:
    import torch
except Exception:
    torch = None

try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Airport Review Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME TOKENS
# =========================================================
THEME = {
    "bg": "#0B1020",
    "panel": "#111A33",
    "panel2": "#0E1730",
    "border": "rgba(255,255,255,0.08)",
    "grid": "rgba(255,255,255,0.10)",
    "axis": "rgba(255,255,255,0.18)",
    "text": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.70)",
    "muted2": "rgba(255,255,255,0.55)",
    "accent": "#20C7FF",
    "accent2": "#3EA8FF",
    "good": "#7CDE77",
    "warn": "#F5C451",
    "bad": "#FF5C6C",
}

SENTIMENT_COLORS = {
    "negative": THEME["bad"],
    "neutral": THEME["warn"],
    "positive": THEME["good"],
}

# =========================================================
# FALLBACK ASPECT MAPS
# =========================================================
FALLBACK_ASPECT_VI = {
    "queue_waiting": "Xếp hàng và chờ đợi",
    "checkin_boarding": "Check-in và lên máy bay",
    "immigration_customs": "Nhập cảnh và hải quan",
    "staff_service": "Nhân viên và phục vụ",
    "transport_parking": "Di chuyển và bãi xe",
    "security_check": "An ninh và soi chiếu",
    "price_value": "Giá cả và giá trị",
    "baggage": "Hành lý và băng chuyền",
    "cleanliness_toilets": "Vệ sinh và nhà vệ sinh",
    "food_beverage": "Đồ ăn và thức uống",
    "signage_information": "Biển chỉ dẫn và thông tin",
    "wifi_internet": "WiFi và Internet",
}

FALLBACK_ASPECT_EN = {
    "queue_waiting": "queue and waiting",
    "checkin_boarding": "check-in and boarding",
    "immigration_customs": "immigration and customs",
    "staff_service": "staff and service",
    "transport_parking": "transport and parking",
    "security_check": "security screening",
    "price_value": "price and value",
    "baggage": "baggage handling",
    "cleanliness_toilets": "cleanliness and toilets",
    "food_beverage": "food and beverage",
    "signage_information": "signage and information",
    "wifi_internet": "wifi and internet",
}

# =========================================================
# CSS
# =========================================================
CUSTOM_CSS = f"""
<style>
:root {{
  --bg: {THEME["bg"]};
  --panel: {THEME["panel"]};
  --panel2: {THEME["panel2"]};
  --border: {THEME["border"]};
  --text: {THEME["text"]};
  --muted: {THEME["muted"]};
  --muted2: {THEME["muted2"]};
  --primary: {THEME["accent"]};
  --primary2: {THEME["accent2"]};
  --good: {THEME["good"]};
  --warn: {THEME["warn"]};
  --bad: {THEME["bad"]};
}}

body, [data-testid="stAppViewContainer"], .stApp {{
  background: radial-gradient(1200px 600px at 30% 0%, rgba(32,199,255,0.10) 0%, rgba(11,16,32,0.00) 55%),
              radial-gradient(900px 500px at 80% 20%, rgba(124,222,119,0.08) 0%, rgba(11,16,32,0.00) 55%),
              var(--bg) !important;
  color: var(--text) !important;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 100%;
}}

h1,h2,h3,h4,h5,h6 {{
  color: var(--text) !important;
  font-weight: 850 !important;
  letter-spacing: -0.02em;
}}

p, span, div, label {{
  color: var(--muted) !important;
}}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(17,26,51,0.95) 0%, rgba(14,23,48,0.95) 100%) !important;
  border-right: 1px solid var(--border) !important;
}}

.kpi, .card {{
  background: linear-gradient(180deg, rgba(17,26,51,0.92) 0%, rgba(14,23,48,0.92) 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.35) !important;
}}

.kpi {{
  padding: 18px 18px !important;
}}

.kpi-title {{
  font-size: 0.82rem !important;
  color: var(--muted2) !important;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 10px !important;
  font-weight: 700 !important;
}}

.kpi-value {{
  font-size: 2.0rem !important;
  font-weight: 900 !important;
  color: var(--text) !important;
  line-height: 1.05 !important;
}}

.kpi-sub {{
  margin-top: 10px !important;
  font-size: 0.85rem !important;
  color: var(--muted2) !important;
  font-weight: 600 !important;
}}

.badge {{
  display:inline-block;
  padding: 8px 14px !important;
  border-radius: 999px !important;
  font-size: 0.86rem !important;
  background: rgba(32,199,255,0.10) !important;
  border: 1px solid rgba(32,199,255,0.25) !important;
  color: var(--primary) !important;
  font-weight: 800 !important;
}}

.small-note {{
  color: var(--muted2) !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
}}

.table-wrap {{
  background: linear-gradient(180deg, rgba(17,26,51,0.92) 0%, rgba(14,23,48,0.92) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}}

.table-scroll {{
  max-height: 420px;
  overflow: auto;
}}

.table-wrap table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}}

.table-wrap thead th {{
  position: sticky;
  top: 0;
  z-index: 5;
  background: rgba(32,199,255,0.08);
  color: var(--text);
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.10);
  white-space: nowrap;
}}

.table-wrap tbody td {{
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  color: var(--muted);
  vertical-align: top;
}}

.table-wrap tbody tr:nth-child(odd) td {{
  background: rgba(255,255,255,0.02);
}}

.table-wrap tbody tr:hover td {{
  background: rgba(32,199,255,0.10);
  color: var(--text);
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# PLOT HELPERS
# =========================================================
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color=THEME["text"], size=12, family="Inter, Arial"),
    margin=dict(l=10, r=10, t=10, b=10),
)

def style_axes(fig, height=None, xgrid=True, ygrid=False, xformat=None):
    fig.update_layout(**PLOT_THEME)
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(
        showgrid=xgrid,
        gridwidth=1,
        gridcolor=THEME["grid"],
        linecolor=THEME["axis"],
        zeroline=False,
        tickfont=dict(color=THEME["muted"], size=11),
        tickformat=xformat if xformat else None,
    )
    fig.update_yaxes(
        showgrid=ygrid,
        gridwidth=1,
        gridcolor=THEME["grid"],
        linecolor=THEME["axis"],
        zeroline=False,
        tickfont=dict(color=THEME["muted"], size=11),
    )
    fig.update_layout(legend=dict(font=dict(color=THEME["muted"])))
    return fig

def pct(x: float) -> str:
    return f"{x*100:.1f}%"

def render_table(df: pd.DataFrame, height: int = 420):
    if df is None or len(df) == 0:
        st.info("Không có dữ liệu để hiển thị.")
        return
    html = df.to_html(index=False, escape=True)
    st.markdown(
        f"""
        <div class="table-wrap">
          <div class="table-scroll" style="max-height:{height}px;">
            {html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def safe_read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default

def path_exists(path_like) -> bool:
    try:
        return Path(path_like).exists()
    except Exception:
        return False

# =========================================================
# LOADERS
# =========================================================
@st.cache_data
def load_absa_data(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_csv(path)

    required = ["label", "aspect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột {missing}. Các cột hiện có: {list(df.columns)}")

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["aspect"] = df["aspect"].astype(str).str.strip()

    if "sentence" in df.columns:
        df["sentence"] = df["sentence"].astype(str)
    if "hit_keywords" in df.columns:
        df["hit_keywords"] = df["hit_keywords"].astype(str)

    return df

@st.cache_data
def discover_batches(artifact_root_str: str) -> List[str]:
    root = Path(artifact_root_str)
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in dirs]

@st.cache_data
def find_latest_transform_dir(transform_root_str: str) -> Optional[str]:
    root = Path(transform_root_str)
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(dirs[0])

@st.cache_data
def load_transform_maps(transform_root_str: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    latest_dir_str = find_latest_transform_dir(transform_root_str)
    if not latest_dir_str:
        return FALLBACK_ASPECT_VI, FALLBACK_ASPECT_EN

    latest_dir = Path(latest_dir_str)
    aspect_vi = safe_read_json(latest_dir / "aspect_vi.json", FALLBACK_ASPECT_VI)
    aspect_en = safe_read_json(latest_dir / "aspect_en.json", FALLBACK_ASPECT_EN)

    if not aspect_vi:
        aspect_vi = FALLBACK_ASPECT_VI
    if not aspect_en:
        aspect_en = FALLBACK_ASPECT_EN

    return aspect_vi, aspect_en

@st.cache_data
def load_runs_from_batch(batch_dir_str: str) -> List[Dict]:
    batch_dir = Path(batch_dir_str)
    runs_dir = batch_dir / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        config = safe_read_json(run_dir / "training_config.json", {})
        metrics = safe_read_json(run_dir / "metrics.json", {})
        report = safe_read_json(run_dir / "classification_report.json", {})
        trainer_state = safe_read_json(run_dir / "trainer_state_summary.json", {})
        cm = safe_read_json(run_dir / "confusion_matrix.json", [])
        label2id = safe_read_json(run_dir / "label2id.json", {})
        id2label = safe_read_json(run_dir / "id2label.json", {})
        charts_dir = run_dir / "charts"

        runs.append({
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "display_name": config.get("display_name", run_dir.name),
            "model_name": config.get("model_name", ""),
            "config": config,
            "metrics": metrics,
            "report": report,
            "trainer_state": trainer_state,
            "confusion_matrix": cm,
            "label2id": label2id,
            "id2label": id2label,
            "model_dir": run_dir / "model",
            "tokenizer_dir": run_dir / "tokenizer",
            "charts": {
                "loss": next(iter(charts_dir.glob("*loss_curve.png")), None),
                "val": next(iter(charts_dir.glob("*val_metrics_curve.png")), None),
                "grad": next(iter(charts_dir.glob("*grad_norm.png")), None),
                "cm": next(iter(charts_dir.glob("*confusion_matrix.png")), None),
                "class_metrics": next(iter(charts_dir.glob("*per_class_metrics.png")), None),
                "overall": next(iter(charts_dir.glob("*overall_test_metrics.png")), None),
            },
            "comparison_dir": batch_dir / "comparison",
        })

    return runs

@st.cache_resource
def load_model_and_tokenizer(model_dir_str: str, tokenizer_dir_str: str):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Thiếu thư viện transformers. Hãy cài: pip install transformers torch"
        ) from e

    if torch is None:
        raise RuntimeError("Thiếu torch. Hãy cài: pip install torch")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_str, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir_str)
    model.eval()
    return tokenizer, model

# =========================================================
# MODEL / INFERENCE HELPERS
# =========================================================
def detect_lang_for_demo(text: str) -> str:
    if detect is None:
        return "vi"
    try:
        return "vi" if detect(text) == "vi" else "en"
    except Exception:
        return "vi"

def get_aspect_texts(lang: str, aspect_vi: Dict[str, str], aspect_en: Dict[str, str], all_aspects: List[str]) -> Dict[str, str]:
    out = {}
    for asp in all_aspects:
        if lang == "vi":
            out[asp] = aspect_vi.get(asp, asp.replace("_", " "))
        else:
            out[asp] = aspect_en.get(asp, asp.replace("_", " "))
    return out

def normalize_id2label(id2label: Dict) -> Dict[int, str]:
    out = {}
    for k, v in id2label.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out

def predict_comment_aspects(
    text: str,
    run_info: Dict,
    threshold: float,
    aspect_vi: Dict[str, str],
    aspect_en: Dict[str, str],
    all_aspects: List[str],
) -> Tuple[str, float, pd.DataFrame]:
    tokenizer, model = load_model_and_tokenizer(str(run_info["model_dir"]), str(run_info["tokenizer_dir"]))
    lang = detect_lang_for_demo(text)
    aspect_texts = get_aspect_texts(lang, aspect_vi, aspect_en, all_aspects)

    id2label = normalize_id2label(run_info.get("id2label", {}))
    if not id2label:
        # fallback common binary
        id2label = {0: "negative", 1: "positive"}

    rows = []
    with torch.no_grad():
        for aspect_key, aspect_text in aspect_texts.items():
            inputs = tokenizer(
                aspect_text,
                text,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_label = id2label.get(pred_idx, str(pred_idx))
            confidence = float(np.max(probs))

            row = {
                "aspect": aspect_key,
                "aspect_text": aspect_text,
                "label": pred_label,
                "confidence": confidence,
            }

            for idx, p in enumerate(probs):
                row[f"prob_{id2label.get(idx, idx)}"] = float(p)

            rows.append(row)

    pred_df = pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)

    # overall = mean probability over aspects
    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    overall_scores = pred_df[prob_cols].mean().to_dict()
    overall_prob_col = max(overall_scores, key=overall_scores.get)
    overall_label = overall_prob_col.replace("prob_", "")
    overall_conf = float(overall_scores[overall_prob_col])

    pred_df["is_strong"] = pred_df.apply(
        lambda r: r.get(f"prob_{r['label']}", 0.0) >= threshold,
        axis=1,
    )

    return overall_label, overall_conf, pred_df

def build_sentiment_score(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    if "prob_positive" in df.columns and "prob_negative" in df.columns:
        df["sentiment_score"] = df["prob_positive"] - df["prob_negative"]
    else:
        df["sentiment_score"] = 0.0
        if "label" in df.columns:
            df.loc[df["label"] == "positive", "sentiment_score"] = df["confidence"]
            df.loc[df["label"] == "negative", "sentiment_score"] = -df["confidence"]
    return df

def heuristic_commentary(text: str, overall_label: str, pred_df: pd.DataFrame) -> str:
    if "prob_negative" in pred_df.columns:
        top_neg = pred_df.sort_values("prob_negative", ascending=False).head(3)
        neg_aspects = [r["aspect_text"] for _, r in top_neg.iterrows() if r.get("prob_negative", 0) >= 0.55]
    else:
        neg_aspects = pred_df[pred_df["label"] == "negative"]["aspect_text"].head(3).tolist()

    if "prob_positive" in pred_df.columns:
        top_pos = pred_df.sort_values("prob_positive", ascending=False).head(3)
        pos_aspects = [r["aspect_text"] for _, r in top_pos.iterrows() if r.get("prob_positive", 0) >= 0.55]
    else:
        pos_aspects = pred_df[pred_df["label"] == "positive"]["aspect_text"].head(3).tolist()

    if overall_label == "negative":
        if neg_aspects:
            return f"Nhận xét tự động: comment này nghiêng về tiêu cực. Các khía cạnh tiêu cực nổi bật là {', '.join(neg_aspects)}. Nên ưu tiên rà soát các pain points này."
        return "Nhận xét tự động: comment này nghiêng về tiêu cực, nhưng tín hiệu theo từng aspect chưa quá mạnh."
    if overall_label == "positive":
        if pos_aspects:
            return f"Nhận xét tự động: comment này nghiêng về tích cực. Các khía cạnh được đánh giá tốt là {', '.join(pos_aspects)}. Đây có thể là những điểm mạnh nên duy trì."
        return "Nhận xét tự động: comment này nghiêng về tích cực, nhưng tín hiệu theo từng aspect chưa thật sự nổi trội."
    return "Nhận xét tự động: comment này mang xu hướng trung tính hoặc pha trộn giữa khen và chê."

def ai_commentary(text: str, overall_label: str, pred_df: pd.DataFrame) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return heuristic_commentary(text, overall_label, pred_df)

    try:
        client = OpenAI(api_key=api_key)

        top_rows = pred_df.head(6).to_dict(orient="records")
        prompt = f"""
Bạn là trợ lý phân tích phản hồi sân bay.

Comment:
{text}

Overall prediction:
{overall_label}

Aspect predictions:
{json.dumps(top_rows, ensure_ascii=False, indent=2)}

Hãy viết:
1) Một đoạn nhận xét ngắn cho người dùng không kỹ thuật
2) Một đoạn gợi ý hành động cho ban quản lý

Ngắn gọn, rõ ràng, không dùng bullet.
"""
        response = client.responses.create(
            model="gpt-5.4",
            input=prompt,
        )
        return response.output_text.strip()
    except Exception:
        return heuristic_commentary(text, overall_label, pred_df)

# =========================================================
# MAIN LOAD
# =========================================================
st.markdown("## ✈️ Airport Review Intelligence Dashboard")
st.markdown(
    "<span class='badge'>ABSA • Sentiment • Aspect Insights • Model Comparison • Demo</span> &nbsp; "
    "<span class='small-note'>Xem dashboard, so sánh mô hình và demo inference comment</span>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## ⚙️ Controls")

data_file = st.sidebar.text_input("ABSA CSV", value=str(DEFAULT_DATA_FILE))
artifact_root_input = st.sidebar.text_input("Artifact root", value=str(DEFAULT_ARTIFACT_ROOT))
transform_root_input = st.sidebar.text_input("Transform root", value=str(DEFAULT_TRANSFORM_ROOT))

# debug light
with st.sidebar.expander("Path status"):
    st.write("DATA_FILE exists:", Path(data_file).exists())
    st.write("ARTIFACT_ROOT exists:", Path(artifact_root_input).exists())
    st.write("TRANSFORM_ROOT exists:", Path(transform_root_input).exists())
    st.write("Torch ready:", torch is not None)

try:
    df = load_absa_data(data_file)
except Exception as e:
    st.error(f"Không load được file dữ liệu: {e}")
    st.stop()

aspect_vi_map, aspect_en_map = load_transform_maps(transform_root_input)

available_labels = sorted(df["label"].dropna().unique().tolist())
label_order = [x for x in ["negative", "neutral", "positive"] if x in available_labels]
if not label_order:
    label_order = available_labels

all_aspects = sorted(df["aspect"].dropna().unique().tolist())

aspect_pick = st.sidebar.multiselect("Aspect", options=all_aspects, default=[])
label_pick = st.sidebar.multiselect("Sentiment", options=available_labels, default=available_labels)
min_rows = st.sidebar.slider("Min samples per aspect", 1, 200, 10, 1)
topN = st.sidebar.slider("Top N aspects", 5, 25, 12)
show_raw = st.sidebar.toggle("Show raw rows", value=False)
show_samples = st.sidebar.toggle("Show sample sentences", value=True)

filtered = df.copy()
if aspect_pick:
    filtered = filtered[filtered["aspect"].isin(aspect_pick)]
if label_pick:
    filtered = filtered[filtered["label"].isin(label_pick)]

if len(filtered) == 0:
    st.warning("Không có dữ liệu sau khi filter.")
    st.stop()

def compute_aspect_summary(data: pd.DataFrame, min_rows_local: int) -> pd.DataFrame:
    ct = pd.crosstab(data["aspect"], data["label"])
    for col in label_order:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[label_order].copy()
    ct["total"] = ct.sum(axis=1)

    ct["neg_rate"] = (ct["negative"] / ct["total"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) if "negative" in ct.columns else 0.0
    ct["neu_rate"] = (ct["neutral"] / ct["total"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) if "neutral" in ct.columns else 0.0
    ct["pos_rate"] = (ct["positive"] / ct["total"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) if "positive" in ct.columns else 0.0
    ct["net_score"] = ct["pos_rate"] - ct["neg_rate"]

    ct = ct[ct["total"] >= min_rows_local].sort_values("total", ascending=False)
    return ct

aspect_summary = compute_aspect_summary(filtered, min_rows)

batch_options = discover_batches(artifact_root_input)
selected_batch = None
if batch_options:
    selected_batch = st.sidebar.selectbox(
        "Model batch",
        options=batch_options,
        index=0,
        format_func=lambda x: Path(x).name,
    )
else:
    st.sidebar.warning("Không tìm thấy batch artifact.")

runs = load_runs_from_batch(selected_batch) if selected_batch else []
run_name_to_info = {r["run_name"]: r for r in runs}

# =========================================================
# KPI
# =========================================================
total = len(filtered)
neg_pct = (filtered["label"] == "negative").mean() if "negative" in filtered["label"].values else 0.0
neu_pct = (filtered["label"] == "neutral").mean() if "neutral" in filtered["label"].values else 0.0
pos_pct = (filtered["label"] == "positive").mean() if "positive" in filtered["label"].values else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f"<div class='kpi'><div class='kpi-title'>Total records</div><div class='kpi-value' style='color:{THEME['accent']} !important'>{total:,}</div><div class='kpi-sub'>rows after filters</div></div>",
    unsafe_allow_html=True,
)
k2.markdown(
    f"<div class='kpi'><div class='kpi-title'>Negative</div><div class='kpi-value' style='color:{THEME['bad']} !important'>{pct(neg_pct)}</div><div class='kpi-sub'>complaints / dissatisfaction</div></div>",
    unsafe_allow_html=True,
)
k3.markdown(
    f"<div class='kpi'><div class='kpi-title'>Neutral</div><div class='kpi-value' style='color:{THEME['warn']} !important'>{pct(neu_pct)}</div><div class='kpi-sub'>informational / mixed</div></div>",
    unsafe_allow_html=True,
)
k4.markdown(
    f"<div class='kpi'><div class='kpi-title'>Positive</div><div class='kpi-value' style='color:{THEME['good']} !important'>{pct(pos_pct)}</div><div class='kpi-sub'>praise / satisfaction</div></div>",
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
tab_overview, tab_pain, tab_strength, tab_models, tab_demo, tab_report = st.tabs(
    ["📊 Overview", "🔥 Pain Points", "🌟 Strengths", "🧪 Model Comparison", "📝 Demo Inference", "📄 Report"]
)

# =========================================================
# OVERVIEW
# =========================================================
with tab_overview:
    cL, cR = st.columns([1, 1])

    with cL:
        st.subheader("Sentiment Distribution")
        s_counts = filtered["label"].value_counts().reindex(label_order).fillna(0).astype(int).reset_index()
        s_counts.columns = ["label", "count"]
        fig = px.pie(
            s_counts,
            names="label",
            values="count",
            hole=0.55,
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_layout(**PLOT_THEME, height=360, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with cR:
        st.subheader(f"Top {topN} Most Mentioned Aspects")
        a_counts = filtered["aspect"].value_counts().head(topN).reset_index()
        a_counts.columns = ["aspect", "count"]
        fig2 = px.bar(a_counts, x="count", y="aspect", orientation="h")
        fig2 = style_axes(fig2, height=360, xgrid=True, ygrid=False)
        fig2.update_traces(marker=dict(color=THEME["accent"], line=dict(color=THEME["axis"], width=1)))
        fig2.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"Sentiment by Aspect (Top {topN})")
    if len(aspect_summary) == 0:
        st.info("Không còn aspect nào đủ dữ liệu sau filter/min_rows.")
    else:
        top_as = aspect_summary.head(topN).copy().reset_index()
        value_vars = [c for c in ["neg_rate", "neu_rate", "pos_rate"] if c in top_as.columns]
        ratio_long = top_as.melt(
            id_vars=["aspect"],
            value_vars=value_vars,
            var_name="metric",
            value_name="rate",
        )
        ratio_long["sentiment"] = ratio_long["metric"].map(
            {"neg_rate": "negative", "neu_rate": "neutral", "pos_rate": "positive"}
        )
        fig3 = px.bar(
            ratio_long,
            x="rate",
            y="aspect",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            orientation="h",
            barmode="stack",
        )
        fig3 = style_axes(fig3, height=520, xgrid=True, ygrid=False, xformat=".0%")
        fig3.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# PAIN POINTS
# =========================================================
with tab_pain:
    st.subheader("🔥 Pain Points")
    if len(aspect_summary) == 0:
        st.info("Không có đủ dữ liệu.")
    else:
        pain = aspect_summary.sort_values(["neg_rate", "total"], ascending=[False, False]).head(topN).reset_index()
        figp = px.bar(
            pain,
            x="neg_rate",
            y="aspect",
            orientation="h",
            text=pain["neg_rate"].map(lambda x: f"{x*100:.1f}%"),
        )
        figp = style_axes(figp, height=520, xgrid=True, ygrid=False, xformat=".0%")
        figp.update_traces(marker=dict(color=THEME["bad"]))
        figp.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(figp, use_container_width=True)

        show_cols = [c for c in ["aspect", "total", "negative", "neutral", "positive", "neg_rate", "pos_rate", "net_score"] if c in pain.columns]
        render_table(pain[show_cols], height=360)

    if show_samples and "sentence" in filtered.columns:
        st.subheader("Sample Negative Sentences")
        neg_rows = filtered[filtered["label"] == "negative"].copy()
        cols = [c for c in ["aspect", "sentence", "hit_keywords"] if c in neg_rows.columns]
        render_table(neg_rows[cols].head(30), height=420)

# =========================================================
# STRENGTHS
# =========================================================
with tab_strength:
    st.subheader("🌟 Strengths")
    if len(aspect_summary) == 0:
        st.info("Không có đủ dữ liệu.")
    else:
        strong = aspect_summary.sort_values(["pos_rate", "total"], ascending=[False, False]).head(topN).reset_index()
        figs = px.bar(
            strong,
            x="pos_rate",
            y="aspect",
            orientation="h",
            text=strong["pos_rate"].map(lambda x: f"{x*100:.1f}%"),
        )
        figs = style_axes(figs, height=520, xgrid=True, ygrid=False, xformat=".0%")
        figs.update_traces(marker=dict(color=THEME["good"]))
        figs.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(figs, use_container_width=True)

        show_cols = [c for c in ["aspect", "total", "negative", "neutral", "positive", "pos_rate", "neg_rate", "net_score"] if c in strong.columns]
        render_table(strong[show_cols], height=360)

    if show_samples and "sentence" in filtered.columns:
        st.subheader("Sample Positive Sentences")
        pos_rows = filtered[filtered["label"] == "positive"].copy()
        cols = [c for c in ["aspect", "sentence", "hit_keywords"] if c in pos_rows.columns]
        render_table(pos_rows[cols].head(30), height=420)

# =========================================================
# MODEL COMPARISON
# =========================================================
with tab_models:
    st.subheader("🧪 Kết quả so sánh giữa các mô hình")

    if not selected_batch:
        st.warning("Chưa tìm thấy batch artifact.")
    elif not runs:
        st.warning("Batch này không có run.")
    else:
        summary_rows = []
        for r in runs:
            summary_rows.append({
                "display_name": r["display_name"],
                "model_name": r["model_name"],
                "best_val_f1": r["trainer_state"].get("best_metric"),
                "test_accuracy": r["metrics"].get("test_accuracy"),
                "test_precision_macro": r["metrics"].get("test_precision_macro"),
                "test_recall_macro": r["metrics"].get("test_recall_macro"),
                "test_f1_macro": r["metrics"].get("test_f1_macro"),
                "test_f1_weighted": r["metrics"].get("test_f1_weighted"),
                "run_name": r["run_name"],
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)

        st.markdown("### Bảng tóm tắt")
        render_table(summary_df, height=260)

        fig_cmp = px.bar(
            summary_df,
            x="display_name",
            y=["test_accuracy", "test_f1_macro", "test_f1_weighted"],
            barmode="group",
            title="So sánh overall metrics",
        )
        fig_cmp = style_axes(fig_cmp, height=420, xgrid=False, ygrid=True)
        st.plotly_chart(fig_cmp, use_container_width=True)

        comp_dir = Path(selected_batch) / "comparison"
        colA, colB = st.columns(2)
        with colA:
            if path_exists(comp_dir / "compare_overall_metrics.png"):
                st.image(str(comp_dir / "compare_overall_metrics.png"), caption="Saved comparison: overall metrics")
        with colB:
            if path_exists(comp_dir / "compare_best_val_f1.png"):
                st.image(str(comp_dir / "compare_best_val_f1.png"), caption="Saved comparison: best val F1")

        st.markdown("### Chi tiết từng mô hình")
        for r in runs:
            with st.expander(f"{r['display_name']} • {r['model_name']} • {r['run_name']}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Best val F1", f"{r['trainer_state'].get('best_metric', 0):.4f}")
                c2.metric("Test Acc", f"{r['metrics'].get('test_accuracy', 0):.4f}")
                c3.metric("Test F1 Macro", f"{r['metrics'].get('test_f1_macro', 0):.4f}")
                c4.metric("Test F1 Weighted", f"{r['metrics'].get('test_f1_weighted', 0):.4f}")

                st.caption(f"Run dir: {r['run_dir']}")

                cc1, cc2 = st.columns(2)
                with cc1:
                    if path_exists(r["charts"]["loss"]):
                        st.image(str(r["charts"]["loss"]), caption="Loss curve")
                    if path_exists(r["charts"]["cm"]):
                        st.image(str(r["charts"]["cm"]), caption="Confusion matrix")
                    if path_exists(r["charts"]["overall"]):
                        st.image(str(r["charts"]["overall"]), caption="Overall test metrics")
                with cc2:
                    if path_exists(r["charts"]["val"]):
                        st.image(str(r["charts"]["val"]), caption="Validation metrics")
                    if path_exists(r["charts"]["grad"]):
                        st.image(str(r["charts"]["grad"]), caption="Gradient norm")
                    if path_exists(r["charts"]["class_metrics"]):
                        st.image(str(r["charts"]["class_metrics"]), caption="Per-class metrics")

                cfg_df = pd.DataFrame([r["config"]]).T.reset_index()
                cfg_df.columns = ["field", "value"]
                st.markdown("#### Training config")
                render_table(cfg_df, height=320)

                if r["report"]:
                    rep_rows = []
                    for cls_name, vals in r["report"].items():
                        if isinstance(vals, dict):
                            rep_rows.append({
                                "class": cls_name,
                                "precision": vals.get("precision"),
                                "recall": vals.get("recall"),
                                "f1-score": vals.get("f1-score"),
                                "support": vals.get("support"),
                            })
                    rep_df = pd.DataFrame(rep_rows)
                    st.markdown("#### Classification report")
                    render_table(rep_df, height=260)

# =========================================================
# DEMO INFERENCE
# =========================================================
with tab_demo:
    st.subheader("📝 Demo inference comment → overall sentiment + aspect sentiment")

    if not runs:
        st.warning("Chưa có run để demo inference. Hãy train xong rồi chọn batch artifact.")
    else:
        selected_run_name = st.selectbox(
            "Chọn mô hình để demo",
            options=[r["run_name"] for r in runs],
            format_func=lambda x: f"{run_name_to_info[x]['display_name']} • {run_name_to_info[x]['model_name']}",
        )
        selected_run = run_name_to_info[selected_run_name]

        if not path_exists(selected_run["model_dir"]) or not path_exists(selected_run["tokenizer_dir"]):
            st.error("Không tìm thấy thư mục model/tokenizer đã train trong artifact run này.")
        elif torch is None:
            st.error("Thiếu torch. Hãy cài: pip install torch transformers")
        else:
            demo_text = st.text_area(
                "Nhập comment",
                height=140,
                value="Nhân viên khá lịch sự nhưng khu soi chiếu quá chậm và nhà vệ sinh chưa sạch.",
            )
            threshold = st.slider("Ngưỡng aspect mạnh", 0.50, 0.90, 0.55, 0.01)

            if st.button("Run demo", type="primary"):
                if not demo_text.strip():
                    st.warning("Bạn cần nhập comment.")
                else:
                    try:
                        with st.spinner("Đang dự đoán..."):
                            overall_label, overall_conf, pred_df = predict_comment_aspects(
                                demo_text.strip(),
                                selected_run,
                                threshold,
                                aspect_vi_map,
                                aspect_en_map,
                                all_aspects,
                            )
                            ai_text = ai_commentary(demo_text.strip(), overall_label, pred_df)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Overall label", overall_label)
                        c2.metric("Overall confidence", f"{overall_conf:.3f}")
                        c3.metric("Model", selected_run["display_name"])

                        strong_df = pred_df[pred_df["is_strong"]].copy()
                        neg_df = strong_df[strong_df["label"] == "negative"].copy()
                        pos_df = strong_df[strong_df["label"] == "positive"].copy()
                        neu_df = strong_df[strong_df["label"] == "neutral"].copy()

                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown("#### Negative aspects")
                            if len(neg_df) == 0:
                                st.info("Chưa có aspect negative vượt ngưỡng.")
                            else:
                                show_cols = [c for c in ["aspect_text", "prob_negative", "confidence"] if c in neg_df.columns]
                                render_table(neg_df[show_cols].head(8), height=260)

                        with cols[1]:
                            st.markdown("#### Neutral aspects")
                            if len(neu_df) == 0:
                                st.info("Chưa có aspect neutral vượt ngưỡng.")
                            else:
                                show_cols = [c for c in ["aspect_text", "prob_neutral", "confidence"] if c in neu_df.columns]
                                render_table(neu_df[show_cols].head(8), height=260)

                        with cols[2]:
                            st.markdown("#### Positive aspects")
                            if len(pos_df) == 0:
                                st.info("Chưa có aspect positive vượt ngưỡng.")
                            else:
                                show_cols = [c for c in ["aspect_text", "prob_positive", "confidence"] if c in pos_df.columns]
                                render_table(pos_df[show_cols].head(8), height=260)

                        plot_df = build_sentiment_score(pred_df)
                        fig_demo = px.bar(
                            plot_df.sort_values("sentiment_score"),
                            x="sentiment_score",
                            y="aspect_text",
                            orientation="h",
                            color="label",
                            color_discrete_map=SENTIMENT_COLORS,
                            title="Aspect sentiment score",
                        )
                        fig_demo = style_axes(fig_demo, height=520, xgrid=True, ygrid=False)
                        st.plotly_chart(fig_demo, use_container_width=True)

                        st.markdown("### AI Commentary")
                        st.info(ai_text)

                        st.markdown("### Full aspect table")
                        show_cols = ["aspect", "aspect_text", "label", "confidence"]
                        extra_cols = [c for c in ["prob_negative", "prob_neutral", "prob_positive"] if c in pred_df.columns]
                        render_table(pred_df[show_cols + extra_cols], height=420)

                    except Exception as e:
                        st.error(f"Demo inference lỗi: {e}")

# =========================================================
# REPORT
# =========================================================
with tab_report:
    st.subheader("📄 Executive Summary")

    if len(aspect_summary) > 0:
        most_mentioned = aspect_summary.index[0]
        worst = aspect_summary.sort_values("neg_rate", ascending=False).index[0] if "neg_rate" in aspect_summary.columns else "-"
        best = aspect_summary.sort_values("pos_rate", ascending=False).index[0] if "pos_rate" in aspect_summary.columns else "-"

        st.markdown(
            f"""
- **Khía cạnh được nhắc nhiều nhất:** `{most_mentioned}`  
- **Pain point lớn nhất:** `{worst}`  
- **Strength nổi bật:** `{best}`  
"""
        )

        st.markdown("#### Aspect summary table")
        out_preview = aspect_summary.reset_index().copy()
        render_table(out_preview.head(50), height=420)
    else:
        st.info("Chưa có aspect đủ dữ liệu để tạo summary.")

    if len(aspect_summary) > 0:
        out = aspect_summary.reset_index().copy()
        for c in ["neg_rate", "neu_rate", "pos_rate", "net_score"]:
            if c in out.columns:
                out[c] = (out[c] * 100).round(2)
        csv1 = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download aspect_summary.csv",
            data=csv1,
            file_name="aspect_summary.csv",
            mime="text/csv",
        )

    if show_raw:
        csv2 = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download filtered_rows.csv",
            data=csv2,
            file_name="filtered_rows.csv",
            mime="text/csv",
        )

# =========================================================
# RAW PREVIEW
# =========================================================
if show_raw:
    st.subheader("Raw rows (preview)")
    render_table(filtered.head(200), height=520)