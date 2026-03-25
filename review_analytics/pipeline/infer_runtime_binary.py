from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.pipeline.preprocess_binary import (
    clean_text,
    safe_detect_lang,
    split_sentences,
    get_kw_for_lang,
    match_aspects,
    get_aspect_text,
)

try:
    from google import genai
except Exception:
    genai = None


SERVING_MODEL_COL = "serving_model_binary_v1"
PRED_SENTENCE_COL = "pred_sentence_aspect_binary_v1"
PRED_COMMENT_COL = "pred_comment_summary_binary_v1"

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}


def get_store() -> MongoStore:
    s = load_settings()
    return MongoStore(s.mongodb_uri, s.mongodb_db)


def get_raw_collection_name() -> str:
    s = load_settings()
    return s.col_raw


def get_active_model_doc(store: Optional[MongoStore] = None) -> Dict:
    store = store or get_store()
    doc = store.collection(SERVING_MODEL_COL).find_one({"slot": "active"}, {"_id": 0})
    if not doc:
        raise RuntimeError(
            f"Không tìm thấy model active trong collection '{SERVING_MODEL_COL}'. "
            "Hãy chạy set_serving_model_binary.py trước."
        )
    return doc


@lru_cache(maxsize=2)
def load_model_and_tokenizer(model_dir: str, tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict_binary_pair(
    sentence: str,
    aspect_text: str,
    model_dir: str,
    tokenizer_dir: str,
    max_length: int = 128,
) -> Dict:
    tokenizer, model = load_model_and_tokenizer(model_dir, tokenizer_dir)

    inputs = tokenizer(
        aspect_text,
        sentence,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    prob_negative = float(probs[0])
    prob_positive = float(probs[1])
    pred_idx = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_idx]
    confidence = float(np.max(probs))

    return {
        "prob_negative": prob_negative,
        "prob_positive": prob_positive,
        "pred_label": pred_label,
        "confidence": confidence,
    }


def build_rows_from_custom_comment(comment_text: str) -> pd.DataFrame:
    text_clean = clean_text(comment_text)
    if not text_clean:
        return pd.DataFrame()

    lang = safe_detect_lang(text_clean)
    kw_dict = get_kw_for_lang(lang)
    sents = split_sentences(text_clean)

    rows = []
    for sid, sent in enumerate(sents):
        hits = match_aspects(sent, kw_dict)
        if not hits:
            continue

        # giữ tất cả aspect match được trong custom input
        for aspect, hit_kws in hits:
            rows.append(
                {
                    "review_id": "manual_input",
                    "lang": lang,
                    "sent_id": sid,
                    "sentence": sent,
                    "aspect": aspect,
                    "aspect_text": get_aspect_text(aspect, lang),
                    "hit_keywords": ", ".join(hit_kws[:10]),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates(subset=["sent_id", "aspect"]).reset_index(drop=True)


def build_comment_text_from_pred_df(pred_df: pd.DataFrame) -> str:
    if pred_df is None or pred_df.empty:
        return ""
    tmp = pred_df[["sent_id", "sentence"]].drop_duplicates().sort_values("sent_id")
    return ". ".join(tmp["sentence"].astype(str).tolist()).strip()


def aggregate_aspect_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(
            columns=[
                "aspect",
                "aspect_text",
                "mentions",
                "negative_count",
                "positive_count",
                "negative_rate",
                "positive_rate",
                "avg_prob_negative",
                "avg_prob_positive",
                "net_score",
                "priority_score",
                "strength_score",
                "avg_confidence",
            ]
        )

    grp = (
        pred_df.groupby(["aspect", "aspect_text"], dropna=False)
        .agg(
            mentions=("aspect", "size"),
            negative_count=("pred_label", lambda s: int((s == "negative").sum())),
            positive_count=("pred_label", lambda s: int((s == "positive").sum())),
            avg_prob_negative=("prob_negative", "mean"),
            avg_prob_positive=("prob_positive", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    grp["negative_rate"] = grp["negative_count"] / grp["mentions"]
    grp["positive_rate"] = grp["positive_count"] / grp["mentions"]
    grp["net_score"] = grp["avg_prob_positive"] - grp["avg_prob_negative"]
    grp["priority_score"] = grp["mentions"] * grp["negative_rate"]
    grp["strength_score"] = grp["mentions"] * grp["positive_rate"]

    float_cols = [
        "avg_prob_negative",
        "avg_prob_positive",
        "avg_confidence",
        "negative_rate",
        "positive_rate",
        "net_score",
        "priority_score",
        "strength_score",
    ]
    grp[float_cols] = grp[float_cols].round(4)

    return grp.sort_values(["mentions", "negative_rate"], ascending=[False, False]).reset_index(drop=True)


def build_comment_summary_doc(pred_df: pd.DataFrame, review_id: str, extra_fields: Optional[Dict] = None) -> Dict:
    if pred_df is None or pred_df.empty:
        doc = {
            "review_id": review_id,
            "n_sentences": 0,
            "n_matched_rows": 0,
            "overall_prob_negative": 0.0,
            "overall_prob_positive": 0.0,
            "overall_label": None,
            "overall_confidence": 0.0,
            "aspects_mentioned": [],
            "aspect_summary": [],
        }
        if extra_fields:
            doc.update(extra_fields)
        return doc

    aspect_df = aggregate_aspect_table(pred_df)

    overall_prob_negative = float(pred_df["prob_negative"].mean())
    overall_prob_positive = float(pred_df["prob_positive"].mean())
    overall_label = "negative" if overall_prob_negative >= overall_prob_positive else "positive"
    overall_confidence = max(overall_prob_negative, overall_prob_positive)

    doc = {
        "review_id": review_id,
        "n_sentences": int(pred_df["sent_id"].nunique()) if "sent_id" in pred_df.columns else int(pred_df["sentence"].nunique()),
        "n_matched_rows": int(len(pred_df)),
        "overall_prob_negative": round(overall_prob_negative, 4),
        "overall_prob_positive": round(overall_prob_positive, 4),
        "overall_label": overall_label,
        "overall_confidence": round(overall_confidence, 4),
        "aspects_mentioned": aspect_df["aspect"].tolist(),
        "aspect_summary": aspect_df.to_dict(orient="records"),
    }
    if extra_fields:
        doc.update(extra_fields)
    return doc


def heuristic_commentary(summary_doc: Dict, aspect_df: pd.DataFrame, scope_name: str) -> str:
    if aspect_df is None or aspect_df.empty:
        return f"Chưa có aspect nào được match trong {scope_name}, nên chưa thể rút ra kết luận đủ mạnh."

    top_neg = aspect_df.sort_values(["negative_rate", "mentions"], ascending=[False, False]).head(3)
    top_pos = aspect_df.sort_values(["positive_rate", "mentions"], ascending=[False, False]).head(3)

    neg_names = ", ".join(top_neg["aspect_text"].tolist()) if not top_neg.empty else "chưa rõ"
    pos_names = ", ".join(top_pos["aspect_text"].tolist()) if not top_pos.empty else "chưa rõ"

    overall = summary_doc.get("overall_label", "unknown")
    if overall == "negative":
        return (
            f"Tổng thể {scope_name} đang nghiêng về tiêu cực. "
            f"Các điểm cần ưu tiên kiểm tra là: {neg_names}. "
            f"Những điểm đang được phản hồi tốt hơn gồm: {pos_names}. "
            f"Ban quản lý nên kiểm tra trực tiếp các pain points trước, sau đó chuẩn hóa quy trình vận hành ở các điểm yếu lặp lại."
        )

    return (
        f"Tổng thể {scope_name} đang nghiêng về tích cực. "
        f"Các điểm được đánh giá tốt gồm: {pos_names}. "
        f"Tuy vậy vẫn cần theo dõi các khía cạnh có tín hiệu tiêu cực như: {neg_names}. "
        f"Ban quản lý nên duy trì các điểm mạnh hiện có và rà soát định kỳ các khía cạnh dễ phát sinh phàn nàn."
    )


def generate_gemini_commentary(
    summary_doc: Dict,
    aspect_df: pd.DataFrame,
    sample_rows: Optional[pd.DataFrame] = None,
    scope_name: str = "tập dữ liệu",
) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "").strip()

    if genai is None or not api_key or not model_name:
        return heuristic_commentary(summary_doc, aspect_df, scope_name)

    if aspect_df is None or aspect_df.empty:
        return heuristic_commentary(summary_doc, aspect_df, scope_name)

    top_neg = aspect_df.sort_values(["negative_rate", "mentions"], ascending=[False, False]).head(5)
    top_pos = aspect_df.sort_values(["positive_rate", "mentions"], ascending=[False, False]).head(5)

    samples = []
    if sample_rows is not None and not sample_rows.empty:
        cols = [c for c in ["sentence", "aspect_text", "pred_label", "prob_negative", "prob_positive"] if c in sample_rows.columns]
        samples = sample_rows[cols].head(12).to_dict(orient="records")

    prompt = f"""
Bạn là chuyên gia phân tích trải nghiệm hành khách tại sân bay. Hãy cung cấp báo cáo chi tiết, chuyên sâu.

Phạm vi phân tích: {scope_name}

Dữ liệu tổng hợp:
{summary_doc}

Top 5 aspect tiêu cực (negative rate cao nhất):
{top_neg.to_dict(orient="records")}

Top 5 aspect tích cực (positive rate cao nhất):
{top_pos.to_dict(orient="records")}

Mẫu phản hồi:
{samples}

HƯỚNG DẪN VIẾT:
- Viết bằng tiếng Việt, chuyên sâu và cụ thể
- Mỗi phần từ 150-200 từ, dùng dẫn chứng từ dữ liệu
- Chia thành ĐÚNG 3 phần với tiêu đề rõ ràng:

### 1. NHẬN ĐỊNH TỔNG QUAN
- Tóm tắt tình hình chung (sentiment ratio, xu hướng chính)
- Xác định những khía cạnh cần ưu tiên nhất
- Liên hệ giữa các vấn đề nếu có

### 2. PHÂN TÍCH CHI TIẾT: VẤNĐỀ VÀ ĐIỂM MẠNH
- Liệt kê TOP vấn đề tiêu cực (tên khía cạnh + % âm + ví dụ)
- Chi tiết từng vấn đề: nguyên nhân, tác động
- Làm nổi bật những điểm mạnh hiện nay
- So sánh tương phản giữa yếu và mạnh

### 3. KHUYẾN NGHỊ HÀNH ĐỘNG
- Ưu tiên xử lý cấp độ (ngay, 1 tháng, 3 tháng)
- Các bước hành động cụ thể, có thể thực hiện
- Chỉ số theo dõi để đánh giá tiến độ
- Liên kết với các điểm mạnh hiện nay

YÊUẦU: Không dùng markdown bullet quá ngắn. Dùng đoạn văn có chiều sâu, lập luận rõ ràng."""

    # Debug logging
    print(f"\n[GEMINI DEBUG] scope_name={scope_name}")
    print(f"[GEMINI DEBUG] aspect_df shape: {aspect_df.shape}")
    print(f"[GEMINI DEBUG] top_neg records: {len(top_neg)}, top_pos records: {len(top_pos)}")
    print(f"[GEMINI DEBUG] samples count: {len(samples)}")
    print(f"[GEMINI DEBUG] prompt length: {len(prompt)} chars")
    print(f"[GEMINI DEBUG] api_key set: {bool(api_key)}, model: {model_name}")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        text = text.strip()
        
        # Debug response
        print(f"[GEMINI DEBUG] response length: {len(text)} chars")
        print(f"[GEMINI DEBUG] response preview: {text[:300]}...")
        
        return text if text else heuristic_commentary(summary_doc, aspect_df, scope_name)
    except Exception as e:
        print(f"[GEMINI ERROR] {type(e).__name__}: {str(e)}")
        return heuristic_commentary(summary_doc, aspect_df, scope_name)


def generate_model_comparison_commentary(compare_df):
    """Generate Gemini commentary comparing mBERT vs XLM-R models."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "").strip()

    if genai is None or not api_key or not model_name:
        return "Không có API key hoặc Gemini package chưa được cài."

    if compare_df is None or compare_df.empty:
        return "Không có model để so sánh."

    models_info = []
    for i, row in compare_df.iterrows():
        models_info.append({
            "model": row.get("model", "Unknown"),
            "trial_name": row.get("trial_name", ""),
            "test_f1_macro": f"{row.get('test_f1_macro', 0):.4f}",
            "test_accuracy": f"{row.get('test_accuracy', 0):.4f}",
            "test_precision_macro": f"{row.get('test_precision_macro', 0):.4f}",
            "test_recall_macro": f"{row.get('test_recall_macro', 0):.4f}",
            "train_runtime": f"{row.get('train_runtime', 0):.2f}s",
            "train_samples_per_second": f"{row.get('train_samples_per_second', 0):.2f}",
            "max_len": row.get("max_len", 80),
            "threshold": f"{row.get('threshold', 0.5):.2f}",
        })

    prompt = f"""Hãy phân tích và so sánh chi tiết hai mô hình Transformer tốt nhất cho bài toán phân tích cảm xúc tiếng Việt:

{models_info}

Cung cấp báo cáo so sánh kỹ thuật chi tiết với 4 phần:

1. TỔNG HỢP HIỆU SUẤT
- So sánh tất cả các chỉ số (F1, Accuracy, Precision, Recall)
- Mô hình nào vượt trội trong những khía cạnh nào?
- Sự chênh lệch hiệu suất giữa hai mô hình là bao nhiêu?

2. PHÂN TÍCH ƯUTHU VÀ NHƯ ƠC ĐIỂM
- XLM-R: những ưu điểm so với mBERT
- mBERT: những ưu điểm riêng của nó
- Sự cân bằng giữa độ chính xác vs tốc độ

3. PHÂN TÍCH HIỆU SUẤT TÍNH TOÁN
- Thời gian huấn luyện và throughput
- Yêu cầu tài nguyên tính toán
- Khả năng triển khai trong thực tế sản xuất

4. KHUYẾN NGHỊ LỰA CHỌN
- Nên chọn mô hình nào cho sản xuất?
- Các yếu tố quyết định (độ chính xác, tốc độ, chi phí)
- Hướng dẫn sử dụng từng mô hình trong các trường hợp khác nhau

Mỗi phần từ 120-150 từ, sử dụng con số cụ thể từ dữ liệu."""

    print(f"[MODEL_COMPARISON] DataFrame shape: {compare_df.shape}, Models: {len(models_info)}")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = getattr(response, "text", "") or ""
        return text.strip() if text else "Không nhận được response từ Gemini."
    except Exception as e:
        print(f"[MODEL_COMPARISON ERROR] {str(e)}")
        return f"Lỗi Gemini: {str(e)}"