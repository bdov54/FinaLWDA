import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd
from langdetect import detect, LangDetectException
from pymongo import UpdateOne

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException

# ===== regex =====
BASIC_PUNCT = r"\.\,\!\?\:\;\-\(\)\/"
RE_KEEP = re.compile(rf"[^\w\s{BASIC_PUNCT}]", flags=re.UNICODE)
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
RE_WS = re.compile(r"\s+")
SENT_SPLIT = re.compile(r"[\.!\?\n\r]+")

def clean_text(s: str, remove_url=True, remove_email=True, remove_special=True) -> str:
    s = str(s).replace("\u200b", "").strip()
    if remove_url:
        s = RE_URL.sub(" ", s)
    if remove_email:
        s = RE_EMAIL.sub(" ", s)
    if remove_special:
        s = RE_KEEP.sub(" ", s)
    s = RE_WS.sub(" ", s).strip()
    return s

def safe_detect_lang(text: str) -> str:
    if not isinstance(text, str):
        return "unk"
    if len(text.split()) < 3:
        return "unk"
    try:
        return detect(text)
    except LangDetectException:
        return "unk"

def split_sentences(text: str, min_len=2):
    parts = [s.strip() for s in SENT_SPLIT.split(str(text)) if s and s.strip()]
    return [s for s in parts if len(s) >= min_len]

def normalize_for_match(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def star_to_label(r: float) -> str:
    r = int(r)
    if r in [1, 2]:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"

label2id = {"negative": 0, "neutral": 1, "positive": 2}

ASPECT_VI = {
    "queue_waiting": "Xếp hàng & chờ đợi",
    "checkin_boarding": "Check-in & lên máy bay",
    "immigration_customs": "Nhập cảnh & hải quan",
    "staff_service": "Nhân viên & phục vụ",
    "transport_parking": "Di chuyển & bãi xe",
    "security_check": "An ninh & soi chiếu",
    "price_value": "Giá cả & chặt chém",
    "baggage": "Hành lý & băng chuyền",
    "cleanliness_toilets": "Vệ sinh & nhà vệ sinh",
    "food_beverage": "Đồ ăn & thức uống",
    "signage_information": "Biển chỉ dẫn & thông tin",
    "wifi_internet": "WiFi & Internet",
}

ASPECT_KEYWORDS_BY_LANG = {
    "en": {
        "security_check": ["security", "security check", "screening", "x-ray"],
        "immigration_customs": ["immigration", "passport control", "customs", "visa"],
        "checkin_boarding": ["check-in", "check in", "boarding", "gate", "flight", "transfer", "connecting flight"],
        "queue_waiting": ["queue", "line", "waiting", "wait time", "crowded"],
        "staff_service": ["staff", "employee", "service", "rude", "helpful"],
        "cleanliness_toilets": ["toilet", "restroom", "bathroom", "clean", "dirty"],
        "price_value": ["price", "expensive", "overpriced", "cheap"],
        "food_beverage": ["food", "restaurant", "cafe", "coffee"],
        "wifi_internet": ["wifi", "wi-fi", "internet", "connection"],
        "baggage": ["baggage", "luggage", "bag", "lost baggage", "baggage claim"],
        "transport_parking": ["taxi", "grab", "bus", "parking", "traffic"],
        "signage_information": ["sign", "signage", "information", "direction", "wayfinding"],
    },
    "vi": {
        "security_check": ["an ninh", "soi chiếu", "kiểm tra an ninh", "x-ray"],
        "immigration_customs": ["nhập cảnh", "xuất cảnh", "hải quan", "hộ chiếu", "visa"],
        "checkin_boarding": ["check-in", "làm thủ tục", "lên máy bay", "cổng", "gate", "chuyến bay", "nối chuyến"],
        "queue_waiting": ["xếp hàng", "chờ", "đợi", "đông", "kẹt"],
        "staff_service": ["nhân viên", "thái độ", "phục vụ", "hỗ trợ", "thô lỗ"],
        "cleanliness_toilets": ["vệ sinh", "sạch", "bẩn", "nhà vệ sinh", "toilet", "wc"],
        "price_value": ["giá", "đắt", "rẻ", "chặt chém"],
        "food_beverage": ["đồ ăn", "ăn uống", "quán", "nhà hàng", "cà phê"],
        "wifi_internet": ["wifi", "internet", "mạng", "kết nối"],
        "baggage": ["hành lý", "vali", "ký gửi", "thất lạc", "băng chuyền"],
        "transport_parking": ["taxi", "grab", "xe buýt", "bãi xe", "gửi xe", "đậu xe"],
        "signage_information": ["chỉ dẫn", "biển", "bảng", "thông tin", "hướng dẫn"],
    },
}

KEEP_LANGS = ['en', 'vi', 'unk', 'ko', 'ja', 'de', 'ru', 'fr', 'th', 'id', 'es', 'it', 'ar', 'zh-cn']

def get_kw_for_lang(lang: str):
    if lang == "unk":
        base_en = ASPECT_KEYWORDS_BY_LANG.get("en", {})
        base_vi = ASPECT_KEYWORDS_BY_LANG.get("vi", {})
        merged = {}
        for asp in set(base_en) | set(base_vi):
            merged[asp] = sorted(set(base_en.get(asp, []) + base_vi.get(asp, [])), key=len, reverse=True)
        return merged

    lang_kw = ASPECT_KEYWORDS_BY_LANG.get(lang, {})
    base_en = ASPECT_KEYWORDS_BY_LANG.get("en", {})
    merged = {}
    for asp in set(lang_kw) | set(base_en):
        merged[asp] = sorted(set(lang_kw.get(asp, []) + base_en.get(asp, [])), key=len, reverse=True)
    return merged

def match_aspects(sentence: str, kw_dict: dict):
    s_norm = normalize_for_match(sentence)
    hits = []
    for asp, kws in kw_dict.items():
        hit_kws = [kw for kw in kws if normalize_for_match(kw) in s_norm]
        if hit_kws:
            hits.append((asp, hit_kws))
    return hits

# ====== Aspect discovery (từ câu chưa match) ======
EN_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were","it","this","that",
    "at","as","be","been","but","by","from","i","we","you","they","he","she","my","our","your","their"
}
VI_STOP = {
    "và","là","thì","mà","có","không","rất","quá","cũng","đã","đang","tôi","mình","bạn","anh","chị","em",
    "như","ở","trên","dưới","với","cho","khi","để","vì","này","đó","nên","vào"
}

def simple_tokens(text: str):
    t = normalize_for_match(text)
    t = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split()

def discover_aspect_candidates(unmatched_by_lang: dict, top_k=30, min_count=15, ngram_range=(1,2)):
    """
    unmatched_by_lang: {lang: [sentence,...]}
    output: list dict {lang, phrase, count, samples[]}
    """
    results = []
    for lang, sents in unmatched_by_lang.items():
        if not sents:
            continue

        stop = EN_STOP if lang == "en" else (VI_STOP if lang == "vi" else EN_STOP)
        cnt = Counter()
        samples = defaultdict(list)

        for sent in sents:
            toks = [w for w in simple_tokens(sent) if w not in stop and len(w) >= 3]
            if not toks:
                continue

            # unigram
            if ngram_range[0] <= 1 <= ngram_range[1]:
                for w in toks:
                    cnt[w] += 1
                    if len(samples[w]) < 3:
                        samples[w].append(sent)

            # bigram
            if ngram_range[0] <= 2 <= ngram_range[1]:
                for i in range(len(toks)-1):
                    bg = f"{toks[i]} {toks[i+1]}"
                    cnt[bg] += 1
                    if len(samples[bg]) < 3:
                        samples[bg].append(sent)

        # filter + top
        cand = [(p,c) for p,c in cnt.items() if c >= min_count]
        cand.sort(key=lambda x: x[1], reverse=True)
        cand = cand[:top_k]

        for phrase, c in cand:
            results.append({
                "lang": lang,
                "phrase": phrase,
                "count": int(c),
                "samples": samples[phrase],
            })

    return results

def run_preprocess(save_to_mongo: bool = True, discover_aspects: bool = False, topk: int = 30):
    s = load_settings()

    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    store.ensure_indexes(s.col_raw, s.col_processed, s.col_raw_web)

    raw_col = store.collection(s.col_raw)
    docs = list(raw_col.find({}, {"_id": 0}))
    if not docs:
        raise RuntimeError("Mongo raw_data đang rỗng. Chạy ingest trước.")

    df_raw = pd.DataFrame(docs)

    for c in ["review_id", "rating", "text"]:
        if c not in df_raw.columns:
            raise ValueError(f"raw_data thiếu cột {c}. Hiện có: {df_raw.columns.tolist()}")

    df = df_raw.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)].copy()

    df["text_raw"] = df["text"].astype("string").fillna("").str.strip()
    bad_literals = {"nan", "none", "null", "na"}
    mask_bad = df["text_raw"].str.lower().isin(bad_literals)
    df.loc[mask_bad, "text_raw"] = ""

    df["text_clean"] = df["text_raw"].apply(clean_text)
    df = df[df["text_clean"].ne("")].copy()

    df["label"] = df["rating"].apply(star_to_label)
    df["label_id"] = df["label"].map(label2id)

    df["lang"] = df["text_clean"].apply(safe_detect_lang)
    df = df[df["lang"].isin(KEEP_LANGS)].copy()

    now = datetime.now(timezone.utc)

    records = []
    unmatched_by_lang = defaultdict(list)

    for _, row in df.iterrows():
        review_id = str(row["review_id"])
        lang = row["lang"]
        label = row["label"]
        label_id = int(row["label_id"])
        review_created_at = row.get("created_at", None)

        kw_dict = get_kw_for_lang(lang)
        sents = split_sentences(row["text_clean"])

        for sid, sent in enumerate(sents):
            hits = match_aspects(sent, kw_dict)
            if not hits:
                if discover_aspects:
                    unmatched_by_lang[lang].append(sent)
                continue

            for asp, hit_kws in hits:
                records.append({
                    "review_id": review_id,
                    "lang": lang,
                    "sent_id": int(sid),
                    "sentence": sent,
                    "aspect": asp,
                    "hit_keywords": ", ".join(hit_kws[:10]),
                    "label_id": label_id,
                    "label": label,
                    "aspect_vi": ASPECT_VI.get(asp, asp),
                    "review_created_at": review_created_at,
                    "created_at": now,
                    "updated_at": now
                })

    absa_df = pd.DataFrame(records)
    logger.info("✅ ABSA rows: %d", len(absa_df))

    # save processed csv
    s.processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    absa_df.to_csv(s.processed_csv_path, index=False, encoding="utf-8-sig")
    logger.info("✅ Saved processed csv: %s", s.processed_csv_path)

    # save transform artifacts
    transform_dir = s.artifact_dir / "transform" / now.strftime("%Y%m%d_%H%M%S")
    transform_dir.mkdir(parents=True, exist_ok=True)
    (transform_dir / "keep_langs.json").write_text(json.dumps(KEEP_LANGS, ensure_ascii=False, indent=2), encoding="utf-8")
    (transform_dir / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (transform_dir / "aspect_vi.json").write_text(json.dumps(ASPECT_VI, ensure_ascii=False, indent=2), encoding="utf-8")
    (transform_dir / "aspect_keywords_by_lang.json").write_text(
        json.dumps(ASPECT_KEYWORDS_BY_LANG, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # ===== discover aspects: export candidates =====
    if discover_aspects:
        candidates = discover_aspect_candidates(unmatched_by_lang, top_k=topk, min_count=15, ngram_range=(1,2))
        cand_path_json = transform_dir / "aspect_candidates.json"
        cand_path_csv  = transform_dir / "aspect_candidates.csv"
        pd.DataFrame(candidates).to_csv(cand_path_csv, index=False, encoding="utf-8-sig")
        cand_path_json.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("✅ Saved aspect candidates: %s | %s", cand_path_csv, cand_path_json)

    logger.info("✅ Saved transform artifacts: %s", transform_dir)

    if save_to_mongo:
        proc_col = store.collection(s.col_processed)

        ops = []
        for d in absa_df.to_dict(orient="records"):
            key = {"review_id": d["review_id"], "sent_id": d["sent_id"], "aspect": d["aspect"]}

            created_at = d.get("created_at", now)

            set_doc = d.copy()
            # ✅ tránh conflict: created_at chỉ để $setOnInsert
            set_doc.pop("created_at", None)
            set_doc["updated_at"] = now

            ops.append(
                UpdateOne(
                    key,
                    {"$set": set_doc, "$setOnInsert": {"created_at": created_at}},
                    upsert=True
                )
            )

        if ops:
            res = proc_col.bulk_write(ops, ordered=False)
            logger.info(
                "✅ Upserted processed docs to Mongo: matched=%d modified=%d upserted=%d",
                res.matched_count, res.modified_count, len(res.upserted_ids)
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Preprocess Mongo raw_data -> processed_data")
    ap.add_argument("--no-mongo", action="store_true", help="Không ghi processed_data vào Mongo")
    ap.add_argument("--discover-aspects", action="store_true", help="Xuất candidate aspect từ câu chưa match")
    ap.add_argument("--topk", type=int, default=30, help="Số candidate phrase mỗi ngôn ngữ")
    args = ap.parse_args()

    if not args.run:
        print("Use: python -m review_analytics.pipeline.preprocess --run")
        return

    try:
        run_preprocess(
            save_to_mongo=(not args.no_mongo),
            discover_aspects=args.discover_aspects,
            topk=args.topk
        )
    except Exception as e:
        logger.error("Preprocess failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)

if __name__ == "__main__":
    main()