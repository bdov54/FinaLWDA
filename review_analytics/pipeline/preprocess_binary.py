import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langdetect import detect, LangDetectException
from pymongo import UpdateOne

from review_analytics.config import load_settings
from review_analytics.db.mongo import MongoStore
from review_analytics.logging import logger
from review_analytics.exception.exception import ReviewAnalyticsException


DEFAULT_BINARY_COLLECTION = "processed_data_binary_123vs45_v2"

BASIC_PUNCT = r"\.\,\!\?\:\;\-\(\)\/"
RE_KEEP = re.compile(rf"[^\w\s{BASIC_PUNCT}]", flags=re.UNICODE)
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
RE_WS = re.compile(r"\s+")
SENT_SPLIT = re.compile(r"[\.!\?\n\r]+")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

KEEP_LANGS = ["en", "vi", "unk", "ko", "ja", "de", "ru", "fr", "th", "id", "es", "it", "ar", "zh-cn"]

ASPECT_VI = {
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

ASPECT_EN = {
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

ASPECT_KEYWORDS_BY_LANG = {
    "en": {
        "security_check": [
            "security",
            "security check",
            "security screening",
            "screening",
            "screening area",
            "security line",
            "security queue",
            "security lane",
            "airport security",
            "x-ray",
            "xray",
            "scanner",
            "body scan",
            "metal detector",
            "bag check",
            "baggage screening",
            "carry-on screening",
            "inspection",
            "security officer",
            "security staff",
        ],
        "immigration_customs": [
            "immigration",
            "passport control",
            "passport check",
            "passport inspection",
            "border control",
            "immigration line",
            "immigration queue",
            "immigration counter",
            "immigration officer",
            "customs",
            "customs check",
            "customs inspection",
            "visa",
            "visa check",
            "entry procedure",
            "exit procedure",
            "arrival immigration",
            "departure immigration",
        ],
        "checkin_boarding": [
            "check-in",
            "check in",
            "self check-in",
            "check-in counter",
            "check-in desk",
            "boarding",
            "boarding gate",
            "boarding pass",
            "boarding process",
            "boarding queue",
            "gate",
            "gate area",
            "gate change",
            "flight gate",
            "transfer",
            "transfer desk",
            "transit",
            "transit desk",
            "connecting flight",
            "connection",
            "layover",
            "missed connection",
        ],
        "queue_waiting": [
            "queue",
            "line",
            "waiting",
            "wait time",
            "waiting time",
            "long queue",
            "long line",
            "long wait",
            "waiting line",
            "stood in line",
            "standing in line",
            "slow line",
            "slow queue",
            "crowded",
            "too crowded",
            "overcrowded",
            "packed",
            "congested",
            "delay in queue",
        ],
        "staff_service": [
            "staff",
            "employee",
            "airport staff",
            "ground staff",
            "counter staff",
            "service",
            "customer service",
            "staff attitude",
            "officer attitude",
            "helpful",
            "very helpful",
            "not helpful",
            "friendly",
            "friendly staff",
            "unfriendly",
            "polite",
            "professional",
            "unprofessional",
            "rude",
            "very rude",
            "supportive",
            "ignored us",
        ],
        "cleanliness_toilets": [
            "toilet",
            "toilets",
            "restroom",
            "rest room",
            "bathroom",
            "washroom",
            "public toilet",
            "toilet area",
            "restroom area",
            "clean",
            "dirty",
            "clean toilet",
            "dirty toilet",
            "clean restroom",
            "dirty restroom",
            "smelly toilet",
            "smelly restroom",
            "bad smell in toilet",
            "restroom hygiene",
            "toilet hygiene",
        ],
        "price_value": [
            "price",
            "prices",
            "expensive",
            "too expensive",
            "overpriced",
            "pricey",
            "cheap",
            "reasonable price",
            "reasonable prices",
            "good value",
            "poor value",
            "worth the price",
            "not worth it",
            "rip-off",
            "rip off",
            "extra charge",
            "hidden fee",
            "cost too much",
        ],
        "food_beverage": [
            "food",
            "food court",
            "food options",
            "restaurant",
            "restaurants",
            "cafe",
            "coffee",
            "coffee shop",
            "drink",
            "drinks",
            "beverage",
            "meal",
            "snack",
            "dining",
            "dining options",
            "expensive food",
            "good food",
            "bad food",
        ],
        "wifi_internet": [
            "wifi",
            "wi-fi",
            "internet",
            "internet access",
            "wireless internet",
            "wifi connection",
            "wifi speed",
            "wifi login",
            "wifi password",
            "free wifi",
            "poor wifi",
            "slow wifi",
            "unstable wifi",
            "cannot connect",
            "could not connect",
            "network connection",
        ],
        "baggage": [
            "baggage",
            "baggage claim",
            "baggage belt",
            "baggage carousel",
            "baggage handling",
            "luggage",
            "bag",
            "bags",
            "checked bag",
            "checked baggage",
            "carry-on bag",
            "carry on bag",
            "lost baggage",
            "lost luggage",
            "delayed baggage",
            "damaged baggage",
            "damaged luggage",
            "bag drop",
            "bag drop counter",
            "bag claim",
        ],
        "transport_parking": [
            "taxi",
            "taxi stand",
            "taxi queue",
            "taxi pickup",
            "taxi fare",
            "grab",
            "grab pickup",
            "uber pickup",
            "bus",
            "airport bus",
            "bus stop",
            "shuttle",
            "shuttle bus",
            "shuttle service",
            "parking",
            "parking lot",
            "car park",
            "parking fee",
            "pickup zone",
            "drop-off zone",
            "drop off area",
            "pick up area",
            "traffic",
            "traffic jam",
        ],
        "signage_information": [
            "sign",
            "signs",
            "signage",
            "sign board",
            "signboards",
            "information",
            "information desk",
            "direction",
            "directions",
            "direction sign",
            "direction board",
            "wayfinding",
            "airport map",
            "unclear signs",
            "poor signage",
            "confusing signs",
            "confusing layout",
            "hard to find gate",
            "hard to navigate",
            "not well marked",
        ],
    },
    "vi": {
        "security_check": [
            "an ninh",
            "kiểm tra an ninh",
            "kiểm soát an ninh",
            "soi chiếu",
            "soi hành lý",
            "máy soi",
            "máy quét",
            "máy scan",
            "cổng từ",
            "x-ray",
            "xray",
            "khu an ninh",
            "hàng an ninh",
            "làn an ninh",
            "nhân viên an ninh",
            "kiểm tra thủ công",
        ],
        "immigration_customs": [
            "nhập cảnh",
            "xuất cảnh",
            "hải quan",
            "hộ chiếu",
            "visa",
            "kiểm tra hộ chiếu",
            "kiểm soát hộ chiếu",
            "quầy nhập cảnh",
            "quầy xuất cảnh",
            "hàng nhập cảnh",
            "hàng xuất cảnh",
            "cán bộ nhập cảnh",
            "cán bộ xuất nhập cảnh",
            "kiểm tra visa",
            "kiểm tra hải quan",
            "thủ tục hải quan",
        ],
        "checkin_boarding": [
            "check-in",
            "check in",
            "làm thủ tục",
            "quầy check-in",
            "quầy làm thủ tục",
            "boarding",
            "lên máy bay",
            "cổng",
            "gate",
            "cổng lên máy bay",
            "cổng ra máy bay",
            "thẻ lên máy bay",
            "boarding pass",
            "đổi cổng",
            "chuyển cổng",
            "nối chuyến",
            "chuyến bay nối",
            "transit",
            "transfer",
            "quầy transit",
            "quầy transfer",
            "lỡ chuyến nối",
        ],
        "queue_waiting": [
            "xếp hàng",
            "chờ",
            "đợi",
            "thời gian chờ",
            "xếp hàng lâu",
            "chờ lâu",
            "đợi lâu",
            "đứng chờ",
            "đứng xếp hàng",
            "đông",
            "quá đông",
            "đông đúc",
            "đông nghịt",
            "chen chúc",
            "kẹt",
            "ùn tắc",
            "quá tải",
            "chậm chạp",
        ],
        "staff_service": [
            "nhân viên",
            "thái độ",
            "thái độ nhân viên",
            "phục vụ",
            "hỗ trợ",
            "nhân viên hỗ trợ",
            "nhân viên quầy",
            "nhân viên sân bay",
            "nhân viên an ninh",
            "cán bộ",
            "thái độ phục vụ",
            "nhiệt tình",
            "hỗ trợ nhiệt tình",
            "không hỗ trợ",
            "thiếu hỗ trợ",
            "thân thiện",
            "lịch sự",
            "chuyên nghiệp",
            "thiếu chuyên nghiệp",
            "thô lỗ",
            "khó chịu",
        ],
        "cleanliness_toilets": [
            "vệ sinh",
            "sạch",
            "bẩn",
            "nhà vệ sinh",
            "toilet",
            "wc",
            "phòng vệ sinh",
            "khu vệ sinh",
            "vệ sinh toilet",
            "nhà vệ sinh bẩn",
            "nhà vệ sinh sạch",
            "toilet bẩn",
            "toilet sạch",
            "mùi hôi",
            "hôi nhà vệ sinh",
            "bốc mùi",
            "thiếu giấy",
            "thiếu xà phòng",
            "vệ sinh kém",
        ],
        "price_value": [
            "giá",
            "đắt",
            "rẻ",
            "giá cao",
            "giá đắt",
            "đắt đỏ",
            "quá đắt",
            "chặt chém",
            "không đáng tiền",
            "đáng tiền",
            "giá hợp lý",
            "mức giá hợp lý",
            "phụ thu",
            "thu thêm",
            "tốn tiền",
            "hợp túi tiền",
        ],
        "food_beverage": [
            "đồ ăn",
            "ăn uống",
            "quán",
            "quán ăn",
            "nhà hàng",
            "khu ăn uống",
            "quầy ăn",
            "cà phê",
            "quán cà phê",
            "coffee",
            "đồ uống",
            "thức ăn",
            "món ăn",
            "đồ ăn đắt",
            "đồ ăn ngon",
            "đồ ăn dở",
        ],
        "wifi_internet": [
            "wifi",
            "wi-fi",
            "internet",
            "mạng",
            "kết nối",
            "kết nối mạng",
            "kết nối internet",
            "wifi miễn phí",
            "wifi chậm",
            "wifi yếu",
            "wifi kém",
            "mạng chậm",
            "không vào được wifi",
            "không kết nối được",
            "mất kết nối",
            "đăng nhập wifi",
        ],
        "baggage": [
            "hành lý",
            "vali",
            "ký gửi",
            "gửi hành lý",
            "quầy ký gửi",
            "thất lạc",
            "thất lạc hành lý",
            "mất hành lý",
            "băng chuyền",
            "băng tải hành lý",
            "nhận hành lý",
            "trễ hành lý",
            "hành lý đến chậm",
            "hành lý hỏng",
            "vali hỏng",
            "bag drop",
        ],
        "transport_parking": [
            "taxi",
            "grab",
            "xe buýt",
            "bus sân bay",
            "xe trung chuyển",
            "shuttle",
            "bãi xe",
            "gửi xe",
            "đậu xe",
            "phí gửi xe",
            "phí đỗ xe",
            "điểm đón taxi",
            "hàng taxi",
            "điểm đón grab",
            "khu đón khách",
            "khu trả khách",
            "điểm đón",
            "điểm trả",
            "kẹt xe",
            "ùn tắc giao thông",
        ],
        "signage_information": [
            "chỉ dẫn",
            "biển",
            "bảng",
            "thông tin",
            "hướng dẫn",
            "biển chỉ dẫn",
            "bảng chỉ dẫn",
            "biển báo",
            "bảng hướng dẫn",
            "quầy thông tin",
            "bản đồ sân bay",
            "khó tìm cổng",
            "khó tìm đường",
            "thiếu biển chỉ dẫn",
            "biển không rõ",
            "chỉ dẫn không rõ",
            "khó định hướng",
            "bố cục khó hiểu",
            "sơ đồ khó hiểu",
        ],
    },
}


def clean_text(s: str, remove_url: bool = True, remove_email: bool = True, remove_special: bool = True) -> str:
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


def split_sentences(text: str, min_len_chars: int = 5, min_tokens: int = 3) -> List[str]:
    parts = [s.strip() for s in SENT_SPLIT.split(str(text)) if s and s.strip()]
    out = []
    for s in parts:
        if len(s) < min_len_chars:
            continue
        if len(WORD_RE.findall(s)) < min_tokens:
            continue
        out.append(s)
    return out


def normalize_for_match(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def star_to_label_binary(r: float):
    r = int(r)
    if r in [1, 2, 3]:
        return "negative"
    if r in [4, 5]:
        return "positive"
    return None


def get_kw_for_lang(lang: str) -> Dict[str, List[str]]:
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


def contains_kw(s_norm: str, kw_norm: str) -> bool:
    if " " in kw_norm:
        return f" {kw_norm} " in f" {s_norm} "
    return re.search(rf"(?<!\w){re.escape(kw_norm)}(?!\w)", s_norm) is not None


def match_aspects(sentence: str, kw_dict: Dict[str, List[str]]):
    s_norm = normalize_for_match(sentence)
    hits = []
    for asp, kws in kw_dict.items():
        hit_kws = [kw for kw in kws if contains_kw(s_norm, normalize_for_match(kw))]
        if hit_kws:
            hits.append((asp, hit_kws))
    return hits


def get_aspect_text(aspect: str, lang: str) -> str:
    if lang == "vi":
        return ASPECT_VI.get(aspect, aspect.replace("_", " "))
    return ASPECT_EN.get(aspect, aspect.replace("_", " "))


def build_output_csv_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def run_preprocess(
    save_to_mongo: bool = True,
    output_col: str = DEFAULT_BINARY_COLLECTION,
    single_aspect_only: bool = True,
):
    s = load_settings()

    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    store.ensure_indexes(
        s.col_raw,
        s.col_processed,
        s.col_raw_web,
        processed_binary_col=output_col,
    )

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

    df["label"] = df["rating"].apply(star_to_label_binary)
    df = df[df["label"].notna()].copy()
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)

    df["lang"] = df["text_clean"].apply(safe_detect_lang)
    df = df[df["lang"].isin(KEEP_LANGS)].copy()

    now = datetime.now(timezone.utc)

    records = []
    stats = Counter()
    label_dist = Counter()
    aspect_dist = Counter()
    lang_dist = Counter()

    for _, row in df.iterrows():
        review_id = str(row["review_id"])
        lang = row["lang"]
        label = row["label"]
        label_id = int(row["label_id"])
        review_rating = int(row["rating"])
        review_created_at = row.get("created_at", None)

        kw_dict = get_kw_for_lang(lang)
        sents = split_sentences(row["text_clean"])

        stats["reviews_seen"] += 1
        stats["sentences_total"] += len(sents)

        for sid, sent in enumerate(sents):
            hits = match_aspects(sent, kw_dict)

            if not hits:
                stats["sentences_unmatched"] += 1
                continue

            if len(hits) > 1:
                stats["sentences_multi_aspect"] += 1
                if single_aspect_only:
                    continue

            if len(hits) == 1:
                stats["sentences_single_aspect"] += 1

            selected_hits = [hits[0]] if (single_aspect_only and len(hits) >= 1) else hits

            for asp, hit_kws in selected_hits:
                aspect_text = get_aspect_text(asp, lang)
                records.append({
                    "review_id": review_id,
                    "lang": lang,
                    "sent_id": int(sid),
                    "sentence": sent,
                    "aspect": asp,
                    "aspect_text": aspect_text,
                    "hit_keywords": ", ".join(hit_kws[:10]),
                    "label_id": label_id,
                    "label": label,
                    "review_rating": review_rating,
                    "review_created_at": review_created_at,
                    "created_at": now,
                    "updated_at": now,
                })
                stats["rows_created"] += 1
                label_dist[label] += 1
                aspect_dist[asp] += 1
                lang_dist[lang] += 1

    absa_df = pd.DataFrame(records)
    if absa_df.empty:
        raise RuntimeError("Không tạo được dòng dữ liệu nào sau preprocess. Hãy kiểm tra dictionary aspect.")

    absa_df = absa_df.drop_duplicates(subset=["review_id", "sent_id", "aspect"]).reset_index(drop=True)

    logger.info("✅ Binary processed rows: %d", len(absa_df))
    logger.info("Stats: %s", dict(stats))
    logger.info("Label dist: %s", dict(label_dist))
    logger.info("Aspect dist top10: %s", dict(aspect_dist.most_common(10)))
    logger.info("Lang dist: %s", dict(lang_dist))

    suffix = "binary_123vs45"
    out_csv = build_output_csv_path(s.processed_csv_path, suffix)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    absa_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("✅ Saved processed csv: %s", out_csv)

    transform_dir = Path(s.artifact_dir) / "transform" / f"{suffix}_{now.strftime('%Y%m%d_%H%M%S')}"
    transform_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "label_mode": "binary_123_vs_45",
        "output_collection": output_col,
        "single_aspect_only": single_aspect_only,
        "rows": int(len(absa_df)),
        "stats": dict(stats),
        "label_dist": dict(label_dist),
        "aspect_dist_top20": dict(aspect_dist.most_common(20)),
        "lang_dist": dict(lang_dist),
        "created_at": now.isoformat(),
    }

    (transform_dir / "preprocess_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (transform_dir / "label2id.json").write_text(
        json.dumps(LABEL2ID, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (transform_dir / "id2label.json").write_text(
        json.dumps(ID2LABEL, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("✅ Saved transform artifacts: %s", transform_dir)

    if save_to_mongo:
        proc_col = store.collection(output_col)
        ops = []

        for d in absa_df.to_dict(orient="records"):
            key = {"review_id": d["review_id"], "sent_id": d["sent_id"], "aspect": d["aspect"]}
            created_at = d.get("created_at", now)

            set_doc = d.copy()
            set_doc.pop("created_at", None)
            set_doc["updated_at"] = now

            ops.append(
                UpdateOne(
                    key,
                    {"$set": set_doc, "$setOnInsert": {"created_at": created_at}},
                    upsert=True,
                )
            )

        if ops:
            res = proc_col.bulk_write(ops, ordered=False)
            logger.info(
                "✅ Upserted docs to Mongo [%s]: matched=%d modified=%d upserted=%d",
                output_col,
                res.matched_count,
                res.modified_count,
                len(res.upserted_ids),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--no-mongo", action="store_true")
    ap.add_argument("--output-col", type=str, default=DEFAULT_BINARY_COLLECTION)
    ap.add_argument("--allow-multi-aspect", action="store_true")
    args = ap.parse_args()

    if not args.run:
        print("Use: python -m review_analytics.pipeline.preprocess_binary --run")
        return

    try:
        run_preprocess(
            save_to_mongo=(not args.no_mongo),
            output_col=args.output_col,
            single_aspect_only=(not args.allow_multi_aspect),
        )
    except Exception as e:
        logger.error("Preprocess binary failed: %s", str(e))
        raise ReviewAnalyticsException(e, sys)


if __name__ == "__main__":
    main()