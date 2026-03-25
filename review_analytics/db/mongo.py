from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from review_analytics.logging import logger

class MongoStore:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def collection(self, name: str) -> Collection:
        return self.db[name]

    def ensure_indexes(
        self,
        raw_col: str,
        processed_col: str,
        raw_web_col: str,
        processed_binary_col: str | None = None,
    ):
        # ---------- RAW ----------
        self.collection(raw_col).create_index([("review_id", ASCENDING)], unique=True)
        self.collection(raw_col).create_index([("created_at", ASCENDING)])
        self.collection(raw_col).create_index([("updated_at", ASCENDING)])

        # ---------- PROCESSED ----------
        self.collection(processed_col).create_index(
            [("review_id", ASCENDING), ("sent_id", ASCENDING), ("aspect", ASCENDING)],
            unique=True
        )
        self.collection(processed_col).create_index([("created_at", ASCENDING)])
        self.collection(processed_col).create_index([("updated_at", ASCENDING)])

        # ---------- PROCESSED BINARY (2 lớp mới) ----------
        if processed_binary_col:
            self.collection(processed_binary_col).create_index(
                [("review_id", ASCENDING), ("sent_id", ASCENDING), ("aspect", ASCENDING)],
                unique=True
            )
            self.collection(processed_binary_col).create_index([("created_at", ASCENDING)])
            self.collection(processed_binary_col).create_index([("updated_at", ASCENDING)])
            self.collection(processed_binary_col).create_index([("label", ASCENDING)])
            self.collection(processed_binary_col).create_index([("label_id", ASCENDING)])
            self.collection(processed_binary_col).create_index([("lang", ASCENDING)])


        # ---------- RAW FROM WEB ----------
        self.collection(raw_web_col).create_index([("created_at", ASCENDING)])
        self.collection(raw_web_col).create_index([("updated_at", ASCENDING)])

        logger.info("✅ Mongo indexes ensured (unique + created_at/updated_at).")