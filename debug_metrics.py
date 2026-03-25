#!/usr/bin/env python
"""Debug script to check what metrics are actually loaded"""
import sys
import json
sys.path.insert(0, '.')

from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings
from review_analytics.logging import logging
import pandas as pd

# Silent logging
logging.basicConfig(level=logging.ERROR)

s = load_settings()
store = MongoStore(s.mongodb_uri, s.mongodb_db)
col = store.collection('model_trials')

# Get one trial document
docs = list(col.find({}, {"_id": 0}).limit(1))

if docs:
    print("Sample document from model_trials collection:")
    print(f"  Keys: {list(docs[0].keys())}")
    
    metrics = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    for m in metrics:
        if m in docs[0]:
            print(f"  ✓ {m}: {docs[0][m]}")
        else:
            print(f"  ✗ {m}: MISSING")
else:
    print("No documents found in model_trials")
