#!/usr/bin/env python
"""Check and update model_trials with precision/recall metrics"""
import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings

def check_and_update():
    s = load_settings()
    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    col = store.collection('model_trials')
    
    # Check first document
    docs = list(col.find({}, {'_id': 0}).limit(1))
    
    if not docs:
        print("No documents in model_trials")
        return
    
    doc = docs[0]
    metrics = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    
    print("Current MongoDB model_trials metrics:")
    for m in metrics:
        has_it = m in doc
        val = doc.get(m, 'N/A')
        print(f"  {'✓' if has_it else '✗'} {m}: {val}")
    
    # Check if precision/recall are missing
    if 'test_precision_macro' not in doc or 'test_recall_macro' not in doc:
        print("\n⚠ Precision/Recall missing! Running update...")
        from TSN_Data.mongodb.push_all_model_trials import push_all_model_trials
        push_all_model_trials()
        print("✓ Update complete")
    else:
        print("\n✓ All metrics present!")

if __name__ == "__main__":
    check_and_update()
