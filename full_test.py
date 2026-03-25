#!/usr/bin/env python
"""Direct test of metrics in MongoDB and chart display"""
import json
from pathlib import Path

# Test 1: Check artifacts have all 4 metrics
print("=" * 60)
print("TEST 1: Check trial artifacts for all 4 metrics")
print("=" * 60)

artifact_path = Path("artifacts/model_batches/b123v45_tuned_20260316_082735/models/mBERT/trials/mbert_t002_len80_ep5_lr1e-05_bs8_ga1_cwsqrt_inv_ls0.03/metrics.json")

if artifact_path.exists():
    with open(artifact_path) as f:
        data = json.load(f)
    
    metrics = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    print(f"\n✓ Found metrics.json at: {artifact_path.name}")
    for m in metrics:
        val = data.get(m, None)
        status = "✓" if val is not None else "✗"
        print(f"  {status} {m}: {val}")
else:
    print(f"✗ metrics.json not found at {artifact_path}")

# Test 2: Check MongoDB model_trials
print("\n" + "=" * 60)
print("TEST 2: Check MongoDB model_trials collection")
print("=" * 60)

import sys
sys.path.insert(0, '.')

from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings

s = load_settings()
store = MongoStore(s.mongodb_uri, s.mongodb_db)
col = store.collection('model_trials')

docs = list(col.find({}, {'_id': 0}).limit(1))
if docs:
    doc = docs[0]
    metrics = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    print(f"\n✓ Found document in model_trials")
    for m in metrics:
        val = doc.get(m, None)
        status = "✓" if val is not None else "✗"
        print(f"  {status} {m}: {val}")
else:
    print("✗ No documents in model_trials")

# Test 3: Check what load_model_comparison_from_mongo returns
print("\n" + "=" * 60)
print("TEST 3: Check load_model_comparison_from_mongo output")
print("=" * 60)

from My_client.app_mongo import load_model_comparison_from_mongo

df = load_model_comparison_from_mongo(top_n=2)
print(f"\n✓ Loaded {len(df)} models")

if not df.empty:
    cols = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    available = [c for c in cols if c in df.columns]
    print(f"\n  Available metrics: {len(available)}/4")
    for c in cols:
        status = "✓" if c in df.columns else "✗"
        print(f"    {status} {c}")
    
    if available:
        print(f"\n  Sample values (first model):")
        for c in available:
            val = df.iloc[0][c]
            print(f"    - {c}: {val:.4f}")
