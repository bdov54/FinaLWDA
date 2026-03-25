#!/usr/bin/env python
"""Debug mBERT metrics loading"""
import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings

# Get data
s = load_settings()
store = MongoStore(s.mongodb_uri, s.mongodb_db)
col = store.collection('model_trials')

# Get top mBERT trial
docs = list(col.find(
    {"display_name": "mBERT"},
    {"_id": 0, "trial_name": 1, "display_name": 1, "test_precision_macro": 1, "test_recall_macro": 1, "test_f1_macro": 1}
).sort("test_f1_macro", -1).limit(1))

if docs:
    doc = docs[0]
    trial_name = doc.get("trial_name")
    
    print(f"mBERT Trial: {trial_name}")
    print(f"  Precision in MongoDB: {doc.get('test_precision_macro', 'NOT FOUND')}")
    print(f"  Recall in MongoDB: {doc.get('test_recall_macro', 'NOT FOUND')}")
    print(f"  F1 in MongoDB: {doc.get('test_f1_macro', 'NOT FOUND')}")
    
    # Check if metrics.json exists
    project_root = Path(__file__).parent
    metrics_path = project_root / "artifacts" / "model_batches" / "b123v45_tuned_20260316_082735" / "models" / "mBERT" / "trials" / trial_name / "metrics.json"
    
    print(f"\nMetrics file path: {metrics_path}")
    print(f"  File exists: {metrics_path.exists()}")
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  Precision in file: {metrics.get('test_precision_macro', 'NOT FOUND')}")
        print(f"  Recall in file: {metrics.get('test_recall_macro', 'NOT FOUND')}")
    else:
        # Try to find the actual trial directory
        trial_base = project_root / "artifacts" / "model_batches" / "b123v45_tuned_20260316_082735" / "models" / "mBERT" / "trials"
        if trial_base.exists():
            dirs = list(trial_base.iterdir())
            print(f"\n  Available trials in {trial_base.name}:")
            for d in sorted(dirs)[:5]:
                print(f"    - {d.name}")
        else:
            print(f"  Trial base directory not found: {trial_base}")
