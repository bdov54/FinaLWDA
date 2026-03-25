#!/usr/bin/env python
"""Quick test to verify 4 metrics are available"""
import json
from pathlib import Path

# Check if artifacts have all 4 metrics
artifact_base = Path("artifacts/model_batches/b123v45_tuned_20260316_082735/models")

# Check mBERT top model
mbert_metrics = artifact_base / "mBERT" / "trials" / "mbert_t002_len80_ep5_lr1e-05_bs8_ga1_cwsqrt_inv_ls0.03" / "metrics.json"
if mbert_metrics.exists():
    with open(mbert_metrics) as f:
        data = json.load(f)
    metrics = ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']
    available = [m for m in metrics if m in data]
    print(f"✓ mBERT metrics: {len(available)}/4 available")
    for m in available:
        print(f"  - {m}: {data[m]:.4f}")
