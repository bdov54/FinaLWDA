import sys
sys.path.insert(0, '.')

from My_client.app_mongo import load_model_comparison_from_mongo

df = load_model_comparison_from_mongo(top_n=2)
print(f"✓ Loaded {len(df)} models\n")

# Check available metric columns
metric_cols = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
available = [col for col in metric_cols if col in df.columns]

print(f"Available metric columns: {available}")
print(f"Total metrics: {len(available)}/4\n")

if not df.empty:
    print("Model data:")
    for col in ["model", "trial_name"] + available:
        if col in df.columns:
            print(f"  {col}:")
            for val in df[col]:
                print(f"    - {val}")
