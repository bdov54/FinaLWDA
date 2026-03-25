import sys
sys.path.insert(0, '.')
from My_client.app_mongo import load_model_comparison_from_mongo

print("Testing load_model_comparison_from_mongo(top_n=2)...")
df = load_model_comparison_from_mongo(top_n=2)
print(f'✓ Loaded {len(df)} best models')
if not df.empty:
    print("\nTop 2 Best Models:")
    for idx, row in df.iterrows():
        print(f"{idx+1}. {row['model']:10} | F1: {row['test_f1_macro']:.6f} | {row['trial_name']}")
else:
    print("No data loaded!")
