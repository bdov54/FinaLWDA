import sys
sys.path.insert(0, '.')
from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings
import pandas as pd
import json

s = load_settings()
store = MongoStore(s.mongodb_uri, s.mongodb_db)
col = store.collection("model_trials")

print("Querying all model_trials from MongoDB...")
docs = list(col.find({}, {"_id": 0}))
print(f"✓ Found {len(docs)} total trials")

# Convert to DataFrame
df = pd.DataFrame(docs)

# Parse config if it's a string
if "config" in df.columns:
    df["config"] = df["config"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

# Standardize column names
if "display_name" in df.columns:
    df["model"] = df["display_name"]

# Extract max_len
if "config" in df.columns:
    df["max_len"] = df["config"].apply(lambda x: x.get("max_len", 80) if isinstance(x, dict) else 80)
else:
    df["max_len"] = 80

# Sort by test_f1_macro and get top 2
if "test_f1_macro" in df.columns:
    df = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
    top2 = df.head(2)

print("\nTop 2 Best Models by test_f1_macro:")
for idx, row in top2.iterrows():
    print(f"{idx+1}. {row['model']:10} | F1: {row['test_f1_macro']:.6f} | Accuracy: {row['test_accuracy']:.6f}")
