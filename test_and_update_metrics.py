import json
import sys
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings
from review_analytics.logging import logging

def check_and_update_metrics():
    """Check MongoDB data and add precision/recall metrics if missing"""
    s = load_settings()
    store = MongoStore(s.mongodb_uri, s.mongodb_db)
    col = store.collection('model_trials')
    
    # Get all documents
    docs = list(col.find({}, {'_id': 1, 'trial_name': 1, 'display_name': 1, 'test_precision_macro': 1, 'test_recall_macro': 1, 'test_f1_macro': 1}))
    
    print(f"✓ Found {len(docs)} documents in model_trials")
    
    # Check first few documents
    if docs:
        print("\nFirst document fields:")
        first_doc = docs[0]
        has_precision = 'test_precision_macro' in first_doc
        has_recall = 'test_recall_macro' in first_doc
        has_f1 = 'test_f1_macro' in first_doc
        has_acc = 'test_accuracy' in first_doc
        
        print(f"  - test_precision_macro: {has_precision}")
        print(f"  - test_recall_macro: {has_recall}")
        print(f"  - test_f1_macro: {has_f1}")
        print(f"  - test_accuracy: {has_acc}")
        
        # If missing, update them
        if not has_precision or not has_recall:
            print("\n⚠ Missing precision/recall in MongoDB. Updating from trial artifacts...")
            from TSN_Data.mongodb.push_all_model_trials import get_all_model_trials
            
            trials = get_all_model_trials()
            print(f"✓ Loaded {len(trials)} trials from artifacts")
            
            # Clear and reinserttribute
            col.drop()
            col.insert_many(trials)
            print(f"✓ Reinserted {len(trials)} documents with complete metrics")
            
            # Verify
            sample = list(col.find({}, {'trial_name': 1, 'test_precision_macro': 1, 'test_recall_macro': 1}).limit(1))
            if sample:
                print("\nSample document after update:")
                print(f"  trial_name: {sample[0].get('trial_name')}")
                print(f"  test_precision_macro: {sample[0].get('test_precision_macro')}")
                print(f"  test_recall_macro: {sample[0].get('test_recall_macro')}")
        else:
            print("✓ All metrics (precision/recall) exist in MongoDB")

if __name__ == "__main__":
    check_and_update_metrics()
