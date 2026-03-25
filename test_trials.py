import sys
sys.path.insert(0, '.')
from review_analytics.db.mongo import MongoStore
from review_analytics.config import load_settings

s = load_settings()
store = MongoStore(s.mongodb_uri, s.mongodb_db)
col = store.collection('model_trials')
count = col.count_documents({})
print(f'✓ Found {count} trial documents in model_trials')

# Top 5 by test_f1_macro
top5 = list(col.find({}, {'_id': 0, 'trial_name': 1, 'display_name': 1, 'test_f1_macro': 1}).sort('test_f1_macro', -1).limit(5))
print('\nTop 5 Best Models by test_f1_macro:')
for i, doc in enumerate(top5, 1):
    print(f"{i}. {doc['display_name']:10} | F1: {doc['test_f1_macro']:.6f} | {doc['trial_name']}")
