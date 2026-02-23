"""
save_model.py  —  Run this ONCE after main.py to persist the trained model.
Usage:
    python save_model.py
Outputs:
    model_artifacts/xgb_model.pkl
    model_artifacts/model_meta.json
"""

import os, json, warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    USE_XGBOOST = False

from sklearn.model_selection import train_test_split
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load feature-engineered dataset ───────────────────────────────────────
CSV_PATH = os.path.join(SCRIPT_DIR, 'outputs', 'dataset_combined.csv')
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        "outputs/dataset_combined.csv not found. Run main.py first.")

combined = pd.read_csv(CSV_PATH)

FEATURES = [
    'duration_sec', 'is_short', 'duration_category',
    'title_length', 'word_count', 'hashtag_count', 'has_hashtag',
    'has_emoji', 'has_sinhala', 'has_english', 'is_bilingual',
    'has_numbers', 'has_question', 'pipe_count', 'exclaim_count',
    'publish_month', 'publish_dow', 'is_weekend',
    'channel_subscribers', 'channel_age_days'
]
TARGET = 'is_high_performer'

X = combined[FEATURES]
y = combined[TARGET]

# ── Train model (same hyperparameters as main.py) ─────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

if USE_XGBOOST:
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, eval_metric='logloss', verbosity=0)
else:
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42)

model.fit(X_train, y_train)

# ── Persist artifacts ──────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, 'model_artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

joblib.dump(model, os.path.join(ARTIFACT_DIR, 'xgb_model.pkl'))

# Channel medians (needed for the "what does this compare to?" UX)
channel_medians = (
    combined.groupby('channel_name')['channel_median'].first().to_dict()
)

meta = {
    "features": FEATURES,
    "use_xgboost": USE_XGBOOST,
    "channel_medians": channel_medians,
    "feature_importances": dict(zip(FEATURES, model.feature_importances_.tolist())),
    "n_train": len(X_train),
    "n_test":  len(X_test),
}

with open(os.path.join(ARTIFACT_DIR, 'model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("Model saved to model_artifacts/xgb_model.pkl")
print("Metadata saved to model_artifacts/model_meta.json")
print(f"\nModel trained on {len(X_train)} videos.")
print(f"Feature importances (top 5):")
fi_sorted = sorted(zip(FEATURES, model.feature_importances_),
                   key=lambda x: x[1], reverse=True)
for fname, fval in fi_sorted[:5]:
    print(f"  {fname:<28} {fval:.4f}")
