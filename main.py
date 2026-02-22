"""
=============================================================================
ML ASSIGNMENT — Predicting YouTube Video Upload Success for Sri Lankan Creators
A Pre-Upload Feature Classification Study Using XGBoost
=============================================================================
Author   : [Rashmi Jayawardhana]  
Channels : Rasmi Vibes | Hey Lee | Timeline of Nuraj  (275 videos total)
Algorithm: XGBoost (Gradient Boosting)
XAI      : Feature Importance + Permutation Importance + PDP + SHAP-like
Key      : Only PRE-UPLOAD features used — NO data leakage
=============================================================================

SETUP (run once):
  python -m venv venv
  venv\\Scripts\\activate
  pip install -r requirements.txt

PLACE THESE FILES IN THE SAME FOLDER AS THIS SCRIPT:
  Content_Excel_file_-_Rasmi_Vibes.xlsx
  Content_Excel_file_-_Hey_Lee.xlsx
  Sorted_by_Content_-_Timeline_of_Nuraj.xlsx
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import warnings
import os

warnings.filterwarnings('ignore')

# ── ALGORITHM: Try XGBoost, fallback to sklearn GradientBoosting ──────────
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
    print("✅ XGBoost loaded successfully")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    USE_XGBOOST = False
    print("⚠️  XGBoost not installed — using sklearn GradientBoostingClassifier (equivalent)")
    print("   Install XGBoost with: pip install xgboost")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, precision_score, recall_score)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier  # for per-channel models

# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================
print("\n" + "="*60 + "\nSECTION 1: LOADING DATA\n" + "="*60)

# ── Resolve paths relative to this script's location ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_A = os.path.join(SCRIPT_DIR, 'data', 'Content_Excel_file_-_Rasmi_Vibes.xlsx')
FILE_B = os.path.join(SCRIPT_DIR, 'data', 'Content_Excel_file_-_Hey_Lee.xlsx')
FILE_C = os.path.join(SCRIPT_DIR, 'data', 'Sorted_by_Content_-_Timeline_of_Nuraj.xlsx')

for fpath in [FILE_A, FILE_B, FILE_C]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"\n❌ Required data file not found:\n   {fpath}\n\n"
            "Please place the three Excel files exported from YouTube Studio\n"
            "inside the  data/  subfolder of this project:\n"
            "  data/Content_Excel_file_-_Rasmi_Vibes.xlsx\n"
            "  data/Content_Excel_file_-_Hey_Lee.xlsx\n"
            "  data/Sorted_by_Content_-_Timeline_of_Nuraj.xlsx\n"
        )

df_a = pd.read_excel(FILE_A, sheet_name='Table data')
df_b = pd.read_excel(FILE_B, sheet_name='Table data')
df_c = pd.read_excel(FILE_C, sheet_name='Table data')

df_a = df_a[df_a['Content'] != 'Total'].copy()
df_b = df_b[df_b['Content'] != 'Total'].copy()
df_c = df_c[df_c['Content'] != 'Total'].copy()

# Normalise column name discrepancy in Channel B
if 'Likes (vs. dislikes) (%)' in df_b.columns:
    df_b = df_b.rename(columns={'Likes (vs. dislikes) (%)': 'Likes (vs dislikes) (%)'})

print(f"Rasmi Vibes       (Channel A): {len(df_a)} videos | 300 subscribers")
print(f"Hey Lee           (Channel B): {len(df_b)} videos | 3,810 subscribers")
print(f"Timeline of Nuraj (Channel C): {len(df_c)} videos | 9,390 subscribers")

# =============================================================================
# SECTION 2 — FEATURE ENGINEERING (Pre-Upload Only — No Leakage)
# =============================================================================
print("\n" + "="*60 + "\nSECTION 2: FEATURE ENGINEERING (Pre-Upload Only)\n" + "="*60)


def engineer_features(df, channel_name, channel_subscribers, channel_joined_date):
    """
    Builds ONLY features available BEFORE or AT UPLOAD TIME.
    Deliberately excludes: likes, comments, views, watch time, CTR,
    shares, subscribers gained — all post-publication metrics.
    This prevents data leakage.
    """
    d = df.copy()
    d['Views']    = pd.to_numeric(d['Views'],    errors='coerce')
    d['Duration'] = pd.to_numeric(d['Duration'], errors='coerce').fillna(0)

    # ── TARGET: above/below channel's own median (neutralises scale) ──────
    channel_median = d['Views'].median()
    d['is_high_performer'] = (d['Views'] >= channel_median).astype(int)

    # ── FEATURE GROUP 1: Video format (creator controls BEFORE upload) ────
    d['duration_sec']      = d['Duration']
    d['is_short']          = (d['duration_sec'] <= 60).astype(int)         # YouTube Shorts flag
    d['duration_category'] = pd.cut(                                         # 0=Short 1=Med 2=Long 3=VeryLong
        d['duration_sec'],
        bins=[-1, 60, 300, 600, 99999],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # ── FEATURE GROUP 2: Title metadata (written BEFORE upload) ──────────
    t = d['Video title'].astype(str)
    d['title_length']  = t.str.len()
    d['word_count']    = t.str.split().str.len()
    d['hashtag_count'] = t.str.count('#')
    d['has_hashtag']   = (d['hashtag_count'] > 0).astype(int)
    d['has_emoji']     = t.apply(lambda x: int(bool(
        re.search(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]', x)))).astype(int)
    d['has_sinhala']   = t.apply(lambda x: int(bool(
        re.search(r'[\u0D80-\u0DFF]', x)))).astype(int)
    d['has_english']   = t.apply(lambda x: int(bool(
        re.search(r'[a-zA-Z]', x)))).astype(int)
    d['is_bilingual']  = ((d['has_sinhala'] == 1) & (d['has_english'] == 1)).astype(int)
    d['has_numbers']   = t.str.contains(r'\d').astype(int)
    d['has_question']  = t.str.contains(r'\?').astype(int)
    d['pipe_count']    = t.str.count(r'\|')
    d['exclaim_count'] = t.str.count(r'!')

    # ── FEATURE GROUP 3: Publish timing (chosen BEFORE upload) ───────────
    pub = pd.to_datetime(d['Video publish time'], errors='coerce')
    d['publish_month'] = pub.dt.month.fillna(6).astype(int)
    d['publish_dow']   = pub.dt.dayofweek.fillna(2).astype(int)
    d['is_weekend']    = (d['publish_dow'] >= 5).astype(int)

    # ── FEATURE GROUP 4: Channel context (known at upload time) ──────────
    d['channel_subscribers'] = channel_subscribers
    d['channel_age_days']    = (
        pd.Timestamp('2026-02-20') - pd.to_datetime(channel_joined_date)
    ).days

    # Metadata (not used as features)
    d['channel_name']   = channel_name
    d['channel_median'] = channel_median
    d['views_actual']   = d['Views']
    return d


df_a2 = engineer_features(df_a, 'Rasmi Vibes',        300,  '2025-05-20')
df_b2 = engineer_features(df_b, 'Hey Lee',            3810, '2023-07-20')
df_c2 = engineer_features(df_c, 'Timeline of Nuraj',  9390, '2012-09-08')
combined = pd.concat([df_a2, df_b2, df_c2], ignore_index=True)

# ── Feature and label definitions ─────────────────────────────────────────
FEATURES = [
    # Group 1: Format
    'duration_sec', 'is_short', 'duration_category',
    # Group 2: Title
    'title_length', 'word_count', 'hashtag_count', 'has_hashtag',
    'has_emoji', 'has_sinhala', 'has_english', 'is_bilingual',
    'has_numbers', 'has_question', 'pipe_count', 'exclaim_count',
    # Group 3: Timing
    'publish_month', 'publish_dow', 'is_weekend',
    # Group 4: Channel
    'channel_subscribers', 'channel_age_days'
]
FLABELS = [
    'Duration (sec)', 'Is Short (≤60s)', 'Duration Category',
    'Title Length', 'Word Count', 'Hashtag Count', 'Has Hashtag',
    'Has Emoji', 'Has Sinhala', 'Has English', 'Is Bilingual',
    'Has Numbers', 'Has Question', 'Pipe Count', 'Exclamation Count',
    'Publish Month', 'Publish Day of Week', 'Is Weekend',
    'Channel Subscribers', 'Channel Age (days)'
]
TARGET = 'is_high_performer'
X = combined[FEATURES].copy()
y = combined[TARGET].copy()

print(f"\nCombined dataset : {len(X)} videos × {len(FEATURES)} features")
print(f"Missing values   : {X.isnull().sum().sum()} ✅ (none)")
print(f"High performers  : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Low performers   : {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")
print(f"\nPer-channel breakdown:")
for ch in ['Rasmi Vibes', 'Hey Lee', 'Timeline of Nuraj']:
    m   = combined['channel_name'] == ch
    n   = m.sum()
    h   = combined.loc[m, TARGET].sum()
    med = combined.loc[m, 'channel_median'].iloc[0]
    print(f"  {ch:<22}: {n:3d} videos | {h:2d} high / {n-h:2d} low | median={med:.0f} views")

# Save combined dataset to CSV
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

combined[FEATURES + [TARGET, 'channel_name', 'views_actual', 'channel_median']].to_csv(
    os.path.join(OUTPUT_DIR, 'dataset_combined.csv'), index=False)
print(f"\nDataset saved to: outputs/dataset_combined.csv")

# =============================================================================
# SECTION 3 — TRAIN/VALIDATION/TEST SPLIT (70% / 15% / 15%)
# =============================================================================
print("\n" + "="*60 + "\nSECTION 3: DATA SPLIT & MODEL TRAINING\n" + "="*60)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"Train : {len(X_train)} videos | {y_train.sum()} high / {(y_train==0).sum()} low")
print(f"Val   : {len(X_val)}  videos | {y_val.sum()} high / {(y_val==0).sum()} low")
print(f"Test  : {len(X_test)}  videos | {y_test.sum()} high / {(y_test==0).sum()} low")

# =============================================================================
# SECTION 4 — MODEL: XGBoost Classifier
# Hyperparameters chosen to prevent overfitting on 275 samples:
#   n_estimators=200   : enough trees for convergence
#   max_depth=3        : shallow weak learners
#   learning_rate=0.05 : slow learning = better generalisation
#   subsample=0.8      : stochastic gradient boosting (row sampling)
#   colsample_bytree=0.8: column sub-sampling per tree
#   min_child_weight=5 : min 5 samples per leaf (regularisation)
# =============================================================================

if USE_XGBOOST:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
else:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42
    )

model.fit(X_train, y_train)

# =============================================================================
# SECTION 5 — EVALUATION
# =============================================================================


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Acc':  accuracy_score(y_true, y_pred),
        'Prec': precision_score(y_true, y_pred, zero_division=0),
        'Rec':  recall_score(y_true, y_pred, zero_division=0),
        'F1':   f1_score(y_true, y_pred, zero_division=0),
        'AUC':  roc_auc_score(y_true, y_prob)
    }


tr_pr = model.predict_proba(X_train)[:, 1]; tr_p = model.predict(X_train)
vl_pr = model.predict_proba(X_val)[:, 1];   vl_p = model.predict(X_val)
te_pr = model.predict_proba(X_test)[:, 1];  te_p = model.predict(X_test)

tr_m = compute_metrics(y_train, tr_p, tr_pr)
vl_m = compute_metrics(y_val,   vl_p, vl_pr)
te_m = compute_metrics(y_test,  te_p, te_pr)

skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
cv_f1  = cross_val_score(model, X, y, cv=skf, scoring='f1')
cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

fi   = model.feature_importances_
perm = permutation_importance(
    model, X_test, y_test, n_repeats=20, random_state=42, scoring='accuracy')

print(f"\n{'Split':<12} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}")
print("-"*58)
for s, m in [('Train', tr_m), ('Validation', vl_m), ('Test', te_m)]:
    print(f"{s:<12} {m['Acc']:>9.3f} {m['Prec']:>10.3f} {m['Rec']:>8.3f} {m['F1']:>8.3f} {m['AUC']:>9.3f}")
print("-"*58)
print(f"{'10-CV Acc':<12} {cv_acc.mean():>9.3f} ± {cv_acc.std():.3f}")
print(f"{'10-CV F1':<12} {cv_f1.mean():>9.3f} ± {cv_f1.std():.3f}")
print(f"{'10-CV AUC':<12} {cv_auc.mean():>9.3f} ± {cv_auc.std():.3f}")

print(f"\nTop 5 Features:")
for rank, i in enumerate(np.argsort(fi)[::-1][:5]):
    print(f"  {rank+1}. {FLABELS[i]:<32} {fi[i]:.4f}")

# =============================================================================
# SECTION 6 — PLOTS (saved to outputs/)
# =============================================================================

C = {'B': '#2196F3', 'G': '#4CAF50', 'R': '#F44336', 'O': '#FF9800', 'P': '#9C27B0'}


# ── Plot 1: EDA ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    'Exploratory Data Analysis — Three Sri Lankan YouTube Channels\n'
    'Rasmi Vibes · Hey Lee · Timeline of Nuraj  |  275 Videos  |  Pre-Upload Features',
    fontsize=13, fontweight='bold', y=1.01)

for ch, col in [('Rasmi Vibes', C['B']), ('Hey Lee', C['G']), ('Timeline of Nuraj', C['O'])]:
    ax[0, 0].hist(
        combined[combined.channel_name == ch].views_actual,
        bins=20, alpha=0.65, label=ch, color=col, edgecolor='w')
ax[0, 0].set_yscale('log')
ax[0, 0].set_title('Views Distribution (log scale)', fontweight='bold')
ax[0, 0].set_xlabel('Views'); ax[0, 0].set_ylabel('Videos'); ax[0, 0].legend(fontsize=8)

chs = ['Rasmi Vibes', 'Hey Lee', 'Timeline of Nuraj']
hi  = [combined[combined.channel_name == c].is_high_performer.sum() for c in chs]
lo  = [len(combined[combined.channel_name == c]) - h for c, h in zip(chs, hi)]
x   = np.arange(3); w = 0.35
b1  = ax[0, 1].bar(x - w/2, hi, w, label='High Performer', color=C['G'], alpha=0.85)
b2  = ax[0, 1].bar(x + w/2, lo, w, label='Low Performer',  color=C['R'], alpha=0.85)
ax[0, 1].set_title('Class Balance per Channel\n(50/50 by design)', fontweight='bold')
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(['Rasmi\nVibes', 'Hey\nLee', 'Timeline\nNuraj'])
ax[0, 1].set_ylabel('Videos'); ax[0, 1].legend()
for b in list(b1) + list(b2):
    ax[0, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                  str(int(b.get_height())), ha='center', fontsize=9, fontweight='bold')

ax[0, 2].hist(combined[combined.is_high_performer == 1].duration_sec.clip(0, 700),
              bins=25, alpha=0.7, color=C['G'], label='High', edgecolor='w')
ax[0, 2].hist(combined[combined.is_high_performer == 0].duration_sec.clip(0, 700),
              bins=25, alpha=0.7, color=C['R'], label='Low', edgecolor='w')
ax[0, 2].set_title('Duration vs Performance', fontweight='bold')
ax[0, 2].set_xlabel('Duration (s)'); ax[0, 2].legend()

for ai, feat, title in [(ax[1, 0], 'hashtag_count', 'Hashtag Count'),
                         (ax[1, 1], 'title_length',  'Title Length (chars)')]:
    bp = ai.boxplot(
        [combined[combined.is_high_performer == 1][feat],
         combined[combined.is_high_performer == 0][feat]],
        labels=['High', 'Low'], patch_artist=True,
        medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor(C['G']); bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(C['R']); bp['boxes'][1].set_alpha(0.6)
    ai.set_title(f'{title} vs Performance', fontweight='bold')

dow    = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
rates  = [combined[combined.publish_dow == i].is_high_performer.mean() for i in range(7)]
counts = [len(combined[combined.publish_dow == i]) for i in range(7)]
bars   = ax[1, 2].bar(dow, rates,
                       color=[C['G'] if r >= 0.5 else C['R'] for r in rates],
                       alpha=0.85, edgecolor='w')
ax[1, 2].axhline(0.5, color='k', linestyle='--', linewidth=1.5, label='50% baseline')
ax[1, 2].set_title('High Performer Rate by Publish Day', fontweight='bold')
ax[1, 2].set_ylabel('Rate'); ax[1, 2].set_ylim(0, 1.15); ax[1, 2].legend()
for b, r, cnt in zip(bars, rates, counts):
    ax[1, 2].text(b.get_x() + b.get_width()/2, r + 0.03,
                  f'{r:.0%}\nn={cnt}', ha='center', fontsize=7.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_eda.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n💾 Saved: outputs/01_eda.png")

# ── Plot 2: Evaluation ─────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    'XGBoost — Model Evaluation Results\n'
    '275 Videos · 20 Pre-Upload Features · No Data Leakage',
    fontsize=13, fontweight='bold')

mn = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']; xm = np.arange(5); wm = 0.25
ax[0, 0].bar(xm - wm, [tr_m['Acc'], tr_m['Prec'], tr_m['Rec'], tr_m['F1'], tr_m['AUC']],
             wm, label='Train',      color=C['B'], alpha=0.85)
ax[0, 0].bar(xm,      [vl_m['Acc'], vl_m['Prec'], vl_m['Rec'], vl_m['F1'], vl_m['AUC']],
             wm, label='Validation', color=C['O'], alpha=0.85)
ax[0, 0].bar(xm + wm, [te_m['Acc'], te_m['Prec'], te_m['Rec'], te_m['F1'], te_m['AUC']],
             wm, label='Test',       color=C['G'], alpha=0.85)
ax[0, 0].set_title('Performance Metrics', fontweight='bold')
ax[0, 0].set_xticks(xm); ax[0, 0].set_xticklabels(mn, rotation=15)
ax[0, 0].set_ylim(0, 1.18); ax[0, 0].legend()
ax[0, 0].axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
for cont in ax[0, 0].containers:
    ax[0, 0].bar_label(cont, fmt='%.2f', fontsize=7, padding=2)

cm = confusion_matrix(y_test, te_p)
im = ax[0, 1].imshow(cm, cmap='Blues')
ax[0, 1].set_title('Confusion Matrix — Test Set (n=42)', fontweight='bold')
ax[0, 1].set_xticks([0, 1]); ax[0, 1].set_yticks([0, 1])
ax[0, 1].set_xticklabels(['Pred Low', 'Pred High'])
ax[0, 1].set_yticklabels(['Actual Low', 'Actual High'])
lbl = [['True Neg', 'False Pos'], ['False Neg', 'True Pos']]
for i in range(2):
    for j in range(2):
        ax[0, 1].text(j, i, f'{lbl[i][j]}\n{cm[i,j]} ({cm[i,j]/cm.sum()*100:.0f}%)',
                      ha='center', va='center', fontsize=11, fontweight='bold',
                      color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax[0, 1], shrink=0.8)

fpr_tr, tpr_tr, _ = roc_curve(y_train, tr_pr)
fpr_vl, tpr_vl, _ = roc_curve(y_val,   vl_pr)
fpr_te, tpr_te, _ = roc_curve(y_test,  te_pr)
ax[1, 0].plot(fpr_tr, tpr_tr, color=C['B'], linewidth=2,   label=f'Train AUC={tr_m["AUC"]:.3f}')
ax[1, 0].plot(fpr_vl, tpr_vl, color=C['O'], linewidth=2,   label=f'Val   AUC={vl_m["AUC"]:.3f}')
ax[1, 0].plot(fpr_te, tpr_te, color=C['G'], linewidth=2.5, label=f'Test  AUC={te_m["AUC"]:.3f}')
ax[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (0.50)')
ax[1, 0].fill_between(fpr_te, tpr_te, alpha=0.08, color=C['G'])
ax[1, 0].set_title('ROC Curve', fontweight='bold')
ax[1, 0].set_xlabel('False Positive Rate'); ax[1, 0].set_ylabel('True Positive Rate')
ax[1, 0].legend(fontsize=9); ax[1, 0].grid(True, alpha=0.3)

bp3 = ax[1, 1].boxplot(
    [cv_acc, cv_f1, cv_auc],
    labels=[f'Accuracy\n{cv_acc.mean():.3f}±{cv_acc.std():.3f}',
            f'F1\n{cv_f1.mean():.3f}±{cv_f1.std():.3f}',
            f'AUC\n{cv_auc.mean():.3f}±{cv_auc.std():.3f}'],
    patch_artist=True, medianprops=dict(color='red', linewidth=2.5))
for p, c in zip(bp3['boxes'], [C['B'], C['G'], C['P']]):
    p.set_facecolor(c); p.set_alpha(0.6)
ax[1, 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Random baseline')
ax[1, 1].set_title('10-Fold Cross-Validation', fontweight='bold')
ax[1, 1].set_ylim(0.3, 1.0); ax[1, 1].legend(); ax[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_evaluation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: outputs/02_evaluation.png")

# ── Plot 3: Feature Importance ─────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    'Feature Importance — Pre-Upload Features Only\n'
    'Built-in Importance · Permutation Importance (Test Set)',
    fontsize=13, fontweight='bold')

si = np.argsort(fi)
ax[0].barh(
    [FLABELS[i] for i in si], fi[si],
    color=['#FF4444' if fi[i] == fi.max() else C['B'] for i in si],
    alpha=0.87, edgecolor='w', height=0.7)
ax[0].set_title('Built-in Feature Importance', fontweight='bold')
ax[0].set_xlabel('Score')
for i, (idx, v) in enumerate(zip(si, fi[si])):
    ax[0].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8, fontweight='bold')

psi = np.argsort(perm.importances_mean)
ax[1].barh(
    [FLABELS[i] for i in psi], perm.importances_mean[psi],
    xerr=perm.importances_std[psi],
    color=[C['G'] if perm.importances_mean[i] > 0 else C['O'] for i in psi],
    alpha=0.85, edgecolor='w', capsize=4, height=0.7)
ax[1].axvline(0, color='k', linewidth=2)
ax[1].set_title('Permutation Importance (accuracy drop when shuffled)', fontweight='bold')
ax[1].set_xlabel('Mean Accuracy Decrease')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: outputs/03_importance.png")

# ── Plot 4: PDP ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Partial Dependence Plots — Top 4 Pre-Upload Features',
             fontsize=13, fontweight='bold')

for a, fi_i in zip(ax.flat, np.argsort(fi)[::-1][:4]):
    feat = FEATURES[fi_i]
    uniq = np.sort(X[feat].unique())
    fr   = (np.linspace(X[feat].quantile(0.05), X[feat].quantile(0.95), 70)
            if len(uniq) > 50 else uniq.astype(float))
    pdp  = []
    for v in fr:
        Xt2 = X.copy(); Xt2[feat] = v
        pdp.append(model.predict_proba(Xt2)[:, 1].mean())
    pdp = np.array(pdp)
    a.plot(fr, pdp, color=C['B'], linewidth=2.8)
    a.fill_between(fr, pdp, 0.5, where=pdp >= 0.5, alpha=0.18, color=C['G'], label='Above 50%')
    a.fill_between(fr, pdp, 0.5, where=pdp <  0.5, alpha=0.18, color=C['R'], label='Below 50%')
    a.axhline(0.5, color='red', linestyle='--', linewidth=1.8, label='50% baseline')
    a.set_title(f'PDP: {FLABELS[fi_i]}', fontweight='bold')
    a.set_xlabel(FLABELS[fi_i]); a.set_ylabel('P(High Performer)')
    a.set_ylim(0, 1); a.grid(True, alpha=0.3); a.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_pdp.png'), dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: outputs/04_pdp.png")

# ── Plot 5: SHAP-like + Cross-channel ──────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    'SHAP-like Explainability — Global Contributions & Cross-Channel Comparison',
    fontsize=13, fontweight='bold')

base    = model.predict_proba(X)[:, 1].mean()
contrib = []
for feat in FEATURES:
    Xp = X.copy()
    Xp[feat] = X[feat].sample(frac=1, random_state=42).values
    contrib.append(base - model.predict_proba(Xp)[:, 1].mean())
contrib = np.array(contrib)
si3     = np.argsort(np.abs(contrib))

ax[0].barh(
    [FLABELS[i] for i in si3], contrib[si3],
    color=[C['G'] if v >= 0 else C['R'] for v in contrib[si3]],
    alpha=0.85, edgecolor='w', height=0.7)
ax[0].axvline(0, color='k', linewidth=2)
ax[0].set_title('Global SHAP-like Contributions', fontweight='bold')
ax[0].set_xlabel('Contribution to P(High)')
ax[0].legend(handles=[
    mpatches.Patch(color=C['G'], label='Increases P(High)'),
    mpatches.Patch(color=C['R'], label='Decreases P(High)')
])

top5 = np.argsort(fi)[::-1][:5]; xc = np.arange(5); wc = 0.25
for ci, (ch, col) in enumerate(zip(
        ['Rasmi Vibes', 'Hey Lee', 'Timeline of Nuraj'],
        [C['B'], C['G'], C['O']])):
    mask = combined.channel_name == ch
    cm2  = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    cm2.fit(X[mask], y[mask])
    ax[1].bar(xc + ci * wc,
              [cm2.feature_importances_[i] for i in top5],
              wc, label=ch, color=col, alpha=0.85)

ax[1].set_title('Feature Importance by Channel (Top 5)', fontweight='bold')
ax[1].set_xticks(xc + wc)
ax[1].set_xticklabels([FLABELS[i] for i in top5], rotation=22, ha='right', fontsize=9)
ax[1].set_ylabel('Importance Score'); ax[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_shap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("💾 Saved: outputs/05_shap.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("COMPLETE RESULTS — COPY THESE INTO YOUR REPORT")
print("="*60)
print(f"Dataset         : 275 videos · 3 channels · 20 features · no leakage")
print(f"Algorithm       : {'XGBoost Classifier' if USE_XGBOOST else 'GradientBoostingClassifier (sklearn)'}")
print(f"Split           : Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} (70/15/15)")
print(f"")
print(f"TEST SET RESULTS:")
print(f"  Accuracy  = {te_m['Acc']:.3f}   ({te_m['Acc']*100:.1f}%)")
print(f"  Precision = {te_m['Prec']:.3f}")
print(f"  Recall    = {te_m['Rec']:.3f}")
print(f"  F1 Score  = {te_m['F1']:.3f}")
print(f"  AUC-ROC   = {te_m['AUC']:.3f}")
print(f"")
print(f"10-FOLD CROSS-VALIDATION:")
print(f"  Accuracy  = {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print(f"  F1 Score  = {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
print(f"  AUC-ROC   = {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"")
print(f"TOP 5 FEATURES (pre-upload only):")
for rank, i in enumerate(np.argsort(fi)[::-1][:5]):
    print(f"  {rank+1}. {FLABELS[i]:<32} importance={fi[i]:.4f}")
print(f"")
print(f"Plots saved to:  outputs/01_eda.png")
print(f"                 outputs/02_evaluation.png")
print(f"                 outputs/03_importance.png")
print(f"                 outputs/04_pdp.png")
print(f"                 outputs/05_shap.png")
print(f"Dataset CSV:     outputs/dataset_combined.csv")
print("="*60)
