"""
app.py  —  YouTube Performance Predictor
=========================================
Predicting YouTube Video Upload Success for Sri Lankan Creators
Manual Light / Dark theme toggle — independent of OS setting.

Run: streamlit run app.py
"""

import os, json, re, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="YouTube Performance Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state ───────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "predicted" not in st.session_state:
    st.session_state.predicted = False
# stored snapshot of the last prediction
if "snap" not in st.session_state:
    st.session_state.snap = {}

T = st.session_state.theme  # "light" | "dark"

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — two complete palettes, fully independent of the OS
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    "light": {
        # Canvas
        "bg_app":          "#f0f4f8",
        "bg_sidebar":      "#ffffff",
        "bg_card":         "#ffffff",
        "bg_card_hover":   "#f8fafc",
        "border":          "#e2e8f0",
        "shadow":          "0 2px 12px rgba(0,0,0,0.07)",
        "shadow_hover":    "0 6px 24px rgba(79,70,229,0.14)",
        # Text
        "text_primary":    "#1e293b",
        "text_secondary":  "#475569",
        "text_muted":      "#94a3b8",
        "text_sidebar":    "#1e293b",
        # Accent / brand
        "accent":          "#4f46e5",
        "accent_light":    "#e0e7ff",
        "accent_text":     "#3730a3",
        "accent2":         "#0ea5e9",
        "accent3":         "#10b981",
        # Status
        "success_bg":      "#f0fdf4",
        "success_border":  "#22c55e",
        "success_text":    "#166534",
        "warn_bg":         "#fffbeb",
        "warn_border":     "#f59e0b",
        "warn_text":       "#92400e",
        "tip_bg":          "#eef2ff",
        "tip_border":      "#6366f1",
        "tip_text":        "#3730a3",
        "result_high_bg":  "linear-gradient(135deg,#f0fdf4,#dcfce7)",
        "result_high_brd": "#22c55e",
        "result_high_clr": "#166534",
        "result_high_num": "#15803d",
        "result_low_bg":   "linear-gradient(135deg,#fff1f2,#fce7e7)",
        "result_low_brd":  "#ef4444",
        "result_low_clr":  "#991b1b",
        "result_low_num":  "#b91c1c",
        # Charts
        "plot_bg":         "rgba(0,0,0,0)",
        "grid":            "rgba(100,116,139,0.13)",
        "axis":            "#64748b",
        "font_clr":        "#1e293b",
        "gauge_bg":        "rgba(241,245,249,0.9)",
        "gauge_brd":       "#e2e8f0",
        "step_low":        "rgba(239,68,68,0.07)",
        "step_high":       "rgba(22,163,74,0.07)",
        "cm_scale":        [[0,"rgba(224,231,255,0.6)"],[1,"rgba(79,70,229,0.85)"]],
        "bar_low":         "rgba(239,68,68,0.75)",
        "bar_high":        "rgba(22,163,74,0.75)",
        "bar_low_line":    "#ef4444",
        "bar_high_line":   "#16a34a",
        "fi_bar":          "rgba(99,102,241,0.70)",
        "fi_bar_top":      "rgba(239,68,68,0.85)",
        "shap_pos":        "rgba(22,163,74,0.80)",
        "shap_neg":        "rgba(220,38,38,0.80)",
        "shap_pos_ln":     "#16a34a",
        "shap_neg_ln":     "#dc2626",
        "vline":           "#94a3b8",
        "hline":           "#6366f1",
        "pie_colors":      ["#6366f1","#f59e0b","#22c55e"],
        # Metric card
        "metric_val_clr":  "#4f46e5",
        # Button
        "btn_bg":          "linear-gradient(135deg,#4f46e5,#7c3aed)",
        "btn_shadow":      "0 4px 14px rgba(79,70,229,0.40)",
        "btn_shadow_hov":  "0 6px 22px rgba(79,70,229,0.55)",
        # Toggle button appearance
        "toggle_label":    "Switch to Dark Mode",
        "toggle_icon":     "dark_mode",
        # Divider / tab marker
        "tab_active":      "#4f46e5",
        "section_border":  "#e0e7ff",
        "header_gradient": "linear-gradient(90deg,#4f46e5,#0ea5e9,#10b981)",
        "hero_sub":        "#64748b",
        "footer_text":     "#94a3b8",
        "footer_border":   "#e2e8f0",
        "info_badge_bg":   "#eef2ff",
        "info_badge_text": "#4338ca",
        "badge_weekend_bg":"#fef3c7",
        "badge_weekend_cl":"#92400e",
        "badge_day_bg":    "#e0e7ff",
        "badge_day_cl":    "#4338ca",
        "sidebar_logo_cl": "#4f46e5",
        "sidebar_sub_cl":  "#64748b",
        "input_bg":        "#f8fafc",
    },
    "dark": {
        # Canvas
        "bg_app":          "#0f1117",
        "bg_sidebar":      "#191c24",
        "bg_card":         "#1e2230",
        "bg_card_hover":   "#252a3a",
        "border":          "rgba(255,255,255,0.08)",
        "shadow":          "0 2px 16px rgba(0,0,0,0.40)",
        "shadow_hover":    "0 6px 28px rgba(99,102,241,0.25)",
        # Text
        "text_primary":    "#e2e8f0",
        "text_secondary":  "#94a3b8",
        "text_muted":      "#64748b",
        "text_sidebar":    "#e2e8f0",
        # Accent / brand
        "accent":          "#818cf8",
        "accent_light":    "rgba(129,140,248,0.15)",
        "accent_text":     "#a5b4fc",
        "accent2":         "#38bdf8",
        "accent3":         "#34d399",
        # Status
        "success_bg":      "rgba(34,197,94,0.10)",
        "success_border":  "#22c55e",
        "success_text":    "#4ade80",
        "warn_bg":         "rgba(245,158,11,0.10)",
        "warn_border":     "#f59e0b",
        "warn_text":       "#fcd34d",
        "tip_bg":          "rgba(129,140,248,0.12)",
        "tip_border":      "#818cf8",
        "tip_text":        "#c7d2fe",
        "result_high_bg":  "linear-gradient(135deg,rgba(34,197,94,0.15),rgba(16,185,129,0.08))",
        "result_high_brd": "#22c55e",
        "result_high_clr": "#4ade80",
        "result_high_num": "#86efac",
        "result_low_bg":   "linear-gradient(135deg,rgba(239,68,68,0.15),rgba(220,38,38,0.08))",
        "result_low_brd":  "#ef4444",
        "result_low_clr":  "#f87171",
        "result_low_num":  "#fca5a5",
        # Charts
        "plot_bg":         "rgba(0,0,0,0)",
        "grid":            "rgba(255,255,255,0.06)",
        "axis":            "#64748b",
        "font_clr":        "#e2e8f0",
        "gauge_bg":        "rgba(30,34,48,0.80)",
        "gauge_brd":       "rgba(255,255,255,0.10)",
        "step_low":        "rgba(239,68,68,0.10)",
        "step_high":       "rgba(34,197,94,0.10)",
        "cm_scale":        [[0,"rgba(30,27,75,0.80)"],[1,"rgba(99,102,241,0.90)"]],
        "bar_low":         "rgba(239,68,68,0.70)",
        "bar_high":        "rgba(34,197,94,0.70)",
        "bar_low_line":    "#ef4444",
        "bar_high_line":   "#22c55e",
        "fi_bar":          "rgba(99,102,241,0.65)",
        "fi_bar_top":      "rgba(239,68,68,0.85)",
        "shap_pos":        "rgba(34,197,94,0.75)",
        "shap_neg":        "rgba(239,68,68,0.75)",
        "shap_pos_ln":     "#22c55e",
        "shap_neg_ln":     "#ef4444",
        "vline":           "rgba(255,255,255,0.25)",
        "hline":           "#818cf8",
        "pie_colors":      ["#818cf8","#f59e0b","#34d399"],
        # Metric card
        "metric_val_clr":  "#818cf8",
        # Button
        "btn_bg":          "linear-gradient(135deg,#4f46e5,#7c3aed)",
        "btn_shadow":      "0 4px 14px rgba(79,70,229,0.45)",
        "btn_shadow_hov":  "0 6px 22px rgba(79,70,229,0.60)",
        # Toggle
        "toggle_label":    "Switch to Light Mode",
        "toggle_icon":     "light_mode",
        # Misc
        "tab_active":      "#818cf8",
        "section_border":  "rgba(129,140,248,0.20)",
        "header_gradient": "linear-gradient(90deg,#818cf8,#38bdf8,#34d399)",
        "hero_sub":        "#64748b",
        "footer_text":     "#475569",
        "footer_border":   "rgba(255,255,255,0.07)",
        "info_badge_bg":   "rgba(129,140,248,0.15)",
        "info_badge_text": "#a5b4fc",
        "badge_weekend_bg":"rgba(245,158,11,0.15)",
        "badge_weekend_cl":"#fcd34d",
        "badge_day_bg":    "rgba(129,140,248,0.15)",
        "badge_day_cl":    "#a5b4fc",
        "sidebar_logo_cl": "#818cf8",
        "sidebar_sub_cl":  "#64748b",
        "input_bg":        "#252a3a",
    },
}

C = THEMES[T]  # active colour tokens

# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION — overrides every Streamlit element, OS-independent
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}}

/* Force background — overrides system-level prefers-color-scheme */
html, body {{ background: {C["bg_app"]} !important; }}

.stApp {{
    background: {C["bg_app"]} !important;
    color: {C["text_primary"]} !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {C["bg_sidebar"]} !important;
    border-right: 1px solid {C["border"]} !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.08) !important;
}}
[data-testid="stSidebar"] * {{
    color: {C["text_sidebar"]} !important;
}}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {{
    background: {C["input_bg"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text_primary"]} !important;
}}

/* Header bar */
header[data-testid="stHeader"] {{
    background: {C["bg_app"]} !important;
    border-bottom: 1px solid {C["border"]};
}}

/* Hide the Deploy button */
[data-testid="stAppDeployButton"] {{
    display: none !important;
}}

/* Main content */
.block-container {{ padding-top: 1.4rem !important; }}

/* General text — exclude .gradient-text so color:transparent is never overridden */
p, label {{ color: {C["text_primary"]}; }}
div:not(.gradient-text-wrap) {{ color: {C["text_primary"]}; }}

/* Gradient heading — must use inline-block so background-clip:text works */
.gradient-text-wrap {{
    display: block;
    text-align: center;
}}
.gradient-text {{
    display: inline-block;
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.2;
    background: {C["header_gradient"]} !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    margin: 0;
    padding: 0 4px;
}}

/* Inputs */
.stTextInput input, .stNumberInput input,
div[data-baseweb="select"] > div {{
    background: {C["input_bg"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text_primary"]} !important;
    border-radius: 8px !important;
}}
.stSlider [data-testid="stMarkdownContainer"] p {{
    color: {C["text_secondary"]} !important;
}}

/* Tabs */
[data-testid="stTabs"] {{
    border-bottom: 1px solid {C["border"]};
}}
[data-testid="stTabs"] button {{
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    color: {C["text_muted"]} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C["accent"]} !important;
    border-bottom: 2px solid {C["accent"]} !important;
}}
[data-testid="stTabs"] button:hover {{
    color: {C["accent"]} !important;
    background: {C["accent_light"]} !important;
    border-radius: 6px 6px 0 0 !important;
}}

/* Sidebar primary button */
[data-testid="stSidebar"] .stButton > button {{
    background: {C["btn_bg"]} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 11px !important;
    box-shadow: {C["btn_shadow"]} !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
    width: 100% !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    box-shadow: {C["btn_shadow_hov"]} !important;
    transform: translateY(-1px) !important;
}}

/* Toggle button (main area) */
.theme-toggle-btn > button {{
    background: {C["accent_light"]} !important;
    color: {C["accent_text"]} !important;
    border: 1px solid {C["accent"]}44 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 5px 14px !important;
}}

/* Dataframe */
[data-testid="stDataFrame"],
[data-testid="dataframe-container"] {{
    background: {C["bg_card"]} !important;
    border-radius: 10px !important;
    border: 1px solid {C["border"]} !important;
    overflow: hidden;
}}

/* Alert boxes (st.info / st.success / st.warning) */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
    background: {C["bg_card"]} !important;
    border: 1px solid {C["border"]} !important;
}}

/* Metric widget */
[data-testid="metric-container"] > div {{
    color: {C["text_primary"]} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C["accent"]} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    color: {C["text_muted"]} !important;
}}

/* Divider */
hr {{ border-color: {C["border"]} !important; }}

/* Spinner */
.stSpinner > div {{ border-top-color: {C["accent"]} !important; }}

/* ── Custom components ── */
.yt-card {{
    background: {C["bg_card"]};
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: {C["shadow"]};
    transition: box-shadow 0.25s ease, transform 0.2s ease;
    margin-bottom: 14px;
}}
.yt-card:hover {{
    box-shadow: {C["shadow_hover"]};
    transform: translateY(-2px);
}}

.metric-card {{
    background: {C["bg_card"]};
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: {C["shadow"]};
    transition: box-shadow 0.2s;
}}
.metric-card:hover {{ box-shadow: {C["shadow_hover"]}; }}
.metric-label {{
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {C["text_muted"]};
    margin-bottom: 6px;
}}
.metric-value {{
    font-size: 1.65rem;
    font-weight: 800;
    color: {C["metric_val_clr"]};
    line-height: 1.1;
}}
.metric-sub {{
    font-size: 0.68rem;
    color: {C["text_muted"]};
    margin-top: 5px;
}}

.section-hdr {{
    font-size: 0.70rem;
    font-weight: 700;
    color: {C["accent"]};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 2px solid {C["section_border"]};
    padding-bottom: 5px;
    margin: 18px 0 11px 0;
}}

.predict-card-high {{
    background: {C["result_high_bg"]};
    border: 2px solid {C["result_high_brd"]};
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(34,197,94,0.18);
}}
.predict-card-low {{
    background: {C["result_low_bg"]};
    border: 2px solid {C["result_low_brd"]};
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(239,68,68,0.18);
}}

.tip-box {{
    background: {C["tip_bg"]};
    border-left: 3px solid {C["tip_border"]};
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: {C["tip_text"]};
    margin: 6px 0;
    line-height: 1.5;
}}
.warn-box {{
    background: {C["warn_bg"]};
    border-left: 3px solid {C["warn_border"]};
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: {C["warn_text"]};
    margin: 6px 0;
    line-height: 1.5;
}}
.ok-box {{
    background: {C["success_bg"]};
    border-left: 3px solid {C["success_border"]};
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: {C["success_text"]};
    margin: 6px 0;
    line-height: 1.5;
}}

.sidebar-info {{
    background: {C["info_badge_bg"]};
    border-radius: 8px;
    padding: 9px 13px;
    font-size: 0.82rem;
    color: {C["info_badge_text"]};
    line-height: 1.6;
}}
</style>
""", unsafe_allow_html=True)

# ─── Model loading ────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "xgb_model.pkl")
META_PATH    = os.path.join(ARTIFACT_DIR, "model_meta.json")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    mdl = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return mdl, meta

model, meta = load_model()

# ─── Helpers ─────────────────────────────────────────────────────────────────
def extract_title_features(title: str):
    t = title
    has_emoji    = int(bool(re.search(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", t)))
    has_sinhala  = int(bool(re.search(r"[\u0D80-\u0DFF]", t)))
    has_english  = int(bool(re.search(r"[a-zA-Z]", t)))
    is_bilingual = int(has_sinhala == 1 and has_english == 1)
    hashtags     = t.count("#")
    return {
        "title_length":  len(t),
        "word_count":    len(t.split()),
        "hashtag_count": hashtags,
        "has_hashtag":   int(hashtags > 0),
        "has_emoji":     has_emoji,
        "has_sinhala":   has_sinhala,
        "has_english":   has_english,
        "is_bilingual":  is_bilingual,
        "has_numbers":   int(bool(re.search(r"\d", t))),
        "has_question":  int("?" in t),
        "pipe_count":    t.count("|"),
        "exclaim_count": t.count("!"),
    }

def compute_shap_like(mdl, X_row, features, n=30):
    base_prob = mdl.predict_proba(X_row)[0, 1]
    contribs  = {}
    rng = np.random.default_rng(seed=42)
    numeric_feats = {
        "duration_sec", "hashtag_count", "title_length", "word_count",
        "pipe_count", "exclaim_count", "publish_month", "publish_dow",
        "channel_subscribers", "channel_age_days", "duration_category",
    }
    for feat in features:
        deltas = []
        for _ in range(n):
            Xp = X_row.copy()
            Xp[feat] = (rng.choice([0, 1, 2, 5, 10, 30, 60, 120, 300, 600], size=1)[0]
                        if feat in numeric_feats
                        else rng.integers(0, 2, size=1)[0])
            deltas.append(mdl.predict_proba(Xp)[0, 1] - base_prob)
        contribs[feat] = -np.mean(deltas)
    return base_prob, contribs

FLABELS = {
    "duration_sec": "Duration (sec)", "is_short": "Is Short (≤60s)",
    "duration_category": "Duration Category", "title_length": "Title Length",
    "word_count": "Word Count", "hashtag_count": "Hashtag Count",
    "has_hashtag": "Has Hashtag", "has_emoji": "Has Emoji",
    "has_sinhala": "Has Sinhala", "has_english": "Has English",
    "is_bilingual": "Is Bilingual", "has_numbers": "Has Numbers",
    "has_question": "Has Question", "pipe_count": "Pipe Count",
    "exclaim_count": "Exclamation Count", "publish_month": "Publish Month",
    "publish_dow": "Publish Day of Week", "is_weekend": "Is Weekend",
    "channel_subscribers": "Channel Subscribers", "channel_age_days": "Channel Age (days)",
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_NAMES   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def plot_layout(**kw):
    base = dict(
        paper_bgcolor=C["plot_bg"], plot_bgcolor=C["plot_bg"],
        font=dict(family="Inter", color=C["font_clr"], size=12),
        margin=dict(l=14, r=14, t=30, b=14),
    )
    base.update(kw)
    return base

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + theme toggle row
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown(f"""
        <div style="padding:10px 0 2px 0;">
          <span style="font-size:1.15rem;font-weight:800;color:{C['sidebar_logo_cl']};">
            YouTube Predictor
          </span><br>
          <span style="font-size:0.74rem;color:{C['sidebar_sub_cl']};">
            Sri Lankan Creator Tool
          </span>
        </div>""", unsafe_allow_html=True)
    with col_toggle:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        icon = "☀️" if T == "dark" else "🌙"
        label = f"{icon}"
        if st.button(label, key="theme_btn", help=C["toggle_label"]):
            st.session_state.theme = "dark" if T == "light" else "light"
            st.rerun()

    st.divider()

    # ── Channel ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Channel Info</div>', unsafe_allow_html=True)
    channel_option = st.selectbox(
        "Select channel",
        ("Rasmi Vibes (300 subs)", "Hey Lee (3,810 subs)",
         "Timeline of Nuraj (9,390 subs)", "Custom Channel"),
    )
    PRESETS = {
        "Rasmi Vibes (300 subs)":         (300,  277),
        "Hey Lee (3,810 subs)":           (3810, 946),
        "Timeline of Nuraj (9,390 subs)": (9390, 4913),
        "Custom Channel":                 None,
    }
    if PRESETS[channel_option] is None:
        channel_subscribers = st.number_input("Subscribers", 0, 10_000_000, 1000, 100)
        channel_age_days    = st.number_input("Channel Age (days)", 1, 20000, 365)
    else:
        channel_subscribers, channel_age_days = PRESETS[channel_option]
        st.markdown(f"""
        <div class="sidebar-info">
          <b>{channel_subscribers:,}</b> subscribers &nbsp;&middot;&nbsp;
          <b>{channel_age_days:,}</b> days old
        </div>""", unsafe_allow_html=True)

    # ── Video Format ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Video Format</div>', unsafe_allow_html=True)
    duration_input = st.slider("Duration (seconds)", 15, 1800, 300, 15)
    if   duration_input <= 60:  dur_label = f"Short — {duration_input}s (YouTube Short)"
    elif duration_input <= 300: dur_label = f"Medium — {duration_input//60}m {duration_input%60}s"
    elif duration_input <= 600: dur_label = f"Long — {duration_input//60}m {duration_input%60}s"
    else:                       dur_label = f"Very Long — {duration_input//60}m {duration_input%60}s"
    st.caption(dur_label)

    # ── Title ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Video Title</div>', unsafe_allow_html=True)
    title_input = st.text_input(
        "Enter your video title",
        value="හොඳම Travel Vlog 2026! | Best Places in Sri Lanka #travel #srilanka",
        max_chars=200,
        help="Title features are extracted automatically.",
    )
    title_feats = extract_title_features(title_input)
    tc1, tc2, tc3, tc4 = st.columns(4)
    lang = ("SIN+ENG" if title_feats["is_bilingual"]
            else "Sinhala" if title_feats["has_sinhala"]
            else "English" if title_feats["has_english"] else "Other")
    tc1.metric("Chars", title_feats["title_length"])
    tc2.metric("Words", title_feats["word_count"])
    tc3.metric("Tags",  title_feats["hashtag_count"])
    tc4.metric("Lang",  lang)

    # ── Timing ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Publish Timing</div>', unsafe_allow_html=True)
    pub_month  = st.selectbox("Month", range(1, 13), index=1,
                              format_func=lambda m: MONTH_NAMES[m-1])
    pub_dow    = st.selectbox("Day of Week", range(7), index=4,
                              format_func=lambda d: DOW_NAMES[d])
    is_weekend = int(pub_dow >= 5)
    day_badge_bg = C["badge_weekend_bg"] if is_weekend else C["badge_day_bg"]
    day_badge_cl = C["badge_weekend_cl"] if is_weekend else C["badge_day_cl"]
    day_label    = "Weekend" if is_weekend else "Weekday"
    st.markdown(
        f'<span style="display:inline-block;background:{day_badge_bg};color:{day_badge_cl};'
        f'padding:3px 10px;border-radius:99px;font-size:0.76rem;font-weight:600;">'
        f'{day_label} &middot; {DOW_NAMES[pub_dow]}, {MONTH_NAMES[pub_month-1]}</span>',
        unsafe_allow_html=True,
    )

    st.divider()
    predict_btn = st.button("Predict Performance", use_container_width=True, type="primary")

# ─── Run prediction on button click ──────────────────────────────────────────
if predict_btn and model is not None:
    features_list = meta["features"]
    _is_short    = int(duration_input <= 60)
    _dur_cat     = 0 if duration_input <= 60 else (1 if duration_input <= 300 else (2 if duration_input <= 600 else 3))
    _input = {
        "duration_sec": duration_input, "is_short": _is_short,
        "duration_category": _dur_cat,
        "title_length": title_feats["title_length"], "word_count": title_feats["word_count"],
        "hashtag_count": title_feats["hashtag_count"], "has_hashtag": title_feats["has_hashtag"],
        "has_emoji": title_feats["has_emoji"], "has_sinhala": title_feats["has_sinhala"],
        "has_english": title_feats["has_english"], "is_bilingual": title_feats["is_bilingual"],
        "has_numbers": title_feats["has_numbers"], "has_question": title_feats["has_question"],
        "pipe_count": title_feats["pipe_count"], "exclaim_count": title_feats["exclaim_count"],
        "publish_month": pub_month, "publish_dow": pub_dow, "is_weekend": is_weekend,
        "channel_subscribers": channel_subscribers, "channel_age_days": channel_age_days,
    }
    _X       = pd.DataFrame([_input])[features_list]
    _prob_hi = float(model.predict_proba(_X)[0, 1])
    st.session_state.snap = {
        "input_dict":           _input,
        "X_input":              _X,
        "features_list":        features_list,
        "prob_high":            _prob_hi,
        "prediction":           int(_prob_hi >= 0.5),
        "prob_low":             1 - _prob_hi,
        "dur_label":            dur_label,
        "lang":                 lang,
        "title_feats":          dict(title_feats),
        "pub_dow":              pub_dow,
        "pub_month":            pub_month,
        "channel_subscribers":  channel_subscribers,
        "duration_input":       duration_input,
    }
    st.session_state.predicted = True


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:28px 0 20px 0;">
  <div class="gradient-text-wrap">
    <span class="gradient-text">YouTube Performance Predictor</span>
  </div>
  <p style="color:{C['hero_sub']};font-size:0.96rem;margin-top:10px;
            letter-spacing:0.01em;line-height:1.6;">
    Pre-upload success prediction for Sri Lankan creators
    &nbsp;&mdash;&nbsp; powered by XGBoost &amp; Explainable AI
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Run `python main.py` then `python save_model.py`, then refresh.")
    st.stop()

# ─── Unpack last prediction snapshot (if any) ────────────────────────────────
_s = st.session_state.snap
if st.session_state.predicted:
    input_dict    = _s["input_dict"]
    X_input       = _s["X_input"]
    features_list = _s["features_list"]
    prob_high     = _s["prob_high"]
    prediction    = _s["prediction"]
    prob_low      = _s["prob_low"]
    _dur_label    = _s["dur_label"]
    _lang         = _s["lang"]
    _title_feats  = _s["title_feats"]
    _pub_dow      = _s["pub_dow"]
    _pub_month    = _s["pub_month"]
    _ch_subs      = _s["channel_subscribers"]
    _dur_input    = _s["duration_input"]

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Prediction  ",
    "  Feature Explanation  ",
    "  Model Overview  ",
    "  How It Works  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not st.session_state.predicted:
        # ── Welcome / placeholder state ────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px 70px 20px;">
          <div style="font-size:3.5rem;margin-bottom:16px;">&#127916;</div>
          <div style="font-size:1.4rem;font-weight:700;color:{C['text_primary']};margin-bottom:10px;">
            Ready to Predict
          </div>
          <div style="font-size:0.95rem;color:{C['text_secondary']};max-width:480px;
                      margin:0 auto;line-height:1.7;">
            Configure your video settings in the sidebar — channel, duration,
            title, and publish timing — then click
            <b style="color:{C['accent']};">Predict Performance</b> to see
            how your video is likely to perform before uploading.
          </div>
          <div style="margin-top:28px;">
            <span style="display:inline-flex;align-items:center;gap:8px;
                         background:{C['accent_light']};color:{C['accent_text']};
                         padding:10px 22px;border-radius:99px;
                         font-size:0.85rem;font-weight:600;
                         border:1px solid {C['accent']}33;">
              &#8592; Configure your settings in the sidebar, then click Predict
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Results ────────────────────────────────────────────────────────
        st.markdown(f"<h3 style='color:{C['text_primary']};margin-top:4px;'>Prediction Result</h3>",
                    unsafe_allow_html=True)
        col_res, col_viz = st.columns([1, 1], gap="large")

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="predict-card-high">
                  <div style="font-size:2.8rem;margin-bottom:6px;">&#127942;</div>
                  <div style="font-size:1.35rem;font-weight:800;color:{C['result_high_clr']};
                              letter-spacing:-0.5px;">HIGH PERFORMER</div>
                  <div style="font-size:2.6rem;font-weight:800;color:{C['result_high_num']};
                              margin:10px 0;line-height:1;">{prob_high*100:.1f}%</div>
                  <div style="font-size:0.83rem;color:{C['text_secondary']};">
                    probability of above-median views
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="predict-card-low">
                  <div style="font-size:2.8rem;margin-bottom:6px;">&#128201;</div>
                  <div style="font-size:1.35rem;font-weight:800;color:{C['result_low_clr']};
                              letter-spacing:-0.5px;">BELOW MEDIAN</div>
                  <div style="font-size:2.6rem;font-weight:800;color:{C['result_low_num']};
                              margin:10px 0;line-height:1;">{prob_low*100:.1f}%</div>
                  <div style="font-size:0.83rem;color:{C['text_secondary']};">
                    probability of below-median views
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown(f"<b style='color:{C['text_primary']};font-size:0.9rem;'>Settings Used</b>",
                        unsafe_allow_html=True)
            settings_df = pd.DataFrame({
                "Setting": ["Duration","Title Length","Hashtags","Language","Publish Day","Channel Size"],
                "Value":   [_dur_label, f"{_title_feats['title_length']} chars",
                            f"{_title_feats['hashtag_count']} hashtags", _lang,
                            f"{DOW_NAMES[_pub_dow]}, {MONTH_NAMES[_pub_month-1]}",
                            f"{_ch_subs:,} subscribers"],
            })
            st.dataframe(settings_df, hide_index=True, use_container_width=True)

        with col_viz:
            g_color = C["bar_high_line"] if prediction == 1 else C["bar_low_line"]
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_high * 100,
                title={"text": "P(High Performer)", "font": {"size": 14, "color": C["axis"]}},
                delta={"reference": 50,
                       "increasing": {"color": C["bar_high_line"]},
                       "decreasing": {"color": C["bar_low_line"]}},
                number={"suffix": "%", "font": {"size": 34, "color": C["font_clr"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": C["axis"],
                             "tickfont": {"color": C["axis"]}},
                    "bar":  {"color": g_color},
                    "bgcolor": C["gauge_bg"],
                    "borderwidth": 1, "bordercolor": C["gauge_brd"],
                    "steps": [
                        {"range": [0,  50], "color": C["step_low"]},
                        {"range": [50,100], "color": C["step_high"]},
                    ],
                    "threshold": {"line": {"color": C["accent"], "width": 3},
                                  "thickness": 0.8, "value": 50},
                },
            ))
            fig_g.update_layout(**plot_layout(height=275))
            st.plotly_chart(fig_g, use_container_width=True)

            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=["Below Median", "High Performer"],
                y=[prob_low * 100, prob_high * 100],
                marker_color=[C["bar_low"], C["bar_high"]],
                marker_line_color=[C["bar_low_line"], C["bar_high_line"]],
                marker_line_width=1.5,
                text=[f"{prob_low*100:.1f}%", f"{prob_high*100:.1f}%"],
                textposition="outside",
                textfont=dict(size=14, color=C["font_clr"], family="Inter"),
            ))
            fig_b.add_hline(y=50, line_dash="dash", line_color=C["hline"],
                            annotation_text="Baseline (50%)", annotation_position="right",
                            annotation_font_color=C["hline"])
            fig_b.update_layout(**plot_layout(
                height=230, showlegend=False,
                yaxis=dict(gridcolor=C["grid"], range=[0, 120],
                           title="Probability (%)", color=C["axis"]),
                xaxis=dict(color=C["axis"]),
            ))
            st.plotly_chart(fig_b, use_container_width=True)

        # Tips
        st.markdown("---")
        st.markdown(f"<h4 style='color:{C['text_primary']};'>Actionable Tips</h4>",
                    unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        with tc1:
            n_tags = _title_feats["hashtag_count"]
            if n_tags == 0:
                st.markdown('<div class="warn-box"><b>No hashtags detected.</b> Adding 3–5 relevant hashtags increases discoverability significantly.</div>', unsafe_allow_html=True)
            elif n_tags > 6:
                st.markdown(f'<div class="warn-box"><b>{n_tags} hashtags is excessive.</b> YouTube recommends 3–5. Too many may be treated as spam.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ok-box"><b>{n_tags} hashtags</b> — within the recommended 3–5 range.</div>', unsafe_allow_html=True)

            if _title_feats["is_bilingual"]:
                st.markdown('<div class="ok-box"><b>Bilingual title detected</b> — Sinhala + English combination is associated with higher reach.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="tip-box"><b>Consider a bilingual title.</b> Sinhala + English titles perform better in Sri Lankan channels.</div>', unsafe_allow_html=True)
        with tc2:
            if _dur_input <= 60:
                st.markdown('<div class="ok-box"><b>YouTube Short (≤60s)</b> — receives an algorithmic boost on the Shorts feed.</div>', unsafe_allow_html=True)
            elif 180 <= _dur_input <= 480:
                st.markdown('<div class="ok-box"><b>Optimal duration (3–8 min)</b> — this range correlates with good watch-through rates.</div>', unsafe_allow_html=True)
            elif _dur_input > 600:
                st.markdown('<div class="warn-box"><b>Very long video (>10 min).</b> Ensure the content justifies the length — retention may drop.</div>', unsafe_allow_html=True)

            tl = _title_feats["title_length"]
            if tl < 40:
                st.markdown(f'<div class="warn-box"><b>Short title ({tl} chars).</b> Longer, keyword-rich titles improve search visibility.</div>', unsafe_allow_html=True)
            elif tl > 100:
                st.markdown(f'<div class="warn-box"><b>Long title ({tl} chars).</b> YouTube truncates titles beyond ~70 chars in search results.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ok-box"><b>Good title length ({tl} chars)</b> — keyword-rich and fully visible in search.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<h3 style='color:{C['text_primary']};margin-top:4px;'>Feature Contribution Analysis</h3>",
                unsafe_allow_html=True)

    if not st.session_state.predicted:
        # ── Placeholder ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center;padding:48px 20px 36px 20px;">
          <div style="font-size:0.95rem;color:{C['text_secondary']};max-width:440px;
                      margin:0 auto;line-height:1.7;">
            Feature contribution charts will appear here after your first prediction.
            Configure your video settings in the sidebar and click
            <b style="color:{C['accent']};">Predict Performance</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── SHAP-like contribution chart ──────────────────────────────────────
        st.caption("Which upload decisions pushed the prediction toward or away from "
                   "'High Performer'. Analogous to SHAP values.")

        with st.spinner("Computing feature contributions…"):
            base_prob, contribs = compute_shap_like(model, X_input, features_list)

        sorted_pairs = sorted(
            zip(features_list,
                [FLABELS.get(f, f) for f in features_list],
                [contribs[f] for f in features_list],
                [input_dict[f] for f in features_list]),
            key=lambda x: abs(x[2]), reverse=True,
        )
        sp_rev = list(reversed(sorted_pairs[:15]))

        shap_colors = [C["shap_pos"] if v >= 0 else C["shap_neg"] for _, _, v, _ in sp_rev]
        shap_lines  = [C["shap_pos_ln"] if v >= 0 else C["shap_neg_ln"] for _, _, v, _ in sp_rev]

        fig_shap = go.Figure(go.Bar(
            x=[v for _, _, v, _ in sp_rev],
            y=[n for _, n, _, _ in sp_rev],
            orientation="h",
            marker_color=shap_colors,
            marker_line_color=shap_lines,
            marker_line_width=1.2,
            customdata=[[round(iv, 2)] for _, _, _, iv in sp_rev],
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.4f}<br>Input value: %{customdata[0]}<extra></extra>",
        ))
        fig_shap.add_vline(x=0, line_width=2, line_color=C["vline"])
        fig_shap.update_layout(**plot_layout(
            height=500,
            xaxis=dict(title="Contribution to P(High Performer)",
                       gridcolor=C["grid"], zerolinecolor=C["vline"], color=C["axis"]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", color=C["font_clr"]),
        ))
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(f"<h5 style='color:{C['text_primary']};'>All Feature Values</h5>",
                    unsafe_allow_html=True)
        rows = [{"Feature": n, "Your Input": iv,
                 "Contribution": f"{cv:+.4f}",
                 "Direction": "Helps" if cv > 0 else ("Hurts" if cv < 0 else "Neutral")}
                for _, n, cv, iv in sorted_pairs]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Global importance always shown (no prediction needed)
    st.markdown("---")
    st.markdown(f"<h4 style='color:{C['text_primary']};'>Global Feature Importance (XGBoost)</h4>",
                unsafe_allow_html=True)
    fi_sorted = sorted(meta["feature_importances"].items(), key=lambda x: x[1], reverse=True)
    top_key   = fi_sorted[0][0]
    fi_rev    = list(reversed(fi_sorted))
    fi_colors = [C["fi_bar_top"] if k == top_key else C["fi_bar"] for k, _ in fi_rev]

    fig_fi = go.Figure(go.Bar(
        x=[v for _, v in fi_rev],
        y=[FLABELS.get(k, k) for k, _ in fi_rev],
        orientation="h",
        marker_color=fi_colors,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(**plot_layout(
        height=520,
        xaxis=dict(title="Feature Importance Score", gridcolor=C["grid"], color=C["axis"]),
        yaxis=dict(color=C["font_clr"]),
    ))
    st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<h3 style='color:{C['text_primary']};margin-top:4px;'>Model Performance Summary</h3>",
                unsafe_allow_html=True)

    metrics = [
        ("Test Accuracy",    "71.4%", "+21.4pp vs random"),
        ("Test F1 Score",    "0.700", "Balanced precision/recall"),
        ("Test AUC-ROC",     "0.814", "Strong discrimination"),
        ("10-CV Accuracy",   "72.7%", "±8.6% — robust estimate"),
        ("Training Samples", "191",   "275 total dataset"),
    ]
    cols = st.columns(5)
    for col, (lbl, val, sub) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"<h5 style='color:{C['text_primary']};'>Train / Validation / Test Split</h5>",
                    unsafe_allow_html=True)
        split_fig = go.Figure(go.Pie(
            labels=["Train (69.5%)", "Validation (15.3%)", "Test (15.3%)"],
            values=[191, 42, 42],
            marker_colors=C["pie_colors"],
            textinfo="label+percent", textfont_size=12, hole=0.44,
        ))
        split_fig.update_layout(**plot_layout(height=290, showlegend=False))
        st.plotly_chart(split_fig, use_container_width=True)

    with col_r:
        st.markdown(f"<h5 style='color:{C['text_primary']};'>XGBoost Hyperparameters</h5>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Parameter": ["n_estimators","max_depth","learning_rate",
                          "subsample","colsample_bytree","min_child_weight"],
            "Value":     [200, 3, 0.05, 0.8, 0.8, 5],
            "Rationale": ["Sufficient trees to converge","Shallow → less overfitting",
                          "Small step → better generalisation",
                          "Row sub-sampling (stochastic boost)",
                          "Column sub-sampling per tree",
                          "Min 5 samples per leaf (regularisation)"],
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h5 style='color:{C['text_primary']};'>Confusion Matrix (Test Set, n=42)</h5>",
                unsafe_allow_html=True)
    cm_fig = go.Figure(go.Heatmap(
        z=[[13, 8], [4, 17]],
        x=["Predicted Low", "Predicted High"],
        y=["Actual Low",    "Actual High"],
        text=[["True Neg<br>13 (31%)", "False Pos<br>8 (19%)"],
              ["False Neg<br>4 (10%)", "True Pos<br>17 (40%)"]],
        texttemplate="%{text}",
        colorscale=C["cm_scale"],
        showscale=False, hoverinfo="skip",
        textfont=dict(size=13, color=C["font_clr"]),
    ))
    cm_fig.update_layout(**plot_layout(
        height=265,
        xaxis=dict(color=C["axis"]),
        yaxis=dict(color=C["axis"]),
    ))
    st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"<h5 style='color:{C['text_primary']};'>Dataset — Three Sri Lankan YouTube Channels</h5>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Channel":     ["Rasmi Vibes", "Hey Lee", "Timeline of Nuraj"],
        "Subscribers": [300, 3810, 9390],
        "Videos":      [50, 86, 139],
        "Niche":       ["Travel / Lifestyle","University / Personal","University Life"],
        "Age (days)":  [277, 946, 4913],
    }), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"<h3 style='color:{C['text_primary']};margin-top:4px;'>How This Tool Works</h3>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div class="yt-card">
<h4 style="color:{C['accent']};margin-top:0;">Algorithm: XGBoost</h4>

**XGBoost** (eXtreme Gradient Boosting) builds decision trees *sequentially*.
Each new tree corrects the residual errors of the previous ones.

1. Start with a base prediction (class prior)
2. Compute residual errors
3. Train a shallow tree to predict residuals
4. Add tree predictions scaled by `learning_rate`
5. Repeat 200 times

Unlike Random Forest (parallel trees) or Logistic Regression (linear boundary),
XGBoost uses **gradient descent in function space** with L1/L2 regularisation.
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
<div class="yt-card">
<h4 style="color:{C['accent']};margin-top:0;">No Data Leakage</h4>

Only features available **before upload** are used:

| Group   | Features |
|---------|----------|
| Format  | Duration, Is Short, Duration Category |
| Title   | Length, Words, Hashtags, Language |
| Timing  | Month, Day of Week, Weekend |
| Channel | Subscribers, Age (days) |

**Excluded features:** Views, Likes, Comments, Watch Time, and CTR.
These are all post-publication metrics that would cause circular reasoning.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
<div class="yt-card">
<h4 style="color:{C['accent']};margin-top:0;">Channel-Relative Labeling</h4>

A video is labelled **"High Performer"** if its views ≥ the channel's own median.
This neutralises subscriber-count differences across channels:

- **Rasmi Vibes:** median ≈ 688 views
- **Hey Lee:** median ≈ 3,604 views
- **Timeline of Nuraj:** median ≈ 4,210 views

A fixed threshold (e.g. "1000 views = viral") would be unfair across channels of
different sizes. The per-channel median ensures a balanced 50/50 class split.

---
<h4 style="color:{C['accent']};">Feature Contribution Method (SHAP-like)</h4>

For each prediction, feature contributions are estimated by randomly perturbing
each feature and measuring how much the predicted probability changes.
Features that cause large changes when perturbed are most influential for
that specific prediction. This is analogous to SHAP's Kernel method applied locally.
</div>""", unsafe_allow_html=True)

    st.success("**Test Accuracy:** 71.4%  ·  **Cross-Validation:** 72.7%\n\n"
               "**Random Baseline:** 50.0%\n\n"
               "**Improvement:** +21.4 percentage points over random prediction")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr style="border-color:{C['footer_border']};margin-top:40px;">
<div style="text-align:center;color:{C['footer_text']};font-size:0.74rem;
            padding:10px 0 18px 0;letter-spacing:0.02em;">
  YouTube Performance Predictor &nbsp;&middot;&nbsp;
  Rashmi Jayawardhana &nbsp;&middot;&nbsp;
  XGBoost + Explainable AI &nbsp;&middot;&nbsp; Pre-Upload Features Only
</div>
""", unsafe_allow_html=True)
