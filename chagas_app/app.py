"""
Chagas Disease Detection - Explainable AI Demo
Optimized XGBoost + SHAP | Conference & Academic Demo
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chagas Disease Detection | XAI Demo",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 50%, #1a1f35 100%);
        border: 1px solid #2a4a7f;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .header-banner h1 {
        color: #4fc3f7;
        font-size: 2.2rem;
        margin: 0 0 8px 0;
        letter-spacing: 1px;
    }
    .header-banner p {
        color: #90a4ae;
        font-size: 1rem;
        margin: 0;
    }

    /* Risk cards */
    .risk-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
    }
    .risk-low    { background: #0d2b1a; border: 2px solid #2e7d32; }
    .risk-medium { background: #2b1f00; border: 2px solid #f57c00; }
    .risk-high   { background: #2b0d0d; border: 2px solid #c62828; }

    .risk-label { font-size: 1.1rem; color: #aaa; margin-bottom: 6px; }
    .risk-value { font-size: 3rem; font-weight: 800; margin: 8px 0; }
    .risk-low    .risk-value { color: #66bb6a; }
    .risk-medium .risk-value { color: #ffa726; }
    .risk-high   .risk-value { color: #ef5350; }
    .risk-verdict { font-size: 1.4rem; font-weight: 700; }
    .risk-low    .risk-verdict { color: #66bb6a; }
    .risk-medium .risk-verdict { color: #ffa726; }
    .risk-high   .risk-verdict { color: #ef5350; }

    /* Metric boxes */
    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2a3050;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-box .label { color: #78909c; font-size: 0.85rem; margin-bottom: 4px; }
    .metric-box .value { color: #e0e0e0; font-size: 1.4rem; font-weight: 700; }

    /* Section headers */
    .section-header {
        border-left: 4px solid #4fc3f7;
        padding-left: 14px;
        margin: 28px 0 16px 0;
        font-size: 1.2rem;
        font-weight: 700;
        color: #cfd8dc;
    }

    /* Feature group header */
    .feature-group {
        background: #161b27;
        border: 1px solid #252b3d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 14px 0 6px 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #4fc3f7;
        letter-spacing: 0.5px;
    }

    /* Info box */
    .info-box {
        background: #0d1b2a;
        border: 1px solid #1565c0;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #90caf9;
    }

    /* Model stats bar */
    .stats-bar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .stat-chip {
        background: #1a2744;
        border: 1px solid #2a4a7f;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #90caf9;
    }

    /* Disclaimer */
    .disclaimer {
        background: #1a1200;
        border: 1px solid #5d4037;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #bcaaa4;
        margin-top: 24px;
    }

    div[data-testid="stSlider"] > label { color: #b0bec5 !important; font-size: 0.88rem; }
    .stSelectbox label { color: #b0bec5 !important; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model   = joblib.load("chagas_model.pkl")
        scaler  = joblib.load("chagas_scaler.pkl")
        with open("feature_names.json") as f:
            features = json.load(f)
        shap_df = pd.read_csv("shap_importance.csv")
        return model, scaler, features, shap_df, True
    except Exception as e:
        return None, None, None, None, False

model, scaler, feature_names, shap_importance, loaded = load_artifacts()

# ─────────────────────────────────────────────
# CLINICAL DESCRIPTIONS
# ─────────────────────────────────────────────
DESCRIPTIONS = {
    "P_wave_duration_mean":  "Mean P-wave duration (atrial depolarization)",
    "P_wave_duration_std":   "Std deviation of P-wave duration",
    "P_wave_duration_min":   "Minimum P-wave duration",
    "P_wave_duration_max":   "Maximum P-wave duration",
    "PR_interval_mean":      "Mean PR interval — AV conduction time",
    "PR_interval_std":       "Std deviation of PR interval",
    "PR_interval_min":       "Minimum PR interval",
    "PR_interval_max":       "Maximum PR interval",
    "PR_segment_mean":       "Mean PR segment duration",
    "PR_segment_std":        "Std deviation of PR segment",
    "PR_segment_min":        "Minimum PR segment",
    "PR_segment_max":        "Maximum PR segment",
    "QRS_duration_mean":     "Mean QRS duration — ventricular depolarization",
    "QRS_duration_std":      "Std deviation of QRS duration",
    "QRS_duration_min":      "Minimum QRS duration",
    "QRS_duration_max":      "Maximum QRS duration",
    "QT_interval_mean":      "Mean QT interval — ventricular repolarization",
    "QT_interval_std":       "Std deviation of QT interval",
    "QT_interval_min":       "Minimum QT interval",
    "QT_interval_max":       "Maximum QT interval",
    "ST_segment_mean":       "Mean ST segment level",
    "ST_segment_std":        "Std deviation of ST segment",
    "ST_segment_min":        "Minimum ST segment",
    "ST_segment_max":        "Maximum ST segment",
    "ST_slope_mean":         "Mean ST slope",
    "ST_slope_std":          "Std deviation of ST slope",
    "ST_slope_min":          "Minimum ST slope",
    "ST_slope_max":          "Maximum ST slope",
    "HRV_MeanNN":            "Mean of RR intervals (ms)",
    "HRV_SDNN":              "Std deviation of all RR intervals — overall HRV",
    "HRV_RMSSD":             "Root mean square of successive RR differences",
    "HRV_SDSD":              "Std deviation of successive RR differences",
    "HRV_CVNN":              "Coefficient of variation of RR intervals",
    "HRV_CVSD":              "Coefficient of variation of successive differences",
    "HRV_MedianNN":          "Median of RR intervals",
    "HRV_MadNN":             "Median absolute deviation of RR intervals",
    "HRV_MCVNN":             "Median-based coefficient of variation",
    "HRV_IQRNN":             "Interquartile range of RR intervals",
    "HRV_SDRMSSD":           "Ratio of SDNN to RMSSD",
    "HRV_Prc20NN":           "20th percentile of RR intervals",
    "HRV_Prc80NN":           "80th percentile of RR intervals",
    "HRV_pNN50":             "Proportion of successive RR differences > 50ms",
    "HRV_pNN20":             "Proportion of successive RR differences > 20ms",
    "HRV_MinNN":             "Minimum RR interval",
    "HRV_MaxNN":             "Maximum RR interval",
    "HRV_HTI":               "HRV triangular index",
    "HRV_TINN":              "Triangular interpolation of RR interval histogram",
    "age":                   "Patient age (years)",
    "is_male":               "Patient sex",
    "PR_QRS_ratio":          "PR interval / QRS ratio (engineered)",
    "QT_HRV_ratio":          "QT interval / HRV SDNN ratio (engineered)",
    "HRV_range":             "HRV range: MaxNN − MinNN (engineered)",
    "ST_variability":        "ST segment std / mean ratio (engineered)",
}

# Realistic default values derived from the high-risk patient in the notebook
DEFAULTS = {
    "P_wave_duration_mean": 110.0, "P_wave_duration_std": 12.0,
    "P_wave_duration_min": 90.0,   "P_wave_duration_max": 130.0,
    "PR_interval_mean": 165.0,     "PR_interval_std": 18.0,
    "PR_interval_min": 140.0,      "PR_interval_max": 195.0,
    "PR_segment_mean": 172.5,      "PR_segment_std": 20.0,
    "PR_segment_min": 145.0,       "PR_segment_max": 200.0,
    "QRS_duration_mean": 95.0,     "QRS_duration_std": 10.0,
    "QRS_duration_min": 10.0,      "QRS_duration_max": 120.0,
    "QT_interval_mean": 400.0,     "QT_interval_std": 25.0,
    "QT_interval_min": 360.0,      "QT_interval_max": 440.0,
    "ST_segment_mean": 362.5,      "ST_segment_std": 30.0,
    "ST_segment_min": 300.0,       "ST_segment_max": 420.0,
    "ST_slope_mean": 0.5,          "ST_slope_std": 0.3,
    "ST_slope_min": -0.5,          "ST_slope_max": 1.5,
    "HRV_MeanNN": 850.0,           "HRV_SDNN": 55.0,
    "HRV_RMSSD": 40.0,             "HRV_SDSD": 40.0,
    "HRV_CVNN": 0.065,             "HRV_CVSD": 0.047,
    "HRV_MedianNN": 845.0,         "HRV_MadNN": 30.0,
    "HRV_MCVNN": 0.035,            "HRV_IQRNN": 75.0,
    "HRV_SDRMSSD": 1.4,            "HRV_Prc20NN": 780.0,
    "HRV_Prc80NN": 920.0,          "HRV_pNN50": 15.0,
    "HRV_pNN20": 35.0,             "HRV_MinNN": 650.0,
    "HRV_MaxNN": 1100.0,           "HRV_HTI": 18.0,
    "HRV_TINN": 350.0,             "age": 59,
}

# Feature ranges for sliders
RANGES = {
    "P_wave_duration_mean": (60.0,  200.0,  1.0),
    "P_wave_duration_std":  (0.0,   60.0,   0.5),
    "P_wave_duration_min":  (40.0,  160.0,  1.0),
    "P_wave_duration_max":  (80.0,  250.0,  1.0),
    "PR_interval_mean":     (80.0,  400.0,  1.0),
    "PR_interval_std":      (0.0,   80.0,   0.5),
    "PR_interval_min":      (60.0,  350.0,  1.0),
    "PR_interval_max":      (100.0, 500.0,  1.0),
    "PR_segment_mean":      (20.0,  300.0,  1.0),
    "PR_segment_std":       (0.0,   80.0,   0.5),
    "PR_segment_min":       (0.0,   250.0,  1.0),
    "PR_segment_max":       (40.0,  400.0,  1.0),
    "QRS_duration_mean":    (40.0,  200.0,  1.0),
    "QRS_duration_std":     (0.0,   50.0,   0.5),
    "QRS_duration_min":     (0.0,   150.0,  1.0),
    "QRS_duration_max":     (60.0,  250.0,  1.0),
    "QT_interval_mean":     (250.0, 600.0,  1.0),
    "QT_interval_std":      (0.0,   80.0,   0.5),
    "QT_interval_min":      (200.0, 500.0,  1.0),
    "QT_interval_max":      (300.0, 700.0,  1.0),
    "ST_segment_mean":      (100.0, 600.0,  1.0),
    "ST_segment_std":       (0.0,   100.0,  0.5),
    "ST_segment_min":       (50.0,  500.0,  1.0),
    "ST_segment_max":       (150.0, 700.0,  1.0),
    "ST_slope_mean":        (-3.0,  3.0,    0.01),
    "ST_slope_std":         (0.0,   3.0,    0.01),
    "ST_slope_min":         (-5.0,  2.0,    0.01),
    "ST_slope_max":         (-2.0,  5.0,    0.01),
    "HRV_MeanNN":           (400.0, 1500.0, 1.0),
    "HRV_SDNN":             (5.0,   250.0,  0.5),
    "HRV_RMSSD":            (5.0,   200.0,  0.5),
    "HRV_SDSD":             (5.0,   200.0,  0.5),
    "HRV_CVNN":             (0.01,  0.3,    0.001),
    "HRV_CVSD":             (0.01,  0.3,    0.001),
    "HRV_MedianNN":         (400.0, 1500.0, 1.0),
    "HRV_MadNN":            (1.0,   150.0,  0.5),
    "HRV_MCVNN":            (0.001, 0.2,    0.001),
    "HRV_IQRNN":            (5.0,   400.0,  1.0),
    "HRV_SDRMSSD":          (0.1,   10.0,   0.01),
    "HRV_Prc20NN":          (300.0, 1400.0, 1.0),
    "HRV_Prc80NN":          (500.0, 1600.0, 1.0),
    "HRV_pNN50":            (0.0,   100.0,  0.1),
    "HRV_pNN20":            (0.0,   100.0,  0.1),
    "HRV_MinNN":            (300.0, 1200.0, 1.0),
    "HRV_MaxNN":            (600.0, 2000.0, 1.0),
    "HRV_HTI":              (2.0,   80.0,   0.1),
    "HRV_TINN":             (50.0,  1000.0, 1.0),
    "age":                  (18,    95,     1),
}

# Feature groups for sidebar organization
GROUPS = {
    "👤 Demographics": ["age", "is_male"],
    "📊 P-Wave": ["P_wave_duration_mean","P_wave_duration_std","P_wave_duration_min","P_wave_duration_max"],
    "📊 PR Interval": ["PR_interval_mean","PR_interval_std","PR_interval_min","PR_interval_max"],
    "📊 PR Segment": ["PR_segment_mean","PR_segment_std","PR_segment_min","PR_segment_max"],
    "📊 QRS Duration": ["QRS_duration_mean","QRS_duration_std","QRS_duration_min","QRS_duration_max"],
    "📊 QT Interval": ["QT_interval_mean","QT_interval_std","QT_interval_min","QT_interval_max"],
    "📊 ST Segment": ["ST_segment_mean","ST_segment_std","ST_segment_min","ST_segment_max"],
    "📊 ST Slope": ["ST_slope_mean","ST_slope_std","ST_slope_min","ST_slope_max"],
    "❤️ HRV — Time Domain": [
        "HRV_MeanNN","HRV_SDNN","HRV_RMSSD","HRV_SDSD",
        "HRV_CVNN","HRV_CVSD","HRV_MedianNN","HRV_MadNN",
        "HRV_MCVNN","HRV_IQRNN","HRV_SDRMSSD",
        "HRV_Prc20NN","HRV_Prc80NN","HRV_pNN50","HRV_pNN20",
        "HRV_MinNN","HRV_MaxNN","HRV_HTI","HRV_TINN"
    ],
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🫀 Chagas Disease Detection</h1>
  <p>Explainable AI · Optimized XGBoost + SHAP · ECG Biomarker Analysis</p>
</div>
""", unsafe_allow_html=True)

# Model stats bar
st.markdown("""
<div class="stats-bar">
  <span class="stat-chip">🎯 Test Accuracy: 72.93%</span>
  <span class="stat-chip">📈 ROC-AUC: 80.59%</span>
  <span class="stat-chip">🔬 Dataset: 13,092 patients</span>
  <span class="stat-chip">🌲 XGBoost · Optuna · 100 Trials</span>
  <span class="stat-chip">🧠 SHAP Explainability</span>
  <span class="stat-chip">📋 53 ECG Features</span>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error("⚠️ Model files not found. Place `chagas_model.pkl`, `chagas_scaler.pkl`, `feature_names.json`, and `shap_importance.csv` in the same folder as `app.py`.")
    st.info("Run the export cell in your Colab notebook, download the 4 files to this folder, then restart the app.")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — PATIENT INPUT
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🩺 Patient ECG Input")
st.sidebar.markdown("---")

input_values = {}

for group_name, features in GROUPS.items():
    with st.sidebar.expander(group_name, expanded=(group_name == "👤 Demographics")):
        for feat in features:
            if feat == "is_male":
                sex = st.selectbox("Sex", ["Male", "Female"], key=feat)
                input_values[feat] = 1 if sex == "Male" else 0
            elif feat == "age":
                r = RANGES[feat]
                input_values[feat] = st.slider(
                    "Age (years)", int(r[0]), int(r[1]),
                    int(DEFAULTS.get(feat, 50)), int(r[2]), key=feat
                )
            else:
                r = RANGES.get(feat, (0.0, 1000.0, 1.0))
                default = float(DEFAULTS.get(feat, (r[0]+r[1])/2))
                input_values[feat] = st.slider(
                    feat, float(r[0]), float(r[1]),
                    default, float(r[2]),
                    help=DESCRIPTIONS.get(feat, ""), key=feat
                )

# ─────────────────────────────────────────────
# COMPUTE ENGINEERED FEATURES
# ─────────────────────────────────────────────
iv = input_values
iv["PR_QRS_ratio"]   = iv["PR_interval_mean"] / (iv["QRS_duration_mean"] + 1e-8)
iv["QT_HRV_ratio"]   = iv["QT_interval_mean"] / (iv["HRV_SDNN"] + 1e-8)
iv["HRV_range"]      = iv["HRV_MaxNN"] - iv["HRV_MinNN"]
iv["ST_variability"] = iv["ST_segment_std"] / (iv["ST_segment_mean"] + 1e-8)

# Build input DataFrame in correct feature order
input_df = pd.DataFrame([{f: iv[f] for f in feature_names}])

# Scale
input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

# Predict
prob      = model.predict_proba(input_df)[0][1]
pred      = model.predict(input_df)[0]
prob_pct  = prob * 100

# Risk level
if prob_pct < 35:
    risk_level = "low";    risk_class = "risk-low";    risk_emoji = "🟢"; verdict = "LOW RISK"
elif prob_pct < 65:
    risk_level = "medium"; risk_class = "risk-medium"; risk_emoji = "🟡"; verdict = "MODERATE RISK"
else:
    risk_level = "high";   risk_class = "risk-high";   risk_emoji = "🔴"; verdict = "HIGH RISK"

# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────
col_risk, col_metrics = st.columns([1, 2])

with col_risk:
    st.markdown(f"""
    <div class="risk-card {risk_class}">
      <div class="risk-label">Chagas Disease Probability</div>
      <div class="risk-value">{prob_pct:.1f}%</div>
      <div class="risk-verdict">{risk_emoji} {verdict}</div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    st.markdown('<div class="section-header">Model Output Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="label">Prediction</div><div class="value">{"Chagas +" if pred else "Chagas −"}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="label">Probability</div><div class="value">{prob_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="label">Age</div><div class="value">{iv["age"]} yrs</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="label">Sex</div><div class="value">{"Male" if iv["is_male"] else "Female"}</div></div>', unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    bar_color = "#ef5350" if risk_level == "high" else ("#ffa726" if risk_level == "medium" else "#66bb6a")
    st.markdown(f"""
    <div style="background:#1a1f2e; border-radius:8px; height:22px; overflow:hidden;">
      <div style="background:{bar_color}; width:{prob_pct:.1f}%; height:100%; border-radius:8px;
                  transition:width 0.5s ease; display:flex; align-items:center; padding-left:8px;">
        <span style="color:#fff; font-size:0.8rem; font-weight:700;">{prob_pct:.1f}%</span>
      </div>
    </div>
    <div style="display:flex; justify-content:space-between; color:#546e7a; font-size:0.75rem; margin-top:4px;">
      <span>0% — No Risk</span><span>50% — Threshold</span><span>100% — Certain</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────────
col_shap, col_top = st.columns([3, 2])

with col_shap:
    st.markdown('<div class="section-header">🧠 SHAP — Why This Prediction?</div>', unsafe_allow_html=True)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Waterfall-style bar chart
        sv     = shap_values[0]
        feat_n = feature_names
        top_n  = 15

        idx_sorted = np.argsort(np.abs(sv))[::-1][:top_n]
        top_feats  = [feat_n[i] for i in idx_sorted]
        top_vals   = [sv[i]     for i in idx_sorted]
        top_feats.reverse(); top_vals.reverse()

        colors = ["#ef5350" if v > 0 else "#42a5f5" for v in top_vals]
        labels = [f"{f}={input_df[f].values[0]:.2f}" for f in top_feats]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        bars = ax.barh(range(top_n), top_vals, color=colors, edgecolor="none", height=0.65)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels, fontsize=8.5, color="#cfd8dc")
        ax.axvline(0, color="#455a64", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value  (red = increases risk | blue = decreases risk)",
                      fontsize=9, color="#90a4ae")
        ax.set_title(f"Top {top_n} Feature Contributions for This Patient",
                     fontsize=11, color="#e0e0e0", pad=12)
        ax.tick_params(colors="#90a4ae")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["left","bottom"]:
            ax.spines[spine].set_color("#2a3050")

        red_patch  = mpatches.Patch(color="#ef5350", label="Increases Chagas risk")
        blue_patch = mpatches.Patch(color="#42a5f5", label="Decreases Chagas risk")
        ax.legend(handles=[red_patch, blue_patch], loc="lower right",
                  facecolor="#161b27", edgecolor="#2a3050", labelcolor="#cfd8dc", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP computation encountered an issue: {e}")

with col_top:
    st.markdown('<div class="section-header">📊 Global Feature Importance</div>', unsafe_allow_html=True)

    if shap_importance is not None:
        top15 = shap_importance.head(15).copy()
        top15 = top15.iloc[::-1]

        fig2, ax2 = plt.subplots(figsize=(5, 6))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#0f1117")

        bars2 = ax2.barh(
            range(len(top15)), top15["Importance"].values,
            color="#4fc3f7", alpha=0.85, edgecolor="none", height=0.65
        )
        ax2.set_yticks(range(len(top15)))
        ax2.set_yticklabels(top15["Feature"].values, fontsize=8.5, color="#cfd8dc")
        ax2.set_xlabel("Mean |SHAP Value|", fontsize=9, color="#90a4ae")
        ax2.set_title("Top 15 Global Features\n(Across All Patients)", fontsize=10, color="#e0e0e0", pad=10)
        ax2.tick_params(colors="#90a4ae")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ["left","bottom"]:
            ax2.spines[spine].set_color("#2a3050")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

st.markdown("---")

# ─────────────────────────────────────────────
# PATIENT REPORT TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Patient Input Summary — Key Features</div>', unsafe_allow_html=True)

try:
    sv_abs = np.abs(shap_values[0])
    top5_idx = np.argsort(sv_abs)[::-1][:10]

    report_rows = []
    for i in top5_idx:
        feat  = feature_names[i]
        val   = input_df[feat].values[0]
        shap_v = shap_values[0][i]
        direction = "↑ Increases Risk" if shap_v > 0 else "↓ Decreases Risk"
        report_rows.append({
            "Feature": feat,
            "Value": f"{val:.3f}" if isinstance(val, float) else str(val),
            "SHAP Impact": f"{shap_v:+.4f}",
            "Clinical Meaning": DESCRIPTIONS.get(feat, feat),
            "Direction": direction
        })

    report_df = pd.DataFrame(report_rows)
    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Direction": st.column_config.TextColumn("Direction", width="medium"),
            "SHAP Impact": st.column_config.TextColumn("SHAP Impact", width="small"),
        }
    )
except Exception as e:
    st.warning(f"Could not generate report table: {e}")

# ─────────────────────────────────────────────
# ENGINEERED FEATURES DISPLAY
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Auto-Computed Engineered Features</div>', unsafe_allow_html=True)
ec1, ec2, ec3, ec4 = st.columns(4)
with ec1:
    st.markdown(f'<div class="metric-box"><div class="label">PR/QRS Ratio</div><div class="value">{iv["PR_QRS_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
with ec2:
    st.markdown(f'<div class="metric-box"><div class="label">QT/HRV Ratio</div><div class="value">{iv["QT_HRV_ratio"]:.3f}</div></div>', unsafe_allow_html=True)
with ec3:
    st.markdown(f'<div class="metric-box"><div class="label">HRV Range (ms)</div><div class="value">{iv["HRV_range"]:.1f}</div></div>', unsafe_allow_html=True)
with ec4:
    st.markdown(f'<div class="metric-box"><div class="label">ST Variability</div><div class="value">{iv["ST_variability"]:.4f}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Research & Academic Use Only.</strong>
  This tool is a demonstration of Explainable AI applied to ECG biomarker analysis for Chagas disease detection.
  It is not a medical device and should not be used for clinical diagnosis or patient management decisions.
  All predictions are probabilistic outputs of a machine learning model trained on the Chagas Disease ECG dataset.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#37474f; font-size:0.8rem;">Chagas Disease Detection · XGBoost + Optuna + SHAP · Academic Conference Demo</p>', unsafe_allow_html=True)
