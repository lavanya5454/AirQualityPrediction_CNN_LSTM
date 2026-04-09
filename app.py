import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

from keras.models import load_model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense · CNN-BiLSTM Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PURPLE & WHITE THEME  (injected CSS)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f9f7ff;
        color: #1a0040;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3b0086 0%, #6a0dad 100%);
        color: #fff;
    }
    [data-testid="stSidebar"] * {
        color: #f0e6ff !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.25);
    }
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: #f0e6ff !important;
    }

    /* ── Main area headings ── */
    h1, h2, h3, h4, h5 { color: #4a0080; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1.5px solid #c084fc;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(107,33,168,0.08);
    }
    [data-testid="stMetricLabel"]  { color: #6b21a8 !important; font-weight: 600; }
    [data-testid="stMetricValue"]  { color: #3b0086 !important; }
    [data-testid="stMetricDelta"]  { color: #7c3aed !important; }

    /* ── Form / input widgets ── */
    [data-testid="stForm"] {
        background: #ffffff;
        border: 1.5px solid #ddd6fe;
        border-radius: 14px;
        padding: 1.5rem 1.5rem 0.5rem;
        box-shadow: 0 3px 12px rgba(107,33,168,0.07);
    }
    .stNumberInput input, .stSelectbox select, .stSlider {
        border-color: #a855f7 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #6a0dad, #9333ea);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.65rem 1.5rem;
        box-shadow: 0 4px 14px rgba(106,13,173,0.35);
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(90deg, #5b009e, #7e22ce);
        box-shadow: 0 6px 20px rgba(106,13,173,0.5);
        transform: translateY(-1px);
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1.5px solid #ddd6fe;
        border-radius: 10px;
        background: #ffffff;
    }

    /* ── Divider ── */
    hr { border-color: #ddd6fe; }

    /* ── Info box ── */
    .stAlert[data-baseweb="notification"] {
        background: #f3e8ff !important;
        border-left: 4px solid #7c3aed !important;
        color: #3b0086 !important;
    }

    /* ── Caption / small text ── */
    .stCaption, small { color: #7c3aed; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border: 1px solid #ddd6fe; border-radius: 8px; }

    /* ── Download button ── */
    .stDownloadButton > button {
        border: 2px solid #7c3aed !important;
        color: #7c3aed !important;
        border-radius: 8px;
        font-weight: 600;
        background: transparent;
    }
    .stDownloadButton > button:hover {
        background: #f3e8ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
LOOKBACK    = 48
FORECAST    = 24
TARGET_COL  = "pm2.5"
MODEL_PATH  = "best_aq_model.h5"
SCALER_PATH = "feature_scaler.pkl"

FEATURE_COLS = [
    "pm2.5", "DEWP", "TEMP", "PRES", "cbwd",
    "Iws", "Is", "Ir",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_roll24",
]

WIND_DIR_MAP = {"NE": 0, "NW": 1, "SE": 2, "SW": 3, "CV (calm)": 4}

AQI_THRESHOLDS = [
    (12.0,         "Good",                  "#16a34a", "😊"),
    (35.4,         "Moderate",              "#ca8a04", "😐"),
    (55.4,         "Unhealthy (Sensitive)", "#ea580c", "😷"),
    (150.4,        "Unhealthy",             "#dc2626", "🤢"),
    (250.4,        "Very Unhealthy",        "#7e22ce", "🤮"),
    (float("inf"), "Hazardous",             "#1e1b4b", "☠️"),
]

UNHEALTHY_THRESHOLD = 55.4   # PM2.5 > this → any "Unhealthy" tier

def aqi_info(pm_value):
    for limit, label, color, icon in AQI_THRESHOLDS:
        if pm_value <= limit:
            return label, color, icon
    return "Hazardous", "#1e1b4b", "☠️"

# ─────────────────────────────────────────────
# CUSTOM KERAS LAYER
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="custom")
class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V  = layers.Dense(1)

    def call(self, query, values):
        q       = tf.expand_dims(query, 1)
        score   = self.V(tf.nn.tanh(self.W1(values) + self.W2(q)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * values, axis=1)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading CNN-BiLSTM model…")
def load_artifacts():
    model  = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
    )
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, scaler.n_features_in_

# ─────────────────────────────────────────────
# BUILD 48-HOUR WINDOW FROM USER INPUTS
# ─────────────────────────────────────────────
def build_48h_window(pm25, dewp, temp, pres, cbwd_code,
                     iws, snow_hrs, rain_hrs, month, current_hour,
                     feature_cols):
    rng = np.random.default_rng(42)
    n   = LOOKBACK

    hour_offsets = np.arange(n)
    abs_hours    = (current_hour - (n - 1 - hour_offsets)) % 24
    abs_months   = np.full(n, month)

    trend    = np.linspace(pm25 * 0.7, pm25, n)
    noise    = rng.normal(0, max(pm25 * 0.06, 2), n)
    diurnal  = 10 * np.sin(2 * np.pi * abs_hours / 24)
    pm25_arr = np.clip(trend + noise + diurnal, 1, 1000)
    pm25_arr[-1] = pm25

    dewp_arr = dewp + rng.normal(0, 0.5, n)
    temp_arr = temp + rng.normal(0, 0.8, n)
    pres_arr = pres + rng.normal(0, 0.3, n)
    iws_arr  = np.clip(iws + rng.normal(0, iws * 0.05 + 0.5, n), 0.45, 600)
    cbwd_arr = np.full(n, float(cbwd_code))
    is_arr   = np.full(n, float(snow_hrs))
    ir_arr   = np.full(n, float(rain_hrs))

    hour_sin  = np.sin(2 * np.pi * abs_hours  / 24)
    hour_cos  = np.cos(2 * np.pi * abs_hours  / 24)
    month_sin = np.sin(2 * np.pi * abs_months / 12)
    month_cos = np.cos(2 * np.pi * abs_months / 12)

    pm25_lag1   = np.roll(pm25_arr, 1);  pm25_lag1[0]  = pm25_arr[0]
    pm25_lag3   = np.roll(pm25_arr, 3);  pm25_lag3[:3] = pm25_arr[0]
    pm25_lag6   = np.roll(pm25_arr, 6);  pm25_lag6[:6] = pm25_arr[0]
    pm25_roll24 = pd.Series(pm25_arr).rolling(24, min_periods=1).mean().values

    raw = {
        "pm2.5": pm25_arr, "DEWP": dewp_arr, "TEMP": temp_arr, "PRES": pres_arr,
        "cbwd": cbwd_arr, "Iws": iws_arr, "Is": is_arr, "Ir": ir_arr,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "pm25_lag1": pm25_lag1, "pm25_lag3": pm25_lag3,
        "pm25_lag6": pm25_lag6, "pm25_roll24": pm25_roll24,
    }

    df = pd.DataFrame({k: raw[k] for k in feature_cols if k in raw})
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols]

# ─────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────
def predict_24h(df_48h, model, scaler, feature_cols):
    scaled      = scaler.transform(df_48h[feature_cols].values)
    inp         = scaled[np.newaxis, ...]
    pred_scaled = model.predict(inp, verbose=0)[0]

    t_idx  = list(feature_cols).index(TARGET_COL)
    n_feat = len(feature_cols)
    dummy  = np.zeros((FORECAST, n_feat))
    dummy[:, t_idx] = pred_scaled
    raw    = np.clip(scaler.inverse_transform(dummy)[:, t_idx], 0, None)

    results = []
    for h, val in enumerate(raw, 1):
        label, color, icon = aqi_info(float(val))
        results.append({"hour": h, "pm25": float(val),
                         "category": label, "color": color, "icon": icon})
    return results

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
CHART_BG   = "#ffffff"
CHART_GRID = "#ede9fe"
CHART_FONT = "#3b0086"
LINE_COLOR = "#7c3aed"

def render_aqi_legend():
    cols = st.columns(len(AQI_THRESHOLDS))
    for col, (limit, label, color, icon) in zip(cols, AQI_THRESHOLDS):
        lim_str = f"≤ {limit}" if limit != float("inf") else "> 250"
        col.markdown(
            f"<div style='background:{color};border-radius:10px;padding:9px 6px;"
            f"text-align:center;color:white;font-size:0.73rem;box-shadow:0 2px 6px rgba(0,0,0,0.15);'>"
            f"{icon}<br><b>{label}</b><br><span style='opacity:.9'>{lim_str} µg/m³</span></div>",
            unsafe_allow_html=True,
        )


def render_metrics(results):
    values  = [r["pm25"] for r in results]
    peak_h  = results[int(np.argmax(values))]
    low_h   = results[int(np.argmin(values))]
    avg_val = float(np.mean(values))
    avg_lbl, _, _ = aqi_info(avg_val)
    unhealthy_hrs = sum(1 for r in results if r["pm25"] > UNHEALTHY_THRESHOLD)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Average PM2.5",   f"{avg_val:.1f} µg/m³",        delta=avg_lbl,                     delta_color="off")
    c2.metric("📈 Peak PM2.5",      f"{peak_h['pm25']:.1f} µg/m³", delta=f"at +{peak_h['hour']}h",    delta_color="inverse")
    c3.metric("📉 Lowest PM2.5",    f"{low_h['pm25']:.1f} µg/m³",  delta=f"at +{low_h['hour']}h",     delta_color="normal")
    c4.metric("⚠️ Unhealthy Hours", f"{unhealthy_hrs} / 24",        delta="hours PM2.5 > 55.4 µg/m³",  delta_color="off")


def render_forecast_chart(results):
    hours  = [r["hour"]     for r in results]
    values = [r["pm25"]     for r in results]
    colors = [r["color"]    for r in results]
    labels = [r["category"] for r in results]

    band_x = hours + hours[::-1]
    band_y = [v * 1.1 for v in values] + [v * 0.9 for v in values[::-1]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself", fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", name="±10% band",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=values,
        mode="lines+markers",
        line=dict(color=LINE_COLOR, width=2.5),
        marker=dict(size=10, color=colors, line=dict(width=1.5, color="white")),
        hovertemplate="Hour +%{x}<br>PM2.5: %{y:.1f} µg/m³<br>%{text}<extra></extra>",
        text=labels,
        name="Model Forecast",
    ))
    for limit, label, color, _ in AQI_THRESHOLDS[:-1]:
        fig.add_hline(y=limit, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=label, annotation_position="right",
                      annotation_font_color=color, annotation_font_size=10)
    fig.update_layout(
        title=dict(text="24-Hour PM2.5 Forecast  ·  CNN-BiLSTM Model Output",
                   font=dict(color=CHART_FONT, size=16)),
        xaxis=dict(title="Hours Ahead", dtick=2, gridcolor=CHART_GRID,
                   color=CHART_FONT, zerolinecolor=CHART_GRID),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor=CHART_GRID,
                   color=CHART_FONT, zerolinecolor=CHART_GRID),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font_color=CHART_FONT, height=430,
        legend=dict(orientation="h", y=-0.18, font_color=CHART_FONT),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_aqi_distribution(results):
    counts     = Counter(r["category"] for r in results)
    colors_map = {label: color for _, label, color, _ in AQI_THRESHOLDS}
    labels_    = list(counts.keys())
    vals_      = [counts[l] for l in labels_]
    clrs_      = [colors_map.get(l, "#aaa") for l in labels_]

    fig = go.Figure(go.Pie(
        labels=labels_, values=vals_,
        marker=dict(colors=clrs_), hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} hours<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="AQI Distribution (24h)", font=dict(color=CHART_FONT)),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font_color=CHART_FONT, height=370, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hourly_table(results):
    rows = []
    for r in results:
        badge = (
            f"<span style='background:{r['color']};color:white;padding:3px 10px;"
            f"border-radius:12px;font-size:0.8rem;font-weight:600;'>"
            f"{r['icon']} {r['category']}</span>"
        )
        rows.append({"Hour": f"+{r['hour']:02d}h",
                     "PM2.5 (µg/m³)": f"{r['pm25']:.1f}",
                     "AQI Category": badge})
    st.write(pd.DataFrame(rows).to_html(escape=False, index=False),
             unsafe_allow_html=True)


def render_input_history(df_48h):
    with st.expander("🔍 View generated 48-hour input window sent to model"):
        pm_vals = df_48h["pm2.5"].values
        fig = go.Figure(go.Scatter(
            x=list(range(LOOKBACK - 1, -1, -1)),
            y=pm_vals,
            mode="lines+markers",
            line=dict(color="#9333ea", width=2),
            marker=dict(size=5, color="#6a0dad"),
            name="PM2.5 (past 48h)",
            hovertemplate="-%{x}h ago<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Historical PM2.5 — 48-Hour Input Window",
                       font=dict(color=CHART_FONT)),
            xaxis=dict(title="Hours Before Now", autorange="reversed",
                       gridcolor=CHART_GRID, color=CHART_FONT),
            yaxis=dict(title="PM2.5 (µg/m³)", gridcolor=CHART_GRID, color=CHART_FONT),
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font_color=CHART_FONT, height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_48h.style.format("{:.3f}"), use_container_width=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    try:
        model, scaler, n_feat = load_artifacts()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        model_loaded = False

    feature_cols = FEATURE_COLS[:n_feat] if model_loaded else FEATURE_COLS

    # ── Sidebar ──
    st.sidebar.markdown("## 🌫️ AirSense")
    st.sidebar.caption("CNN · BiLSTM · Bahdanau Attention")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Architecture**")
    st.sidebar.caption("1D-CNN → BiLSTM × 2 → Attention → Dense")
    st.sidebar.markdown("**Training Data**")
    st.sidebar.caption("UCI Beijing PM2.5 (2010 – 2014)")
    st.sidebar.markdown("**I/O**")
    st.sidebar.caption("Input: 48-hour window × 16 features")
    st.sidebar.caption("Output: 24-hour PM2.5 forecast")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Enter current weather conditions and click **Run Forecast**. "
        "The 48-hour history is built internally. "
        "Output is the raw neural network prediction."
    )

    # ── Hero Banner ──
    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#3b0086 0%,#6a0dad 50%,#9333ea 100%);
                    border-radius:16px;padding:2.5rem 2rem 2rem;margin-bottom:2rem;
                    box-shadow:0 8px 30px rgba(106,13,173,0.3);'>
            <h1 style='font-size:2.6rem;color:#ffffff;margin:0;font-weight:800;'>
                🌫️ AirSense — Air Quality Predictor
            </h1>
            <p style='color:#e9d5ff;margin:0.5rem 0 0;font-size:1rem;'>
                CNN + BiLSTM + Bahdanau Attention &nbsp;·&nbsp; 24-Hour PM2.5 Forecast
                &nbsp;·&nbsp; Pure model output — no rule-based logic
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.warning("Ensure **best_aq_model.h5** and **feature_scaler.pkl** are in the project root.")
        return

    # ── AQI Scale ──
    st.markdown("##### AQI Reference Scale")
    render_aqi_legend()
    st.markdown("---")

    # ── INPUT FORM ──
    st.markdown("## 📥 Enter Current Conditions")
    st.caption(
        "Fill in the present-hour weather readings. "
        "The app builds a 48-hour input window internally and runs the CNN-BiLSTM model."
    )

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🌫️ Air Quality")
            pm25 = st.number_input(
                "Current PM2.5 (µg/m³)",
                min_value=0.0, max_value=1000.0, value=75.0, step=1.0,
                help="Particulate matter concentration right now",
            )
            month = st.selectbox(
                "Month",
                list(range(1, 13)), index=2,
                format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                        "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
            )
            hour = st.slider("Current Hour (0–23)", 0, 23, 8)

        with col2:
            st.markdown("#### 🌡️ Temperature & Humidity")
            temp = st.number_input("Temperature (°C)",
                                   min_value=-40.0, max_value=60.0, value=10.0, step=0.5)
            dewp = st.number_input("Dew Point (°C)",
                                   min_value=-40.0, max_value=40.0, value=-2.0, step=0.5,
                                   help="Lower = drier air")
            pres = st.number_input("Atmospheric Pressure (hPa)",
                                   min_value=900.0, max_value=1100.0, value=1013.0, step=0.5)

        with col3:
            st.markdown("#### 💨 Wind & Precipitation")
            wind_dir = st.selectbox("Wind Direction", list(WIND_DIR_MAP.keys()), index=0)
            iws = st.number_input("Cumulated Wind Speed (m/s)",
                                  min_value=0.0, max_value=600.0, value=45.0, step=1.0,
                                  help="Cumulative wind speed over the hour")
            snow_hrs = st.number_input("Hours of Snow Today", min_value=0, max_value=24, value=0, step=1)
            rain_hrs = st.number_input("Hours of Rain Today", min_value=0, max_value=24, value=0, step=1)

        submitted = st.form_submit_button(
            "🚀  Run CNN-BiLSTM Model Forecast",
            type="primary",
            use_container_width=True,
        )

    # ── Run model ──
    if submitted:
        cbwd_code = WIND_DIR_MAP[wind_dir]
        with st.spinner("Building 48-hour window & running model inference…"):
            try:
                df_48h  = build_48h_window(pm25, dewp, temp, pres, cbwd_code,
                                            iws, snow_hrs, rain_hrs, month, hour, feature_cols)
                results = predict_24h(df_48h, model, scaler, feature_cols)
                st.session_state["results"]  = results
                st.session_state["df_input"] = df_48h
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ── Display Results ──
    if "results" in st.session_state:
        results = st.session_state["results"]
        df_48h  = st.session_state.get("df_input")

        st.markdown("---")
        st.markdown("## 📊 Model Forecast Results")
        st.caption("✅ Raw inverse-transformed output from the CNN-BiLSTM network — no overrides applied.")

        render_metrics(results)
        st.markdown("---")

        col_l, col_r = st.columns([2, 1])
        with col_l:
            render_forecast_chart(results)
        with col_r:
            render_aqi_distribution(results)

        st.markdown("---")
        st.markdown("### 🕐 Hourly Forecast Breakdown")
        render_hourly_table(results)
        st.markdown("")

        if df_48h is not None:
            render_input_history(df_48h)

        st.markdown("---")
        df_dl = pd.DataFrame([{
            "Hour Ahead":   f"+{r['hour']:02d}h",
            "PM2.5 µg/m³": round(r["pm25"], 2),
            "AQI Category": r["category"],
        } for r in results])
        st.download_button(
            "⬇️ Download Forecast CSV",
            data=df_dl.to_csv(index=False).encode(),
            file_name="pm25_24h_forecast.csv",
            mime="text/csv",
        )
    else:
        st.markdown("---")
        st.info("Fill in the conditions above and click **Run CNN-BiLSTM Model Forecast** to see the 24-hour prediction.")

if __name__ == "__main__":
    main()
