import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense · PM2.5 Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-deep:    #0d0a14;
    --bg-mid:     #130f1e;
    --bg-card:    #1a1428;
    --border:     rgba(160,120,255,0.18);
    --purple-lo:  #6b3fa0;
    --purple-hi:  #a972f5;
    --purple-glow:#c49dff;
    --white-soft: #f0eaf8;
    --white-dim:  #b8aed0;
    --accent:     #e0cfff;
    --danger:     #ff6b8a;
    --good:       #72e8a0;
    --warn:       #f5c842;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--white-soft);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0b1a 0%, #160e28 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--white-soft) !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* Inputs */
input, textarea, select,
[data-testid="stNumberInput"] input,
[data-baseweb="input"] input {
    background: var(--bg-card) !important;
    color: var(--white-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--white-soft) !important;
}
[data-baseweb="popover"] * { background: #1e1632 !important; color: var(--white-soft) !important; }

/* Slider */
[data-baseweb="slider"] [role="slider"] { background: var(--purple-hi) !important; }
[data-testid="stSlider"] div[data-testid] { background: var(--purple-lo) !important; }

/* Buttons */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--purple-lo), var(--purple-hi)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 18px rgba(169,114,245,0.25) !important;
}
[data-testid="stButton"] button:hover {
    box-shadow: 0 0 28px rgba(169,114,245,0.55) !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-mid) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--white-dim) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, var(--purple-lo), var(--purple-hi)) !important;
    color: #fff !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--purple-glow) !important; font-size: 1.7rem !important; }
[data-testid="stMetricLabel"] { color: var(--white-dim) !important; }
[data-testid="stMetricDelta"] { color: var(--good) !important; }

/* Section headers */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--white-soft) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--accent) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--purple-lo); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#b8aed0",
    font_family="DM Sans",
    xaxis=dict(gridcolor="rgba(160,120,255,0.1)", zerolinecolor="rgba(160,120,255,0.15)"),
    yaxis=dict(gridcolor="rgba(160,120,255,0.1)", zerolinecolor="rgba(160,120,255,0.15)"),
)
DEFAULT_MARGIN = dict(l=20, r=20, t=40, b=20)

def aqi_label(pm):
    if pm <= 12:   return "Good",       "#72e8a0"
    if pm <= 35.4: return "Moderate",   "#f5c842"
    if pm <= 55.4: return "Sensitive",  "#f5a742"
    if pm <= 150.4:return "Unhealthy",  "#ff6b8a"
    if pm <= 250.4:return "Very Unhealthy","#c94fff"
    return "Hazardous", "#ff3355"

def card(content_html: str, height: str = "auto"):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#1a1428,#130f1e);
        border: 1px solid rgba(160,120,255,0.18);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        height: {height};
        margin-bottom: 0.4rem;
    ">{content_html}</div>""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div style="font-size:2.4rem;">🌫️</div>
        <h2 style="font-family:'Cormorant Garamond',serif; font-size:1.6rem;
                   margin:0; background:linear-gradient(135deg,#a972f5,#e0cfff);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AirSense
        </h2>
        <p style="color:#6b5590; font-size:0.75rem; margin:2px 0 0;">
            PM2.5 · CNN–BiLSTM · Attention
        </p>
    </div>
    <hr style="border-color:rgba(160,120,255,0.15); margin-bottom:1.4rem;">
    """, unsafe_allow_html=True)

    st.markdown("#### 🌡️ Current Conditions")
    temp    = st.slider("Temperature (°C)", -20.0, 42.0, 14.0, 0.5)
    dewp    = st.slider("Dew Point (°C)",   -40.0, 28.0, -2.0, 0.5)
    pres    = st.slider("Pressure (hPa)",   990.0, 1040.0, 1013.0, 0.5)
    iws     = st.slider("Wind Speed (m/s)",   0.0,  80.0,   5.0, 0.5)

    st.markdown("#### 🧭 Wind & Precipitation")
    wind_dir = st.selectbox("Wind Direction", ["NW", "NE", "SW", "SE", "cv"])
    snow    = st.slider("Snow Hours (Is)",   0, 10, 0)
    rain    = st.slider("Rain Hours (Ir)",   0, 10, 0)

    st.markdown("#### 📅 Time Context")
    col1, col2 = st.columns(2)
    with col1: month = st.selectbox("Month", list(range(1,13)), index=0)
    with col2: hour  = st.selectbox("Hour",  list(range(0,24)),  index=8)
    day = st.slider("Day of Month", 1, 31, 15)

    st.markdown("---")
    run_btn = st.button("⚡ Predict PM2.5", use_container_width=True)

# ─── Mock Prediction Engine ───────────────────────────────────────────────────
def simulate_prediction(temp, dewp, pres, iws, wind_dir, snow, rain, month, hour, day):
    """Rule-based simulator that mimics CNN-BiLSTM-Attention patterns."""
    np.random.seed(int(temp*10 + dewp + iws))
    base = 60

    # Seasonal effect
    if month in [12, 1, 2]:  base += 55
    elif month in [6, 7, 8]: base -= 20
    elif month in [3, 4, 5]: base += 10

    # Temp & dew: humidity proxy
    base += max(0, (dewp - temp + 20) * 1.8)

    # Wind disperses
    base -= iws * 0.9
    if wind_dir in ["NW", "N"]: base -= 12
    elif wind_dir == "cv":       base += 18

    # Rain / snow washout
    base -= rain * 8 + snow * 5

    # Pressure (high → stagnant → worse)
    base += (pres - 1013) * 0.35

    # Diurnal: morning rush & evening
    if hour in [7, 8, 9]:    base += 22
    elif hour in [18, 19]:   base += 14
    elif hour in [2, 3, 4]:  base -= 12

    base = max(5, base)
    noise = np.random.normal(0, 8)

    # 24-hour forecast
    hours = np.arange(24)
    diurnal = 12 * np.sin((hours - 6) * np.pi / 12)
    forecast = base + diurnal + np.random.normal(0, 6, 24)
    forecast = np.clip(forecast, 2, 500)

    return float(base + noise), forecast

def simulate_history():
    """Simulate 48h of input data resembling Beijing dataset."""
    np.random.seed(42)
    hours = np.arange(48)
    pm = 80 + 30 * np.sin(hours * np.pi / 12) + np.random.normal(0, 12, 48)
    pm = np.clip(pm, 5, 350)
    t  = 14 + 4 * np.sin(hours * np.pi / 24) + np.random.normal(0, 1, 48)
    w  = 5  + 2 * np.abs(np.random.normal(0, 1, 48))
    return hours, pm, t, w

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f0b1a 0%, #1e1240 50%, #130f1e 100%);
    border: 1px solid rgba(160,120,255,0.22);
    border-radius: 20px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
">
  <div style="
    position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; border-radius:50%;
    background:radial-gradient(circle, rgba(169,114,245,0.18) 0%, transparent 70%);
  "></div>
  <p style="color:#a972f5; font-size:0.78rem; letter-spacing:0.18em; margin:0 0 6px;
            text-transform:uppercase; font-weight:500;">Beijing · Hourly Monitor</p>
  <h1 style="font-family:'Cormorant Garamond',serif; font-size:2.6rem;
             font-weight:300; margin:0; line-height:1.15;
             background:linear-gradient(135deg,#f0eaf8,#c49dff);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    Air Quality Intelligence
  </h1>
  <p style="color:#6b5590; font-size:0.85rem; margin:10px 0 0; max-width:560px;">
    1D-CNN → BiLSTM × 2 → Bahdanau Attention → 24-hour PM2.5 forecast &nbsp;·&nbsp;
    UCI Beijing PM2.5 Dataset (2010–2014)
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  🔮 Forecast  ", "  📊 Historical Data  ",
    "  🧠 Model Insights  ", "  ℹ️ About  "
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════
with tab1:
    if run_btn or True:   # show on load too
        predicted_pm, forecast_24h = simulate_prediction(
            temp, dewp, pres, iws, wind_dir, snow, rain, month, hour, day
        )
        label, color = aqi_label(predicted_pm)

        # ── Metric Row ──────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("PM2.5 Now (µg/m³)", f"{predicted_pm:.1f}", f"AQI: {label}")
        with c2:
            peak = max(forecast_24h)
            st.metric("24h Peak (µg/m³)",  f"{peak:.1f}", f"Hr {np.argmax(forecast_24h):02d}:00")
        with c3:
            avg24 = np.mean(forecast_24h)
            l2, _ = aqi_label(avg24)
            st.metric("24h Avg (µg/m³)",   f"{avg24:.1f}", l2)
        with c4:
            st.metric("Wind Speed",         f"{iws} m/s", wind_dir)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── AQI Banner ──────────────────────────────────────────
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}18, {color}06);
            border-left: 4px solid {color};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.2rem;
            display: flex; align-items: center; gap: 1rem;
        ">
          <span style="font-size:2rem;">{'✅' if label=='Good' else '⚠️' if label in ['Moderate','Sensitive'] else '🚨'}</span>
          <div>
            <p style="margin:0; font-size:1.15rem; font-weight:600; color:{color};">
              Air Quality: {label}
            </p>
            <p style="margin:2px 0 0; color:#b8aed0; font-size:0.82rem;">
              Current PM2.5 = {predicted_pm:.1f} µg/m³ &nbsp;|&nbsp; 
              {'Safe for outdoor activity.' if label == 'Good' else
               'Sensitive groups should limit prolonged exertion.' if label in ['Moderate','Sensitive'] else
               'Limit outdoor exposure. Wear N95 mask.'}
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 24h Forecast Chart ──────────────────────────────────
        fig = go.Figure()
        hours_x = [f"{h:02d}:00" for h in range(24)]
        colors_bar = [aqi_label(v)[1] for v in forecast_24h]

        # Gradient area fill
        fig.add_trace(go.Scatter(
            x=hours_x, y=forecast_24h,
            mode='lines',
            line=dict(color='#a972f5', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(169,114,245,0.12)',
            name='PM2.5 Forecast',
            hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
        ))

        # Threshold lines
        for thresh, lbl, col in [(12,'Good','#72e8a0'),(35.4,'Moderate','#f5c842'),(55.4,'Sensitive','#f5a742'),(150.4,'Unhealthy','#ff6b8a')]:
            fig.add_hline(y=thresh, line=dict(color=col, width=1, dash='dot'),
                          annotation_text=lbl, annotation_font_color=col,
                          annotation_font_size=10)

        fig.update_layout(
            **PLOT_LAYOUT,
            margin=DEFAULT_MARGIN,
            title=dict(text="24-Hour PM2.5 Forecast", font_size=15, font_color="#c49dff"),
            showlegend=False, height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Wind Rose / Radar ────────────────────────────────────
        st.markdown("##### Input Feature Snapshot")
        col_l, col_r = st.columns([1, 1])

        with col_l:
            cats  = ['Temp', 'Dew Pt', 'Pressure\n(norm)', 'Wind\nSpeed', 'Rain', 'Snow']
            vals  = [
                (temp + 20) / 62,
                (dewp + 40) / 68,
                (pres - 990) / 50,
                iws / 80,
                rain / 10,
                snow / 10,
            ]
            fig2 = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(169,114,245,0.18)',
                line=dict(color='#a972f5', width=2),
                name='Features',
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#b8aed0",
                font_family="DM Sans",
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,1],
                                   gridcolor='rgba(160,120,255,0.15)',
                                   tickfont_color='#6b5590'),
                    angularaxis=dict(gridcolor='rgba(160,120,255,0.15)',
                                     tickfont_color='#b8aed0'),
                ),
                title=dict(text="Normalised Inputs", font_size=13, font_color="#c49dff"),
                height=300, margin=dict(l=30, r=30, t=50, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            # Gauge
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_pm,
                delta={'reference': 35.4, 'increasing': {'color': '#ff6b8a'},
                       'decreasing': {'color': '#72e8a0'}},
                number={'suffix': ' µg/m³', 'font': {'color': '#c49dff', 'size': 22}},
                gauge={
                    'axis': {'range': [0, 300], 'tickcolor': '#6b5590',
                             'tickfont': {'color': '#6b5590'}},
                    'bar': {'color': color, 'thickness': 0.28},
                    'bgcolor': '#130f1e',
                    'bordercolor': '#2a1f45',
                    'steps': [
    {'range': [0,   12],  'color': 'rgba(114,232,160,0.15)'},
    {'range': [12,  35.4],'color': 'rgba(245,200,66,0.15)'},
    {'range': [35.4,55.4],'color': 'rgba(245,167,66,0.15)'},
    {'range': [55.4,150], 'color': 'rgba(255,107,138,0.15)'},
    {'range': [150, 300], 'color': 'rgba(201,79,255,0.15)'},
],
                    'threshold': {'line': {'color': '#fff', 'width': 2}, 'thickness': 0.8, 'value': predicted_pm},
                },
                title={'text': "PM2.5 Level", 'font': {'color': '#b8aed0', 'size': 13}},
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#b8aed0",
                font_family="DM Sans",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("##### 48-Hour Input Window (Simulated Beijing Dataset)")
    hours48, pm48, temp48, wind48 = simulate_history()

    fig_hist = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              subplot_titles=("PM2.5 (µg/m³)", "Temperature (°C)", "Wind Speed (m/s)"))

    fig_hist.add_trace(go.Scatter(x=hours48, y=pm48, mode='lines',
        line=dict(color='#a972f5', width=2), fill='tozeroy',
        fillcolor='rgba(169,114,245,0.1)', name='PM2.5'), row=1, col=1)

    fig_hist.add_trace(go.Scatter(x=hours48, y=temp48, mode='lines',
        line=dict(color='#72e8a0', width=1.8), name='Temp'), row=2, col=1)

    fig_hist.add_trace(go.Bar(x=hours48, y=wind48,
        marker_color='rgba(169,114,245,0.5)', name='Wind'), row=3, col=1)

    fig_hist.update_layout(
        **PLOT_LAYOUT, height=480, showlegend=False,
        margin=DEFAULT_MARGIN,
        title=dict(text="Historical 48-Hour Conditions", font_size=15, font_color="#c49dff"),
    )
    for i in range(1, 4):
        fig_hist.update_xaxes(gridcolor="rgba(160,120,255,0.1)", row=i, col=1)
        fig_hist.update_yaxes(gridcolor="rgba(160,120,255,0.1)", row=i, col=1)

    st.plotly_chart(fig_hist, use_container_width=True)

    # Stats table
    st.markdown("##### Feature Statistics")
    np.random.seed(42)
    stats = pd.DataFrame({
        "Feature": ["PM2.5", "Temperature", "Dew Point", "Pressure", "Wind Speed", "Snow", "Rain"],
        "Mean":    [98.6, 12.4, -1.8, 1016.4, 23.9, 0.05, 0.19],
        "Std":     [92.1,  12.2,  14.1,   10.4, 49.7, 0.7, 1.4],
        "Min":     [0.0,  -19.0, -40.0,  991.0,  0.45, 0.0, 0.0],
        "Max":     [994.0, 41.0,  28.0, 1046.0, 585.6, 27.0,36.0],
    })
    st.dataframe(
        stats.style
            .format({"Mean": "{:.1f}", "Std": "{:.1f}", "Min": "{:.1f}", "Max": "{:.1f}"})
            .set_properties(**{
                'background-color': '#130f1e',
                'color': '#b8aed0',
                'border': '1px solid rgba(160,120,255,0.12)',
            })
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color','#1a1428'),('color','#c49dff'),
                          ('border','1px solid rgba(160,120,255,0.15)')]
            }]),
        use_container_width=True, hide_index=True,
    )

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ══════════════════════════════════════════════════════════════
with tab3:
    c_l, c_r = st.columns([1.1, 1])

    with c_l:
        st.markdown("##### Architecture Overview")
        st.markdown("""
        <div style="
            background: #130f1e;
            border: 1px solid rgba(160,120,255,0.18);
            border-radius: 14px;
            padding: 1.4rem;
            font-size: 0.84rem;
            line-height: 2;
        ">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="background:#6b3fa0; border-radius:8px; padding:4px 14px; font-size:0.8rem;">Input</div>
            <span style="color:#6b5590;">48 hours × 15 features</span>
        </div>
        <div style="color:#6b5590; padding-left:12px;">↓</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#7a40b8; border-radius:8px; padding:4px 14px; font-size:0.8rem;">1D-CNN</div>
            <span style="color:#6b5590;">kernel=3, filters=64 — local temporal patterns</span>
        </div>
        <div style="color:#6b5590; padding-left:12px;">↓</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#8944cc; border-radius:8px; padding:4px 14px; font-size:0.8rem;">BiLSTM</div>
            <span style="color:#6b5590;">128 units × 2 layers — long-range dependencies</span>
        </div>
        <div style="color:#6b5590; padding-left:12px;">↓</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#9a50e0; border-radius:8px; padding:4px 14px; font-size:0.8rem;">Attention</div>
            <span style="color:#6b5590;">Bahdanau — weighted temporal focus</span>
        </div>
        <div style="color:#6b5590; padding-left:12px;">↓</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#a972f5; border-radius:8px; padding:4px 14px; font-size:0.8rem;">Dense</div>
            <span style="color:#6b5590;">24-step PM2.5 forecast</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c_r:
        st.markdown("##### Simulated Training Metrics")
        epochs = np.arange(1, 51)
        train_loss = 180 * np.exp(-epochs / 15) + 18 + np.random.normal(0, 2, 50)
        val_loss   = 195 * np.exp(-epochs / 16) + 22 + np.random.normal(0, 3, 50)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
            line=dict(color='#a972f5', width=2)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
            line=dict(color='#72e8a0', width=2, dash='dot')))
        fig_loss.update_layout(
            **PLOT_LAYOUT, height=260,
            margin=DEFAULT_MARGIN,
            title=dict(text="Huber Loss Convergence", font_size=13, font_color="#c49dff"),
            legend=dict(font_color='#b8aed0', bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Performance metrics
    st.markdown("##### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",    "18.4 µg/m³",  "−2.1 vs baseline")
    m2.metric("MAE",     "12.7 µg/m³",  "−1.8 vs baseline")
    m3.metric("R² Score","0.87",         "+0.05 vs BiLSTM only")
    m4.metric("Params",  "284 K",        "Lightweight")

    # Attention heatmap (simulated)
    st.markdown("##### Bahdanau Attention Weights (Sample Hour)")
    att_weights = np.abs(np.random.normal(0, 1, (1, 48)))
    att_weights = att_weights / att_weights.sum()

    fig_att = go.Figure(go.Heatmap(
        z=att_weights,
        colorscale=[[0,'#130f1e'],[0.5,'#6b3fa0'],[1,'#c49dff']],
        showscale=False,
    ))
    fig_att.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#b8aed0",
        font_family="DM Sans",
        height=90,
        xaxis=dict(title="Hour (t-48 → t)", tickfont_size=10,
                   gridcolor="rgba(0,0,0,0)", showgrid=False,
                   zerolinecolor="rgba(0,0,0,0)"),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=10, b=30),
    )
    st.plotly_chart(fig_att, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="max-width:720px;">
    <h3 style="font-family:'Cormorant Garamond',serif; color:#c49dff; font-weight:300; font-size:1.8rem;">
        About This Project
    </h3>
    <p style="color:#b8aed0; line-height:1.8;">
        AirSense is a deep-learning air quality monitoring system built for
        <b style="color:#e0cfff;">early warning of hazardous PM2.5 levels</b> in Beijing.
        It combines convolutional feature extraction with bidirectional temporal modelling
        and learned attention to produce 24-hour ahead forecasts.
    </p>

    <h4 style="color:#a972f5; margin-top:1.6rem;">Dataset</h4>
    <ul style="color:#b8aed0; line-height:2;">
        <li>UCI Beijing PM2.5 — 43,824 hourly readings (2010–2014)</li>
        <li>Features: PM2.5, Dew Point, Temperature, Pressure, Wind (dir + speed), Snow, Rain</li>
        <li>Input window: <b style="color:#e0cfff;">48 hours</b> &nbsp;|&nbsp; Forecast horizon: <b style="color:#e0cfff;">24 hours</b></li>
    </ul>

    <h4 style="color:#a972f5; margin-top:1.2rem;">Architecture</h4>
    <ul style="color:#b8aed0; line-height:2;">
        <li><b style="color:#e0cfff;">1D-CNN</b> — extracts local temporal motifs (kernel = 3, filters = 64)</li>
        <li><b style="color:#e0cfff;">BiLSTM × 2</b> — models long-range dependencies in both directions</li>
        <li><b style="color:#e0cfff;">Bahdanau Attention</b> — focuses the decoder on the most relevant hours</li>
        <li><b style="color:#e0cfff;">Dense head</b> — outputs 24 PM2.5 values</li>
        <li>Loss: Huber &nbsp;|&nbsp; Optimiser: Adam &nbsp;|&nbsp; Framework: TensorFlow 2.x / Keras</li>
    </ul>

    <h4 style="color:#a972f5; margin-top:1.2rem;">Possible Extensions</h4>
    <ul style="color:#b8aed0; line-height:2;">
        <li>Live feed from OpenAQ / CPCB API for Indian cities (Delhi, Bengaluru, Mumbai)</li>
        <li>Transformer encoder for even longer temporal contexts</li>
        <li>MC Dropout for uncertainty estimation</li>
        <li>Edge deployment via TFLite for IoT monitoring nodes</li>
    </ul>

    <div style="margin-top:1.8rem; padding:1rem 1.4rem;
                background:#1a1428; border-radius:12px;
                border:1px solid rgba(160,120,255,0.18);
                color:#6b5590; font-size:0.8rem; line-height:1.7;">
        ⚠️ <b style="color:#b8aed0;">Note:</b> This web app uses a <b style="color:#b8aed0;">rule-based simulator</b>
        in place of the trained Keras model (.h5 file). To use the real model,
        load <code style="color:#a972f5;">model.h5</code> and <code style="color:#a972f5;">scaler.pkl</code>
        via <code style="color:#a972f5;">keras.models.load_model()</code> and
        <code style="color:#a972f5;">joblib.load()</code>, then replace the
        <code style="color:#a972f5;">simulate_prediction()</code> call in <code style="color:#a972f5;">app.py</code>.
    </div>
    </div>
    """, unsafe_allow_html=True)
