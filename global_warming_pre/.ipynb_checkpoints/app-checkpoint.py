"""
🌍 Climate AI Dashboard - Global Warming Prediction System
A SaaS-level ML-powered climate analysis and forecasting application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="ClimateAI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e2e8f0;
}
.stApp { background-color: #0a0f1e; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1428 0%, #0a0f1e 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #38bdf8 !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
h1 { color: #f0f9ff; font-weight: 800; letter-spacing: -0.5px; }
h2 { color: #e2e8f0; font-weight: 700; }
h3 { color: #cbd5e1; font-weight: 600; }

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1f3d 0%, #132040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(14,165,233,0.5);
}

/* ── Selectbox / Sliders ── */
.stSelectbox > div > div,
.stSlider > div > div {
    background-color: #0f1f3d !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

/* ── DataFrames ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Alerts / Info boxes ── */
.stAlert { border-radius: 10px; }

/* ── Divider ── */
hr { border-color: #1e3a5f; }

/* ── Custom card ── */
.custom-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #132040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.badge-rising  { background:#ef4444; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-stable  { background:#22c55e; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-critical{ background:#f97316; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

/* ── Logo strip ── */
.logo-strip {
    display:flex; align-items:center; gap:10px; margin-bottom:28px;
}
.logo-icon {
    font-size:2.2rem; line-height:1;
}
.logo-text {
    font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
    background: linear-gradient(90deg,#38bdf8,#818cf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-strip">
        <span class="logo-icon">🌍</span>
        <span class="logo-text">ClimateAI</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🌍 Global Overview", "📊 EDA", "🤖 Model Training", "🔮 Prediction", "📥 Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.6;'>
    📁 Dataset: Global Warming Trends<br>
    📅 Period: 1961 – 2022<br>
    🤖 Models: LR · RF · XGB · LGBM<br>
    🔄 Last updated: 2024
    </div>
    """, unsafe_allow_html=True)

# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load dataset from /data/dataset.csv or generate synthetic demo data."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # normalise common column name variants
        df.columns = [c.strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ","_")
            if "year" in cl:              col_map[c] = "Year"
            elif "country" in cl or "area" in cl: col_map[c] = "Country"
            elif "change" in cl and "temp" in cl: col_map[c] = "TempChange"
            elif "temp" in cl:             col_map[c] = "AvgTemp"
            elif "co2" in cl or "emission" in cl: col_map[c] = "CO2"
        df.rename(columns=col_map, inplace=True)
        if "AvgTemp" not in df.columns and "TempChange" in df.columns:
            df["AvgTemp"] = df["TempChange"]
        return df
    else:
        return _synthetic_data()

def _synthetic_data():
    """Generate realistic synthetic climate dataset."""
    np.random.seed(42)
    years  = np.arange(1961, 2023)
    countries = [
        "United States","China","India","Germany","Brazil",
        "Australia","Russia","Canada","Japan","France",
        "United Kingdom","South Africa","Nigeria","Indonesia","Mexico",
        "Saudi Arabia","Argentina","South Korea","Italy","Turkey",
    ]
    rows = []
    base_temps = {c: np.random.uniform(5, 28) for c in countries}
    for country in countries:
        base = base_temps[country]
        for i, year in enumerate(years):
            trend    = 0.018 * i + 0.0004 * i**2
            noise    = np.random.normal(0, 0.25)
            avg_temp = base + trend + noise
            change   = trend + noise * 0.5
            co2_base = 250 + 1.8 * i
            co2      = co2_base + np.random.normal(0, 10)
            rows.append({
                "Year":       year,
                "Country":    country,
                "AvgTemp":    round(avg_temp, 3),
                "TempChange": round(change,   3),
                "CO2":        round(co2,      2),
            })
    return pd.DataFrame(rows)

@st.cache_data
def preprocess(df):
    from src.preprocess import preprocess_data
    return preprocess_data(df)

df_raw = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <h1 style='margin-bottom:4px;'>Climate AI Dashboard</h1>
    <p style='color:#64748b; font-size:1.05rem; margin-bottom:32px;'>
    Global Warming Prediction & Analysis System · 1961 – 2022
    </p>
    """, unsafe_allow_html=True)

    # KPI row
    global_avg  = df_raw["AvgTemp"].mean()
    latest_year = df_raw["Year"].max()
    earliest    = df_raw[df_raw["Year"] == df_raw["Year"].min()]["AvgTemp"].mean()
    latest_avg  = df_raw[df_raw["Year"] == latest_year]["AvgTemp"].mean()
    delta_total = latest_avg - earliest
    hottest_year_idx = df_raw.groupby("Year")["AvgTemp"].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡️ Global Avg Temp", f"{global_avg:.2f} °C")
    c2.metric("📈 Total Warming", f"+{delta_total:.2f} °C", f"since {df_raw['Year'].min()}")
    c3.metric("🔥 Hottest Year on Record", str(hottest_year_idx))
    c4.metric("📅 Dataset Coverage", f"{df_raw['Year'].min()} – {latest_year}", f"{df_raw['Country'].nunique()} countries")

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini time-series
    annual = df_raw.groupby("Year")["AvgTemp"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["Year"], y=annual["AvgTemp"],
        mode="lines", name="Avg Temp",
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
    ))
    # trend line
    z = np.polyfit(annual["Year"], annual["AvgTemp"], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=annual["Year"], y=p(annual["Year"]),
        mode="lines", name="Trend",
        line=dict(color="#f97316", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Global Average Temperature Trend (1961–2022)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e2d4a", color="#64748b"),
        yaxis=dict(gridcolor="#1e2d4a", color="#64748b", title="Avg Temp (°C)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div class='custom-card'>
    <h3>🚀 Getting Started</h3>
    <p style='color:#94a3b8; font-size:0.92rem;'>
    Use the sidebar to navigate between sections. Place your dataset at
    <code style='background:#0a0f1e; padding:2px 6px; border-radius:4px;'>data/dataset.csv</code>
    — the app auto-detects columns. If no file is found, a realistic synthetic dataset is used.
    </p>
    <ul style='color:#94a3b8; font-size:0.9rem; line-height:2;'>
    <li>📊 <b>EDA</b> — Explore raw data, distributions, and correlations</li>
    <li>🤖 <b>Model Training</b> — Train & compare LR, RF, XGBoost, LightGBM</li>
    <li>🔮 <b>Prediction</b> — Forecast future temperatures interactively</li>
    <li>📥 <b>Export</b> — Download predictions as CSV</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GLOBAL OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌍 Global Overview":
    st.markdown("<h2>🌍 Global Climate Overview</h2>", unsafe_allow_html=True)

    # Annual warming by decade
    df_raw["Decade"] = (df_raw["Year"] // 10) * 10
    decade_avg = df_raw.groupby(["Decade", "Country"])["AvgTemp"].mean().reset_index()
    annual_mean = df_raw.groupby("Year")["AvgTemp"].mean().reset_index()

    # ── Row 1 ──
    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            annual_mean, x="Year", y="AvgTemp",
            title="Annual Mean Temperature",
            color_discrete_sequence=["#38bdf8"],
        )
        fig.update_traces(line_width=2.5)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0, r=0, t=40, b=0), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        country_trend = (
            df_raw.groupby("Country")["TempChange"].mean()
            .sort_values(ascending=False).head(15).reset_index()
        )
        fig2 = px.bar(
            country_trend, x="TempChange", y="Country",
            orientation="h", title="Top 15 Countries by Avg Temp Change",
            color="TempChange", color_continuous_scale="RdYlBu_r",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ──
    c3, c4 = st.columns(2)

    with c3:
        # CO2 vs TempChange scatter
        if "CO2" in df_raw.columns:
            co2_df = df_raw.dropna(subset=["CO2", "TempChange"])
            fig3 = px.scatter(
                co2_df, x="CO2", y="TempChange",
                color="Year", title="CO₂ vs Temperature Change",
                color_continuous_scale="Plasma",
                opacity=0.6,
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
                margin=dict(l=0, r=0, t=40, b=0), height=300,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Decade box-plot
        fig4 = px.box(
            df_raw, x="Decade", y="AvgTemp",
            title="Temperature Distribution by Decade",
            color="Decade", color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), showlegend=False,
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0, r=0, t=40, b=0), height=300,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Country selector ──
    st.markdown("### Country Deep-Dive")
    selected_countries = st.multiselect(
        "Select countries to compare",
        options=sorted(df_raw["Country"].unique()),
        default=list(df_raw["Country"].unique())[:5],
    )
    if selected_countries:
        filtered = df_raw[df_raw["Country"].isin(selected_countries)]
        country_annual = filtered.groupby(["Year", "Country"])["AvgTemp"].mean().reset_index()
        fig5 = px.line(
            country_annual, x="Year", y="AvgTemp", color="Country",
            title="Country-Level Temperature Trends",
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=40, b=0), height=350,
        )
        st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown("<h2>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Distributions", "🔗 Correlations"])

    with tab1:
        st.markdown(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(100), use_container_width=True, height=360)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(df_raw.describe().round(3), use_container_width=True)
        with c2:
            st.markdown("**Missing Values**")
            miss = df_raw.isnull().sum().rename("Missing").to_frame()
            miss["Pct"] = (miss["Missing"] / len(df_raw) * 100).round(2)
            st.dataframe(miss, use_container_width=True)

    with tab2:
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        sel = st.selectbox("Feature", num_cols)
        fig_h = px.histogram(
            df_raw, x=sel, nbins=50,
            title=f"Distribution of {sel}",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=40,b=0), height=300,
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Rolling average
        annual2 = df_raw.groupby("Year", as_index=False)[sel].mean()
        annual2["Rolling5"] = annual2[sel].rolling(5, center=True).mean()
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=annual2["Year"], y=annual2[sel],
            mode="lines", name=sel, line=dict(color="#64748b", width=1.5)))
        fig_r.add_trace(go.Scatter(x=annual2["Year"], y=annual2["Rolling5"],
            mode="lines", name="5-yr Rolling Avg", line=dict(color="#f97316", width=2.5)))
        fig_r.update_layout(
            title=f"{sel} — Annual Mean + 5-Year Rolling Average",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=40,b=0), height=300,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with tab3:
        num_df = df_raw[num_cols].dropna()
        corr = num_df.corr()
        fig_c = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
            zmin=-1, zmax=1,
        )
        fig_c.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            margin=dict(l=0,r=0,t=40,b=0), height=420,
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.markdown("<h2>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)

    if st.button("🚀 Train All Models"):
        with st.spinner("Preprocessing data and training models…"):
            from src.preprocess import preprocess_data
            from src.train import train_models

            X_train, X_test, y_train, y_test, feature_names, scaler, le = preprocess_data(df_raw)
            results, best_name, best_model, fi_df = train_models(
                X_train, X_test, y_train, y_test, feature_names
            )

        st.success(f"✅ Training complete! Best model: **{best_name}**")

        # ── Leaderboard ──
        results_df = pd.DataFrame(results).T.rename_axis("Model").reset_index()
        results_df = results_df.sort_values("RMSE")
        st.markdown("### 🏆 Model Leaderboard")
        st.dataframe(results_df.style.highlight_min(subset=["RMSE","MAE"], color="#0d2e4a")
                                     .highlight_max(subset=["R2"], color="#0d2e4a")
                                     .format({"RMSE":"{:.4f}","MAE":"{:.4f}","R2":"{:.4f}"}),
                     use_container_width=True)

        # ── Bar chart ──
        fig_lb = go.Figure()
        for metric, color in [("RMSE","#ef4444"),("MAE","#f97316"),("R2","#22c55e")]:
            fig_lb.add_trace(go.Bar(
                name=metric, x=results_df["Model"], y=results_df[metric],
                marker_color=color, opacity=0.85,
            ))
        fig_lb.update_layout(
            barmode="group", title="Model Comparison",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0,r=0,t=40,b=0), height=320,
        )
        st.plotly_chart(fig_lb, use_container_width=True)

        # ── Feature Importance ──
        if fi_df is not None:
            st.markdown("### 📌 Feature Importance (Best Model)")
            fig_fi = px.bar(
                fi_df.head(15), x="Importance", y="Feature",
                orientation="h", color="Importance",
                color_continuous_scale="Blues",
                title=f"Top Features — {best_name}",
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), coloraxis_showscale=False,
                xaxis=dict(gridcolor="#1e2d4a"), yaxis=dict(gridcolor="#1e2d4a"),
                margin=dict(l=0,r=0,t=40,b=0), height=380,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # store in session
        st.session_state["model_trained"] = True
        st.session_state["best_model_name"] = best_name
        st.session_state["scaler"] = scaler
        st.session_state["le"] = le
        st.session_state["feature_names"] = feature_names
    else:
        st.info("Click **Train All Models** to start training. This may take ~30 seconds.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Prediction":
    st.markdown("<h2>🔮 Climate Prediction Engine</h2>", unsafe_allow_html=True)

    from src.predict import load_best_model, predict_temperature

    model, scaler, le, feature_names = load_best_model()

    if model is None:
        st.warning("⚠️ No trained model found. Please go to **🤖 Model Training** and train the models first.")
    else:
        st.markdown(f"Model loaded: **{type(model).__name__}**")

        countries = sorted(df_raw["Country"].unique())
        c1, c2, c3 = st.columns(3)
        with c1:
            year_input = st.slider("📅 Year", 1961, 2050, 2030)
        with c2:
            country_input = st.selectbox("🌍 Country / Region", countries)
        with c3:
            co2_input = st.number_input("🏭 CO₂ Emission (ppm)", min_value=200.0, max_value=600.0, value=420.0, step=1.0)

        prev_temp = st.slider("🌡️ Previous Year Avg Temp (°C)", -5.0, 40.0, 14.0, 0.1)

        if st.button("🔮 Predict Temperature"):
            pred, confidence = predict_temperature(
                model, scaler, le, feature_names,
                year=year_input, country=country_input,
                co2=co2_input, prev_temp=prev_temp,
            )

            # Trend classification
            delta = pred - prev_temp
            if delta > 1.5:
                trend_label = "<span class='badge-critical'>⚠️ CRITICAL</span>"
                trend_text  = "Rapid temperature increase detected."
                advice      = "🚨 Urgent climate mitigation measures are critical!"
            elif delta > 0.3:
                trend_label = "<span class='badge-rising'>📈 RISING</span>"
                trend_text  = "Continued warming trend observed."
                advice      = "⚠️ Sustained emission reduction required."
            else:
                trend_label = "<span class='badge-stable'>📉 STABLE</span>"
                trend_text  = "Temperature within stable range."
                advice      = "✅ Continue current environmental policies."

            m1, m2, m3 = st.columns(3)
            m1.metric("🌡️ Predicted Temperature", f"{pred:.2f} °C")
            m2.metric("📊 Confidence Score", f"{confidence:.1f}%")
            m3.metric("📈 Change from Prev Year", f"{delta:+.2f} °C")

            st.markdown(f"""
            <div class='custom-card' style='margin-top:16px;'>
            <h3>Prediction Summary — {country_input} · {year_input}</h3>
            <p>Climate Trend: {trend_label}</p>
            <p style='color:#94a3b8;'>{trend_text}</p>
            <p style='color:#f59e0b;'>{advice}</p>
            </div>
            """, unsafe_allow_html=True)

            # Forecast chart: next 20 years
            st.markdown("### 📈 20-Year Temperature Forecast")
            future_years = list(range(year_input, year_input + 21))
            future_preds = []
            pt = prev_temp
            for fy in future_years:
                fp, _ = predict_temperature(model, scaler, le, feature_names,
                                            year=fy, country=country_input,
                                            co2=co2_input + (fy - year_input)*1.5,
                                            prev_temp=pt)
                future_preds.append(fp)
                pt = fp

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=future_years, y=future_preds,
                mode="lines+markers", name="Forecast",
                line=dict(color="#f97316", width=2.5),
                marker=dict(size=6, color="#f97316"),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
            ))
            fig_f.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#1e2d4a", color="#64748b"),
                yaxis=dict(gridcolor="#1e2d4a", color="#64748b", title="Predicted Temp (°C)"),
                margin=dict(l=0,r=0,t=20,b=0), height=300,
            )
            st.plotly_chart(fig_f, use_container_width=True)

            # store for export
            forecast_df = pd.DataFrame({"Year": future_years, "Predicted_Temp": future_preds,
                                         "Country": country_input, "CO2_ppm": [
                                             co2_input + (y - year_input)*1.5 for y in future_years]})
            st.session_state["forecast_df"] = forecast_df

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPORT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📥 Export":
    st.markdown("<h2>📥 Export Predictions</h2>", unsafe_allow_html=True)

    if "forecast_df" in st.session_state:
        df_export = st.session_state["forecast_df"]
        st.dataframe(df_export, use_container_width=True)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv_data,
            file_name="climate_forecast.csv",
            mime="text/csv",
        )
    else:
        st.info("No forecast data yet. Run a prediction in **🔮 Prediction** first.")

    st.markdown("---")
    st.markdown("### 📦 Export Raw Dataset")
    csv_raw = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Raw Dataset CSV", data=csv_raw,
                       file_name="climate_raw_data.csv", mime="text/csv")
