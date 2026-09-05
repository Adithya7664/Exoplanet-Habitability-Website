import os
import base64
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from scipy.stats import gaussian_kde

from src.features import (
    FEATURE_COLS, FEATURE_LABELS, compute_esi_components,
    EARTH_EQT_K, EARTH_RADIUS_RE, EARTH_MASS_ME, EARTH_INSOL
)
from src.model import load_model, shap_for_planet
from src.renderer import render_planet_3d
from src.terraformer import generate_terraforming_plan

load_dotenv()

st.set_page_config(
    page_title="PlanetScope",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def _bg_css() -> str:
    for name, mime in [("background.jpg", "jpeg"), ("background.png", "png"), ("background.webp", "webp")]:
        p = Path(name)
        if p.exists():
            data = base64.b64encode(p.read_bytes()).decode()
            return f"url('data:image/{mime};base64,{data}') center/cover fixed no-repeat"
    return "radial-gradient(ellipse at bottom, #0d1b2e 0%, #050a0f 100%)"

@st.cache_data
def _build_css():
    bg = _bg_css()
    return f"""
    <style>
    /* Hide sidebar toggle and sidebar */
    [data-testid="stSidebar"] {{ display: none; }}
    [data-testid="collapsedControl"] {{ display: none; }}
    
    /* Hide Streamlit default toolbar background */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    header[data-testid="stHeader"]::before {{
        background: transparent !important;
    }}

    /* Background */
    .stApp {{
        background: {bg};
    }}

    /* Semi-transparent content layer */
    .block-container {{
        background: transparent;
        border-radius: 12px;
        padding-top: 1rem !important;
        margin-top: 0.5rem;
    }}

    /* Top header bar */
    .ps-header {{
        display: flex;
        align-items: center;
        padding: 12px 24px;
        background: rgba(0, 0, 0, 0.03);
        border-bottom: 1px solid rgba(80, 140, 255, 0.25);
        margin-bottom: 0.5rem;
        transform: translateZ(0);
        will-change: transform;
    }}
    .ps-logo {{
        font-size: 1.6rem;
        margin-right: 10px;
    }}
    .ps-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: #a0c4ff;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .ps-subtitle {{
        font-size: 0.7rem;
        color: #5577aa;
        margin-left: 6px;
        margin-top: 4px;
    }}

    /* Style tabs as nav bar */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid rgba(80, 140, 255, 0.2);
        padding: 4px 16px 0 16px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #7799cc;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(80, 140, 255, 0.15) !important;
        color: #a0c4ff !important;
        border-bottom: 2px solid #4488ff !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 1.5rem;
    }}
    .stTabs [data-baseweb="tab-list"]::after {{
        content: "✦  ✧  ✦  ✧  ✦";
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        color: rgba(120, 170, 255, 0.75);
        font-size: 1.2rem;
        letter-spacing: 3px;
        pointer-events: none;
    }}
    

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(80, 140, 255, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] p {{
        color: rgba(160, 190, 255, 0.85) !important;
        font-size: 0.8rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #ffffff !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(80, 140, 255, 0.15);
        border-radius: 8px;
    }}
    
    /* Animations */
    @keyframes ps-pulse {{
        0%, 100% {{ opacity: 0.5; transform: scale(1); }}
        50% {{ opacity: 1; transform: scale(1.3); }}
    }}
    @keyframes ps-twinkle {{
        0%, 100% {{ opacity: 0.15; }}
        50% {{ opacity: 0.75; }}
    }}
    @keyframes ps-bar {{
        0%, 100% {{ transform: scaleY(0.35); opacity: 0.5; }}
        50% {{ transform: scaleY(1.0); opacity: 1.0; }}
    }}
    .ps-star-1 {{ animation: ps-twinkle 2.1s ease-in-out infinite; }}
    .ps-star-2 {{ animation: ps-twinkle 3.3s ease-in-out infinite 0.7s; }}
    .ps-star-3 {{ animation: ps-twinkle 2.7s ease-in-out infinite 1.4s; }}
    
    /* Plotly toolbar visibility on dark bg */
    .modebar {{
        background: rgba(10, 20, 40, 0.6) !important;
        border-radius: 4px;
    }}
    .modebar-btn path {{
        fill: rgba(140, 180, 255, 0.7) !important;
    }}
    .modebar-btn:hover path {{
        fill: rgba(200, 220, 255, 1.0) !important;
    }}
    
    /* Shooting stars */
    @keyframes ps-shoot {{
        0%   {{ transform: rotate(-35deg) translateX(-250px); opacity: 0; }}
        8%   {{ opacity: 1; }}
        28%  {{ opacity: 0; }}
        100% {{ transform: rotate(-35deg) translateX(2000px); opacity: 0; }}
    }}
    .ps-shooting-star {{
        position: fixed;
        height: 1.5px;
        background: linear-gradient(90deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.3) 60%, transparent 100%);
        border-radius: 2px;
        pointer-events: none;
        z-index: 0;
        animation: ps-shoot linear infinite;
        will-change: transform, opacity;
    }}

    /* Bottom twinkling dots */
    @keyframes ps-twinkle-dot {{
        0%, 100% {{ opacity: 0.15; transform: scale(0.9); }}
        50% {{ opacity: 1; transform: scale(1.4); }}
    }}
    .ps-tdot {{
        position: fixed;
        border-radius: 50%;
        background: white;
        pointer-events: none;
        z-index: 0;
        animation: ps-twinkle-dot ease-in-out infinite;
        will-change: transform, opacity;
    }}
    </style>
    <div class="ps-shooting-star" style="width:160px;top:8%;left:15%;animation-duration:5s;animation-delay:0s;"></div>
    <div class="ps-shooting-star" style="width:120px;top:22%;left:55%;animation-duration:6s;animation-delay:4s;"></div>
    <div class="ps-shooting-star" style="width:200px;top:5%;left:70%;animation-duration:7s;animation-delay:9s;"></div>
    <div class="ps-shooting-star" style="width:140px;top:40%;left:8%;animation-duration:5.5s;animation-delay:14s;"></div>
    <div class="ps-shooting-star" style="width:180px;top:12%;left:38%;animation-duration:6.5s;animation-delay:2s;"></div>
    <div class="ps-shooting-star" style="width:130px;top:50%;left:62%;animation-duration:5s;animation-delay:19s;"></div>
    <div class="ps-shooting-star" style="width:150px;top:3%;left:25%;animation-duration:5.5s;animation-delay:7s;"></div>
    <div class="ps-shooting-star" style="width:190px;top:18%;left:80%;animation-duration:6s;animation-delay:11s;"></div>
    <div class="ps-shooting-star" style="width:110px;top:33%;left:45%;animation-duration:4.5s;animation-delay:16s;"></div>
    <div class="ps-shooting-star" style="width:170px;top:6%;left:5%;animation-duration:7s;animation-delay:22s;"></div>
    <div class="ps-shooting-star" style="width:135px;top:55%;left:30%;animation-duration:5s;animation-delay:25s;"></div>
    <div class="ps-shooting-star" style="width:200px;top:10%;left:90%;animation-duration:6.5s;animation-delay:3s;"></div>
    <div class="ps-shooting-star" style="width:125px;top:42%;left:72%;animation-duration:5s;animation-delay:28s;"></div>
    <div class="ps-shooting-star" style="width:155px;top:28%;left:18%;animation-duration:7.5s;animation-delay:33s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:72%;left:8%;animation-duration:2.1s;animation-delay:0s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:78%;left:15%;animation-duration:3.4s;animation-delay:0.8s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:65%;left:22%;animation-duration:2.8s;animation-delay:1.5s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:83%;left:30%;animation-duration:1.9s;animation-delay:0.3s;"></div>
    <div class="ps-tdot" style="width:3px;height:3px;top:70%;left:38%;animation-duration:3.1s;animation-delay:2.1s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:88%;left:45%;animation-duration:2.5s;animation-delay:0.6s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:75%;left:52%;animation-duration:3.7s;animation-delay:1.2s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:62%;left:58%;animation-duration:2.2s;animation-delay:2.8s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:80%;left:65%;animation-duration:2.9s;animation-delay:0.4s;"></div>
    <div class="ps-tdot" style="width:3px;height:3px;top:68%;left:72%;animation-duration:3.3s;animation-delay:1.9s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:85%;left:78%;animation-duration:2.0s;animation-delay:3.2s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:73%;left:85%;animation-duration:3.6s;animation-delay:0.9s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:90%;left:91%;animation-duration:2.4s;animation-delay:1.7s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:66%;left:95%;animation-duration:3.0s;animation-delay:2.5s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:82%;left:3%;animation-duration:2.7s;animation-delay:1.1s;"></div>
    <div class="ps-tdot" style="width:3px;height:3px;top:77%;left:48%;animation-duration:3.8s;animation-delay:0.2s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:93%;left:20%;animation-duration:2.3s;animation-delay:3.5s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:87%;left:60%;animation-duration:2.6s;animation-delay:4.0s;"></div>
    <div class="ps-tdot" style="width:1px;height:1px;top:95%;left:75%;animation-duration:3.2s;animation-delay:2.2s;"></div>
    <div class="ps-tdot" style="width:2px;height:2px;top:71%;left:33%;animation-duration:2.8s;animation-delay:1.4s;"></div>
    """


def inject_css():
    st.markdown(_build_css(), unsafe_allow_html=True)


@st.cache_data
def _build_header_html() -> str:
    bar = lambda h, c, d: (
        f'<div style="width:5px;height:{h}px;background:{c};border-radius:2px 2px 0 0;'
        f'transform-origin:bottom;will-change:transform,opacity;'
        f'animation:ps-bar 1.4s ease-in-out infinite {d}s"></div>'
    )
    bars = "".join([
        bar(10,  "rgba(80,160,255,0.85)",  0.0),
        bar(28,  "rgba(130,200,255,0.95)", 0.2),
        bar(18,  "rgba(100,175,255,0.9)",  0.1),
        bar(32,  "rgba(150,215,255,0.95)", 0.4),
        bar(14,  "rgba(90,165,255,0.85)",  0.6),
        bar(26,  "rgba(140,205,255,0.95)", 0.8),
        bar(20,  "rgba(110,185,255,0.9)",  1.0),
        bar(22,  "rgba(120,195,255,0.9)",  1.3),
    ])
    return f"""
    <div class="ps-header">
        <span class="ps-logo">🪐</span>
        <span class="ps-title">PlanetScope</span>
        <span class="ps-subtitle">NASA Exoplanet Archive · Habitability AI</span>
        <div style="flex:1"></div>
        <div style="display:flex;align-items:flex-end;gap:4px;height:36px;padding-right:16px">{bars}</div>
    </div>
    """

def render_header():
    st.markdown(_build_header_html(), unsafe_allow_html=True)

DATA_PATH = Path("data/exoplanets_engineered.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Run `python train.py` first.")
        st.stop()
    return pd.read_csv(DATA_PATH, low_memory=False)


@st.cache_resource
def load_trained_model():
    return load_model()


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Set ANTHROPIC_API_KEY in a .env file in the project root.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

def radar_chart(components: dict, title: str = "") -> go.Figure:
    categories = list(components.keys()) + [list(components.keys())[0]]
    values = list(components.values()) + [list(components.values())[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(categories), theta=categories,
        fill="toself", name="Earth",
        line_color="rgba(100,200,100,0.8)",
        fillcolor="rgba(100,200,100,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself", name=title or "Planet",
        line_color="rgba(100,150,255,0.9)",
        fillcolor="rgba(100,150,255,0.2)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=380,
        margin=dict(t=40, b=20)
    )
    return fig

def shap_waterfall_chart(shap_vals: np.ndarray, expected: float, feature_names: list) -> go.Figure:
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    indices = np.argsort(np.abs(shap_vals))
    sorted_labels = [labels[i] for i in indices]
    sorted_vals = shap_vals[indices]
    colors = ["#e84444" if v < 0 else "#44aa66" for v in sorted_vals]
    fig = go.Figure(go.Bar(
        x=sorted_vals, y=sorted_labels,
        orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in sorted_vals],
        textposition="outside"
    ))
    fig.update_layout(
        title=f"SHAP Feature Contributions (baseline: {expected:.3f})",
        xaxis_title="Impact on ESI Score",
        height=420, margin=dict(l=20, r=60, t=50, b=20)
    )
    return fig

@st.cache_data
def _build_esi_chart(esi_tuple: tuple) -> go.Figure:
    esi_vals = np.array(esi_tuple)
    kde = gaussian_kde(esi_vals, bw_method=0.08)
    n_seg = 40
    x_pts = np.linspace(0, 1, n_seg + 1)
    y_pts = kde(x_pts)

    def esi_color(v):
        v = max(0.0, min(1.0, v))
        if v <= 0.5:
            t = v * 2
            return 210, int(60 + 140 * t), 50
        else:
            t = (v - 0.5) * 2
            return int(210 - 160 * t), 200, int(50 + 20 * t)

    fig = go.Figure()
    for i in range(n_seg):
        x_mid = (x_pts[i] + x_pts[i + 1]) / 2
        r, g, b = esi_color(x_mid)
        for width, alpha in [(14, 0.07), (6, 0.22), (2, 0.75), (1, 1.0)]:
            fig.add_trace(go.Scatter(
                x=[x_pts[i], x_pts[i + 1]], y=[y_pts[i], y_pts[i + 1]],
                mode="lines",
                line=dict(color=f"rgba({r},{g},{b},{alpha})", width=width),
                showlegend=False, hoverinfo="skip"
            ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="rgba(100,220,100,0.7)",
                annotation_text="Earth", annotation_font_color="rgba(100,220,100,0.9)")
    fig.update_layout(
        height=320, margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(title="Earth Similarity Index", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False
    )
    return fig

@st.fragment
def page_home(df: pd.DataFrame):
    title_col, m1, m2, m3, m4 = st.columns([2.5, 1, 1, 1, 1])
    with title_col:
        st.title("🌍 Planetary Habitability Explorer")
        st.caption("NASA Exoplanet Archive · ESI Ensemble Model · Terragenesis AI")
    m1.metric("Total Planets", f"{len(df):,}")
    m2.metric("In Habitable Zone", f"{int(df['in_hz'].sum()):,}")
    m3.metric("ESI > 0.5", f"{(df['esi'] > 0.5).sum():,}")
    m4.metric("Highest ESI", f"{df['esi'].max():.3f}")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Top 10 Most Earth-like Planets")
        top10 = df.nlargest(10, "esi")[
            ["pl_name", "hostname", "esi", "in_hz", "sy_dist", "pl_orbper", "disc_year"]
        ].round(3)
        for _, row in top10.iterrows():
            dist_ly = f"{row['sy_dist']*3.26:.0f} ly" if pd.notna(row.get("sy_dist")) else "unknown"
            period = f"{row['pl_orbper']:.0f} days" if pd.notna(row.get("pl_orbper")) else "unknown"
            year = f"{int(row['disc_year'])}" if pd.notna(row.get("disc_year")) else "unknown"
            st.markdown(
                f"• <span style='color:#7ec8ff;font-weight:600'>{row['pl_name']}</span> "
                f"<span style='color:#aaaaaa'>· {row['hostname']}</span> "
                f"&nbsp;|&nbsp; ESI: <span style='color:#44ee88'>{row['esi']:.3f}</span> "
                f"&nbsp;|&nbsp; {dist_ly} "
                f"&nbsp;|&nbsp; {period} "
                f"&nbsp;|&nbsp; {year}",
                unsafe_allow_html=True
            )

    with col2:
        st.subheader("ESI Distribution")
        esi_tuple = tuple(df["esi"].dropna().values.tolist())
        fig = _build_esi_chart(esi_tuple)
        st.plotly_chart(fig, width='stretch')

    col_orb, col_sky = st.columns(2)
    with col_orb:
        st.subheader("Orbital Map — Period vs Radius")
        st.caption("Angle = orbital period (log scale)  ·  Distance from center = planet radius  ·  Color = stellar temperature")

        ring_df = df[
            df["pl_orbper"].notna() & df["pl_rade"].notna() & df["st_teff"].notna()
        ].copy()

        log_per = np.log10(ring_df["pl_orbper"].clip(0.5, 50000))
        per_min, per_max = log_per.min(), log_per.max()
        ring_df["theta"] = (log_per - per_min) / (per_max - per_min) * 360
        ring_df["r_plot"] = ring_df["pl_rade"].clip(0.1, 20)
        earth_theta = float((np.log10(365) - per_min) / (per_max - per_min) * 360)

        tick_periods = [1, 10, 100, 365, 1000, 10000]
        tick_thetas = [(np.log10(p) - per_min) / (per_max - per_min) * 360 for p in tick_periods]
        tick_labels = ["1d", "10d", "100d", "1yr", "3yr", "27yr"]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=ring_df["r_plot"],
            theta=ring_df["theta"],
            mode="markers",
            marker=dict(
                size=4,
                color=ring_df["st_teff"],
                colorscale="RdYlBu",
                cmin=2500, cmax=8500,
                colorbar=dict(
                    title=dict(text="Star Temp (K)", font=dict(color="white", size=11)),
                    tickfont=dict(color="white", size=10),
                    thickness=12, len=0.7
                ),
                opacity=0.65,
                line=dict(width=0),
            ),
            text=ring_df["pl_name"],
            customdata=ring_df[["pl_orbper", "esi"]].values,
            hovertemplate="<b>%{text}</b><br>Period: %{customdata[0]:.1f} days<br>Radius: %{r:.2f} R⊕<br>ESI: %{customdata[1]:.3f}<extra></extra>",
            showlegend=False,
        ))
        fig2.add_trace(go.Scatterpolar(
            r=[1.0], theta=[earth_theta],
            mode="markers+text",
            marker=dict(size=10, color="#44ff88", symbol="star", line=dict(width=1, color="white")),
            text=["Earth"], textposition="top center",
            textfont=dict(color="#44ff88", size=10),
            showlegend=False, hoverinfo="skip",
        ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text="Radius (R⊕)", font=dict(color="rgba(200,220,255,0.9)", size=11)),
                    range=[0, 21],
                    gridcolor="rgba(255,255,255,0.2)",
                    linecolor="rgba(255,255,255,0.25)",
                    tickfont=dict(color="rgba(210,230,255,0.9)", size=10),
                    tickcolor="rgba(255,255,255,0.4)",
                    showgrid=True,
                    showline=True,
                ),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=tick_thetas,
                    ticktext=tick_labels,
                    gridcolor="rgba(255,255,255,0.15)",
                    linecolor="rgba(255,255,255,0.2)",
                    tickfont=dict(color="rgba(210,230,255,0.9)", size=11),
                    direction="clockwise",
                    showgrid=True,
                    showline=True,
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=40, b=20, l=60, r=80),
            showlegend=False,
        )
        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": True, "doubleClick": "reset+autosize"})
        
    with col_sky:
        st.subheader("Nearest Earth-like Planets")
        st.caption("Count of ESI > 0.5 planets by distance from Earth")

        dist_df = df[df["sy_dist"].notna() & (df["esi"] > 0.5)].copy()
        dist_df["dist_ly"] = dist_df["sy_dist"] * 3.2616

        bands = [
            ("> 1k ly",    1000, 99999),
            ("500–1k ly",  500,  1000),
            ("100–500 ly", 100,  500),
            ("50–100 ly",  50,   100),
            ("10–50 ly",   10,   50),
            ("< 10 ly",    0,    10),
        ]
        labels = [b[0] for b in bands]
        counts = [int(((dist_df["dist_ly"] >= b[1]) & (dist_df["dist_ly"] < b[2])).sum()) for b in bands]

        core_colors = [
            "rgba(16,72,30,0.30)",
            "rgba(22,105,42,0.42)",
            "rgba(28,138,53,0.55)",
            "rgba(35,175,67,0.68)",
            "rgba(42,215,82,0.84)",
            "rgba(50,255,100,1.0)",
        ]
        glow_colors = [c.replace(c.split(",")[-1], " 0.08)") for c in core_colors]

        fig_dist = go.Figure()

        # glow layer
        fig_dist.add_trace(go.Bar(
            y=labels, x=counts, orientation="h",
            marker=dict(color=glow_colors, line=dict(width=0)),
            width=0.75, showlegend=False, hoverinfo="skip",
        ))
        # core bars
        fig_dist.add_trace(go.Bar(
            y=labels, x=counts, orientation="h",
            marker=dict(color=core_colors, line=dict(color="rgba(255,255,255,0.08)", width=0.5)),
            width=0.38,
            text=[str(c) for c in counts],
            textposition="outside",
            textfont=dict(color="rgba(180,255,210,0.9)", size=11),
            showlegend=False,
            hovertemplate="%{y}: %{x} planets<extra></extra>",
        ))

        fig_dist.update_layout(
            height=480, barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(
                title="Earth-like Planets (ESI > 0.5)",
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(color="rgba(180,220,200,0.8)", size=10),
                zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(color="rgba(180,255,210,0.9)", size=11),
                gridcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=20, b=50, l=110, r=70),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, width='stretch', config={"displayModeBar": False})

@st.fragment    
def page_explorer(df: pd.DataFrame):
    st.title("🔭 Planet Explorer")

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            esi_range = st.slider("ESI Range", 0.0, 1.0, (0.0, 1.0), 0.01)
            hz_only = st.checkbox("Habitable Zone only")
        with c2:
            max_dist = st.slider("Max Distance (pc)", 1, 5000, 5000, 10)
        with c3:
            min_temp = st.slider("Min Stellar Temp (K)", 2000, 10000, 2000, 100)
            max_temp = st.slider("Max Stellar Temp (K)", 2000, 10000, 10000, 100)

    mask = (
        (df["esi"] >= esi_range[0]) & (df["esi"] <= esi_range[1]) &
        (df["sy_dist"].fillna(0) <= max_dist) &
        (df["st_teff"].fillna(5778) >= min_temp) &
        (df["st_teff"].fillna(5778) <= max_temp)
    )
    if hz_only:
        mask = mask & (df["in_hz"] == 1.0)

    filtered = df[mask].copy()
    st.caption(f"{len(filtered):,} planets match your filters")
    
    col_density, col_funnel = st.columns(2)
    with col_density:
        st.subheader("3D Habitability Density Map")
        st.caption("Planet density around Earth's position · Earth at center · Rings = equal distance from Earth")

        density_df = filtered[["pl_eqt", "pl_rade"]].dropna().copy()
        earth_temp, earth_rad = 255.0, 1.0
        std_temp = max(density_df["pl_eqt"].std(), 1.0)
        std_rad  = max(density_df["pl_rade"].std(), 0.01)
        density_df["xn"] = (density_df["pl_eqt"] - earth_temp) / std_temp
        density_df["yn"] = (density_df["pl_rade"] - earth_rad)  / std_rad

        rng = 3
        H, xe, ye = np.histogram2d(
            density_df["xn"].clip(-rng, rng),
            density_df["yn"].clip(-rng, rng),
            bins=40, range=[[-rng, rng], [-rng, rng]]
        )
        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2
        X, Y = np.meshgrid(xc, yc)

        fig3d = go.Figure()

        fig3d.add_trace(go.Surface(
            x=X, y=Y, z=H.T,
            colorscale="Turbo",
            opacity=0.93,
            colorbar=dict(
                title=dict(text="Planet Count", font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=10),
                thickness=12, len=0.6,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="rgba(255,255,255,0.6)", project_z=True)
            ),
        ))

        # Earth marker
        fig3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[float(H.max()) * 0.04],
            mode="markers+text",
            marker=dict(size=9, color="#44ff88", symbol="diamond",
                        line=dict(width=1, color="white")),
            text=["Earth"], textposition="top center",
            textfont=dict(color="#44ff88", size=11),
            showlegend=False, hoverinfo="skip",
        ))

        # Concentric rings on floor
        theta = np.linspace(0, 2 * np.pi, 150)
        ring_specs = [
            (0.5, "rgba(80,255,140,0.6)"),
            (1.0, "rgba(80,220,255,0.5)"),
            (1.5, "rgba(180,80,255,0.45)"),
            (2.0, "rgba(255,140,60,0.40)"),
            (2.8, "rgba(255,60,60,0.35)"),
        ]
        for r, clr in ring_specs:
            fig3d.add_trace(go.Scatter3d(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                z=np.zeros(150),
                mode="lines",
                line=dict(color=clr, width=2),
                showlegend=False, hoverinfo="skip",
            ))

        fig3d.update_layout(
            height=580,
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(title="Δ Temperature (σ from Earth)",
                            gridcolor="rgba(255,255,255,0.18)",
                            backgroundcolor="rgba(0,0,10,0.15)",
                            color="white", nticks=8, range=[-rng, rng]),
                yaxis=dict(title="Δ Radius (σ from Earth)",
                            gridcolor="rgba(255,255,255,0.18)",
                            backgroundcolor="rgba(0,0,10,0.15)",
                            color="white", nticks=8, range=[-rng, rng]),
                zaxis=dict(title="Planet Count",
                            gridcolor="rgba(255,255,255,0.18)",
                            backgroundcolor="rgba(0,0,10,0.15)",
                            color="white", nticks=8),
                bgcolor="rgba(0,0,10,0.15)",
                aspectratio=dict(x=1, y=1, z=0.6),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
            ),
            margin=dict(t=20, b=20, l=0, r=0),
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig3d, width='stretch', config={"displayModeBar": True, "doubleClick": "reset+autosize"})
        
    with col_funnel:
        st.subheader("Habitability Funnel")
        st.caption("How many planets survive each filter step")
        
        total = len(df)
        after_dist = int((df["sy_dist"].fillna(0) <= max_dist).sum())
        after_temp = int(
            ((df["sy_dist"].fillna(0) <= max_dist) &
            (df["st_teff"].fillna(5778) >= min_temp) &
            (df["st_teff"].fillna(5778) <= max_temp)).sum()
        )
        after_esi = int(
            ((df["sy_dist"].fillna(0) <= max_dist) &
            (df["st_teff"].fillna(5778) >= min_temp) &
            (df["st_teff"].fillna(5778) <= max_temp) &
            (df["esi"] >= esi_range[0]) & (df["esi"] <= esi_range[1])).sum()
        )
        after_hz = int(
            ((df["sy_dist"].fillna(0) <= max_dist) &
            (df["st_teff"].fillna(5778) >= min_temp) &
            (df["st_teff"].fillna(5778) <= max_temp) &
            (df["esi"] >= esi_range[0]) & (df["esi"] <= esi_range[1]) &
            (df["in_hz"] == 1.0 if hz_only else True)).sum()
        )
        
        funnel_labels = [
            "All Planets",
            "Distance Filter",
            "Stellar Temp Filter",
            "ESI Filter",
            "HZ Filter" if hz_only else "Final Set"
        ]
        funnel_values = [total, after_dist, after_temp, after_esi, after_hz]

        funnel_colors = [
            "rgba(255,80,80,0.85)",
            "rgba(255,160,60,0.85)",
            "rgba(255,220,60,0.85)",
            "rgba(80,220,120,0.85)",
            "rgba(60,180,255,0.85)",
        ]

        fig_funnel = go.Figure(go.Funnel(
            y=funnel_labels,
            x=funnel_values,
            textinfo="value+percent initial",
            textfont=dict(color="white", size=12),
            marker=dict(
                color=funnel_colors,
                line=dict(width=1.5, color="rgba(255,255,255,0.15)"),
            ),
            connector=dict(
                line=dict(color="rgba(255,255,255,0.08)", width=1),
                fillcolor="rgba(255,255,255,0.03)",
            ),
        ))

        fig_funnel.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=20, b=20, l=20, r=20),
            funnelmode="stack",
        )
        st.plotly_chart(fig_funnel, width='stretch', config={"displayModeBar": False})
        
        
@st.cache_data(show_spinner=False)
def _cached_shap(planet_name: str):
    df = load_data()
    row = df[df["pl_name"] == planet_name].iloc[0]
    return shap_for_planet(load_trained_model(), pd.DataFrame([row.to_dict()]))

@st.fragment    
def page_planet_details(df: pd.DataFrame, model):
    # render_planet_3d.clear()
    st.title("🪐 Planet Details")

    def _cluster(hostname: str) -> str:
        m = re.match(r'^([A-Za-z]+)', hostname.strip())
        return m.group(1).upper() if m else "OTHER"

    all_hosts = df["hostname"].dropna().unique().tolist()
    clusters  = sorted(set(_cluster(h) for h in all_hosts))

    cluster = st.selectbox(
        "Cluster",
        clusters,
        index=clusters.index("KEPLER") if "KEPLER" in clusters else 0,
    )

    cluster_hosts = sorted(h for h in all_hosts if _cluster(h) == cluster)
    host = st.selectbox(
        "Star System",
        cluster_hosts,
        index=cluster_hosts.index("Kepler-442") if "Kepler-442" in cluster_hosts else 0,
    )

    system_planets = sorted(df[df["hostname"] == host]["pl_name"].dropna().unique().tolist())
    st.markdown("**Select planet:**")
    n_cols = min(len(system_planets), 6)
    cols   = st.columns(n_cols)
    for i, pl in enumerate(system_planets):
        if cols[i % n_cols].button(pl, use_container_width=True, key=f"pl_btn_{pl}"):
            st.session_state["selected_planet"] = pl

    if "selected_planet" not in st.session_state or st.session_state["selected_planet"] not in system_planets:
        st.session_state["selected_planet"] = system_planets[0]
    selected = st.session_state["selected_planet"]

    row = df[df["pl_name"] == selected].iloc[0]
    planet = row.to_dict()
    esi = float(planet.get("esi", 0))

    st.markdown(f"## {selected}")
    dist = planet.get('sy_dist')
    dist_str = f"{dist:.1f} pc" if pd.notna(dist) else "unknown"
    yr = int(planet.get('disc_year') or 0)
    st.caption(f"Star: **{planet.get('hostname', '?')}** · Discovered: **{yr}** · Distance: **{dist_str}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ESI Score", f"{esi:.3f}")
    eqt = planet.get('pl_eqt')
    c2.metric("Eq. Temperature", f"{eqt:.0f} K" if pd.notna(eqt) else "?")
    rad = planet.get('pl_rade')
    c3.metric("Radius", f"{rad:.2f} R⊕" if pd.notna(rad) else "?")
    c4.metric("In Habitable Zone", "Yes" if planet.get("in_hz") == 1.0 else "No")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ESI Components vs Earth")
        esi_comps = compute_esi_components(planet)
        st.plotly_chart(radar_chart(esi_comps, selected), width='stretch')

    with col2:
        st.subheader("Planet Visual")
        st.iframe(render_planet_3d(planet, height=420), height=420)
        

    st.subheader("SHAP Feature Contributions")
    st.caption("How much each feature pushed the ESI score up (green) or down (red) from the baseline")
    shap_vals, expected = _cached_shap(selected)
    st.plotly_chart(shap_waterfall_chart(shap_vals, expected, FEATURE_COLS), width='stretch')

@st.fragment    
def page_terraforming(df: pd.DataFrame):
    st.title("🛸 Terraforming AI")
    st.caption("Select a planet and Claude will generate a Terragenesis-style terraforming roadmap")

    planet_names = sorted(df["pl_name"].dropna().unique().tolist())
    selected = st.selectbox("Select a planet to terraform", planet_names)

    row = df[df["pl_name"] == selected].iloc[0]
    planet = row.to_dict()
    esi = float(planet.get("esi", 0))

    st.subheader("Parameters vs Earth")
    params = {
        "Eq. Temp (K)": (float(planet.get("pl_eqt") or 255), 255),
        "Radius (R⊕)": (float(planet.get("pl_rade") or 1), 1),
        "Mass (M⊕)": (float(min(planet.get("pl_bmasse") or 1, 100)), 1),
        "Insol. Flux": (float(min(planet.get("insol_flux") or 1, 10)), 1),
    }
    names = list(params.keys())
    planet_vals = [v[0] for v in params.values()]
    earth_vals = [v[1] for v in params.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Planet", x=names, y=planet_vals, marker_color="#4488ff"))
    fig.add_trace(go.Bar(name="Earth", x=names, y=earth_vals, marker_color="#44aa66"))
    fig.update_layout(barmode="group", height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width='stretch')

    st.markdown(f"**ESI Score:** {esi:.3f}")
    
    if st.button("Generate Terraforming plan", type="primary"):
        client = get_anthropic_client()
        with st.spinner("Claude is analyzing the planet and generating a roadmap..."):
            plan = generate_terraforming_plan(planet, client)

        difficulty_colors = {
            "Easy": "green", "Moderate": "orange",
            "Hard": "red", "Extreme": "violet", "Impossible": "gray"
        }
        color = difficulty_colors.get(plan.difficulty, "gray")

        st.markdown(f"### Terraforming Plan: {plan.planet_name}")
        col1, col2 = st.columns(2)
        col1.markdown(f"**Difficulty:** :{color}[{plan.difficulty}]")
        col2.markdown(f"**Estimated Timeline:** {plan.timeline_years}")

        st.markdown("**Key Challenges:**")
        for challenge in plan.key_challenges:
            st.markdown(f"- {challenge}")

        st.markdown("---")
        st.markdown("### Terraforming Phases")
        for phase in plan.phases:
            with st.expander(f"Phase {phase.phase}: {phase.name} ({phase.duration})", expanded=True):
                st.markdown(f"**Objective:** {phase.objective}")
                st.markdown(f"**Terragenesis Mechanic:** {phase.terragenesis_mechanic}")
                st.markdown("**Technologies:**")
                for tech in phase.technologies:
                    st.markdown(f"- {tech}")

@st.fragment       
def page_custom(df: pd.DataFrame, model):
    st.title("✏️ Custom Planet Designer")
    st.caption("Design your own planet and see its predicted habitability score in real time")

    col1, col2, col3 = st.columns(3)
    with col1:
        radius = st.slider("Radius (Earth radii)", 0.1, 5.0, 1.0, 0.05)
        mass = st.slider("Mass (Earth masses)", 0.1, 10.0, 1.0, 0.05)
        eq_temp = st.slider("Equilibrium Temperature (K)", 50, 800, 255, 5)
        period = st.slider("Orbital Period (days)", 1, 1000, 365, 1)
    with col2:
        insol = st.slider("Insolation Flux (Earth=1)", 0.1, 5.0, 1.0, 0.05)
        eccentricity = st.slider("Orbital Eccentricity", 0.0, 0.9, 0.0, 0.01)
        st_temp = st.slider("Stellar Temperature (K)", 2500, 10000, 5778, 100)
        st_age = st.slider("Stellar Age (Gyr)", 0.1, 13.0, 4.6, 0.1)
    with col3:
        st.markdown("**Atmospheric Composition**")
        atm_pressure = st.slider("Atmospheric Pressure (atm)", 0.0, 10.0, 1.0, 0.05)
        atm_oxygen   = st.slider("Oxygen Level (%)", 0, 100, 21, 1)

    lum_proxy = (st_temp / 5778) ** 4
    density = (mass / max(radius, 0.01) ** 3) * 5.51
    in_hz = 1.0 if 0.36 <= insol <= 1.11 else 0.0

    custom = {
        "pl_rade": radius, "pl_bmasse": mass, "pl_eqt": eq_temp,
        "pl_orbper": period, "pl_insol": insol, "pl_orbeccen": eccentricity,
        "st_teff": st_temp, "st_age": st_age, "pl_dens": density,
        "in_hz": in_hz, "insol_flux": insol,
        "log_mass": float(np.log1p(mass)),
        "log_radius": float(np.log1p(radius)),
        "log_period": float(np.log1p(period)),
        "insol_clipped": min(insol, 200),
        "stellar_temp_ratio": st_temp / 5778,
        "stellar_lum_log": float(np.log10(max(lum_proxy, 1e-6))),
        "temp_ratio": eq_temp / 255.0,
        "density_ratio": density / 5.51,
        "eccentricity": eccentricity,
        "st_logg_filled": 4.44,
        "st_met_filled": 0.0,
        "st_age_filled": st_age,
    }

    row_df = pd.DataFrame([custom])
    score = float(model.predict(row_df)[0])
    custom["esi"] = score

    press_ok = 0.5 <= atm_pressure <= 2.5
    o2_ok    = 15 <= atm_oxygen <= 35
    if press_ok and o2_ok:
        atm_label = "Breathable"
    elif press_ok or o2_ok:
        atm_label = "Marginal"
    else:
        atm_label = "Hostile"

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted ESI", f"{score:.3f}")
    c2.metric("In Habitable Zone", "Yes" if in_hz == 1.0 else "No")
    c3.metric("Habitability", "High" if score > 0.7 else "Medium" if score > 0.4 else "Low")
    c4.metric("Atmosphere", atm_label)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ESI Components vs Earth")
        st.plotly_chart(radar_chart(compute_esi_components(custom), "Custom Planet"), width='stretch')

    with col2:
        st.subheader("Planet Visual")
        if st.button("Render Planet", type="primary"):
            st.session_state["custom_render"] = render_planet_3d(
                custom, height=420, pressure=atm_pressure, oxygen=float(atm_oxygen)
            )
        if "custom_render" in st.session_state:
            st.iframe(st.session_state["custom_render"], height=420)

    st.subheader("Most Similar Real Planets")
    dist_df = df.copy().assign(_dist=(
        (df["pl_rade"].fillna(1) - radius) ** 2 +
        (df["pl_eqt"].fillna(255) - eq_temp) ** 2 / 10000 +
        (df["pl_bmasse"].fillna(1) - mass) ** 2
    ))
    similar = dist_df.nsmallest(3, "_dist")[["pl_name", "hostname", "esi", "pl_eqt", "pl_rade"]].round(3)
    similar.columns = ["Planet", "Star", "ESI", "Eq Temp (K)", "Radius"]
    st.dataframe(similar, width='stretch', hide_index=True)
        
def main():
    inject_css()
    render_header()

    df = load_data()
    model = load_trained_model()

    tabs = st.tabs([
        "🏠 Home",
        "🔭 Explorer",
        "🪐 Planet Details",
        "🛸 Terraforming AI",
        "✏️ Custom Planet"
    ])

    with tabs[0]:
        page_home(df)
    with tabs[1]:
        page_explorer(df)
    with tabs[2]:
        page_planet_details(df, model)
    with tabs[3]:
        page_terraforming(df)
    with tabs[4]:
        page_custom(df, model)


if __name__ == "__main__":
    main()