"""
Multimodal Crime Incident Report Analyzer — Streamlit Dashboard
===============================================================
Interactive web dashboard for exploring and filtering the unified
master incident dataset produced by integration/build_dataset.py.

Run from the repository root:
    streamlit run integration/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "integration" / "master_incident_dataset.csv"

SEVERITY_ORDER = ["Low", "Medium", "High"]
SEVERITY_COLORS = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
}
SEVERITY_BG = {
    "Low":    "rgba(46,204,113,0.12)",
    "Medium": "rgba(243,156,18,0.12)",
    "High":   "rgba(231,76,60,0.12)",
}

SOURCE_COLORS = {
    "Audio": "#3498db",
    "PDF":   "#9b59b6",
    "Image": "#e67e22",
    "Video": "#1abc9c",
    "Text":  "#e74c3c",
}
SOURCE_ICONS = {
    "Audio": "🎙️",
    "PDF":   "📄",
    "Image": "🖼️",
    "Video": "📹",
    "Text":  "💬",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
)

_M = dict(l=0, r=0, t=10, b=0)   # default margin shorthand

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=str)
    df["Severity_Score"] = pd.to_numeric(df["Severity_Score"], errors="coerce").fillna(0.0)
    df["Severity_Level"] = pd.Categorical(
        df["Severity_Level"], categories=SEVERITY_ORDER, ordered=True
    )
    return df


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multimodal Crime Incident Analyzer",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ─── Global ─── */
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #0a0d13; border-right: 1px solid #21262d; }
[data-testid="stHeader"] { background-color: #0d1117; }

/* ─── All headings — bright white so nothing looks faint ─── */
h1, h2, h3, h4, h5, h6 {
    color: #e6edf3 !important;
    font-weight: 700 !important;
}
/* Streamlit wraps markdown in stMarkdownContainer */
[data-testid="stMarkdownContainer"] h1 { font-size: 1.9rem !important; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.45rem !important; }
[data-testid="stMarkdownContainer"] h3 {
    font-size: 1.2rem !important;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin-bottom: 14px;
}
[data-testid="stMarkdownContainer"] h4 {
    font-size: 1.05rem !important;
    color: #e6edf3 !important;
    margin-bottom: 10px;
}

/* ─── Body / paragraph text ─── */
[data-testid="stMarkdownContainer"] p { color: #c9d1d9 !important; }

/* ─── Sidebar labels and widget text ─── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stMarkdown strong { color: #e6edf3 !important; }

/* ─── Filter / widget labels in main area ─── */
label { color: #c9d1d9 !important; }

/* ─── Metric cards ─── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #b0bec5 !important; font-size: 0.82rem !important; }

/* ─── Info / warning banners ─── */
[data-testid="stAlert"] p { color: #c9d1d9 !important; }

/* ─── Section divider ─── */
hr { border-color: #21262d !important; }

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

/* ─── Tab strip ─── */
[data-testid="stTabs"] button { color: #b0bec5 !important; font-size: 0.9rem; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
    font-weight: 600 !important;
}

/* ─── Caption / small text ─── */
[data-testid="stCaptionContainer"] p { color: #8b949e !important; }

/* ─── Sidebar custom classes ─── */
.sidebar-title {
    font-size: 1.15rem; font-weight: 700; color: #e6edf3;
    margin-bottom: 2px;
}
.sidebar-caption { font-size: 0.75rem; color: #8b949e; margin-bottom: 16px; }

/* ─── KPI accent colours ─── */
.kpi-high   { color: #e74c3c; }
.kpi-medium { color: #f39c12; }
.kpi-low    { color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="sidebar-title">🚨 Incident Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-caption">Multimodal Crime Report Dashboard</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Filters**")

    all_sources = sorted(df["Source"].unique().tolist())
    selected_sources = st.multiselect(
        "Data Source",
        options=all_sources,
        default=all_sources,
        format_func=lambda s: f"{SOURCE_ICONS.get(s, '')}  {s}",
    )

    selected_severities = st.multiselect(
        "Severity Level",
        options=SEVERITY_ORDER,
        default=SEVERITY_ORDER,
    )

    score_min, score_max = st.slider(
        "Severity Score (0 – 10)",
        min_value=0.0,
        max_value=10.0,
        value=(0.0, 10.0),
        step=0.5,
    )

    all_events = sorted(
        [e for e in df["Event"].dropna().unique() if e != "N/A"]
    )
    selected_events = st.multiselect(
        "Event Type",
        options=all_events,
        default=[],
        placeholder="All event types",
    )

    search_query = st.text_input("🔍  Search (event / location / details)", "")

    st.divider()
    st.caption(f"Source file: `master_incident_dataset.csv`")
    st.caption(f"Total rows: **{len(df):,}**")

# ---------------------------------------------------------------------------
# Build filtered view
# ---------------------------------------------------------------------------

mask = (
    df["Source"].isin(selected_sources)
    & df["Severity_Level"].isin(selected_severities)
    & (df["Severity_Score"] >= score_min)
    & (df["Severity_Score"] <= score_max)
)

if selected_events:
    mask &= df["Event"].isin(selected_events)

if search_query.strip():
    q = search_query.strip().lower()
    mask &= (
        df["Event"].str.lower().str.contains(q, na=False)
        | df["Location"].str.lower().str.contains(q, na=False)
        | df["Details"].str.lower().str.contains(q, na=False)
    )

filtered = df[mask].copy()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown("# 🚨 Multimodal Crime Incident Report Analyzer")
st.markdown(
    "Unified incident dashboard ingesting data from **5 AI-powered pipelines**: "
    "audio 911 calls, police PDFs, crime-scene images, CCTV footage, and social-media text."
)

filters_active = (
    len(selected_sources) < len(all_sources)
    or len(selected_severities) < 3
    or (score_min, score_max) != (0.0, 10.0)
    or bool(selected_events)
    or bool(search_query.strip())
)
if filters_active:
    st.info(f"**{len(filtered):,}** of {len(df):,} incidents match your filters.")

st.divider()

# ---------------------------------------------------------------------------
# KPI metrics row
# ---------------------------------------------------------------------------

high_n   = int((filtered["Severity_Level"] == "High").sum())
medium_n = int((filtered["Severity_Level"] == "Medium").sum())
low_n    = int((filtered["Severity_Level"] == "Low").sum())
avg_score = filtered["Severity_Score"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📋 Total Incidents",      f"{len(filtered):,}")
k2.metric("🔴 High Severity",        f"{high_n:,}")
k3.metric("🟡 Medium Severity",      f"{medium_n:,}")
k4.metric("🟢 Low Severity",         f"{low_n:,}")
k5.metric("📊 Avg Severity Score",   f"{avg_score:.2f}" if len(filtered) else "—")

st.divider()

# ---------------------------------------------------------------------------
# Row 1 — Donut by source  |  Stacked bar severity by source
# ---------------------------------------------------------------------------

col_a, col_b = st.columns([1, 1.6])

with col_a:
    st.markdown("#### Incidents by Data Source")
    src_counts = filtered["Source"].value_counts().reset_index()
    src_counts.columns = ["Source", "Count"]

    fig_donut = px.pie(
        src_counts,
        names="Source",
        values="Count",
        color="Source",
        color_discrete_map=SOURCE_COLORS,
        hole=0.52,
    )
    fig_donut.update_traces(
        textinfo="label+percent",
        textfont_size=12,
        hovertemplate="<b>%{label}</b><br>Incidents: %{value:,}<br>Share: %{percent}<extra></extra>",
    )
    fig_donut.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False,
        height=300,
        margin=_M,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_b:
    st.markdown("#### Severity Distribution by Source")

    sev_src = (
        filtered.groupby(["Source", "Severity_Level"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    sev_src["Severity_Level"] = pd.Categorical(
        sev_src["Severity_Level"], categories=SEVERITY_ORDER, ordered=True
    )
    sev_src = sev_src.sort_values("Severity_Level")

    fig_sev_src = px.bar(
        sev_src,
        x="Source",
        y="Count",
        color="Severity_Level",
        barmode="stack",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"Severity_Level": SEVERITY_ORDER},
        text="Count",
    )
    fig_sev_src.update_traces(textposition="inside", textfont_size=11)
    fig_sev_src.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title=None,
        yaxis_title="Incidents",
        legend_title="Severity",
        height=300,
        margin=_M,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_sev_src, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2 — Top events  |  Severity score histogram
# ---------------------------------------------------------------------------

col_c, col_d = st.columns([1.4, 1])

with col_c:
    st.markdown("#### Top 15 Event Types")

    event_df = (
        filtered[filtered["Event"] != "N/A"]["Event"]
        .value_counts()
        .head(15)
        .reset_index()
    )
    event_df.columns = ["Event", "Count"]

    fig_events = px.bar(
        event_df.sort_values("Count"),
        x="Count",
        y="Event",
        orientation="h",
        color="Count",
        color_continuous_scale=["#21262d", "#3498db", "#e74c3c"],
        text="Count",
    )
    fig_events.update_traces(textposition="outside", textfont_size=11)
    fig_events.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Incidents",
        yaxis_title=None,
        showlegend=False,
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=0, r=50, t=10, b=0),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_events, use_container_width=True)

with col_d:
    st.markdown("#### Severity Score Distribution")

    fig_hist = px.histogram(
        filtered,
        x="Severity_Score",
        color="Severity_Level",
        color_discrete_map=SEVERITY_COLORS,
        nbins=20,
        category_orders={"Severity_Level": SEVERITY_ORDER},
        barmode="stack",
        labels={"Severity_Score": "Severity Score (0–10)", "count": "Incidents"},
    )
    fig_hist.add_vline(
        x=3.0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.25)",
        annotation_text="Low → Med",
        annotation_font_color="#8b949e",
        annotation_font_size=10,
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=7.0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.25)",
        annotation_text="Med → High",
        annotation_font_color="#8b949e",
        annotation_font_size=10,
        annotation_position="top right",
    )
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Severity Score (0–10)",
        yaxis_title="Incidents",
        legend_title="Severity",
        height=400,
        margin=_M,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Severity over modalities — grouped bar
# ---------------------------------------------------------------------------

st.divider()
st.markdown("#### Severity Breakdown Across All Modalities")

sev_full = (
    filtered.groupby(["Source", "Severity_Level"], observed=True)
    .size()
    .unstack(fill_value=0)
)
# Reorder columns to Low / Medium / High
for col in SEVERITY_ORDER:
    if col not in sev_full.columns:
        sev_full[col] = 0
sev_full = sev_full[SEVERITY_ORDER].reset_index()

fig_grouped = go.Figure()
for level in SEVERITY_ORDER:
    fig_grouped.add_trace(go.Bar(
        name=level,
        x=sev_full["Source"],
        y=sev_full[level],
        marker_color=SEVERITY_COLORS[level],
        text=sev_full[level],
        textposition="outside",
        textfont_size=11,
    ))

fig_grouped.update_layout(
    **PLOTLY_LAYOUT,
    barmode="group",
    xaxis_title=None,
    yaxis_title="Incidents",
    legend_title="Severity",
    height=320,
    margin=_M,
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#21262d"),
)
st.plotly_chart(fig_grouped, use_container_width=True)

# ---------------------------------------------------------------------------
# Modality spotlight — one tab per source
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Modality Spotlight")

tab_audio, tab_pdf, tab_image, tab_video, tab_text = st.tabs(
    [f"{SOURCE_ICONS[s]}  {s}" for s in ["Audio", "PDF", "Image", "Video", "Text"]]
)

for tab, source in zip(
    [tab_audio, tab_pdf, tab_image, tab_video, tab_text],
    ["Audio", "PDF", "Image", "Video", "Text"],
):
    with tab:
        src_df = filtered[filtered["Source"] == source]
        if src_df.empty:
            st.warning(f"No **{source}** incidents match the active filters.")
            continue

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",       len(src_df))
        m2.metric("🔴 High",    int((src_df["Severity_Level"] == "High").sum()))
        m3.metric("🟡 Medium",  int((src_df["Severity_Level"] == "Medium").sum()))
        m4.metric("Avg Score",  f"{src_df['Severity_Score'].mean():.2f}")

        col_e, col_f = st.columns([1.2, 1])

        with col_e:
            top_ev = src_df["Event"].value_counts().head(8).reset_index()
            top_ev.columns = ["Event", "Count"]

            fig_top = px.bar(
                top_ev,
                x="Event",
                y="Count",
                color="Count",
                color_continuous_scale=["#21262d", SOURCE_COLORS.get(source, "#3498db")],
                text="Count",
            )
            fig_top.update_traces(textposition="outside", textfont_size=11)
            fig_top.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Top Events — {source}",
                title_font_color="#c9d1d9",
                title_font_size=13,
                xaxis_title=None,
                yaxis_title="Incidents",
                showlegend=False,
                coloraxis_showscale=False,
                height=280,
                margin=_M,
                xaxis=dict(showgrid=False, tickangle=-20),
                yaxis=dict(gridcolor="#21262d"),
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_f:
            sev_pie = src_df["Severity_Level"].value_counts().reset_index()
            sev_pie.columns = ["Severity_Level", "Count"]

            fig_sev_pie = px.pie(
                sev_pie,
                names="Severity_Level",
                values="Count",
                color="Severity_Level",
                color_discrete_map=SEVERITY_COLORS,
                hole=0.45,
            )
            fig_sev_pie.update_traces(
                textinfo="label+percent",
                textfont_size=12,
            )
            fig_sev_pie.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Severity Mix — {source}",
                title_font_color="#c9d1d9",
                title_font_size=13,
                showlegend=False,
                height=280,
                margin=_M,
            )
            st.plotly_chart(fig_sev_pie, use_container_width=True)

        # Show top 5 high-severity records for this source
        high_src = src_df[src_df["Severity_Level"] == "High"].head(5)
        if not high_src.empty:
            with st.expander(f"🔴 Top High-Severity {source} Incidents"):
                st.dataframe(
                    high_src[["Incident_ID", "Event", "Location", "Time", "Severity_Score", "Details"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# Full incident table with download
# ---------------------------------------------------------------------------

st.divider()
st.markdown(f"### All Incident Records &nbsp;&nbsp;<small style='color:#8b949e;font-weight:400'>({len(filtered):,} rows)</small>", unsafe_allow_html=True)

display_cols = [
    "Incident_ID", "Source", "Event", "Location",
    "Time", "Severity_Score", "Severity_Level", "Details",
]

# Highlight severity column
def _sev_color(val: str) -> str:
    return f"background-color: {SEVERITY_BG.get(val, '')}; color: {SEVERITY_COLORS.get(val, '#c9d1d9')}"

display_df = filtered[display_cols].copy()

try:
    styled = display_df.style.map(_sev_color, subset=["Severity_Level"])
except AttributeError:
    # pandas < 2.1 fallback
    styled = display_df.style.applymap(_sev_color, subset=["Severity_Level"])

styled = styled.format({"Severity_Score": "{:.1f}"})

st.dataframe(
    styled,
    use_container_width=True,
    height=440,
    hide_index=True,
    column_config={
        "Incident_ID":    st.column_config.TextColumn("Incident ID",   width=110),
        "Source":         st.column_config.TextColumn("Source",        width=75),
        "Event":          st.column_config.TextColumn("Event",         width=180),
        "Location":       st.column_config.TextColumn("Location",      width=130),
        "Time":           st.column_config.TextColumn("Time",          width=85),
        "Severity_Score": st.column_config.NumberColumn("Score",       width=65,  format="%.1f"),
        "Severity_Level": st.column_config.TextColumn("Severity",      width=85),
        "Details":        st.column_config.TextColumn("Details",       width=420),
    },
)

col_dl1, col_dl2 = st.columns([1, 6])
with col_dl1:
    st.download_button(
        label="⬇  Download CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_incidents.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Pipeline summary sidebar panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.markdown("**Pipeline Summary**")
    for src in ["Audio", "PDF", "Image", "Video", "Text"]:
        n = int((df["Source"] == src).sum())
        icon = SOURCE_ICONS.get(src, "")
        st.caption(f"{icon} {src}: **{n:,}** records")

    st.divider()
    st.markdown("**Severity Scale**")
    st.caption("Score = confidence × 10")
    st.caption("🟢 Low:    0 – 3")
    st.caption("🟡 Medium: 3 – 7")
    st.caption("🔴 High:   7 – 10")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "🚨 **Multimodal Crime Incident Report Analyzer** — AI Engineering Group Project &nbsp;|&nbsp; "
    "Pipeline: Audio · PDF · Image · Video · Text · Integration",
    unsafe_allow_html=True,
)
