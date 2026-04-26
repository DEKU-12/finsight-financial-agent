"""
app.py — FinSight Streamlit Frontend

The user-facing dashboard for the FinSight Autonomous Financial Research Agent.

Tabs:
  1. Analysis  — Run the agent on any ticker and display results live
  2. Past Runs  — Browse all MLflow experiment runs as a filterable table
  3. Monitoring — View the latest Evidently data quality / drift report

How to run:
    streamlit run app.py

Environment:
    Requires a .env file with GROQ_API_KEY, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY.
    MLflow server must be running:  mlflow server --host 0.0.0.0 --port 5001
"""

import os
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import mlflow
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FinSight — Autonomous Financial Research Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load project config ───────────────────────────────────────────────────────
# sys.path is already correct when streamlit runs from the project root
from config import config

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] { background-color: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0; }

/* Risk badges */
.risk-low      { background:#1a472a; color:#6fcf97; padding:6px 18px;
                  border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
.risk-medium   { background:#3d2b00; color:#f6c90e; padding:6px 18px;
                  border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
.risk-high     { background:#4a1010; color:#f87171; padding:6px 18px;
                  border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }
.risk-critical { background:#2a0000; color:#ff4444; padding:6px 18px;
                  border-radius:20px; font-weight:700; font-size:1.1rem; display:inline-block; }

/* Section headers */
.section-header { color:#60a5fa; font-size:1.05rem; font-weight:600;
                   margin-top:1.2rem; margin-bottom:0.3rem; }

/* Anomaly flags */
.flag-high   { background:#4a1010; border-left:4px solid #f87171;
                padding:8px 12px; border-radius:4px; margin:4px 0; }
.flag-medium { background:#3d2b00; border-left:4px solid #f6c90e;
                padding:8px 12px; border-radius:4px; margin:4px 0; }
.flag-low    { background:#1a2d1a; border-left:4px solid #6fcf97;
                padding:8px 12px; border-radius:4px; margin:4px 0; }

/* Sentiment badge */
.sent-positive { color:#6fcf97; font-weight:600; }
.sent-negative { color:#f87171; font-weight:600; }
.sent-neutral  { color:#9ca3af; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Helper: risk badge HTML ───────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    cls = f"risk-{level.lower()}" if level.lower() in ("low","medium","high","critical") else "risk-medium"
    return f'<span class="{cls}">⚠ {level.upper()} RISK</span>'


def sentiment_html(label: str, score: float) -> str:
    cls = "sent-positive" if label == "positive" else ("sent-negative" if label == "negative" else "sent-neutral")
    icon = "📈" if label == "positive" else ("📉" if label == "negative" else "➖")
    return f'<span class="{cls}">{icon} {label.upper()} ({score:+.2f})</span>'


def flag_html(flag: dict) -> str:
    sev = flag.get("severity","low").lower()
    cls = f"flag-{sev}" if sev in ("high","medium","low") else "flag-low"
    icon = "🔴" if sev == "high" else ("🟡" if sev == "medium" else "🟢")
    return f'<div class="{cls}">{icon} <b>{flag.get("name","")}</b> — {flag.get("description","")}</div>'


def fmt(val, suffix="", prefix="", decimals=2, fallback="N/A"):
    """Format a numeric value nicely, or show fallback if None."""
    if val is None:
        return fallback
    return f"{prefix}{val:,.{decimals}f}{suffix}"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 FinSight")
    st.markdown("*Autonomous Financial Research Agent*")
    st.divider()

    ticker_input = st.text_input(
        "Stock Ticker Symbol",
        placeholder="e.g. AAPL, TSLA, NVDA",
        help="Enter any NYSE/NASDAQ ticker symbol.",
    ).strip().upper()

    company_input = st.text_input(
        "Company Name (optional)",
        placeholder="e.g. Apple, Tesla",
        help="Improves news search quality. Leave blank to auto-detect.",
    ).strip()

    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- 📊 yfinance (price & technicals)")
    st.markdown("- 📋 Alpha Vantage (fundamentals)")
    st.markdown("- 📰 NewsAPI (headlines & sentiment)")
    st.markdown("- 🤖 Groq / Llama-3 (AI narrative)")
    st.divider()
    st.caption(f"Model: `{config.LLM_MODEL}`")
    st.caption(f"MLflow: `{config.MLFLOW_TRACKING_URI}`")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# Detect if running on Streamlit Cloud (no MLflow server available)
# ═══════════════════════════════════════════════════════════════════════════════
IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit-sharing" or \
           "streamlit.io" in os.environ.get("HOSTNAME", "") or \
           not os.environ.get("MLFLOW_TRACKING_URI", "").startswith("http://localhost")

if IS_CLOUD:
    tab_analysis, tab_monitoring = st.tabs([
        "🔍 Analysis",
        "🩺 Monitoring",
    ])
    tab_runs = None
else:
    tab_analysis, tab_runs, tab_monitoring = st.tabs([
        "🔍 Analysis",
        "📊 Past Runs",
        "🩺 Monitoring",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_analysis:

    # ── Landing state (no run yet) ────────────────────────────────────────────
    if not run_button and "last_result" not in st.session_state:
        st.markdown("## Welcome to FinSight 👋")
        st.markdown(
            "Enter a stock ticker in the sidebar and click **Run Analysis** "
            "to generate a full AI-powered financial research report. "
            "The agent will:"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**📥 Fetch Data**\nPrice history, fundamentals, and live news headlines")
        with c2:
            st.info("**🧮 Analyze**\nRSI, Bollinger Bands, momentum, volatility, anomaly detection")
        with c3:
            st.info("**📄 Generate Report**\nGroq/Llama-3 AI narrative + downloadable PDF")
        st.stop()

    # ── Run the agent ─────────────────────────────────────────────────────────
    if run_button:
        if not ticker_input:
            st.error("Please enter a ticker symbol.")
            st.stop()

        # Validate API keys before running
        try:
            config.validate()
        except ValueError as e:
            st.error(f"**Configuration Error:** {e}")
            st.stop()

        with st.spinner(f"🔄 Running FinSight agent for **{ticker_input}**… this takes ~30 seconds"):
            from agent.graph import run_agent
            result = run_agent(ticker_input, company_input or None)
            st.session_state["last_result"] = result

    # ── Display results ───────────────────────────────────────────────────────
    result = st.session_state.get("last_result", {})
    if not result:
        st.stop()

    if result.get("report_status") == "error":
        st.error(f"Agent error: {result.get('error', 'Unknown error')}")
        st.stop()

    ticker     = result.get("ticker", "")
    company    = result.get("company_name", ticker)
    risk_level = result.get("risk_level", "unknown")

    # ── Header row ────────────────────────────────────────────────────────────
    h1, h2, h3 = st.columns([3, 1, 1])
    with h1:
        st.markdown(f"## {company} ({ticker})")
        st.markdown(f"{result.get('sector','—')} · {result.get('industry','—')}")
    with h2:
        st.markdown(risk_badge(risk_level), unsafe_allow_html=True)
    with h3:
        report_path = result.get("report_path","")
        if report_path and Path(report_path).exists():
            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f.read(),
                    file_name=Path(report_path).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.caption("PDF not available")

    st.divider()

    # ── Price & Market ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Price & Market</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Current Price",   fmt(result.get("current_price"),    prefix="$"))
    p2.metric("52W High",        fmt(result.get("week_52_high"),     prefix="$"))
    p3.metric("52W Low",         fmt(result.get("week_52_low"),      prefix="$"))
    p4.metric("% from 52W High", fmt(result.get("pct_from_52w_high"), suffix="%"))
    p5.metric("% from 52W Low",  fmt(result.get("pct_from_52w_low"),  suffix="%"))
    mc = result.get("market_cap")
    p6.metric("Market Cap", f"${mc/1e9:.1f}B" if mc else "N/A")

    # ── Technical Analysis ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Technical Analysis</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric("RSI (14)",       fmt(result.get("rsi_14"), decimals=1),
              delta=result.get("rsi_signal",""))
    t2.metric("Volatility 30d", fmt(result.get("volatility_30d_pct"), suffix="%", decimals=1),
              delta=result.get("volatility_signal",""))
    t3.metric("Momentum 10d",   fmt(result.get("momentum_10d_pct"), suffix="%", decimals=2),
              delta=result.get("momentum_signal",""))
    t4.metric("BB %B",          fmt(result.get("bb_pct_b"), decimals=2),
              delta=result.get("bb_signal",""))
    t5.metric("MA Cross",       result.get("ma_cross_signal","—").replace("_"," ").title())
    t6.metric("Price vs MA200", result.get("price_vs_ma200","—").replace("_"," ").title())

    # ── Fundamentals ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Fundamentals</div>', unsafe_allow_html=True)
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("P/E Ratio",    fmt(result.get("pe_ratio"),          decimals=1))
    f2.metric("Forward P/E",  fmt(result.get("forward_pe"),        decimals=1))
    f3.metric("EPS",          fmt(result.get("eps"),               prefix="$"))
    f4.metric("Profit Margin",fmt(result.get("profit_margin"),     suffix="%", decimals=1) if result.get("profit_margin") else "N/A")
    f5.metric("Debt/Equity",  fmt(result.get("debt_to_equity"),    decimals=2))
    f6.metric("Beta",         fmt(result.get("beta"),              decimals=2))

    # ── News Sentiment ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📰 News Sentiment</div>', unsafe_allow_html=True)
    ns1, ns2, ns3 = st.columns([1,1,2])
    ns1.metric("Articles Analyzed", result.get("article_count", 0))
    ns2.metric("Sentiment Score",   fmt(result.get("average_sentiment_score"), decimals=2))
    with ns3:
        label = result.get("sentiment_label","neutral")
        score = result.get("average_sentiment_score", 0.0) or 0.0
        st.markdown(
            f"**Sentiment:** {sentiment_html(label, score)}",
            unsafe_allow_html=True,
        )

    # Show article headlines if available
    articles = result.get("articles", [])
    if articles:
        with st.expander(f"📰 View {len(articles)} Headlines"):
            for art in articles:
                score = art.get("sentiment_score", 0)
                icon  = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
                title = art.get("title","No title")
                url   = art.get("url","")
                src   = art.get("source","")
                st.markdown(f"{icon} [{title}]({url}) — *{src}*")

    # ── Anomaly Flags ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Anomaly Detection</div>', unsafe_allow_html=True)
    flags      = result.get("flags", [])
    flag_count = result.get("flag_count", 0)
    high_count = result.get("high_severity_count", 0)

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Flags",        flag_count)
    a2.metric("High-Severity Flags", high_count)
    a3.metric("Anomaly Detected",   "Yes" if result.get("anomaly_detected") else "No")

    if flags:
        for flag in sorted(flags, key=lambda x: {"high":0,"medium":1,"low":2}.get(x.get("severity","low").lower(),3)):
            st.markdown(flag_html(flag), unsafe_allow_html=True)
    else:
        st.success("✅ No anomalies detected.")

    # ── AI Narrative ──────────────────────────────────────────────────────────
    narrative = result.get("llm_narrative","")
    if narrative:
        st.divider()
        st.markdown('<div class="section-header">🤖 AI Research Narrative (Llama-3)</div>',
                    unsafe_allow_html=True)
        with st.expander("View full AI narrative", expanded=True):
            st.markdown(narrative)

    # ── Run metadata ──────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.caption(f"⏱ Agent latency: {fmt(result.get('agent_latency_seconds'), suffix='s', decimals=1)}")
    m2.caption(f"🪙 Tokens used: {result.get('llm_tokens_used','—')}")
    m3.caption(f"🔬 MLflow run: `{result.get('mlflow_run_id','—')[:8]}…`" if result.get('mlflow_run_id') else "🔬 MLflow: not logged")
    m4.caption(f"📄 Report: `{result.get('report_filename','—')}`")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PAST RUNS (MLflow) — local only
# ─────────────────────────────────────────────────────────────────────────────
if not IS_CLOUD and tab_runs is not None:
  with tab_runs:
    st.markdown("## 📊 Past Experiment Runs")
    st.markdown(f"Pulling from MLflow at `{config.MLFLOW_TRACKING_URI}`")

    @st.cache_data(ttl=30)   # refresh every 30 seconds
    def load_mlflow_runs():
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            runs = mlflow.search_runs(
                experiment_names=[config.MLFLOW_EXPERIMENT_NAME],
                order_by=["start_time DESC"],
                max_results=200,
            )
            return runs
        except Exception as e:
            return str(e)

    if st.button("🔄 Refresh", key="refresh_runs"):
        st.cache_data.clear()

    runs_data = load_mlflow_runs()

    if isinstance(runs_data, str):
        st.error(f"Could not connect to MLflow: {runs_data}")
        st.info("Make sure MLflow is running:  `mlflow server --host 0.0.0.0 --port 5001`")
    elif runs_data.empty:
        st.info("No runs found yet. Run an analysis first!")
    else:
        # ── Filters ───────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            # Ticker filter from tags.company_name or run_name
            if "tags.mlflow.runName" in runs_data.columns:
                tickers_in_runs = sorted(
                    runs_data["tags.mlflow.runName"]
                    .str.extract(r"^([A-Z]+)_")[0]
                    .dropna()
                    .unique()
                )
                sel_ticker = st.multiselect("Filter by ticker", tickers_in_runs)
            else:
                sel_ticker = []
        with fc2:
            if "tags.risk_level" in runs_data.columns:
                risk_opts = sorted(runs_data["tags.risk_level"].dropna().unique())
                sel_risk = st.multiselect("Filter by risk level", risk_opts)
            else:
                sel_risk = []
        with fc3:
            if "tags.sentiment_label" in runs_data.columns:
                sent_opts = sorted(runs_data["tags.sentiment_label"].dropna().unique())
                sel_sent = st.multiselect("Filter by sentiment", sent_opts)
            else:
                sel_sent = []

        # Apply filters
        df = runs_data.copy()
        if sel_ticker and "tags.mlflow.runName" in df.columns:
            df = df[df["tags.mlflow.runName"].str.startswith(tuple(sel_ticker))]
        if sel_risk and "tags.risk_level" in df.columns:
            df = df[df["tags.risk_level"].isin(sel_risk)]
        if sel_sent and "tags.sentiment_label" in df.columns:
            df = df[df["tags.sentiment_label"].isin(sel_sent)]

        # ── Build display table ───────────────────────────────────────────────
        display_cols = {}

        if "tags.mlflow.runName" in df.columns:
            display_cols["Run"] = df["tags.mlflow.runName"]
        if "tags.company_name" in df.columns:
            display_cols["Company"] = df["tags.company_name"]
        if "start_time" in df.columns:
            display_cols["Date"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
        if "tags.risk_level" in df.columns:
            display_cols["Risk"] = df["tags.risk_level"]
        if "tags.sentiment_label" in df.columns:
            display_cols["Sentiment"] = df["tags.sentiment_label"]
        if "metrics.current_price" in df.columns:
            display_cols["Price ($)"] = df["metrics.current_price"].round(2)
        if "metrics.rsi_14" in df.columns:
            display_cols["RSI 14"] = df["metrics.rsi_14"].round(1)
        if "metrics.volatility_30d_pct" in df.columns:
            display_cols["Volatility %"] = df["metrics.volatility_30d_pct"].round(1)
        if "metrics.sentiment_score" in df.columns:
            display_cols["Sent. Score"] = df["metrics.sentiment_score"].round(2)
        if "metrics.anomaly_detected" in df.columns:
            display_cols["Anomaly"] = df["metrics.anomaly_detected"].map({1:"Yes",0:"No"})
        if "metrics.agent_latency_seconds" in df.columns:
            display_cols["Latency (s)"] = df["metrics.agent_latency_seconds"].round(1)
        if "tags.report_status" in df.columns:
            display_cols["Status"] = df["tags.report_status"]

        if display_cols:
            display_df = pd.DataFrame(display_cols)
            st.dataframe(display_df, use_container_width=True, height=500)
            st.caption(f"Showing {len(display_df)} of {len(runs_data)} runs")
        else:
            st.dataframe(df, use_container_width=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Summary Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Runs", len(runs_data))

        if "metrics.agent_latency_seconds" in runs_data.columns:
            avg_lat = runs_data["metrics.agent_latency_seconds"].mean()
            s2.metric("Avg Latency", f"{avg_lat:.1f}s")

        if "metrics.anomaly_detected" in runs_data.columns:
            pct_anomaly = runs_data["metrics.anomaly_detected"].mean() * 100
            s3.metric("Anomaly Rate", f"{pct_anomaly:.0f}%")

        if "metrics.sentiment_score" in runs_data.columns:
            avg_sent = runs_data["metrics.sentiment_score"].mean()
            s4.metric("Avg Sentiment", f"{avg_sent:+.2f}")

        st.markdown(
            f"🔗 [Open MLflow UI]({config.MLFLOW_TRACKING_URI}) for full charts and artifact downloads",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MONITORING (Evidently)
# ─────────────────────────────────────────────────────────────────────────────
with tab_monitoring:
    st.markdown("## 🩺 Data Quality & Drift Monitoring")
    st.markdown(
        "Each agent run generates an **Evidently AI** monitoring report "
        "that checks data quality and detects statistical drift vs. the reference dataset."
    )

    # Find the most recent monitoring report
    result = st.session_state.get("last_result", {})
    monitoring_path = result.get("monitoring_report_path","")

    # Also scan the reports directory for any monitoring HTML files
    reports_dir = config.REPORTS_DIR
    monitoring_files = sorted(
        reports_dir.glob("monitoring_*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if reports_dir.exists() else []

    if not monitoring_path and monitoring_files:
        monitoring_path = str(monitoring_files[0])

    # ── File picker ───────────────────────────────────────────────────────────
    if monitoring_files:
        file_names = [p.name for p in monitoring_files]
        chosen = st.selectbox(
            "Select monitoring report",
            file_names,
            index=0,
            help="Reports are generated automatically after each agent run.",
        )
        monitoring_path = str(reports_dir / chosen)

    # ── Show the report ───────────────────────────────────────────────────────
    if monitoring_path and Path(monitoring_path).exists():
        # Show last-run drift status from session state
        if result:
            d1, d2, d3 = st.columns(3)
            d1.metric("Quality Issues",  result.get("quality_issue_count", 0))
            d2.metric("Drift Detected",  "Yes" if result.get("drift_detected") else "No")
            status = result.get("monitoring_status","—")
            d3.metric("Monitor Status",  status)

        st.divider()
        st.markdown(f"**Viewing:** `{Path(monitoring_path).name}`")

        with open(monitoring_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        import streamlit.components.v1 as components
        components.html(html_content, height=900, scrolling=True)

    elif monitoring_files:
        st.warning("Selected report file not found.")
    else:
        st.info(
            "No monitoring reports yet. Run an analysis first — "
            "a report is automatically generated after each run."
        )
        st.markdown(
            "The monitoring report checks:\n"
            "- ✅ Data completeness (no missing price/volume data)\n"
            "- ✅ Value range validity (RSI 0–100, price > 0, etc.)\n"
            "- ✅ Statistical drift vs. reference dataset (z-score method)\n"
        )
