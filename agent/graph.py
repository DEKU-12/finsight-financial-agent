"""
agent/graph.py — LangGraph Agent Definition

This is the brain of FinSight. It defines a stateful LangGraph pipeline
that connects all 6 nodes in sequence, passing a shared state dict
from one node to the next.

Pipeline:
    START
      │
      ▼
  fetch_price       ← yfinance: price, MA30, MA200, daily returns
      │
      ▼
  fetch_fundamentals ← Alpha Vantage: P/E, EPS, margins, debt-to-equity
      │
      ▼
  fetch_news        ← NewsAPI: headlines + sentiment scores
      │
      ▼
  analyze           ← RSI, Bollinger Bands, volatility, momentum
      │
      ▼
  detect_anomaly    ← flag unusual patterns and risk signals
      │
      ▼
  generate_report   ← Groq/Llama3 narrative + reportlab PDF
      │
      ▼
    END

Why LangGraph?
  LangGraph (built on LangChain) manages the state dict automatically,
  merges each node's output into the shared state, and handles the
  routing between nodes. It's the industry-standard framework for
  building multi-step AI agents in 2025/2026.

  For this pipeline the graph is linear (no branching), but the
  LangGraph structure makes it easy to add conditional routing later —
  for example: if anomaly_detected=1, route to an "alert" node
  before generating the report.

Usage:
    from agent.graph import run_agent
    result = run_agent("AAPL")
    print(result["report_path"])
"""

import logging
import time
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, START, END

from agent.nodes.fetch_price import fetch_price_data
from agent.nodes.fetch_fundamentals import fetch_fundamentals
from agent.nodes.fetch_news import fetch_news
from agent.nodes.analyze import analyze
from agent.nodes.detect_anomaly import detect_anomaly
from agent.nodes.generate_report import generate_report

logger = logging.getLogger(__name__)


# ── State definition ──────────────────────────────────────────────────────────
# LangGraph requires a TypedDict that defines the shape of the shared state.
# We use Any for most fields since the data dicts are flexible.
# Every key that any node might write to should be listed here.

class AgentState(TypedDict, total=False):
    # Input
    ticker: str
    company_name_input: str   # user-provided name (may differ from yfinance name)
    run_start_time: float

    # fetch_price output
    company_name: str
    sector: str
    industry: str
    currency: str
    market_cap: Optional[int]
    current_price: Optional[float]
    week_52_high: Optional[float]
    week_52_low: Optional[float]
    avg_volume: Optional[int]
    current_volume: Optional[int]
    ma_30: Optional[float]
    ma_200: Optional[float]
    daily_returns: list
    close_prices: list
    dates: list
    status: str

    # fetch_fundamentals output
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    eps: Optional[float]
    profit_margin: Optional[float]
    operating_margin: Optional[float]
    revenue_ttm: Optional[float]
    debt_to_equity: Optional[float]
    beta: Optional[float]
    analyst_target_price: Optional[float]
    dividend_yield: Optional[float]
    book_value: Optional[float]

    # fetch_news output
    articles: list
    article_count: int
    average_sentiment_score: float
    sentiment_label: str

    # analyze output
    rsi_14: Optional[float]
    rsi_signal: str
    bb_upper: Optional[float]
    bb_middle: Optional[float]
    bb_lower: Optional[float]
    bb_width: Optional[float]
    bb_signal: str
    bb_pct_b: Optional[float]
    volatility_30d: Optional[float]
    volatility_30d_pct: Optional[float]
    volatility_signal: str
    momentum_10d: Optional[float]
    momentum_10d_pct: Optional[float]
    momentum_signal: str
    price_vs_ma30: str
    price_vs_ma200: str
    ma_cross_signal: str
    pct_from_52w_high: Optional[float]
    pct_from_52w_low: Optional[float]
    analysis_status: str

    # detect_anomaly output
    flags: list
    flag_count: int
    high_severity_count: int
    anomaly_detected: int
    risk_level: str
    anomaly_summary: str

    # generate_report output
    report_path: str
    report_filename: str
    llm_narrative: str
    llm_tokens_used: int
    llm_latency_seconds: float
    report_status: str

    # Timing
    agent_latency_seconds: float


# ── Node wrappers ─────────────────────────────────────────────────────────────
# Each LangGraph node receives the full state dict and returns a dict
# of the keys it wants to update. We wrap each function to log timing.

def node_fetch_price(state: AgentState) -> dict:
    logger.info("[Node 1/6] fetch_price → %s", state["ticker"])
    result = fetch_price_data(state["ticker"])
    return result


def node_fetch_fundamentals(state: AgentState) -> dict:
    logger.info("[Node 2/6] fetch_fundamentals → %s", state["ticker"])
    result = fetch_fundamentals(state["ticker"])
    # Avoid overwriting company_name if yfinance already got a better one
    if state.get("company_name") and not result.get("company_name"):
        result["company_name"] = state["company_name"]
    return result


def node_fetch_news(state: AgentState) -> dict:
    logger.info("[Node 3/6] fetch_news → %s", state["ticker"])
    company = state.get("company_name") or state.get("company_name_input") or state["ticker"]
    result = fetch_news(company_name=company, ticker=state["ticker"])
    return result


def node_analyze(state: AgentState) -> dict:
    logger.info("[Node 4/6] analyze → %s", state["ticker"])
    result = analyze(dict(state))
    return result


def node_detect_anomaly(state: AgentState) -> dict:
    logger.info("[Node 5/6] detect_anomaly → %s", state["ticker"])
    result = detect_anomaly(dict(state))
    return result


def node_generate_report(state: AgentState) -> dict:
    logger.info("[Node 6/6] generate_report → %s", state["ticker"])
    result = generate_report(dict(state))
    # Compute total agent latency
    start = state.get("run_start_time", time.time())
    result["agent_latency_seconds"] = round(time.time() - start, 2)
    return result


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph.

    Nodes are added in pipeline order, then connected with directed edges.
    The graph is compiled once and reused for all runs.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("fetch_price", node_fetch_price)
    graph.add_node("fetch_fundamentals", node_fetch_fundamentals)
    graph.add_node("fetch_news", node_fetch_news)
    graph.add_node("analyze", node_analyze)
    graph.add_node("detect_anomaly", node_detect_anomaly)
    graph.add_node("generate_report", node_generate_report)

    # Define edges (linear pipeline)
    graph.add_edge(START, "fetch_price")
    graph.add_edge("fetch_price", "fetch_fundamentals")
    graph.add_edge("fetch_fundamentals", "fetch_news")
    graph.add_edge("fetch_news", "analyze")
    graph.add_edge("analyze", "detect_anomaly")
    graph.add_edge("detect_anomaly", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# Compile once at module import time — reuse for all calls
_agent = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(ticker: str, company_name: Optional[str] = None) -> dict:
    """
    Run the full FinSight agent pipeline for a given ticker.

    Args:
        ticker:       Stock ticker symbol, e.g. "AAPL". Case-insensitive.
        company_name: Optional human-readable company name, e.g. "Apple".
                      Used to improve news search quality.
                      If omitted, yfinance's company name is used.

    Returns:
        The final state dict with all fields from all 6 nodes,
        including report_path (PDF location) and all metrics
        needed for MLflow logging.

    Example:
        result = run_agent("TSLA", "Tesla")
        print(result["report_path"])
        print(result["rsi_14"])
        print(result["risk_level"])
    """
    ticker = ticker.strip().upper()
    logger.info("=" * 60)
    logger.info("FinSight Agent | Starting run for %s", ticker)
    logger.info("=" * 60)

    initial_state: AgentState = {
        "ticker": ticker,
        "company_name_input": company_name or ticker,
        "run_start_time": time.time(),
    }

    try:
        final_state = _agent.invoke(initial_state)
        logger.info(
            "Agent run complete for %s | status=%s | latency=%.2fs",
            ticker,
            final_state.get("report_status"),
            final_state.get("agent_latency_seconds", 0),
        )
        return dict(final_state)

    except Exception as exc:
        logger.error("Agent run failed for %s: %s", ticker, exc, exc_info=True)
        return {
            "ticker": ticker,
            "report_status": "error",
            "error": str(exc),
            "agent_latency_seconds": 0,
        }


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    company = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_agent(ticker, company)

    # Print a clean summary (skip large lists and full narrative)
    skip = {"close_prices", "daily_returns", "dates", "llm_narrative",
            "description", "report_sections"}
    summary = {k: v for k, v in result.items() if k not in skip}
    print("\n" + "=" * 60)
    print("AGENT RUN SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n✅ PDF report: {result.get('report_path', 'NOT GENERATED')}")
