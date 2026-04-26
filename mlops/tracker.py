"""
mlops/tracker.py — MLflow Experiment Tracker

Logs every FinSight agent run as a reproducible MLflow experiment.
This is the core MLops piece of the project — it's what lets you
open a dashboard and compare 50 runs across different tickers.

What gets logged per run:
─────────────────────────────────────────────────────────────
PARAMETERS (inputs — what you asked for)
  ticker, run_date, llm_model, data_sources

METRICS (outputs — what the agent found)
  current_price, pe_ratio, eps, beta
  rsi_14, volatility_30d, momentum_10d
  sentiment_score, article_count
  anomaly_detected, flag_count, risk_level_score
  llm_tokens_used, llm_latency_seconds, agent_latency_seconds

TAGS (metadata for filtering in the UI)
  sector, industry, sentiment_label, risk_level,
  ma_cross_signal, rsi_signal

ARTIFACTS (files attached to the run)
  PDF report
  Evidently HTML monitoring report (if available)
  Raw state JSON (all data in one file for reproducibility)
─────────────────────────────────────────────────────────────

Usage:
    from mlops.tracker import log_run
    run_id = log_run(state)   # call after generate_report node
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow

from config import config

logger = logging.getLogger(__name__)

# Map risk level strings to numeric scores for MLflow metric tracking
RISK_LEVEL_SCORE = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
    "unknown": -1,
}


def log_run(state: dict) -> Optional[str]:
    """
    Log a completed agent run to MLflow.

    Args:
        state: The final state dict returned by the LangGraph agent
               after all nodes (including generate_report) have run.

    Returns:
        The MLflow run_id string if logging succeeded, else None.
        The run_id is added to the state dict so it can be displayed
        in the Streamlit UI.
    """
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Logging MLflow run for %s", ticker)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    try:
        with mlflow.start_run(run_name=f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id

            # ── Parameters (inputs) ───────────────────────────────────────────
            mlflow.log_params({
                "ticker": ticker,
                "run_date": datetime.now().strftime("%Y-%m-%d"),
                "llm_model": config.LLM_MODEL,
                "data_sources": "yfinance,alpha_vantage,newsapi",
                "rsi_period": config.RSI_PERIOD,
                "bb_window": config.BB_WINDOW,
                "anomaly_zscore_threshold": config.ANOMALY_ZSCORE_THRESHOLD,
            })

            # ── Metrics (outputs) ─────────────────────────────────────────────
            metrics = _build_metrics(state)
            # MLflow requires all metric values to be floats and non-None
            clean_metrics = {
                k: float(v) for k, v in metrics.items()
                if v is not None
            }
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

            # ── Tags (categorical metadata for UI filtering) ──────────────────
            tags = _build_tags(state)
            mlflow.set_tags(tags)

            # ── Artifacts ─────────────────────────────────────────────────────
            _log_artifacts(state, run_id)

            logger.info("MLflow run logged: run_id=%s ticker=%s", run_id, ticker)
            return run_id

    except Exception as exc:
        logger.error("MLflow logging failed for %s: %s", ticker, exc, exc_info=True)
        return None


# ── Metric builders ───────────────────────────────────────────────────────────

def _build_metrics(state: dict) -> dict:
    """Extract all numeric metrics from the state dict."""
    risk_score = RISK_LEVEL_SCORE.get(
        state.get("risk_level", "unknown").lower(), -1
    )

    return {
        # Price
        "current_price":        state.get("current_price"),
        "week_52_high":         state.get("week_52_high"),
        "week_52_low":          state.get("week_52_low"),
        "pct_from_52w_high":    state.get("pct_from_52w_high"),
        "pct_from_52w_low":     state.get("pct_from_52w_low"),
        "market_cap_billions":  _to_billions(state.get("market_cap")),
        # Technical
        "rsi_14":               state.get("rsi_14"),
        "volatility_30d":       state.get("volatility_30d"),
        "volatility_30d_pct":   state.get("volatility_30d_pct"),
        "momentum_10d_pct":     state.get("momentum_10d_pct"),
        "bb_pct_b":             state.get("bb_pct_b"),
        "bb_width":             state.get("bb_width"),
        "ma_30":                state.get("ma_30"),
        "ma_200":               state.get("ma_200"),
        # Fundamental
        "pe_ratio":             state.get("pe_ratio"),
        "forward_pe":           state.get("forward_pe"),
        "eps":                  state.get("eps"),
        "profit_margin":        state.get("profit_margin"),
        "operating_margin":     state.get("operating_margin"),
        "debt_to_equity":       state.get("debt_to_equity"),
        "beta":                 state.get("beta"),
        "analyst_target_price": state.get("analyst_target_price"),
        "dividend_yield":       state.get("dividend_yield"),
        # News
        "sentiment_score":      state.get("average_sentiment_score"),
        "article_count":        state.get("article_count"),
        # Anomaly
        "anomaly_detected":     state.get("anomaly_detected"),
        "flag_count":           state.get("flag_count"),
        "high_severity_flags":  state.get("high_severity_count"),
        "risk_level_score":     risk_score,
        # Performance
        "llm_tokens_used":      state.get("llm_tokens_used"),
        "llm_latency_seconds":  state.get("llm_latency_seconds"),
        "agent_latency_seconds": state.get("agent_latency_seconds"),
    }


def _build_tags(state: dict) -> dict:
    """Build categorical tags for MLflow run filtering."""
    return {
        "sector":           str(state.get("sector", "Unknown")),
        "industry":         str(state.get("industry", "Unknown")),
        "company_name":     str(state.get("company_name", state.get("ticker", "Unknown"))),
        "sentiment_label":  str(state.get("sentiment_label", "neutral")),
        "risk_level":       str(state.get("risk_level", "unknown")),
        "rsi_signal":       str(state.get("rsi_signal", "neutral")),
        "ma_cross_signal":  str(state.get("ma_cross_signal", "neutral")),
        "bb_signal":        str(state.get("bb_signal", "within_bands")),
        "momentum_signal":  str(state.get("momentum_signal", "neutral")),
        "volatility_signal": str(state.get("volatility_signal", "moderate")),
        "report_status":    str(state.get("report_status", "unknown")),
        "anomaly_detected": "yes" if state.get("anomaly_detected") else "no",
    }


def _log_artifacts(state: dict, run_id: str) -> None:
    """Log PDF report, monitoring report, and raw state JSON as MLflow artifacts."""

    # ── PDF report ────────────────────────────────────────────────────────────
    report_path = state.get("report_path")
    if report_path and Path(report_path).exists():
        mlflow.log_artifact(report_path, artifact_path="reports")
        logger.debug("Logged PDF artifact: %s", report_path)
    else:
        logger.warning("No PDF report found to log as artifact")

    # ── Evidently HTML monitoring report ──────────────────────────────────────
    monitoring_report_path = state.get("monitoring_report_path")
    if monitoring_report_path and Path(monitoring_report_path).exists():
        mlflow.log_artifact(monitoring_report_path, artifact_path="monitoring")
        logger.debug("Logged Evidently report: %s", monitoring_report_path)

    # ── Raw state JSON (full reproducibility) ─────────────────────────────────
    # Save a JSON snapshot of all data (excluding long lists) so you can
    # perfectly reproduce what the agent saw during this run.
    skip_keys = {"close_prices", "daily_returns", "dates", "llm_narrative",
                 "description", "report_sections"}
    state_snapshot = {
        k: v for k, v in state.items()
        if k not in skip_keys and _is_json_serialisable(v)
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"state_{state.get('ticker', 'unknown')}_",
        delete=False,
    ) as f:
        json.dump(state_snapshot, f, indent=2, default=str)
        tmp_path = f.name

    try:
        mlflow.log_artifact(tmp_path, artifact_path="state")
    finally:
        os.unlink(tmp_path)

    logger.debug("Logged state JSON artifact for run %s", run_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_billions(value: Optional[float]) -> Optional[float]:
    """Convert a raw market cap value to billions."""
    if value is None:
        return None
    return round(float(value) / 1e9, 2)


def _is_json_serialisable(value) -> bool:
    """Check if a value can be serialised to JSON."""
    try:
        json.dumps(value, default=str)
        return True
    except (TypeError, ValueError):
        return False


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    # Minimal dummy state to test logging without running the full agent
    dummy_state = {
        "ticker": "TEST",
        "company_name": "Test Corp",
        "sector": "Technology",
        "industry": "Software",
        "currency": "USD",
        "current_price": 150.0,
        "week_52_high": 200.0,
        "week_52_low": 100.0,
        "pct_from_52w_high": -25.0,
        "pct_from_52w_low": 50.0,
        "market_cap": 500_000_000_000,
        "rsi_14": 55.0,
        "rsi_signal": "bullish",
        "volatility_30d": 0.22,
        "volatility_30d_pct": 22.0,
        "volatility_signal": "moderate",
        "momentum_10d_pct": 3.0,
        "momentum_signal": "bullish",
        "bb_pct_b": 0.65,
        "bb_width": 0.10,
        "bb_signal": "within_bands",
        "ma_30": 148.0,
        "ma_200": 140.0,
        "ma_cross_signal": "golden_cross",
        "pe_ratio": 28.0,
        "forward_pe": 25.0,
        "eps": 5.36,
        "profit_margin": 0.21,
        "operating_margin": 0.28,
        "debt_to_equity": 1.2,
        "beta": 1.15,
        "analyst_target_price": 175.0,
        "dividend_yield": 0.005,
        "average_sentiment_score": 0.4,
        "sentiment_label": "positive",
        "article_count": 5,
        "anomaly_detected": 0,
        "flag_count": 0,
        "high_severity_count": 0,
        "risk_level": "low",
        "llm_tokens_used": 850,
        "llm_latency_seconds": 1.2,
        "agent_latency_seconds": 2.1,
        "report_status": "success",
    }

    run_id = log_run(dummy_state)
    print(f"\nMLflow run logged: {run_id}")
    print(f"View at: {config.MLFLOW_TRACKING_URI}")
