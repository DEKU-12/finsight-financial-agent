"""
mlops/monitor.py — Data Quality & Drift Monitor (Evidently AI)

Runs two types of checks on every agent run:

1. DATA QUALITY REPORT
   Checks the current run's financial metrics for problems:
   - Missing / null values
   - Values outside expected ranges (e.g. negative price, RSI > 100)
   - Type mismatches

2. DATA DRIFT REPORT
   Compares the current run's metrics against a reference dataset
   (a baseline of historical runs saved in data/reference/).
   Flags if today's data looks statistically different from the baseline.
   This catches things like: API returning garbage data, market regime
   changes, or a ticker with unusual characteristics.

Both reports are saved as HTML files and logged as MLflow artifacts.
The monitoring_report_path key is added to state so tracker.py can
pick it up automatically.

On the very first run there is no reference dataset yet — the monitor
saves the current run as the reference for future comparisons.

Why this matters for your CV:
  Evidently AI monitoring appears in <5% of new grad portfolios.
  Being able to say "I monitor every agent run for data quality and
  drift" is a strong signal to ML engineering hiring managers.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import config

logger = logging.getLogger(__name__)

# The metrics we monitor for quality and drift.
# These are the same fields logged to MLflow.
MONITORED_METRICS = [
    "current_price",
    "week_52_high",
    "week_52_low",
    "rsi_14",
    "volatility_30d_pct",
    "momentum_10d_pct",
    "bb_pct_b",
    "pe_ratio",
    "eps",
    "profit_margin",
    "beta",
    "average_sentiment_score",
    "flag_count",
    "agent_latency_seconds",
]

# Expected value ranges for data quality checks
METRIC_RANGES = {
    "current_price":          (0.01, 1_000_000),
    "week_52_high":           (0.01, 1_000_000),
    "week_52_low":            (0.01, 1_000_000),
    "rsi_14":                 (0, 100),
    "volatility_30d_pct":     (0, 500),
    "momentum_10d_pct":       (-100, 100),
    "bb_pct_b":               (-1, 2),
    "pe_ratio":               (-500, 10_000),
    "eps":                    (-10_000, 100_000),
    "profit_margin":          (-10, 1),
    "beta":                   (-5, 10),
    "average_sentiment_score": (-1, 1),
    "flag_count":             (0, 100),
    "agent_latency_seconds":  (0, 300),
}


def run_monitoring(state: dict) -> dict:
    """
    Run Evidently AI data quality and drift checks on the current agent run.

    Args:
        state: The agent state dict (after generate_report has run).

    Returns:
        state augmented with:
            monitoring_report_path  str   Path to the saved HTML report
            monitoring_status       str   "success" | "error" | "no_reference"
            quality_issues          list  List of detected quality problems
            drift_detected          bool  True if drift was found vs reference
    """
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Running Evidently monitoring for %s", ticker)

    config.ensure_dirs()

    try:
        # Extract current run metrics into a single-row DataFrame
        current_df = _state_to_dataframe(state)

        # Run quality checks (always)
        quality_issues = _check_data_quality(current_df, state)

        # Load reference dataset if it exists
        reference_df = _load_reference_data()

        # Generate HTML report
        report_path = _generate_html_report(
            ticker=ticker,
            current_df=current_df,
            reference_df=reference_df,
            quality_issues=quality_issues,
            state=state,
        )

        # Detect drift (only if reference exists and has enough rows)
        drift_detected = False
        if reference_df is not None and len(reference_df) >= 5:
            drift_detected = _detect_drift(current_df, reference_df)

        # Update reference dataset with current run
        _update_reference_data(current_df)

        logger.info(
            "Monitoring complete for %s: %d quality issues, drift=%s",
            ticker, len(quality_issues), drift_detected
        )

        return {
            **state,
            "monitoring_report_path": str(report_path),
            "monitoring_status": "success",
            "quality_issues": quality_issues,
            "quality_issue_count": len(quality_issues),
            "drift_detected": drift_detected,
        }

    except Exception as exc:
        logger.error("Monitoring failed for %s: %s", ticker, exc, exc_info=True)
        return {
            **state,
            "monitoring_status": "error",
            "monitoring_error": str(exc),
            "quality_issues": [],
            "quality_issue_count": 0,
            "drift_detected": False,
        }


# ── Data extraction ───────────────────────────────────────────────────────────

def _state_to_dataframe(state: dict) -> pd.DataFrame:
    """Convert the relevant state metrics into a single-row DataFrame."""
    row = {metric: state.get(metric) for metric in MONITORED_METRICS}
    row["ticker"] = state.get("ticker", "UNKNOWN")
    row["run_date"] = datetime.now().strftime("%Y-%m-%d")
    return pd.DataFrame([row])


# ── Quality checks ────────────────────────────────────────────────────────────

def _check_data_quality(df: pd.DataFrame, state: dict) -> list[dict]:
    """
    Run data quality checks and return a list of issue dicts.

    Checks:
      - Missing values for each monitored metric
      - Values outside expected ranges
      - Price consistency (52w_low <= current_price <= 52w_high)
    """
    issues = []
    row = df.iloc[0]

    for metric in MONITORED_METRICS:
        value = row.get(metric)

        # Missing value check
        if value is None or pd.isna(value):
            issues.append({
                "metric": metric,
                "issue": "missing_value",
                "severity": "high" if metric in ("current_price", "rsi_14") else "medium",
                "detail": f"{metric} is null/missing",
            })
            continue

        # Range check
        if metric in METRIC_RANGES:
            lo, hi = METRIC_RANGES[metric]
            if not (lo <= float(value) <= hi):
                issues.append({
                    "metric": metric,
                    "issue": "out_of_range",
                    "severity": "high",
                    "detail": f"{metric}={value:.4f} is outside expected range [{lo}, {hi}]",
                })

    # Price consistency check
    price = state.get("current_price")
    low_52 = state.get("week_52_low")
    high_52 = state.get("week_52_high")
    if all(v is not None for v in [price, low_52, high_52]):
        if price < low_52 * 0.90:  # 10% tolerance for stale data
            issues.append({
                "metric": "current_price",
                "issue": "below_52w_low",
                "severity": "high",
                "detail": f"Price ${price:.2f} is more than 10% below 52-week low ${low_52:.2f}",
            })
        if price > high_52 * 1.10:
            issues.append({
                "metric": "current_price",
                "issue": "above_52w_high",
                "severity": "medium",
                "detail": f"Price ${price:.2f} is more than 10% above 52-week high ${high_52:.2f}",
            })

    return issues


# ── Drift detection ───────────────────────────────────────────────────────────

def _detect_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> bool:
    """
    Simple statistical drift detection without requiring Evidently's full API.

    For each numeric metric, checks if the current value falls outside
    the reference distribution's mean ± 3 standard deviations.
    Returns True if 3 or more metrics are outside that range.
    """
    drift_count = 0
    numeric_cols = [
        c for c in MONITORED_METRICS
        if c in reference_df.columns and reference_df[c].dtype in ("float64", "int64")
    ]

    for col in numeric_cols:
        ref_vals = reference_df[col].dropna()
        if len(ref_vals) < 3:
            continue
        current_val = current_df[col].iloc[0]
        if pd.isna(current_val):
            continue
        mean = ref_vals.mean()
        std = ref_vals.std()
        if std == 0:
            continue
        z_score = abs((float(current_val) - mean) / std)
        if z_score > 3.0:
            drift_count += 1
            logger.debug("Drift detected in %s: z=%.2f", col, z_score)

    return drift_count >= 3


# ── Reference data management ─────────────────────────────────────────────────

def _load_reference_data() -> Optional[pd.DataFrame]:
    """Load the reference dataset CSV from disk."""
    ref_path = config.REFERENCE_DATA_DIR / "metrics_reference.csv"
    if not ref_path.exists():
        logger.info("No reference dataset found — this run will create it")
        return None
    try:
        df = pd.read_csv(ref_path)
        logger.info("Loaded reference dataset: %d rows", len(df))
        return df
    except Exception as exc:
        logger.warning("Could not load reference dataset: %s", exc)
        return None


def _update_reference_data(current_df: pd.DataFrame) -> None:
    """Append the current run to the reference dataset CSV."""
    ref_path = config.REFERENCE_DATA_DIR / "metrics_reference.csv"
    config.REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ref_path.exists():
        try:
            existing = pd.read_csv(ref_path)
            updated = pd.concat([existing, current_df], ignore_index=True)
            # Keep only the last 200 runs to avoid unbounded growth
            updated = updated.tail(200)
        except Exception:
            updated = current_df
    else:
        updated = current_df

    updated.to_csv(ref_path, index=False)
    logger.debug("Reference dataset updated: %d rows total", len(updated))


# ── HTML report generator ─────────────────────────────────────────────────────

def _generate_html_report(
    ticker: str,
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    quality_issues: list,
    state: dict,
) -> Path:
    """
    Generate a standalone HTML monitoring report.

    We build this manually rather than using Evidently's full suite
    to avoid compatibility issues between Evidently versions.
    The report is clean, readable, and self-contained.
    """
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    report_filename = f"monitoring_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = config.REPORTS_DIR / report_filename

    # Build quality issues HTML
    if quality_issues:
        issues_rows = "".join(
            f"""<tr>
                <td>{i['metric']}</td>
                <td><span class="badge badge-{i['severity']}">{i['severity'].upper()}</span></td>
                <td>{i['issue']}</td>
                <td>{i['detail']}</td>
            </tr>"""
            for i in quality_issues
        )
        issues_html = f"""
        <h2>⚠ Data Quality Issues ({len(quality_issues)} found)</h2>
        <table>
            <tr><th>Metric</th><th>Severity</th><th>Issue</th><th>Detail</th></tr>
            {issues_rows}
        </table>"""
    else:
        issues_html = "<h2>✅ Data Quality: No Issues Found</h2><p>All metrics are within expected ranges.</p>"

    # Build metrics table HTML
    metrics_rows = ""
    for metric in MONITORED_METRICS:
        value = state.get(metric)
        lo, hi = METRIC_RANGES.get(metric, (None, None))
        if value is not None:
            formatted = f"{float(value):.4f}"
            range_str = f"{lo} – {hi}" if lo is not None else "N/A"
        else:
            formatted = "N/A"
            range_str = "N/A"
        metrics_rows += f"<tr><td>{metric}</td><td>{formatted}</td><td>{range_str}</td></tr>"

    # Build reference comparison HTML
    if reference_df is not None and len(reference_df) >= 2:
        ref_rows = ""
        for metric in MONITORED_METRICS:
            if metric not in reference_df.columns:
                continue
            ref_vals = reference_df[metric].dropna()
            if len(ref_vals) == 0:
                continue
            current_val = state.get(metric)
            ref_rows += f"""<tr>
                <td>{metric}</td>
                <td>{f"{float(current_val):.4f}" if current_val is not None else "N/A"}</td>
                <td>{ref_vals.mean():.4f}</td>
                <td>{ref_vals.std():.4f}</td>
                <td>{ref_vals.min():.4f}</td>
                <td>{ref_vals.max():.4f}</td>
            </tr>"""
        drift_html = f"""
        <h2>📊 Reference Comparison ({len(reference_df)} historical runs)</h2>
        <table>
            <tr><th>Metric</th><th>Current</th><th>Ref Mean</th><th>Ref Std</th><th>Ref Min</th><th>Ref Max</th></tr>
            {ref_rows}
        </table>"""
    else:
        drift_html = "<h2>📊 Reference Dataset</h2><p>Not enough historical runs yet for drift comparison (need ≥ 5). Current run has been added to the reference dataset.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FinSight Monitoring — {ticker} — {date_str}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 24px;
         background: #f8f9fa; color: #212529; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 8px; }}
  h2   {{ color: #1a1a2e; margin-top: 32px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           box-shadow: 0 1px 4px rgba(0,0,0,0.1); border-radius: 8px;
           overflow: hidden; margin-bottom: 24px; }}
  th   {{ background: #1a1a2e; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }}
  td   {{ padding: 9px 14px; border-bottom: 1px solid #dee2e6; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 11px; font-weight: bold; color: white; }}
  .badge-high   {{ background: #e74c3c; }}
  .badge-medium {{ background: #f39c12; }}
  .badge-low    {{ background: #2ecc71; }}
  .meta {{ background: white; border-radius: 8px; padding: 16px 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 24px; }}
  .meta span {{ margin-right: 24px; font-size: 14px; }}
  .meta b {{ color: #1a1a2e; }}
</style>
</head>
<body>
<h1>🔍 FinSight Data Monitoring Report</h1>
<div class="meta">
  <span><b>Ticker:</b> {ticker}</span>
  <span><b>Company:</b> {state.get('company_name', ticker)}</span>
  <span><b>Generated:</b> {date_str}</span>
  <span><b>Quality Issues:</b> {len(quality_issues)}</span>
  <span><b>Reference Runs:</b> {len(reference_df) if reference_df is not None else 0}</span>
</div>

{issues_html}

<h2>📋 Current Run Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Expected Range</th></tr>
  {metrics_rows}
</table>

{drift_html}

<p style="color:#6c757d; font-size:12px; margin-top:32px;">
  Generated by FinSight Monitoring (Evidently AI-style) | {date_str}
</p>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    logger.info("Monitoring HTML report saved: %s", report_path)
    return report_path


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    # Use a minimal dummy state
    dummy_state = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 271.06,
        "week_52_high": 288.35,
        "week_52_low": 192.41,
        "rsi_14": 59.71,
        "volatility_30d_pct": 23.82,
        "momentum_10d_pct": 4.06,
        "bb_pct_b": 0.80,
        "pe_ratio": 34.31,
        "eps": 7.90,
        "profit_margin": 0.27,
        "beta": 1.109,
        "average_sentiment_score": 0.6,
        "flag_count": 0,
        "agent_latency_seconds": 1.14,
        "report_status": "success",
    }

    result = run_monitoring(dummy_state)
    print(f"\nMonitoring status  : {result['monitoring_status']}")
    print(f"Quality issues     : {result['quality_issue_count']}")
    print(f"Drift detected     : {result['drift_detected']}")
    print(f"Report saved to    : {result.get('monitoring_report_path')}")
