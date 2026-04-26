"""
agent/nodes/detect_anomaly.py — Anomaly & Risk Flag Detector

Examines the combined price + analysis data and raises flags when
something looks unusual or risky. This is what makes FinSight more
than just a data dashboard — it actively alerts on warning signs.

Flags raised by this module:

  PRICE FLAGS
  ├── price_near_52w_high     Price is within 3% of its 52-week high
  ├── price_near_52w_low      Price is within 3% of its 52-week low
  └── price_outside_bb        Price has broken outside Bollinger Bands

  MOMENTUM & TREND FLAGS
  ├── rsi_overbought          RSI > 70 — momentum stretched high
  ├── rsi_oversold            RSI < 30 — momentum stretched low
  ├── death_cross             MA30 crossed below MA200 — bearish long-term
  └── strong_negative_momentum  10-day momentum < -5%

  VOLATILITY FLAGS
  └── high_volatility         Annualised volatility > 40%

  VOLUME FLAGS
  └── volume_spike            Current volume > 2× average volume

  FUNDAMENTAL FLAGS (needs data from fetch_fundamentals)
  ├── negative_eps            Company is losing money (EPS < 0)
  ├── high_pe_ratio           P/E > 50 — expensive relative to earnings
  ├── high_debt               Debt-to-equity > 2.0 — heavily leveraged
  └── negative_profit_margin  Company's revenue isn't covering costs

Each flag is a dict with: name, severity ("low"|"medium"|"high"), description.
The overall anomaly_detected field is 1 if any HIGH severity flag fires, else 0.
This is what gets logged to MLflow as a metric.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Severity levels
HIGH = "high"
MEDIUM = "medium"
LOW = "low"


def detect_anomaly(state: dict) -> dict:
    """
    Scan state dict for anomalies and risk signals.

    Args:
        state: Combined dict from fetch_price + fetch_fundamentals +
               fetch_news + analyze. All keys are optional — the function
               gracefully skips checks when data is missing.

    Returns:
        The input state dict augmented with:
            flags               list[dict]  All raised flags
            flag_count          int         Total number of flags
            high_severity_count int         Number of HIGH severity flags
            anomaly_detected    int         1 if any HIGH flag fired, else 0
            risk_level          str         "low" | "medium" | "high" | "critical"
            anomaly_summary     str         Human-readable summary of flags
            anomaly_status      str         "success" | "error"
    """
    ticker = state.get("ticker", "UNKNOWN")
    logger.info("Running anomaly detection for %s", ticker)

    flags: list[dict] = []

    try:
        # ── Price flags ───────────────────────────────────────────────────────
        _check_52w_proximity(state, flags)
        _check_bollinger_breakout(state, flags)

        # ── Momentum & trend flags ────────────────────────────────────────────
        _check_rsi_extremes(state, flags)
        _check_ma_cross(state, flags)
        _check_momentum(state, flags)

        # ── Volatility flags ──────────────────────────────────────────────────
        _check_volatility(state, flags)

        # ── Volume flags ──────────────────────────────────────────────────────
        _check_volume_spike(state, flags)

        # ── Fundamental flags ─────────────────────────────────────────────────
        _check_fundamentals(state, flags)

        # ── Aggregate results ─────────────────────────────────────────────────
        high_flags = [f for f in flags if f["severity"] == HIGH]
        medium_flags = [f for f in flags if f["severity"] == MEDIUM]
        anomaly_detected = 1 if high_flags else 0

        # Risk level
        if len(high_flags) >= 3:
            risk_level = "critical"
        elif len(high_flags) >= 1:
            risk_level = "high"
        elif len(medium_flags) >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Human-readable summary
        if flags:
            flag_names = ", ".join(f["name"] for f in flags)
            anomaly_summary = f"{len(flags)} flag(s) raised: {flag_names}."
        else:
            anomaly_summary = "No anomalies detected. All indicators within normal ranges."

        logger.info(
            "Anomaly detection complete for %s: %d flags (%d high), risk=%s",
            ticker,
            len(flags),
            len(high_flags),
            risk_level,
        )

        return {
            **state,
            "flags": flags,
            "flag_count": len(flags),
            "high_severity_count": len(high_flags),
            "anomaly_detected": anomaly_detected,
            "risk_level": risk_level,
            "anomaly_summary": anomaly_summary,
            "anomaly_status": "success",
        }

    except Exception as exc:
        logger.error("Anomaly detection failed for %s: %s", ticker, exc, exc_info=True)
        return {
            **state,
            "flags": [],
            "flag_count": 0,
            "high_severity_count": 0,
            "anomaly_detected": 0,
            "risk_level": "unknown",
            "anomaly_summary": f"Anomaly detection error: {exc}",
            "anomaly_status": "error",
        }


# ── Individual flag checks ────────────────────────────────────────────────────

def _check_52w_proximity(state: dict, flags: list) -> None:
    """Flag if price is near the 52-week high or low."""
    price = state.get("current_price")
    high_52 = state.get("week_52_high")
    low_52 = state.get("week_52_low")

    if price and high_52 and high_52 > 0:
        pct_from_high = abs((price - high_52) / high_52) * 100
        if pct_from_high <= 3.0:
            flags.append(_flag(
                name="price_near_52w_high",
                severity=MEDIUM,
                description=(
                    f"Price (${price:.2f}) is within {pct_from_high:.1f}% of its "
                    f"52-week high (${high_52:.2f}). Stock may be overextended."
                ),
            ))

    if price and low_52 and low_52 > 0:
        pct_from_low = abs((price - low_52) / low_52) * 100
        if pct_from_low <= 3.0:
            flags.append(_flag(
                name="price_near_52w_low",
                severity=HIGH,
                description=(
                    f"Price (${price:.2f}) is within {pct_from_low:.1f}% of its "
                    f"52-week low (${low_52:.2f}). Stock is in significant distress."
                ),
            ))


def _check_bollinger_breakout(state: dict, flags: list) -> None:
    """Flag if price has broken outside Bollinger Bands."""
    bb_signal = state.get("bb_signal")
    if bb_signal == "above_upper_band":
        flags.append(_flag(
            name="price_outside_bb_upper",
            severity=MEDIUM,
            description=(
                "Price has broken above the upper Bollinger Band. "
                "Statistically unusual — often precedes a reversion to the mean."
            ),
        ))
    elif bb_signal == "below_lower_band":
        flags.append(_flag(
            name="price_outside_bb_lower",
            severity=HIGH,
            description=(
                "Price has broken below the lower Bollinger Band. "
                "Significant downward pressure — may indicate panic selling or bad news."
            ),
        ))


def _check_rsi_extremes(state: dict, flags: list) -> None:
    """Flag RSI overbought/oversold conditions."""
    rsi = state.get("rsi_14")
    if rsi is None:
        return
    if rsi > 70:
        flags.append(_flag(
            name="rsi_overbought",
            severity=MEDIUM,
            description=(
                f"RSI is {rsi:.1f} — above the overbought threshold of 70. "
                "Buying momentum is stretched. Risk of short-term pullback."
            ),
        ))
    elif rsi < 30:
        flags.append(_flag(
            name="rsi_oversold",
            severity=MEDIUM,
            description=(
                f"RSI is {rsi:.1f} — below the oversold threshold of 30. "
                "Selling momentum is extreme. May signal a buying opportunity or continued decline."
            ),
        ))


def _check_ma_cross(state: dict, flags: list) -> None:
    """Flag Death Cross (bearish long-term signal)."""
    cross = state.get("ma_cross_signal")
    if cross == "death_cross":
        flags.append(_flag(
            name="death_cross",
            severity=HIGH,
            description=(
                "The 30-day moving average has crossed below the 200-day moving average "
                "(Death Cross). This is a widely watched bearish signal indicating a potential "
                "long-term downtrend."
            ),
        ))


def _check_momentum(state: dict, flags: list) -> None:
    """Flag strong negative momentum."""
    mom = state.get("momentum_10d")
    if mom is not None and mom < -0.05:
        flags.append(_flag(
            name="strong_negative_momentum",
            severity=HIGH,
            description=(
                f"10-day price momentum is {mom * 100:.1f}% — stock has lost more than 5% "
                "in the past 10 trading days. Sustained selling pressure detected."
            ),
        ))


def _check_volatility(state: dict, flags: list) -> None:
    """Flag high annualised volatility."""
    vol = state.get("volatility_30d")
    if vol is not None and vol > 0.40:
        flags.append(_flag(
            name="high_volatility",
            severity=MEDIUM,
            description=(
                f"Annualised volatility is {vol * 100:.1f}% — significantly above the "
                "typical 20–30% range for large-cap stocks. Higher risk environment."
            ),
        ))


def _check_volume_spike(state: dict, flags: list) -> None:
    """Flag if today's volume is more than 2× the average."""
    current_vol = state.get("current_volume")
    avg_vol = state.get("avg_volume")
    if current_vol and avg_vol and avg_vol > 0:
        ratio = current_vol / avg_vol
        if ratio > 2.0:
            flags.append(_flag(
                name="volume_spike",
                severity=MEDIUM,
                description=(
                    f"Today's volume ({current_vol:,}) is {ratio:.1f}× the 3-month average "
                    f"({avg_vol:,}). Unusual trading activity — may signal institutional "
                    "buying/selling or an imminent news event."
                ),
            ))


def _check_fundamentals(state: dict, flags: list) -> None:
    """Flag fundamental warning signs from Alpha Vantage data."""
    eps = state.get("eps")
    pe = state.get("pe_ratio")
    debt_to_equity = state.get("debt_to_equity")
    profit_margin = state.get("profit_margin")

    if eps is not None and eps < 0:
        flags.append(_flag(
            name="negative_eps",
            severity=HIGH,
            description=(
                f"Earnings per share is ${eps:.2f} — the company is currently unprofitable. "
                "Negative EPS means losses are being passed to shareholders."
            ),
        ))

    if pe is not None and pe > 50:
        flags.append(_flag(
            name="high_pe_ratio",
            severity=MEDIUM,
            description=(
                f"P/E ratio is {pe:.1f} — significantly above the market average of ~20. "
                "The stock is priced for very high future growth. Any disappointment could "
                "cause a sharp correction."
            ),
        ))

    if debt_to_equity is not None and debt_to_equity > 2.0:
        flags.append(_flag(
            name="high_debt_to_equity",
            severity=HIGH,
            description=(
                f"Debt-to-equity ratio is {debt_to_equity:.2f} — the company carries "
                "more than twice as much debt as equity. High leverage increases risk "
                "during economic downturns."
            ),
        ))

    if profit_margin is not None and profit_margin < 0:
        flags.append(_flag(
            name="negative_profit_margin",
            severity=HIGH,
            description=(
                f"Profit margin is {profit_margin * 100:.1f}% — the company is spending "
                "more than it earns. Sustained negative margins lead to cash burn and "
                "potential solvency risk."
            ),
        ))


# ── Helper ────────────────────────────────────────────────────────────────────

def _flag(name: str, severity: str, description: str) -> dict:
    """Create a standardised flag dict."""
    return {"name": name, "severity": severity, "description": description}


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from agent.nodes.fetch_price import fetch_price_data
    from agent.nodes.fetch_fundamentals import fetch_fundamentals
    from agent.nodes.analyze import analyze

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    state = fetch_price_data(ticker)
    state.update(fetch_fundamentals(ticker))
    state = analyze(state)
    result = detect_anomaly(state)

    skip = {"close_prices", "daily_returns", "dates", "description"}
    display = {k: v for k, v in result.items() if k not in skip}
    print(json.dumps(display, indent=2, default=str))
