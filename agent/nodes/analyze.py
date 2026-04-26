"""
agent/nodes/analyze.py — Quantitative Analysis Engine

Takes the raw price data returned by fetch_price.py and computes the
technical indicators that traders and quant analysts actually use:

  - RSI (Relative Strength Index, 14-day)
      Momentum oscillator: 0–100 scale.
      > 70  = overbought (stock may be due for a pullback)
      < 30  = oversold   (stock may be undervalued / due for a bounce)
      30–70 = neutral zone

  - Bollinger Bands (20-day, 2 std devs)
      A price envelope around a 20-day moving average.
      When price touches the upper band → stretched high.
      When price touches the lower band → stretched low.
      Band width tells you how volatile the market is right now.

  - Volatility (30-day annualised)
      Standard deviation of daily returns × √252 (trading days/year).
      High volatility = risky / uncertain.
      Low volatility  = calm / stable.

  - Momentum (10-day price rate of change)
      How fast is the price moving right now?
      Positive = accelerating upward, negative = accelerating downward.

  - Price vs MA signals
      Is price above or below its 30-day and 200-day moving averages?
      Price > MA200 = long-term uptrend (bullish).
      Price < MA200 = long-term downtrend (bearish).
      MA30 crossing above MA200 = "Golden Cross" (strong buy signal).
      MA30 crossing below MA200 = "Death Cross" (strong sell signal).

All calculations are pure numpy/pandas — deterministic and reproducible.
No API calls. No randomness.

Input:  the dict returned by fetch_price.fetch_price_data()
Output: a new dict with all analysis results, ready for detect_anomaly.py
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


def analyze(price_data: dict) -> dict:
    """
    Run all technical analysis on price data from fetch_price_data().

    Args:
        price_data: dict returned by fetch_price.fetch_price_data().
                    Must have status="success" and contain close_prices
                    and daily_returns lists.

    Returns:
        dict with all analysis results. Merges with price_data so
        downstream nodes receive a single unified state dict.
        Adds an "analysis_status" key: "success" | "error" | "insufficient_data".
    """
    ticker = price_data.get("ticker", "UNKNOWN")
    logger.info("Running analysis for %s", ticker)

    # ── Guard: only analyse if we actually have price data ───────────────────
    # We check for close_prices directly rather than relying on the "status"
    # key, because downstream nodes (e.g. fetch_news) may overwrite "status"
    # in the shared LangGraph state dict with their own error status.
    close_prices: list = price_data.get("close_prices", [])
    daily_returns: list = price_data.get("daily_returns", [])
    if not close_prices:
        logger.warning("Skipping analysis for %s — no close_prices in state", ticker)
        return {**price_data, "analysis_status": "skipped"}

    if len(close_prices) < 30:
        logger.warning("Insufficient price history for %s (%d days)", ticker, len(close_prices))
        return {
            **price_data,
            "analysis_status": "insufficient_data",
            "analysis_error": f"Need at least 30 days of price data, got {len(close_prices)}.",
        }

    closes = pd.Series(close_prices, dtype=float)
    returns = pd.Series(daily_returns, dtype=float)

    try:
        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_value = _compute_rsi(closes, period=config.RSI_PERIOD)
        rsi_signal = _rsi_signal(rsi_value)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb = _compute_bollinger_bands(closes, window=config.BB_WINDOW, num_std=config.BB_STD)

        # ── Volatility (annualised) ───────────────────────────────────────────
        volatility_30d = _compute_volatility(returns, window=30)
        volatility_signal = _volatility_signal(volatility_30d)

        # ── Momentum (10-day rate of change) ─────────────────────────────────
        momentum_10d = _compute_momentum(closes, period=10)
        momentum_signal = "bullish" if momentum_10d > 0 else "bearish"

        # ── Price vs Moving Average signals ───────────────────────────────────
        current_price: float = price_data.get("current_price") or closes.iloc[-1]
        ma_30: Optional[float] = price_data.get("ma_30")
        ma_200: Optional[float] = price_data.get("ma_200")

        price_vs_ma30 = _price_vs_ma(current_price, ma_30, "MA30")
        price_vs_ma200 = _price_vs_ma(current_price, ma_200, "MA200")
        ma_cross_signal = _golden_death_cross(ma_30, ma_200)

        # ── 52-week position ──────────────────────────────────────────────────
        week_52_high: Optional[float] = price_data.get("week_52_high")
        week_52_low: Optional[float] = price_data.get("week_52_low")
        pct_from_52w_high = _pct_from(current_price, week_52_high)
        pct_from_52w_low = _pct_from(current_price, week_52_low)

        logger.info(
            "Analysis complete for %s: RSI=%.1f (%s), Vol=%.1f%%, BB=%s, Mom=%s",
            ticker,
            rsi_value,
            rsi_signal,
            volatility_30d * 100,
            bb["bb_signal"],
            momentum_signal,
        )

        return {
            # Pass through everything from price_data
            **price_data,
            # RSI
            "rsi_14": round(rsi_value, 2),
            "rsi_signal": rsi_signal,
            # Bollinger Bands
            "bb_upper": bb["upper"],
            "bb_middle": bb["middle"],
            "bb_lower": bb["lower"],
            "bb_width": bb["width"],
            "bb_signal": bb["bb_signal"],
            "bb_pct_b": bb["pct_b"],
            # Volatility
            "volatility_30d": round(volatility_30d, 4),
            "volatility_30d_pct": round(volatility_30d * 100, 2),
            "volatility_signal": volatility_signal,
            # Momentum
            "momentum_10d": round(momentum_10d, 4),
            "momentum_10d_pct": round(momentum_10d * 100, 2),
            "momentum_signal": momentum_signal,
            # MA signals
            "price_vs_ma30": price_vs_ma30,
            "price_vs_ma200": price_vs_ma200,
            "ma_cross_signal": ma_cross_signal,
            # 52-week position
            "pct_from_52w_high": round(pct_from_52w_high, 2) if pct_from_52w_high is not None else None,
            "pct_from_52w_low": round(pct_from_52w_low, 2) if pct_from_52w_low is not None else None,
            # Status
            "analysis_status": "success",
        }

    except Exception as exc:
        logger.error("Analysis failed for %s: %s", ticker, exc, exc_info=True)
        return {**price_data, "analysis_status": "error", "analysis_error": str(exc)}


# ── Technical indicator implementations ──────────────────────────────────────

def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing method.

    RSI = 100 - (100 / (1 + RS))
    where RS = average gain / average loss over `period` days.

    Wilder's smoothing uses exponential weighting (com = period - 1),
    which is the industry standard used by Bloomberg, TradingView, etc.
    """
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Avoid division by zero when avg_loss is 0 (pure uptrend)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1])


def _rsi_signal(rsi: float) -> str:
    """Translate RSI value into a human-readable signal."""
    if rsi >= 70:
        return "overbought"
    elif rsi <= 30:
        return "oversold"
    elif rsi >= 55:
        return "bullish"
    elif rsi <= 45:
        return "bearish"
    else:
        return "neutral"


def _compute_bollinger_bands(
    closes: pd.Series, window: int = 20, num_std: float = 2.0
) -> dict:
    """
    Compute Bollinger Bands.

    Middle band  = simple moving average over `window` days
    Upper band   = middle + num_std × rolling std
    Lower band   = middle - num_std × rolling std

    %B = (price - lower) / (upper - lower)
         > 1.0 = price above upper band (stretched high)
         < 0.0 = price below lower band (stretched low)
         = 0.5 = price at middle band

    Band width = (upper - lower) / middle  (normalised)
    """
    rolling_mean = closes.rolling(window=window).mean()
    rolling_std = closes.rolling(window=window).std()

    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std

    current_price = float(closes.iloc[-1])
    upper_val = float(upper.iloc[-1])
    lower_val = float(lower.iloc[-1])
    middle_val = float(rolling_mean.iloc[-1])

    band_range = upper_val - lower_val
    pct_b = (current_price - lower_val) / band_range if band_range != 0 else 0.5
    bb_width = band_range / middle_val if middle_val != 0 else 0.0

    # Signal
    if pct_b > 1.0:
        bb_signal = "above_upper_band"
    elif pct_b < 0.0:
        bb_signal = "below_lower_band"
    elif pct_b > 0.8:
        bb_signal = "near_upper_band"
    elif pct_b < 0.2:
        bb_signal = "near_lower_band"
    else:
        bb_signal = "within_bands"

    return {
        "upper": round(upper_val, 2),
        "middle": round(middle_val, 2),
        "lower": round(lower_val, 2),
        "width": round(bb_width, 4),
        "pct_b": round(pct_b, 4),
        "bb_signal": bb_signal,
    }


def _compute_volatility(returns: pd.Series, window: int = 30) -> float:
    """
    Compute annualised volatility as the rolling standard deviation of
    daily returns, scaled by √252 (number of trading days in a year).

    A volatility of 0.20 means 20% annualised — typical for large-cap stocks.
    Values > 0.40 are considered high risk.
    """
    recent_returns = returns.tail(window).dropna()
    if len(recent_returns) < 5:
        return 0.0
    daily_std = float(recent_returns.std())
    return daily_std * math.sqrt(252)


def _volatility_signal(volatility: float) -> str:
    """Translate annualised volatility into a risk label."""
    if volatility < 0.15:
        return "low"
    elif volatility < 0.30:
        return "moderate"
    elif volatility < 0.50:
        return "high"
    else:
        return "very_high"


def _compute_momentum(closes: pd.Series, period: int = 10) -> float:
    """
    Compute price momentum as the Rate of Change (ROC) over `period` days.

    ROC = (current_price - price_n_days_ago) / price_n_days_ago

    Positive ROC = price is higher than it was `period` days ago (upward momentum).
    Negative ROC = price is lower (downward momentum).
    """
    if len(closes) <= period:
        return 0.0
    price_now = float(closes.iloc[-1])
    price_then = float(closes.iloc[-(period + 1)])
    if price_then == 0:
        return 0.0
    return (price_now - price_then) / price_then


def _price_vs_ma(price: float, ma: Optional[float], ma_name: str) -> str:
    """Return whether price is above or below a given moving average."""
    if ma is None:
        return f"insufficient_data_for_{ma_name}"
    diff_pct = ((price - ma) / ma) * 100
    if diff_pct > 5:
        return f"well_above_{ma_name}"
    elif diff_pct > 0:
        return f"above_{ma_name}"
    elif diff_pct > -5:
        return f"below_{ma_name}"
    else:
        return f"well_below_{ma_name}"


def _golden_death_cross(ma_30: Optional[float], ma_200: Optional[float]) -> str:
    """
    Detect Golden Cross or Death Cross based on MA30 vs MA200.

    Golden Cross: MA30 > MA200 — long-term bullish signal.
    Death Cross:  MA30 < MA200 — long-term bearish signal.
    """
    if ma_30 is None or ma_200 is None:
        return "insufficient_data"
    if ma_30 > ma_200 * 1.01:
        return "golden_cross"
    elif ma_30 < ma_200 * 0.99:
        return "death_cross"
    else:
        return "neutral"


def _pct_from(price: float, reference: Optional[float]) -> Optional[float]:
    """Return percentage difference between price and a reference level."""
    if reference is None or reference == 0:
        return None
    return ((price - reference) / reference) * 100


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys
    import os

    logging.basicConfig(level=logging.INFO)

    # Add project root to path so imports work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from agent.nodes.fetch_price import fetch_price_data

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    price_data = fetch_price_data(ticker)
    result = analyze(price_data)

    # Print just the analysis fields (skip the long lists)
    skip = {"close_prices", "daily_returns", "dates"}
    display = {k: v for k, v in result.items() if k not in skip}
    print(json.dumps(display, indent=2))
