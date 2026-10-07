"""
agent/nodes/fetch_price.py — Stock Price Data Fetcher

Uses yfinance (no API key required) to retrieve:
  - Current price, 52-week high/low, volume
  - 30-day and 200-day moving averages
  - 1-year of daily closing prices and returns
  - Basic company metadata (name, sector, industry, market cap)

This is the FIRST node the LangGraph agent calls. Its output feeds
directly into analyze.py and detect_anomaly.py.

Return contract:
    On success: dict with status="success" and all price fields.
    On error:   dict with status="error" and an "error" key describing what went wrong.
    On no-data: dict with status="no_data".

Always check result["status"] before using the data downstream.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import config

logger = logging.getLogger(__name__)


def fetch_price_data(ticker: str) -> dict:
    """
    Fetch price data from yfinance, falling back to Polygon when Yahoo
    returns nothing (Yahoo often blocks cloud hosts like Streamlit Cloud/Render).
    """
    result = _fetch_yfinance(ticker)
    if result["status"] != "success":
        fallback = _fetch_polygon(result["ticker"])
        if fallback:
            return fallback
    return result


def _fetch_yfinance(ticker: str) -> dict:
    """
    Fetch comprehensive price data for a stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "MSFT", "GOOGL".
                Case-insensitive — will be uppercased internally.

    Returns:
        dict with the following keys on success:
            ticker          (str)   Uppercased ticker symbol
            company_name    (str)   Full company name from Yahoo Finance
            sector          (str)   Sector, e.g. "Technology"
            industry        (str)   Industry, e.g. "Consumer Electronics"
            currency        (str)   Trading currency, e.g. "USD"
            market_cap      (int|None) Market capitalisation in currency units
            current_price   (float|None) Most recent closing / regular market price
            week_52_high    (float|None) Highest price in the past 52 weeks
            week_52_low     (float|None) Lowest price in the past 52 weeks
            avg_volume      (int|None)   3-month average daily volume
            current_volume  (int|None)   Most recent day's volume
            ma_30           (float|None) 30-day simple moving average of Close
            ma_200          (float|None) 200-day simple moving average of Close
            daily_returns   (list[float]) List of daily % returns for past year
            close_prices    (list[float]) List of daily closing prices for past year
            dates           (list[str])  Corresponding dates in "YYYY-MM-DD" format
            status          (str) "success"
    """
    ticker = ticker.strip().upper()
    logger.info("Fetching price data for %s", ticker)

    try:
        stock = yf.Ticker(ticker)

        # ── 1. Company metadata ──────────────────────────────────────────
        info = stock.info

        # yfinance returns an empty dict with a single key if the ticker
        # is invalid. Guard against that before touching any fields.
        if not info or len(info) <= 1:
            logger.warning("No info returned for ticker %s", ticker)
            return {
                "ticker": ticker,
                "status": "no_data",
                "error": f"Ticker '{ticker}' not found on Yahoo Finance.",
            }

        company_name: str = info.get("longName") or info.get("shortName") or ticker
        sector: str = info.get("sector", "Unknown")
        industry: str = info.get("industry", "Unknown")
        currency: str = info.get("currency", "USD")
        market_cap: Optional[int] = info.get("marketCap")

        # ── 2. Current price ─────────────────────────────────────────────
        # Yahoo Finance uses different keys depending on market state.
        current_price: Optional[float] = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        week_52_high: Optional[float] = info.get("fiftyTwoWeekHigh")
        week_52_low: Optional[float] = info.get("fiftyTwoWeekLow")
        avg_volume: Optional[int] = info.get("averageVolume")
        current_volume: Optional[int] = info.get("volume") or info.get(
            "regularMarketVolume"
        )

        # ── 3. Historical price data (1 year) ────────────────────────────
        hist: pd.DataFrame = stock.history(period="1y")

        if hist.empty:
            logger.warning("No historical data returned for %s", ticker)
            return {
                "ticker": ticker,
                "company_name": company_name,
                "status": "no_data",
                "error": f"No historical price data available for '{ticker}'.",
            }

        hist = hist.dropna(subset=["Close"])

        # ── 4. Moving averages ───────────────────────────────────────────
        hist["MA30"] = hist["Close"].rolling(window=30).mean()
        hist["MA200"] = hist["Close"].rolling(window=200).mean()

        ma_30: Optional[float] = _last_valid(hist["MA30"])
        ma_200: Optional[float] = _last_valid(hist["MA200"])

        # ── 5. Daily returns ─────────────────────────────────────────────
        hist["Daily_Return"] = hist["Close"].pct_change()
        hist = hist.dropna(subset=["Daily_Return"])

        daily_returns: list[float] = [
            round(r, 6) for r in hist["Daily_Return"].tolist()
        ]
        close_prices: list[float] = [
            round(p, 4) for p in hist["Close"].tolist()
        ]
        dates: list[str] = [
            str(d.date()) for d in hist.index.tolist()
        ]

        logger.info(
            "Price data fetched for %s: price=%.2f, MA30=%s, MA200=%s",
            ticker,
            current_price or 0.0,
            f"{ma_30:.2f}" if ma_30 else "N/A",
            f"{ma_200:.2f}" if ma_200 else "N/A",
        )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "currency": currency,
            "market_cap": market_cap,
            "current_price": _round(current_price),
            "week_52_high": _round(week_52_high),
            "week_52_low": _round(week_52_low),
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "ma_30": _round(ma_30),
            "ma_200": _round(ma_200),
            "daily_returns": daily_returns,
            "close_prices": close_prices,
            "dates": dates,
            "status": "success",
        }

    except Exception as exc:
        logger.error("Error fetching price data for %s: %s", ticker, exc, exc_info=True)
        return {
            "ticker": ticker,
            "status": "error",
            "error": str(exc),
        }


# ── Polygon fallback ──────────────────────────────────────────────────────────

POLYGON_BASE_URL = "https://api.polygon.io"


def _fetch_polygon(ticker: str) -> Optional[dict]:
    """
    One year of daily prices plus company details from Polygon (Massive).
    Two calls; the free tier allows 5 per minute. Returns None on failure.
    """
    if not config.POLYGON_API_KEY:
        logger.warning("Polygon fallback skipped for %s: POLYGON_API_KEY not set", ticker)
        return None

    today = date.today()
    try:
        aggs = requests.get(
            f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{today - timedelta(days=365)}/{today}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000,
                    "apiKey": config.POLYGON_API_KEY},
            timeout=15,
        ).json()
    except Exception as exc:
        logger.warning("Polygon price request failed for %s: %s", ticker, exc)
        return None

    bars = aggs.get("results")
    if not bars:
        logger.warning(
            "Polygon returned no prices for %s: %s",
            ticker, aggs.get("error") or aggs.get("message") or aggs.get("status"),
        )
        return None

    hist = pd.DataFrame(
        {
            "Close": [b["c"] for b in bars],
            "High": [b["h"] for b in bars],
            "Low": [b["l"] for b in bars],
            "Volume": [b["v"] for b in bars],
        },
        index=pd.to_datetime([b["t"] for b in bars], unit="ms"),
    )

    # Company name and market cap are nice-to-have; prices are enough to go on
    details: dict = {}
    try:
        details = requests.get(
            f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}",
            params={"apiKey": config.POLYGON_API_KEY},
            timeout=15,
        ).json().get("results") or {}
    except Exception as exc:
        logger.warning("Polygon details request failed for %s: %s", ticker, exc)

    returns = hist["Close"].pct_change().dropna()
    closes = hist["Close"].iloc[1:]

    logger.info("Using Polygon prices for %s (%d days)", ticker, len(hist))
    return {
        "ticker": ticker,
        "company_name": details.get("name") or ticker,
        "sector": "Unknown",
        "industry": details.get("sic_description", "Unknown"),
        "currency": (details.get("currency_name") or "usd").upper(),
        "market_cap": details.get("market_cap"),
        "current_price": _round(hist["Close"].iloc[-1]),
        "week_52_high": _round(hist["High"].max()),
        "week_52_low": _round(hist["Low"].min()),
        "avg_volume": int(hist["Volume"].tail(63).mean()),
        "current_volume": int(hist["Volume"].iloc[-1]),
        "ma_30": _round(_last_valid(hist["Close"].rolling(window=30).mean())),
        "ma_200": _round(_last_valid(hist["Close"].rolling(window=200).mean())),
        "daily_returns": [round(r, 6) for r in returns.tolist()],
        "close_prices": [round(p, 4) for p in closes.tolist()],
        "dates": [str(d.date()) for d in closes.index],
        "status": "success",
    }


# ── Private helpers ──────────────────────────────────────────────────────────

def _last_valid(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value from a pandas Series, or None."""
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _round(value: Optional[float], decimals: int = 2) -> Optional[float]:
    """Round a float to `decimals` places, or return None if value is None."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


# ── Quick manual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    result = fetch_price_data("AAPL")

    # Print everything except the long lists to keep output readable
    summary = {k: v for k, v in result.items() if k not in ("daily_returns", "close_prices", "dates")}
    summary["daily_returns_count"] = len(result.get("daily_returns", []))
    summary["close_prices_count"] = len(result.get("close_prices", []))

    print(json.dumps(summary, indent=2))
