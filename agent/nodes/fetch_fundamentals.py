"""
agent/nodes/fetch_fundamentals.py — Fundamental Financial Data Fetcher

Uses the Alpha Vantage OVERVIEW endpoint (free tier: 25 calls/day) to retrieve
key valuation and profitability metrics for a stock.

Data returned:
  - Valuation   : P/E ratio, Forward P/E, Price-to-Book, EV/EBITDA
  - Profitability: EPS, Profit Margin, Operating Margin
  - Growth      : Revenue (TTM), Revenue per Share
  - Risk        : Beta, Debt-to-Equity
  - Income      : Dividend Yield
  - Analyst     : Target Price

⚠️  Rate-limit protection:
    Alpha Vantage free tier allows 25 API calls/day.
    Set AV_USE_CACHE=true in your .env during active development to read from
    a local JSON cache (data/av_cache.json) instead of burning your daily quota.
    The cache is keyed by ticker and written automatically after each live call.

Return contract:
    On success      : dict with status="success" and all fundamental fields.
    On rate_limited : dict with status="rate_limited" — use cached/fallback data.
    On error        : dict with status="error" and an "error" key.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import requests
import yfinance as yf

from config import config

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamentals from Alpha Vantage, falling back to yfinance when
    Alpha Vantage returns nothing usable (rate limit, premium-only, no data).
    """
    result = _fetch_alpha_vantage(ticker)
    if result["status"] != "success":
        fallback = _fetch_yfinance(result["ticker"])
        if fallback:
            return fallback
    return result


def _fetch_alpha_vantage(ticker: str) -> dict:
    """
    Fetch fundamental financial data for a stock ticker from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL". Case-insensitive.

    Returns:
        dict with fields described in the module docstring.
        Always check result["status"] before using downstream.
    """
    ticker = ticker.strip().upper()
    logger.info("Fetching fundamentals for %s", ticker)

    # ── Cache read (development mode) ────────────────────────────────────────
    if config.AV_USE_CACHE:
        cached = _read_cache(ticker)
        if cached:
            logger.info("Returning cached fundamentals for %s", ticker)
            cached["status"] = "success"
            cached["from_cache"] = True
            return cached

    # ── Live API call ─────────────────────────────────────────────────────────
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": config.ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(
            ALPHA_VANTAGE_BASE_URL, params=params, timeout=15
        )
        response.raise_for_status()
        data: dict = response.json()

    except requests.exceptions.Timeout:
        logger.error("Alpha Vantage request timed out for %s", ticker)
        return {"ticker": ticker, "status": "error", "error": "Request timed out."}

    except requests.exceptions.RequestException as exc:
        logger.error("Network error fetching fundamentals for %s: %s", ticker, exc)
        return {"ticker": ticker, "status": "error", "error": str(exc)}

    # ── Parse API response ────────────────────────────────────────────────────

    # Rate limit or premium-only: Alpha Vantage returns a "Note" or "Information" key
    if "Note" in data or "Information" in data:
        logger.warning("Alpha Vantage refused %s: %s", ticker, data.get("Note") or data.get("Information"))
        return {
            "ticker": ticker,
            "status": "rate_limited",
            "error": (
                "Alpha Vantage free tier limit reached (25 calls/day). "
                "Set AV_USE_CACHE=true and re-run after 24 hours, "
                "or use cached data."
            ),
        }

    # Invalid ticker: Alpha Vantage returns {"Error Message": "..."}
    if "Error Message" in data:
        logger.warning("Alpha Vantage error for %s: %s", ticker, data["Error Message"])
        return {
            "ticker": ticker,
            "status": "error",
            "error": data["Error Message"],
        }

    # Empty response: ticker exists on some exchanges but AV has no data
    if not data or len(data) < 5:
        logger.warning("Empty or minimal response from Alpha Vantage for %s", ticker)
        return {
            "ticker": ticker,
            "status": "no_data",
            "error": f"Alpha Vantage returned no fundamental data for '{ticker}'.",
        }

    # ── Build result dict ─────────────────────────────────────────────────────
    result = {
        "ticker": ticker,
        "company_name": data.get("Name", ticker),
        "sector": data.get("Sector", "Unknown"),
        "industry": data.get("Industry", "Unknown"),
        "description": data.get("Description", ""),
        "exchange": data.get("Exchange", "Unknown"),
        "currency": data.get("Currency", "USD"),
        "country": data.get("Country", "Unknown"),
        # Valuation
        "pe_ratio": _float(data.get("PERatio")),
        "forward_pe": _float(data.get("ForwardPE")),
        "price_to_book": _float(data.get("PriceToBookRatio")),
        "ev_to_ebitda": _float(data.get("EVToEBITDA")),
        "price_to_sales_ttm": _float(data.get("PriceToSalesRatioTTM")),
        # Earnings & profitability
        "eps": _float(data.get("EPS")),
        "diluted_eps_ttm": _float(data.get("DilutedEPSTTM")),
        "profit_margin": _float(data.get("ProfitMargin")),
        "operating_margin": _float(data.get("OperatingMarginTTM")),
        "return_on_equity": _float(data.get("ReturnOnEquityTTM")),
        "return_on_assets": _float(data.get("ReturnOnAssetsTTM")),
        # Revenue & growth
        "revenue_ttm": _float(data.get("RevenueTTM")),
        "revenue_per_share": _float(data.get("RevenuePerShareTTM")),
        "quarterly_revenue_growth": _float(data.get("QuarterlyRevenueGrowthYOY")),
        "quarterly_earnings_growth": _float(data.get("QuarterlyEarningsGrowthYOY")),
        # Balance sheet risk
        "debt_to_equity": _float(data.get("DebtToEquity")),
        "book_value": _float(data.get("BookValue")),
        "current_ratio": _float(data.get("CurrentRatio")),
        "quick_ratio": _float(data.get("QuickRatio")),
        # Market risk
        "beta": _float(data.get("Beta")),
        "market_cap": _float(data.get("MarketCapitalization")),
        # Income
        "dividend_yield": _float(data.get("DividendYield")),
        "dividend_per_share": _float(data.get("DividendPerShare")),
        # 52-week range (AV version — may differ slightly from yfinance)
        "week_52_high": _float(data.get("52WeekHigh")),
        "week_52_low": _float(data.get("52WeekLow")),
        # Analyst consensus
        "analyst_target_price": _float(data.get("AnalystTargetPrice")),
        "status": "success",
        "from_cache": False,
    }

    logger.info(
        "Fundamentals fetched for %s: PE=%.2f, EPS=%.2f, Beta=%s",
        ticker,
        result["pe_ratio"] or 0.0,
        result["eps"] or 0.0,
        result["beta"],
    )

    # ── Write to cache for future dev runs ────────────────────────────────────
    _write_cache(ticker, result)

    return result


# ── yfinance fallback ─────────────────────────────────────────────────────────

def _fetch_yfinance(ticker: str) -> Optional[dict]:
    """Same fields as the Alpha Vantage result, from yfinance. None on failure."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        logger.warning("yfinance fundamentals failed for %s: %s", ticker, exc)
        return None
    if not info or info.get("trailingPE") is None and info.get("marketCap") is None:
        return None

    def pct_to_ratio(value):
        # yfinance reports these two as percentages (16.97 = 0.1697)
        value = _float(value)
        return value / 100 if value is not None else None

    logger.info("Using yfinance fundamentals for %s", ticker)
    return {
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "description": info.get("longBusinessSummary", ""),
        "exchange": info.get("exchange", "Unknown"),
        "currency": info.get("currency", "USD"),
        "country": info.get("country", "Unknown"),
        "pe_ratio": _float(info.get("trailingPE")),
        "forward_pe": _float(info.get("forwardPE")),
        "price_to_book": _float(info.get("priceToBook")),
        "ev_to_ebitda": _float(info.get("enterpriseToEbitda")),
        "price_to_sales_ttm": _float(info.get("priceToSalesTrailing12Months")),
        "eps": _float(info.get("trailingEps")),
        "diluted_eps_ttm": _float(info.get("trailingEps")),
        "profit_margin": _float(info.get("profitMargins")),
        "operating_margin": _float(info.get("operatingMargins")),
        "return_on_equity": _float(info.get("returnOnEquity")),
        "return_on_assets": _float(info.get("returnOnAssets")),
        "revenue_ttm": _float(info.get("totalRevenue")),
        "revenue_per_share": _float(info.get("revenuePerShare")),
        "quarterly_revenue_growth": _float(info.get("revenueGrowth")),
        "quarterly_earnings_growth": _float(info.get("earningsQuarterlyGrowth")),
        "debt_to_equity": pct_to_ratio(info.get("debtToEquity")),
        "book_value": _float(info.get("bookValue")),
        "current_ratio": _float(info.get("currentRatio")),
        "quick_ratio": _float(info.get("quickRatio")),
        "beta": _float(info.get("beta")),
        "market_cap": _float(info.get("marketCap")),
        "dividend_yield": pct_to_ratio(info.get("dividendYield")),
        "dividend_per_share": _float(info.get("dividendRate")),
        "week_52_high": _float(info.get("fiftyTwoWeekHigh")),
        "week_52_low": _float(info.get("fiftyTwoWeekLow")),
        "analyst_target_price": _float(info.get("targetMeanPrice")),
        "status": "success",
        "from_cache": False,
        "source": "yfinance",
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _read_cache(ticker: str) -> Optional[dict]:
    """Read ticker data from the local JSON cache file."""
    cache_path: Path = config.AV_CACHE_PATH
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache: dict = json.load(f)
        return cache.get(ticker)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read AV cache: %s", exc)
        return None


def _write_cache(ticker: str, data: dict) -> None:
    """Write/update ticker data in the local JSON cache file."""
    cache_path: Path = config.AV_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    cache[ticker] = data
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.debug("AV cache updated for %s", ticker)
    except OSError as exc:
        logger.warning("Could not write AV cache: %s", exc)


# ── Private helper ────────────────────────────────────────────────────────────

def _float(value) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if value is None or value == "None" or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    logging.basicConfig(level=logging.INFO)

    # Make sure your .env file has ALPHA_VANTAGE_API_KEY set before running this.
    result = fetch_fundamentals("AAPL")

    # Trim description for readability
    display = dict(result)
    if display.get("description"):
        display["description"] = display["description"][:80] + "..."

    print(_json.dumps(display, indent=2))
