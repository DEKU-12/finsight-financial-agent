"""
agent/nodes/fetch_fundamentals.py — Fundamental Financial Data Fetcher

Uses yfinance to retrieve key valuation and profitability metrics for a stock:
  - Valuation    : P/E ratio, Forward P/E, Price-to-Book, EV/EBITDA
  - Profitability: EPS, Profit Margin, Operating Margin
  - Growth       : Revenue (TTM), Revenue per Share
  - Risk         : Beta, Debt-to-Equity
  - Income       : Dividend Yield
  - Analyst      : Target Price

Note: Yahoo Finance often refuses requests from cloud hosts (Render,
Streamlit Cloud), so on those platforms fundamentals may come back empty.

Return contract:
    On success : dict with status="success" and all fundamental fields.
    On no data : dict with status="no_data" and an "error" key.
"""

import logging
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental financial data for a stock ticker from yfinance.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL". Case-insensitive.

    Returns:
        dict with fields described in the module docstring.
        Always check result["status"] before using downstream.
    """
    ticker = ticker.strip().upper()
    logger.info("Fetching fundamentals for %s", ticker)
    return _fetch_yfinance(ticker) or {
        "ticker": ticker,
        "status": "no_data",
        "error": f"No fundamental data available for '{ticker}'.",
    }


def _fetch_yfinance(ticker: str) -> Optional[dict]:
    """Fundamental fields from yfinance. None on failure."""
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

    logger.info("Fundamentals fetched for %s", ticker)
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
    }


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
    import json

    logging.basicConfig(level=logging.INFO)
    result = fetch_fundamentals("AAPL")

    # Trim description for readability
    display = dict(result)
    if display.get("description"):
        display["description"] = display["description"][:80] + "..."

    print(json.dumps(display, indent=2))
