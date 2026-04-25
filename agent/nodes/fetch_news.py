"""
agent/nodes/fetch_news.py — News Headlines Fetcher with Sentiment

Uses the NewsAPI /v2/everything endpoint (free tier: 100 requests/day)
to fetch recent news about a company, then classifies each headline's
sentiment using a fast keyword-based approach.

Why keyword sentiment instead of an LLM here?
  - The LLM is reserved for the final report narrative (generate_report.py).
  - Keyword sentiment is deterministic → MLflow experiments are reproducible.
  - It's fast and uses zero API quota.

Sentiment output:
  - Each article gets: "positive", "negative", or "neutral"
  - The result dict includes an average_sentiment_score (-1.0 to +1.0)
    and a sentiment_label ("positive" / "neutral" / "negative").

Return contract:
    On success  : dict with status="success" and all news fields.
    On no news  : dict with status="success", articles=[], article_count=0.
    On error    : dict with status="error" and an "error" key.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import config

logger = logging.getLogger(__name__)

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# Sentiment mapping used to compute the numeric average score
_SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Threshold for labelling the aggregate sentiment
_POSITIVE_THRESHOLD = 0.2
_NEGATIVE_THRESHOLD = -0.2


def fetch_news(company_name: str, ticker: Optional[str] = None) -> dict:
    """
    Fetch recent news headlines for a company and classify their sentiment.

    Args:
        company_name: Human-readable company name, e.g. "Apple" or "Tesla".
                      Used as the primary search query.
        ticker:       Optional ticker symbol, e.g. "AAPL".
                      When provided, broadens the query to catch financial news
                      that uses the ticker rather than the full name.

    Returns:
        dict with:
            company             (str)   The company_name passed in
            ticker              (str|None)
            articles            (list)  Up to 5 processed article dicts — see below
            article_count       (int)   Number of articles returned (0–5)
            average_sentiment_score (float)  Mean of article sentiment scores (-1 to 1)
            sentiment_label     (str)   "positive" | "neutral" | "negative"
            query_used          (str)   The exact search query sent to NewsAPI
            status              (str)   "success" | "error"

        Each article dict contains:
            title           (str)
            description     (str|None)
            source          (str)   Publication name
            published_at    (str)   ISO 8601 datetime string
            url             (str)
            sentiment       (str)   "positive" | "neutral" | "negative"
            sentiment_score (float) 1.0 | 0.0 | -1.0
    """
    logger.info("Fetching news for company='%s' ticker=%s", company_name, ticker)

    # ── Build search query ────────────────────────────────────────────────────
    # Example: "Apple OR AAPL stock" — catches both editorial and financial news
    if ticker:
        query = f'"{company_name}" OR "{ticker}" stock'
    else:
        query = f'"{company_name}"'

    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 5,
        "apiKey": config.NEWS_API_KEY,
    }

    # ── Make request ──────────────────────────────────────────────────────────
    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data: dict = response.json()

    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out for '%s'", company_name)
        return _error_result(company_name, ticker, "Request timed out.")

    except requests.exceptions.RequestException as exc:
        logger.error("Network error fetching news for '%s': %s", company_name, exc)
        return _error_result(company_name, ticker, str(exc))

    # ── Parse response ────────────────────────────────────────────────────────
    if data.get("status") != "ok":
        error_msg = data.get("message", "NewsAPI returned a non-ok status.")
        logger.warning(
            "NewsAPI error for '%s': code=%s message=%s",
            company_name,
            data.get("code", "unknown"),
            error_msg,
        )
        return _error_result(company_name, ticker, error_msg)

    raw_articles: list = data.get("articles", [])

    # ── Process articles ──────────────────────────────────────────────────────
    processed: list[dict] = []
    for article in raw_articles[:5]:
        title: str = article.get("title") or ""
        description: str = article.get("description") or ""
        full_text: str = f"{title} {description}"

        sentiment_label: str = classify_sentiment(full_text)
        sentiment_score: float = _SENTIMENT_SCORE[sentiment_label]

        processed.append(
            {
                "title": title,
                "description": description,
                "source": (article.get("source") or {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", ""),
                "url": article.get("url", ""),
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
            }
        )

    # ── Aggregate sentiment ───────────────────────────────────────────────────
    if processed:
        avg_score: float = sum(a["sentiment_score"] for a in processed) / len(
            processed
        )
    else:
        avg_score = 0.0

    avg_score = round(avg_score, 4)

    if avg_score >= _POSITIVE_THRESHOLD:
        agg_label = "positive"
    elif avg_score <= _NEGATIVE_THRESHOLD:
        agg_label = "negative"
    else:
        agg_label = "neutral"

    logger.info(
        "News fetched for '%s': %d articles, avg_sentiment=%.2f (%s)",
        company_name,
        len(processed),
        avg_score,
        agg_label,
    )

    return {
        "company": company_name,
        "ticker": ticker,
        "articles": processed,
        "article_count": len(processed),
        "average_sentiment_score": avg_score,
        "sentiment_label": agg_label,
        "query_used": query,
        "status": "success",
    }


def classify_sentiment(text: str) -> str:
    """
    Classify text as "positive", "negative", or "neutral" using keyword matching.

    This is intentionally simple and deterministic — good enough for financial
    headline sentiment and ensures MLflow runs are fully reproducible.

    The algorithm:
      1. Tokenise text to lowercase.
      2. Count hits in a curated positive keyword list.
      3. Count hits in a curated negative keyword list.
      4. Return "positive" if pos > neg, "negative" if neg > pos, else "neutral".

    Args:
        text: Any string — typically article title + description concatenated.

    Returns:
        "positive", "negative", or "neutral"
    """
    text_lower = text.lower()

    positive_keywords = [
        # Price / market action
        "surge", "soar", "rally", "rise", "gain", "climb", "jump", "spike",
        "rebound", "recover", "bounce", "breakout", "bullish",
        # Financial performance
        "beat", "exceed", "outperform", "record", "profit", "growth",
        "revenue", "earnings", "strong", "positive", "upgrade", "buy",
        # Business milestones
        "launch", "partnership", "acquisition", "expansion", "innovation",
        "breakthrough", "approval", "win", "award", "deal", "invest",
        "dividend", "buyback", "increase",
    ]

    negative_keywords = [
        # Price / market action
        "fall", "drop", "decline", "plunge", "crash", "tumble", "sink",
        "slip", "lose", "miss", "weak", "bearish", "sell", "downgrade",
        # Financial performance
        "loss", "deficit", "below", "cut", "reduce", "layoff", "restructure",
        "debt", "default", "bankruptcy", "insolvency", "write-off", "impair",
        # Legal / regulatory risk
        "lawsuit", "fine", "penalty", "investigation", "fraud", "scandal",
        "violation", "recall", "warning", "concern", "risk", "uncertainty",
        "delay", "fail", "withdraw",
    ]

    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


# ── Private helpers ───────────────────────────────────────────────────────────

def _error_result(
    company_name: str, ticker: Optional[str], error: str
) -> dict:
    """Return a standardised error result dict."""
    return {
        "company": company_name,
        "ticker": ticker,
        "articles": [],
        "article_count": 0,
        "average_sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "status": "error",
        "error": error,
    }


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # Make sure your .env file has NEWS_API_KEY set before running this.
    result = fetch_news("Apple", ticker="AAPL")

    # Pretty-print without being noisy
    display = dict(result)
    for article in display.get("articles", []):
        article["description"] = (article["description"] or "")[:80]

    print(json.dumps(display, indent=2))
