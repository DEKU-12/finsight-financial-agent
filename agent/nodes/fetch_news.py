"""
agent/nodes/fetch_news.py — News Headlines Fetcher with Sentiment

Uses the NewsAPI /v2/everything endpoint (free tier: 100 requests/day)
to fetch recent news about a company, then classifies each headline's
sentiment using FinBERT — a BERT model fine-tuned on 10,000+ financial
sentences (Malo et al., 2014 / ProsusAI/finbert on HuggingFace).

Classifier selection (automatic):
  1. FinBERT  — used when `transformers` + `torch` are available.
               Best accuracy for financial text. Understands domain-
               specific language and financial context that general-
               purpose tools miss.
  2. VADER    — fallback when FinBERT cannot load (e.g. Streamlit Cloud,
               no torch, insufficient memory). Still far better than
               keyword matching.

Why not an LLM here?
  - The LLM is reserved for the final report narrative (generate_report.py).
  - FinBERT/VADER are deterministic → MLflow experiments are reproducible.
  - They're fast and use zero API quota.

Benchmark history (89 financial headlines, 20 stocks):
  Keyword matching → 58% agreement vs VADER
  VADER            → 39% agreement vs FinBERT  (positivity bias on fin. text)
  FinBERT          → production classifier (domain-tuned gold standard)

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
from datetime import datetime, timedelta, timezone
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


# ── Classifier setup (lazy-loaded, auto-selects FinBERT or VADER) ─────────────

_finbert = None          # HuggingFace pipeline, loaded on first call
_vader   = None          # VADER analyser, loaded as fallback
_classifier_name = None  # "finbert" or "vader" — set once on first call


def _load_classifier():
    """
    Load FinBERT if possible, otherwise fall back to VADER.
    Called once on the first classify_sentiment() call, then cached.
    """
    global _finbert, _vader, _classifier_name

    # ── Try FinBERT ────────────────────────────────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT (ProsusAI/finbert)...")
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        _classifier_name = "finbert"
        logger.info("FinBERT loaded successfully — using as sentiment classifier.")
        return

    except Exception as exc:
        logger.warning(
            "FinBERT unavailable (%s) — falling back to VADER.", exc
        )

    # ── Fall back to VADER ─────────────────────────────────────────────────────
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        _classifier_name = "vader"
        logger.info("VADER loaded as fallback sentiment classifier.")

    except Exception as exc:
        logger.error(
            "Neither FinBERT nor VADER could be loaded: %s. "
            "Sentiment will default to 'neutral'.", exc
        )
        _classifier_name = "none"


def classify_sentiment(text: str) -> tuple[str, float]:
    """
    Classify text sentiment using FinBERT (primary) or VADER (fallback).

    FinBERT (ProsusAI/finbert):
      - Fine-tuned on 10,000+ financial news sentences
      - Understands financial context: "price target raised despite miss"
        is correctly read as mixed/neutral, not positive
      - Returns "positive", "negative", or "neutral" directly

    VADER fallback (Hutto & Gilbert, 2014):
      - Peer-reviewed general-purpose sentiment tool
      - Fast and deterministic; good on general news text
      - Thresholds: compound >= 0.05 → positive, <= -0.05 → negative

    Args:
        text: Any string — typically article title + description concatenated.

    Returns:
        Tuple of (label, score) where:
          label  — "positive" | "negative" | "neutral"
          score  — FinBERT confidence (0–1) or VADER compound (-1 to +1)
    """
    global _finbert, _vader, _classifier_name

    # Load on first call
    if _classifier_name is None:
        _load_classifier()

    # ── FinBERT ────────────────────────────────────────────────────────────────
    if _classifier_name == "finbert" and _finbert is not None:
        try:
            result = _finbert(text[:512])[0]
            label  = result["label"].lower()   # already "positive"/"negative"/"neutral"
            score  = round(float(result["score"]), 4)
            return label, score
        except Exception as exc:
            logger.warning("FinBERT inference failed (%s), using VADER fallback.", exc)

    # ── VADER fallback ─────────────────────────────────────────────────────────
    if _vader is not None:
        compound = _vader.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return label, round(compound, 4)

    # ── Last resort: neutral ───────────────────────────────────────────────────
    return "neutral", 0.0


def get_classifier_name() -> str:
    """Return which classifier is active ('finbert', 'vader', or 'none')."""
    if _classifier_name is None:
        _load_classifier()
    return _classifier_name or "none"


# ── Main fetcher ──────────────────────────────────────────────────────────────

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
            company                 (str)   The company_name passed in
            ticker                  (str|None)
            articles                (list)  Up to 5 processed article dicts
            article_count           (int)   Number of articles returned (0–5)
            average_sentiment_score (float) Mean of article sentiment scores (-1 to 1)
            sentiment_label         (str)   "positive" | "neutral" | "negative"
            sentiment_classifier    (str)   "finbert" or "vader" (which was used)
            query_used              (str)   The exact search query sent to NewsAPI
            status                  (str)   "success" | "error"

        Each article dict contains:
            title               (str)
            description         (str|None)
            source              (str)   Publication name
            published_at        (str)   ISO 8601 datetime string
            url                 (str)
            sentiment           (str)   "positive" | "neutral" | "negative"
            sentiment_score     (float) 1.0 | 0.0 | -1.0
            sentiment_confidence(float) FinBERT confidence or VADER compound
    """
    logger.info("Fetching news for company='%s' ticker=%s", company_name, ticker)

    # ── Build search query ────────────────────────────────────────────────────
    # Use AND with financial terms to avoid matching unrelated uses of the
    # company name or ticker (e.g. "Apple" matching food articles, "stock"
    # matching "stock car" racing results).
    if ticker:
        query = (
            f'("{company_name}" OR "{ticker}")'
            f' AND (stock OR shares OR earnings OR investor OR market)'
        )
    else:
        query = (
            f'"{company_name}"'
            f' AND (stock OR shares OR earnings OR investor OR market)'
        )

    from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 5,
        "apiKey":   config.NEWS_API_KEY,
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

    except Exception as exc:
        logger.error("Unexpected error fetching news for '%s': %s", company_name, exc)
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
        title:       str = article.get("title") or ""
        description: str = article.get("description") or ""
        full_text:   str = f"{title} {description}"

        sentiment_label, sentiment_confidence = classify_sentiment(full_text)
        sentiment_score: float = _SENTIMENT_SCORE[sentiment_label]

        processed.append({
            "title":                title,
            "description":          description,
            "source":               (article.get("source") or {}).get("name", "Unknown"),
            "published_at":         article.get("publishedAt", ""),
            "url":                  article.get("url", ""),
            "sentiment":            sentiment_label,
            "sentiment_score":      sentiment_score,
            "sentiment_confidence": sentiment_confidence,
        })

    # ── Aggregate sentiment ───────────────────────────────────────────────────
    if processed:
        avg_score: float = sum(a["sentiment_score"] for a in processed) / len(processed)
    else:
        avg_score = 0.0

    avg_score = round(avg_score, 4)

    if avg_score >= _POSITIVE_THRESHOLD:
        agg_label = "positive"
    elif avg_score <= _NEGATIVE_THRESHOLD:
        agg_label = "negative"
    else:
        agg_label = "neutral"

    classifier_used = get_classifier_name()

    logger.info(
        "News fetched for '%s': %d articles, avg_sentiment=%.2f (%s) via %s",
        company_name, len(processed), avg_score, agg_label, classifier_used,
    )

    return {
        "company":                company_name,
        "ticker":                 ticker,
        "articles":               processed,
        "article_count":          len(processed),
        "average_sentiment_score": avg_score,
        "sentiment_label":        agg_label,
        "sentiment_classifier":   classifier_used,
        "query_used":             query,
        "status":                 "success",
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _error_result(company_name: str, ticker: Optional[str], error: str) -> dict:
    """Return a standardised error result dict."""
    return {
        "company":                company_name,
        "ticker":                 ticker,
        "articles":               [],
        "article_count":          0,
        "average_sentiment_score": 0.0,
        "sentiment_label":        "neutral",
        "sentiment_classifier":   "none",
        "status":                 "error",
        "error":                  error,
    }


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    result = fetch_news("Apple", ticker="AAPL")

    display = dict(result)
    for article in display.get("articles", []):
        article["description"] = (article["description"] or "")[:80]

    print(json.dumps(display, indent=2))
    print(f"\nClassifier used: {result.get('sentiment_classifier')}")
