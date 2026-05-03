"""
scripts/measure_sentiment_accuracy.py — FinSight Sentiment Accuracy Measurement

Measures how accurately our VADER-based sentiment classifier labels
financial news headlines, using FinBERT as the reference benchmark.

Why FinBERT as reference?
  - Fine-tuned on 10,000+ financial news sentences (Malo et al., 2014)
  - The gold standard for financial NLP sentiment classification
  - Understands financial context that general-purpose tools miss

Why VADER as our production classifier?
  - Zero latency (no model download, no GPU needed)
  - Fully deterministic → MLflow runs are reproducible
  - Peer-reviewed (Hutto & Gilbert, 2014)
  - Previously validated: improved from 58% (keyword) to this level vs VADER,
    and now re-validated here against FinBERT

How it works:
  1. Pull up to 5 headlines per ticker via NewsAPI for 20 large-cap stocks.
  2. Run our VADER classifier on every headline.
  3. Run FinBERT on every headline (downloads ~400MB model on first run).
  4. Compute agreement rate = labels that match / total headlines.

Run:
    pip install transformers torch vaderSentiment
    python scripts/measure_sentiment_accuracy.py

Note: First run downloads the FinBERT model (~400MB). Subsequent runs use
the cached model and are much faster.

Output:
    Agreement report printed to console + saved as sentiment_accuracy_report.txt
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter

# ── Add project root ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("ERROR: vaderSentiment not installed. Run:  pip install vaderSentiment")
    sys.exit(1)

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    print("ERROR: transformers not installed. Run:  pip install transformers torch")
    sys.exit(1)

import requests
from config import config
from agent.nodes.fetch_news import classify_sentiment

# ── Settings ──────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "TSLA", "AMZN", "JPM",  "JNJ",  "SPY",
    "QQQ",  "BAC",  "PFE",  "COIN", "GS",
    "NFLX", "AMD",  "INTC", "DIS",  "UNH",
]

HEADLINES_PER_TICKER = 5

# FinBERT label map → our 3-class schema
_FINBERT_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
}


# ── Load FinBERT ──────────────────────────────────────────────────────────────
print("Loading FinBERT model (first run downloads ~400MB)...")
try:
    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )
    print("FinBERT loaded.\n")
except Exception as e:
    print(f"ERROR loading FinBERT: {e}")
    print("Make sure you have an internet connection and torch installed.")
    sys.exit(1)


def finbert_label(text: str) -> str:
    """Return FinBERT's 3-class label for a piece of financial text."""
    try:
        result = finbert(text[:512])[0]
        raw_label = result["label"].lower()
        return _FINBERT_LABEL_MAP.get(raw_label, "neutral")
    except Exception:
        return "neutral"


# ── NewsAPI fetcher ───────────────────────────────────────────────────────────

def fetch_headlines(ticker: str) -> list[str]:
    """Fetch up to HEADLINES_PER_TICKER headlines for a ticker via NewsAPI."""
    from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "q":        f'"{ticker}" stock',
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": HEADLINES_PER_TICKER,
        "apiKey":   config.NEWS_API_KEY,
    }
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=15
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        texts = []
        for a in articles[:HEADLINES_PER_TICKER]:
            title = a.get("title") or ""
            desc  = a.get("description") or ""
            combined = f"{title} {desc}".strip()
            if combined:
                texts.append(combined)
        return texts
    except Exception as e:
        print(f"  ⚠ NewsAPI error for {ticker}: {e}")
        return []


# ── Main report ───────────────────────────────────────────────────────────────

def run_sentiment_report():
    all_results: list[dict] = []
    skipped_tickers = []

    print("\n" + "=" * 70)
    print("  FinSight Sentiment Classifier Accuracy Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Our classifier: VADER (Hutto & Gilbert, 2014)")
    print("  Reference:      FinBERT (Malo et al., 2014 — financial NLP gold standard)")
    print("=" * 70)
    print(f"\n{'Ticker':<8} {'Headlines':>10} {'Agreement':>11} {'Ours (VADER)':>16}  {'FinBERT':>16}")
    print("-" * 68)

    for ticker in TICKERS:
        headlines = fetch_headlines(ticker)
        if not headlines:
            skipped_tickers.append(ticker)
            print(f"{ticker:<8} {'NO DATA':>10}")
            continue

        ticker_results = []
        for text in headlines:
            our_label, _ = classify_sentiment(text)
            ref_label     = finbert_label(text)
            match         = our_label == ref_label
            ticker_results.append({
                "ticker":    ticker,
                "text":      text[:80],
                "our_label": our_label,
                "ref_label": ref_label,
                "match":     match,
            })

        all_results.extend(ticker_results)

        n       = len(ticker_results)
        matches = sum(r["match"] for r in ticker_results)
        pct     = matches / n * 100

        our_dist = Counter(r["our_label"] for r in ticker_results)
        ref_dist = Counter(r["ref_label"] for r in ticker_results)
        our_str  = f"P{our_dist['positive']} N{our_dist['negative']} ={our_dist['neutral']}"
        ref_str  = f"P{ref_dist['positive']} N{ref_dist['negative']} ={ref_dist['neutral']}"

        print(f"{ticker:<8} {n:>10}  {pct:>9.1f}%  {our_str:>16}  {ref_str:>16}")

    print("-" * 68)

    if not all_results:
        print("\n❌ No headlines fetched. Check your NEWS_API_KEY in .env")
        return

    total   = len(all_results)
    matches = sum(r["match"] for r in all_results)
    overall = matches / total * 100

    our_dist = Counter(r["our_label"] for r in all_results)
    ref_dist = Counter(r["ref_label"] for r in all_results)

    print(f"\n{'SUMMARY':}")
    print(f"  Total headlines analysed  : {total}")
    print(f"  Matching labels           : {matches}")
    print(f"  Overall agreement rate    : {overall:.1f}%")
    print(f"\n  VADER (ours) — positive: {our_dist['positive']}  "
          f"negative: {our_dist['negative']}  neutral: {our_dist['neutral']}")
    print(f"  FinBERT (ref) — positive: {ref_dist['positive']}  "
          f"negative: {ref_dist['negative']}  neutral: {ref_dist['neutral']}")

    if skipped_tickers:
        print(f"\n  Tickers skipped (no data): {', '.join(skipped_tickers)}")

    print("=" * 70)

    # ── Sample disagreements ──────────────────────────────────────────────────
    disagreements = [r for r in all_results if not r["match"]]
    if disagreements:
        print(f"\n🔍 Sample disagreements (VADER → FinBERT):")
        for r in disagreements[:5]:
            print(f"  [{r['our_label']:8s} → {r['ref_label']:8s}]  {r['text'][:70]}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = Path(__file__).parent.parent / "sentiment_accuracy_report.txt"
    with open(report_path, "w") as f:
        f.write(f"FinSight Sentiment Accuracy Report — {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Our classifier: VADER (Hutto & Gilbert, 2014)\n")
        f.write(f"Reference:      FinBERT (ProsusAI/finbert)\n")
        f.write(f"Headlines analysed: {total}\n")
        f.write(f"Overall agreement rate: {overall:.1f}%\n\n")
        f.write(f"VADER distribution:\n")
        f.write(f"  positive: {our_dist['positive']}\n")
        f.write(f"  negative: {our_dist['negative']}\n")
        f.write(f"  neutral:  {our_dist['neutral']}\n\n")
        f.write(f"FinBERT distribution:\n")
        f.write(f"  positive: {ref_dist['positive']}\n")
        f.write(f"  negative: {ref_dist['negative']}\n")
        f.write(f"  neutral:  {ref_dist['neutral']}\n\n")
        f.write("Per-headline results:\n")
        for r in all_results:
            status = "✓" if r["match"] else "✗"
            f.write(
                f"  {status} [{r['ticker']}] vader={r['our_label']:8s} "
                f"finbert={r['ref_label']:8s}  {r['text'][:70]}\n"
            )

    print(f"\n✅ Report saved to: sentiment_accuracy_report.txt")
    print(f"\n📋 CV-ready statement:")
    print(f'   "Upgraded sentiment classifier from keyword matching to VADER;')
    print(f'    re-validated against FinBERT (financial NLP gold standard) across')
    print(f'    {total} real headlines from {len(TICKERS) - len(skipped_tickers)} stocks,')
    print(f'    achieving {overall:.0f}% label agreement."')


if __name__ == "__main__":
    run_sentiment_report()
