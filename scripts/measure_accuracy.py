"""
scripts/measure_accuracy.py — FinSight Accuracy Measurement Script

Measures how accurately our agent calculates technical indicators
compared to the industry-standard `ta` library as ground truth.

Metrics measured:
  - RSI(14)          — Relative Strength Index
  - Bollinger Upper  — Upper Bollinger Band
  - Bollinger Lower  — Lower Bollinger Band
  - MA30             — 30-day Moving Average
  - MA200            — 200-day Moving Average

Run:
    python scripts/measure_accuracy.py

Output:
    Accuracy report printed to console + saved as accuracy_report.txt
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from agent.nodes.fetch_price import fetch_price_data
from agent.nodes.analyze import analyze

# ── Tickers to test ───────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "TSLA", "AMZN", "JPM", "JNJ", "SPY",
    "QQQ",  "BAC",  "PFE", "COIN", "GS",
    "NFLX", "AMD",  "INTC","DIS",  "UNH",
]

# ── Accuracy thresholds ───────────────────────────────────────────────────────
RSI_TOLERANCE     = 1.0   # accept within 1 RSI point
BB_TOLERANCE_PCT  = 1.0   # accept within 1% of band value
MA_TOLERANCE_PCT  = 0.5   # accept within 0.5% of MA value


def get_ground_truth(ticker: str) -> dict:
    """Calculate ground truth values using the `ta` library."""
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if df.empty or len(df) < 30:
            return None

        close = df["Close"]

        # RSI(14)
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

        # Bollinger Bands(20, 2)
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]

        # Moving Averages
        ma30  = close.rolling(30).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        return {
            "rsi":      round(float(rsi), 4),
            "bb_upper": round(float(bb_upper), 4),
            "bb_lower": round(float(bb_lower), 4),
            "ma30":     round(float(ma30), 4),
            "ma200":    round(float(ma200), 4) if ma200 else None,
        }
    except Exception as e:
        print(f"  ⚠ Ground truth failed for {ticker}: {e}")
        return None


def get_our_values(ticker: str) -> dict:
    """Get values from our agent."""
    try:
        price_state = fetch_price_data(ticker)
        analysis    = analyze(price_state)
        return {
            "rsi":      analysis.get("rsi_14"),
            "bb_upper": analysis.get("bb_upper"),
            "bb_lower": analysis.get("bb_lower"),
            "ma30":     price_state.get("ma_30"),
            "ma200":    price_state.get("ma_200"),
        }
    except Exception as e:
        print(f"  ⚠ Our agent failed for {ticker}: {e}")
        return None


def pct_error(ours, truth):
    """Calculate percentage error between our value and ground truth."""
    if ours is None or truth is None:
        return None
    if truth == 0:
        return None
    return abs(ours - truth) / abs(truth) * 100


def run_accuracy_report():
    results = []

    print("\n" + "="*65)
    print("  FinSight Technical Indicator Accuracy Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)
    print(f"\n{'Ticker':<8} {'RSI Err':>9} {'BB Up Err':>10} {'BB Lo Err':>10} {'MA30 Err':>9} {'MA200 Err':>10}")
    print("-"*60)

    for ticker in TICKERS:
        truth = get_ground_truth(ticker)
        ours  = get_our_values(ticker)

        if not truth or not ours:
            print(f"{ticker:<8} {'SKIP':>9}")
            continue

        rsi_err    = pct_error(ours["rsi"],      truth["rsi"])
        bbu_err    = pct_error(ours["bb_upper"], truth["bb_upper"])
        bbl_err    = pct_error(ours["bb_lower"], truth["bb_lower"])
        ma30_err   = pct_error(ours["ma30"],     truth["ma30"])
        ma200_err  = pct_error(ours["ma200"],    truth["ma200"])

        def fmt(v):
            return f"{v:.3f}%" if v is not None else "  N/A"

        print(f"{ticker:<8} {fmt(rsi_err):>9} {fmt(bbu_err):>10} {fmt(bbl_err):>10} {fmt(ma30_err):>9} {fmt(ma200_err):>10}")

        results.append({
            "ticker":   ticker,
            "rsi_err":  rsi_err,
            "bbu_err":  bbu_err,
            "bbl_err":  bbl_err,
            "ma30_err": ma30_err,
            "ma200_err":ma200_err,
        })

    print("-"*60)

    # ── Summary statistics ─────────────────────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.mean(vals) if vals else None

    def accuracy(avg_err):
        return 100 - avg_err if avg_err is not None else None

    rsi_acc   = accuracy(avg("rsi_err"))
    bbu_acc   = accuracy(avg("bbu_err"))
    bbl_acc   = accuracy(avg("bbl_err"))
    ma30_acc  = accuracy(avg("ma30_err"))
    ma200_acc = accuracy(avg("ma200_err"))

    overall = np.mean([x for x in [rsi_acc, bbu_acc, bbl_acc, ma30_acc, ma200_acc] if x])

    print(f"\n{'METRIC':<25} {'AVG ERROR':>10} {'ACCURACY':>10}")
    print("-"*47)
    print(f"{'RSI(14)':<25} {avg('rsi_err'):>9.4f}% {rsi_acc:>9.2f}%")
    print(f"{'Bollinger Upper':<25} {avg('bbu_err'):>9.4f}% {bbu_acc:>9.2f}%")
    print(f"{'Bollinger Lower':<25} {avg('bbl_err'):>9.4f}% {bbl_acc:>9.2f}%")
    print(f"{'MA30':<25} {avg('ma30_err'):>9.4f}% {ma30_acc:>9.2f}%")
    print(f"{'MA200':<25} {avg('ma200_err'):>9.4f}% {ma200_acc:>9.2f}%")
    print("-"*47)
    print(f"{'OVERALL ACCURACY':<25} {'':>10} {overall:>9.2f}%")
    print(f"\nTickers tested: {len(results)}/{len(TICKERS)}")
    print("Ground truth:   `ta` library (industry standard)")
    print("="*65)

    # ── Save report ────────────────────────────────────────────────────────────
    report_path = Path(__file__).parent.parent / "accuracy_report.txt"
    with open(report_path, "w") as f:
        f.write(f"FinSight Accuracy Report — {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Tickers tested: {len(results)}\n")
        f.write(f"Overall accuracy: {overall:.2f}%\n\n")
        f.write(f"RSI(14) accuracy:         {rsi_acc:.2f}%\n")
        f.write(f"Bollinger Upper accuracy: {bbu_acc:.2f}%\n")
        f.write(f"Bollinger Lower accuracy: {bbl_acc:.2f}%\n")
        f.write(f"MA30 accuracy:            {ma30_acc:.2f}%\n")
        f.write(f"MA200 accuracy:           {ma200_acc:.2f}%\n")

    print(f"\n✅ Report saved to: accuracy_report.txt")
    print(f"\n📋 CV-ready statement:")
    print(f'   "Validated technical indicators (RSI, Bollinger Bands, MA30/MA200)')
    print(f'    against industry-standard `ta` library across {len(results)} stocks,')
    print(f'    achieving {overall:.1f}% average accuracy."')


if __name__ == "__main__":
    run_accuracy_report()
