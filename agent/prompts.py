"""
agent/prompts.py — LLM Prompt Templates for FinSight

This module contains the prompt templates sent to Groq/Llama3.
The LLM is used ONLY in generate_report.py to write the final
narrative section of the report — all analysis is done in Python.

Design principles:
  - The prompt injects real numbers from the analysis pipeline.
  - It instructs the LLM to write in a specific structured format
    so reportlab can parse and render sections cleanly.
  - Temperature is set low (0.3) for factual, consistent output.
  - The LLM is told to stay within the data — no hallucination of
    figures not provided in the prompt.
"""

from string import Template


# ── Main report prompt ────────────────────────────────────────────────────────

REPORT_PROMPT_TEMPLATE = Template("""You are a professional financial analyst writing a structured research report.
You have been given real market data for $company_name ($ticker).
Write a clear, factual, and professional report based ONLY on the data provided below.
Do NOT invent numbers, percentages, or facts not present in the data.

=== MARKET DATA ===
Company: $company_name ($ticker)
Sector: $sector | Industry: $industry
Current Price: $$$current_price ($currency)
52-Week High: $$$week_52_high | 52-Week Low: $$$week_52_low
% From 52W High: $pct_from_52w_high% | % From 52W Low: $pct_from_52w_low%
Market Cap: $market_cap

=== TECHNICAL ANALYSIS ===
RSI (14-day): $rsi_14 → $rsi_signal
Bollinger Bands: Upper=$$$bb_upper | Middle=$$$bb_middle | Lower=$$$bb_lower
Bollinger Signal: $bb_signal (%%B = $bb_pct_b)
30-Day Moving Average: $$$ma_30
200-Day Moving Average: $$$ma_200
MA Cross Signal: $ma_cross_signal
Price vs MA30: $price_vs_ma30
Price vs MA200: $price_vs_ma200
10-Day Momentum: $momentum_10d_pct% → $momentum_signal
30-Day Annualised Volatility: $volatility_30d_pct% → $volatility_signal risk

=== FUNDAMENTAL DATA ===
P/E Ratio: $pe_ratio | Forward P/E: $forward_pe
EPS: $$$eps | Profit Margin: $profit_margin_pct%
Operating Margin: $operating_margin_pct%
Revenue (TTM): $revenue_ttm
Debt-to-Equity: $debt_to_equity | Beta: $beta
Analyst Target Price: $$$analyst_target_price

=== NEWS SENTIMENT ===
Recent Headlines Sentiment: $sentiment_label (score: $average_sentiment_score)
Number of Articles Reviewed: $article_count
Headlines:
$headlines_text

=== RISK FLAGS ===
Risk Level: $risk_level
Anomaly Detected: $anomaly_detected_text
Flags Raised ($flag_count):
$flags_text

=== YOUR TASK ===
Write a structured financial research report with EXACTLY these four sections.
Use the section headers exactly as shown (the PDF renderer depends on them).

## EXECUTIVE SUMMARY
Write 2-3 sentences summarising the company's current market position,
whether the stock looks bullish or bearish overall, and the most important
takeaway for an investor.

## TECHNICAL ANALYSIS
Discuss the RSI, Bollinger Bands, moving averages, momentum, and volatility
in plain English. Explain what each indicator is telling us about the stock's
price behaviour. Use the actual numbers from the data above.

## FUNDAMENTAL ANALYSIS
Discuss the company's profitability, valuation (P/E ratio), revenue,
debt levels, and analyst price target. Is the stock cheap or expensive
relative to its earnings? Is the balance sheet healthy?

## RISK ASSESSMENT
Discuss the risk flags raised. If anomaly_detected is YES, explain which
flags are most concerning and why. If risk is low, confirm what's working
in the stock's favour. End with a balanced view — what could go right
and what could go wrong.

Keep each section between 80-150 words. Use professional but accessible language.
Do not add any sections beyond the four listed above.
Do not include disclaimers about not being a financial advisor.
""")


def build_report_prompt(state: dict) -> str:
    """
    Build the final LLM prompt by injecting all analysis data into the template.

    Args:
        state: The fully populated state dict after all 5 pipeline nodes have run.

    Returns:
        A formatted string ready to be sent to Groq as the user message.
    """
    def fmt(value, prefix="", suffix="", decimals=2, default="N/A"):
        """Format a value safely, returning default if None."""
        if value is None:
            return default
        if isinstance(value, float):
            return f"{prefix}{value:.{decimals}f}{suffix}"
        return f"{prefix}{value}{suffix}"

    def fmt_pct(value, decimals=2, default="N/A"):
        """Format a percentage value (already in percentage points)."""
        if value is None:
            return default
        return f"{value:.{decimals}f}"

    def fmt_large(value, default="N/A"):
        """Format large numbers with B/M suffixes."""
        if value is None:
            return default
        try:
            v = float(value)
            if v >= 1e12:
                return f"${v/1e12:.2f}T"
            elif v >= 1e9:
                return f"${v/1e9:.2f}B"
            elif v >= 1e6:
                return f"${v/1e6:.2f}M"
            else:
                return f"${v:,.0f}"
        except (TypeError, ValueError):
            return default

    # Format headlines
    articles = state.get("articles", [])
    if articles:
        headlines_text = "\n".join(
            f"  [{a.get('sentiment', 'neutral').upper()}] {a.get('title', '')}"
            for a in articles
        )
    else:
        headlines_text = "  No recent headlines available."

    # Format flags
    flags = state.get("flags", [])
    if flags:
        flags_text = "\n".join(
            f"  [{f['severity'].upper()}] {f['name']}: {f['description']}"
            for f in flags
        )
    else:
        flags_text = "  None — all indicators within normal ranges."

    # Profit margin as percentage
    profit_margin = state.get("profit_margin")
    profit_margin_pct = fmt_pct(profit_margin * 100 if profit_margin is not None else None)

    operating_margin = state.get("operating_margin")
    operating_margin_pct = fmt_pct(operating_margin * 100 if operating_margin is not None else None)

    return REPORT_PROMPT_TEMPLATE.substitute(
        # Identity
        company_name=state.get("company_name", state.get("ticker", "Unknown")),
        ticker=state.get("ticker", "N/A"),
        sector=state.get("sector", "Unknown"),
        industry=state.get("industry", "Unknown"),
        currency=state.get("currency", "USD"),
        # Price
        current_price=fmt(state.get("current_price")),
        week_52_high=fmt(state.get("week_52_high")),
        week_52_low=fmt(state.get("week_52_low")),
        pct_from_52w_high=fmt(state.get("pct_from_52w_high")),
        pct_from_52w_low=fmt(state.get("pct_from_52w_low")),
        market_cap=fmt_large(state.get("market_cap")),
        # Technical
        rsi_14=fmt(state.get("rsi_14")),
        rsi_signal=state.get("rsi_signal", "N/A"),
        bb_upper=fmt(state.get("bb_upper")),
        bb_middle=fmt(state.get("bb_middle")),
        bb_lower=fmt(state.get("bb_lower")),
        bb_signal=state.get("bb_signal", "N/A"),
        bb_pct_b=fmt(state.get("bb_pct_b")),
        ma_30=fmt(state.get("ma_30")),
        ma_200=fmt(state.get("ma_200")),
        ma_cross_signal=state.get("ma_cross_signal", "N/A"),
        price_vs_ma30=state.get("price_vs_ma30", "N/A"),
        price_vs_ma200=state.get("price_vs_ma200", "N/A"),
        momentum_10d_pct=fmt_pct(state.get("momentum_10d_pct")),
        momentum_signal=state.get("momentum_signal", "N/A"),
        volatility_30d_pct=fmt_pct(state.get("volatility_30d_pct")),
        volatility_signal=state.get("volatility_signal", "N/A"),
        # Fundamentals
        pe_ratio=fmt(state.get("pe_ratio")),
        forward_pe=fmt(state.get("forward_pe")),
        eps=fmt(state.get("eps")),
        profit_margin_pct=profit_margin_pct,
        operating_margin_pct=operating_margin_pct,
        revenue_ttm=fmt_large(state.get("revenue_ttm")),
        debt_to_equity=fmt(state.get("debt_to_equity")),
        beta=fmt(state.get("beta")),
        analyst_target_price=fmt(state.get("analyst_target_price")),
        # News
        sentiment_label=state.get("sentiment_label", "neutral"),
        average_sentiment_score=fmt(state.get("average_sentiment_score")),
        article_count=state.get("article_count", 0),
        headlines_text=headlines_text,
        # Anomaly
        risk_level=state.get("risk_level", "unknown").upper(),
        anomaly_detected_text="YES" if state.get("anomaly_detected") else "NO",
        flag_count=state.get("flag_count", 0),
        flags_text=flags_text,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Print a sample prompt with dummy data so you can review the format
    sample_state = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currency": "USD",
        "market_cap": 3_200_000_000_000,
        "current_price": 189.50,
        "week_52_high": 199.62,
        "week_52_low": 164.08,
        "pct_from_52w_high": -5.07,
        "pct_from_52w_low": 15.49,
        "rsi_14": 58.3,
        "rsi_signal": "bullish",
        "bb_upper": 195.0,
        "bb_middle": 183.0,
        "bb_lower": 171.0,
        "bb_signal": "within_bands",
        "bb_pct_b": 0.72,
        "ma_30": 186.0,
        "ma_200": 179.0,
        "ma_cross_signal": "golden_cross",
        "price_vs_ma30": "above_MA30",
        "price_vs_ma200": "above_MA200",
        "momentum_10d_pct": 2.3,
        "momentum_signal": "bullish",
        "volatility_30d_pct": 22.1,
        "volatility_signal": "moderate",
        "pe_ratio": 29.5,
        "forward_pe": 27.1,
        "eps": 6.43,
        "profit_margin": 0.253,
        "operating_margin": 0.298,
        "revenue_ttm": 385_000_000_000,
        "debt_to_equity": 1.73,
        "beta": 1.25,
        "analyst_target_price": 210.0,
        "sentiment_label": "positive",
        "average_sentiment_score": 0.6,
        "article_count": 3,
        "articles": [
            {"title": "Apple reports record iPhone sales", "sentiment": "positive"},
            {"title": "Apple Vision Pro gains traction", "sentiment": "positive"},
            {"title": "Supply chain concerns linger for Apple", "sentiment": "negative"},
        ],
        "risk_level": "low",
        "anomaly_detected": 0,
        "flag_count": 0,
        "flags": [],
    }

    prompt = build_report_prompt(sample_state)
    print(prompt)
