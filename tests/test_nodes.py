"""
tests/test_nodes.py — Unit Tests for FinSight Agent Nodes

Tests every major node in the pipeline using:
  - Real yfinance data for fetch_price (AAPL — always available)
  - Mocked API responses for Alpha Vantage and NewsAPI (no key needed)
  - Synthetic state dicts for analyze, detect_anomaly, generate_report

Run:
    pytest tests/test_nodes.py -v
    pytest tests/test_nodes.py -v --tb=short   # shorter tracebacks
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES — reusable test data
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_price_state():
    """Minimal state dict after fetch_price runs successfully."""
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currency": "USD",
        "current_price": 175.0,
        "week_52_high": 200.0,
        "week_52_low": 130.0,
        "avg_volume": 50_000_000,
        "current_volume": 45_000_000,
        "ma_30": 172.0,
        "ma_200": 160.0,
        "close_prices": [150.0 + i * 0.5 for i in range(60)],
        "daily_returns": [0.001 * (i % 5 - 2) for i in range(59)],
        "dates": [f"2024-01-{str(i+1).zfill(2)}" for i in range(60)],
        "status": "success",
    }


@pytest.fixture
def sample_full_state(sample_price_state):
    """Full state dict with all node outputs — used for report/anomaly tests."""
    return {
        **sample_price_state,
        # Fundamentals
        "pe_ratio": 28.0,
        "forward_pe": 25.0,
        "eps": 6.13,
        "profit_margin": 0.25,
        "operating_margin": 0.30,
        "revenue_ttm": 390_000_000_000,
        "debt_to_equity": 1.5,
        "beta": 1.2,
        "analyst_target_price": 200.0,
        "dividend_yield": 0.005,
        "book_value": 4.0,
        "market_cap": 2_700_000_000_000,
        # News
        "articles": [
            {"title": "Apple hits new high", "sentiment_score": 0.6,
             "source": "Reuters", "url": "https://reuters.com/1"},
            {"title": "iPhone sales strong", "sentiment_score": 0.4,
             "source": "Bloomberg", "url": "https://bloomberg.com/1"},
        ],
        "article_count": 2,
        "average_sentiment_score": 0.5,
        "sentiment_label": "positive",
        # Analysis
        "rsi_14": 58.0,
        "rsi_signal": "neutral",
        "bb_upper": 180.0,
        "bb_middle": 172.0,
        "bb_lower": 164.0,
        "bb_width": 0.093,
        "bb_signal": "within_bands",
        "bb_pct_b": 0.68,
        "volatility_30d": 0.18,
        "volatility_30d_pct": 18.0,
        "volatility_signal": "moderate",
        "momentum_10d": 3.5,
        "momentum_10d_pct": 2.0,
        "momentum_signal": "bullish",
        "price_vs_ma30": "above",
        "price_vs_ma200": "above",
        "ma_cross_signal": "golden_cross",
        "pct_from_52w_high": -12.5,
        "pct_from_52w_low": 34.6,
        "analysis_status": "success",
        # Anomaly
        "flags": [],
        "flag_count": 0,
        "high_severity_count": 0,
        "anomaly_detected": 0,
        "risk_level": "low",
        "anomaly_summary": "No anomalies detected.",
        # Report
        "report_path": "",
        "report_filename": "",
        "llm_narrative": "Apple shows strong technical and fundamental signals.",
        "llm_tokens_used": 800,
        "llm_latency_seconds": 1.5,
        "report_status": "success",
        "agent_latency_seconds": 12.3,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: fetch_price
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchPrice:

    def test_returns_dict(self):
        """fetch_price_data must return a dict."""
        from agent.nodes.fetch_price import fetch_price_data
        result = fetch_price_data("AAPL")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        """Result must contain all required keys."""
        from agent.nodes.fetch_price import fetch_price_data
        result = fetch_price_data("AAPL")
        required = [
            "company_name", "current_price", "close_prices",
            "daily_returns", "dates", "ma_30", "ma_200",
            "week_52_high", "week_52_low",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_close_prices_is_list(self):
        """close_prices must be a non-empty list."""
        from agent.nodes.fetch_price import fetch_price_data
        result = fetch_price_data("AAPL")
        assert isinstance(result["close_prices"], list)
        assert len(result["close_prices"]) > 0

    def test_current_price_is_positive(self):
        """Current price must be a positive number."""
        from agent.nodes.fetch_price import fetch_price_data
        result = fetch_price_data("AAPL")
        price = result.get("current_price")
        if price is not None:
            assert price > 0, "Current price should be positive"

    def test_invalid_ticker_returns_dict(self):
        """Invalid ticker must not crash — returns a dict with error info."""
        from agent.nodes.fetch_price import fetch_price_data
        result = fetch_price_data("INVALIDTICKER999")
        assert isinstance(result, dict)

    def test_ma30_less_than_or_equal_week52high(self):
        """MA30 should be a reasonable price (not above 52w high by a lot)."""
        from agent.nodes.fetch_price import fetch_price_data
        result = fetch_price_data("AAPL")
        ma30 = result.get("ma_30")
        high = result.get("week_52_high")
        if ma30 and high:
            assert ma30 <= high * 1.1, "MA30 unreasonably above 52W high"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: analyze
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyze:

    def test_returns_dict(self, sample_price_state):
        """analyze must return a dict."""
        from agent.nodes.analyze import analyze
        result = analyze(sample_price_state)
        assert isinstance(result, dict)

    def test_rsi_in_valid_range(self, sample_price_state):
        """RSI must be between 0 and 100 (or nan for flat price data)."""
        import math
        from agent.nodes.analyze import analyze
        result = analyze(sample_price_state)
        rsi = result.get("rsi_14")
        if rsi is not None and not math.isnan(rsi):
            assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"

    def test_bollinger_bands_ordering(self, sample_price_state):
        """BB lower < BB middle < BB upper."""
        from agent.nodes.analyze import analyze
        result = analyze(sample_price_state)
        lower  = result.get("bb_lower")
        middle = result.get("bb_middle")
        upper  = result.get("bb_upper")
        if all(v is not None for v in [lower, middle, upper]):
            assert lower < middle < upper, "Bollinger Band ordering violated"

    def test_volatility_non_negative(self, sample_price_state):
        """Volatility must be non-negative."""
        from agent.nodes.analyze import analyze
        result = analyze(sample_price_state)
        vol = result.get("volatility_30d_pct")
        if vol is not None:
            assert vol >= 0, "Volatility cannot be negative"

    def test_rsi_signal_valid(self, sample_price_state):
        """RSI signal must be one of the expected labels."""
        from agent.nodes.analyze import analyze
        result = analyze(sample_price_state)
        valid_signals = {"overbought", "oversold", "bullish", "bearish", "neutral"}
        signal = result.get("rsi_signal", "neutral")
        assert signal in valid_signals, f"Unexpected RSI signal: {signal}"

    def test_ma_cross_signal_valid(self, sample_price_state):
        """MA cross signal must be one of the expected labels."""
        from agent.nodes.analyze import analyze
        result = analyze(sample_price_state)
        valid = {"golden_cross", "death_cross", "neutral"}
        signal = result.get("ma_cross_signal", "neutral")
        assert signal in valid, f"Unexpected MA cross signal: {signal}"

    def test_skips_gracefully_without_prices(self):
        """analyze must not crash when close_prices is missing."""
        from agent.nodes.analyze import analyze
        result = analyze({"ticker": "TEST"})
        assert isinstance(result, dict)
        assert result.get("analysis_status") == "skipped"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: detect_anomaly
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectAnomaly:

    def test_returns_dict(self, sample_full_state):
        """detect_anomaly must return a dict."""
        from agent.nodes.detect_anomaly import detect_anomaly
        result = detect_anomaly(sample_full_state)
        assert isinstance(result, dict)

    def test_required_keys(self, sample_full_state):
        """Result must have all anomaly output keys."""
        from agent.nodes.detect_anomaly import detect_anomaly
        result = detect_anomaly(sample_full_state)
        for key in ["flags", "flag_count", "high_severity_count",
                    "anomaly_detected", "risk_level", "anomaly_summary"]:
            assert key in result, f"Missing key: {key}"

    def test_flag_count_matches_flags(self, sample_full_state):
        """flag_count must equal len(flags)."""
        from agent.nodes.detect_anomaly import detect_anomaly
        result = detect_anomaly(sample_full_state)
        assert result["flag_count"] == len(result["flags"])

    def test_anomaly_detected_is_binary(self, sample_full_state):
        """anomaly_detected must be 0 or 1."""
        from agent.nodes.detect_anomaly import detect_anomaly
        result = detect_anomaly(sample_full_state)
        assert result["anomaly_detected"] in (0, 1)

    def test_risk_level_valid(self, sample_full_state):
        """risk_level must be one of the expected values."""
        from agent.nodes.detect_anomaly import detect_anomaly
        result = detect_anomaly(sample_full_state)
        assert result["risk_level"] in ("low", "medium", "high", "critical")

    def test_high_rsi_triggers_flag(self, sample_full_state):
        """RSI > 75 should trigger an overbought flag."""
        from agent.nodes.detect_anomaly import detect_anomaly
        state = {**sample_full_state, "rsi_14": 82.0, "rsi_signal": "overbought"}
        result = detect_anomaly(state)
        flag_names = [f["name"] for f in result["flags"]]
        assert any("rsi" in name.lower() or "overbought" in name.lower()
                   for name in flag_names), "Expected RSI flag for RSI=82"

    def test_low_rsi_triggers_flag(self, sample_full_state):
        """RSI < 25 should trigger an oversold flag."""
        from agent.nodes.detect_anomaly import detect_anomaly
        state = {**sample_full_state, "rsi_14": 18.0, "rsi_signal": "oversold"}
        result = detect_anomaly(state)
        flag_names = [f["name"] for f in result["flags"]]
        assert any("rsi" in name.lower() or "oversold" in name.lower()
                   for name in flag_names), "Expected RSI flag for RSI=18"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: fetch_news (mocked — no API key needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchNews:

    @patch("agent.nodes.fetch_news.requests.get")
    def test_returns_dict(self, mock_get):
        """fetch_news must return a dict even with mocked API."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "ok",
                "articles": [
                    {
                        "title": "Apple stock surges",
                        "description": "Apple hits record high",
                        "url": "https://example.com/1",
                        "source": {"name": "Reuters"},
                        "publishedAt": "2024-01-15T10:00:00Z",
                    }
                ],
            },
        )
        from agent.nodes.fetch_news import fetch_news
        result = fetch_news("Apple", "AAPL")
        assert isinstance(result, dict)

    @patch("agent.nodes.fetch_news.requests.get")
    def test_sentiment_label_valid(self, mock_get):
        """sentiment_label must be positive, negative, or neutral."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "ok",
                "articles": [
                    {
                        "title": "Apple profit soars to record high",
                        "description": "Strong earnings beat expectations",
                        "url": "https://example.com/1",
                        "source": {"name": "Bloomberg"},
                        "publishedAt": "2024-01-15T10:00:00Z",
                    }
                ],
            },
        )
        from agent.nodes.fetch_news import fetch_news
        result = fetch_news("Apple", "AAPL")
        assert result.get("sentiment_label") in ("positive", "negative", "neutral")

    @patch("agent.nodes.fetch_news.requests.get")
    def test_sentiment_score_in_range(self, mock_get):
        """Sentiment score must be between -1 and 1."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ok", "articles": []},
        )
        from agent.nodes.fetch_news import fetch_news
        result = fetch_news("Apple", "AAPL")
        score = result.get("average_sentiment_score", 0)
        assert -1.0 <= score <= 1.0

    @patch("agent.nodes.fetch_news.requests.get")
    def test_api_error_returns_dict(self, mock_get):
        """API errors must not crash — returns dict with defaults."""
        mock_get.side_effect = Exception("Network error")
        from agent.nodes.fetch_news import fetch_news
        result = fetch_news("Apple", "AAPL")
        assert isinstance(result, dict)
        assert "articles" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: config
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_config_imports(self):
        """Config singleton must import without errors."""
        from config import config
        assert config is not None

    def test_default_values_present(self):
        """Config must have all expected attributes."""
        from config import config
        assert hasattr(config, "LLM_MODEL")
        assert hasattr(config, "RSI_PERIOD")
        assert hasattr(config, "BB_WINDOW")
        assert hasattr(config, "REPORTS_DIR")

    def test_rsi_period_positive(self):
        """RSI period must be a positive integer."""
        from config import config
        assert config.RSI_PERIOD > 0

    def test_bb_window_positive(self):
        """BB window must be a positive integer."""
        from config import config
        assert config.BB_WINDOW > 0

    def test_llm_model_is_string(self):
        """LLM model must be a non-empty string."""
        from config import config
        assert isinstance(config.LLM_MODEL, str)
        assert len(config.LLM_MODEL) > 0
