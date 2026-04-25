"""
config.py — Centralized configuration for FinSight.

All API keys and settings are loaded from environment variables.
Never hard-code secrets here. Use a .env file locally (see .env.example).

Usage:
    from config import config
    print(config.GROQ_API_KEY)
    config.validate()  # raises ValueError if any required key is missing
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file into os.environ.
# If .env doesn't exist (e.g. in production/Docker), this is a no-op —
# the real env vars are expected to already be set.
load_dotenv()


class Config:
    """
    Singleton-style configuration object.

    Attributes are read once at import time from environment variables.
    Use config.validate() at app startup to fail fast if keys are missing.
    """

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "finsight-runs"
    )

    # ------------------------------------------------------------------
    # File Paths
    # ------------------------------------------------------------------
    # Base directory = the folder containing this file (project root)
    BASE_DIR: Path = Path(__file__).parent

    REPORTS_DIR: Path = BASE_DIR / os.getenv("REPORTS_DIR", "data/reports")
    REFERENCE_DATA_DIR: Path = BASE_DIR / os.getenv(
        "REFERENCE_DATA_DIR", "data/reference"
    )

    # ------------------------------------------------------------------
    # LLM Settings
    # ------------------------------------------------------------------
    # The Groq model used to write the final narrative report.
    # Alternatives: "mixtral-8x7b-32768", "gemma2-9b-it"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-70b-8192")

    # Maximum tokens to request from Groq for the report narrative
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # Temperature for LLM generation (0 = deterministic, 1 = creative)
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ------------------------------------------------------------------
    # Alpha Vantage rate-limit guard
    # ------------------------------------------------------------------
    # Free tier: 25 calls/day. Set to True during development to read
    # from a local cache file instead of making live API calls.
    AV_USE_CACHE: bool = os.getenv("AV_USE_CACHE", "false").lower() == "true"
    AV_CACHE_PATH: Path = BASE_DIR / "data" / "av_cache.json"

    # ------------------------------------------------------------------
    # Analysis Parameters
    # ------------------------------------------------------------------
    # RSI lookback period in days
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))

    # Bollinger Bands: rolling window and number of standard deviations
    BB_WINDOW: int = int(os.getenv("BB_WINDOW", "20"))
    BB_STD: float = float(os.getenv("BB_STD", "2.0"))

    # Anomaly detection: z-score threshold above which a value is flagged
    ANOMALY_ZSCORE_THRESHOLD: float = float(
        os.getenv("ANOMALY_ZSCORE_THRESHOLD", "2.0")
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Check that all required API keys are present.

        Call this once at app startup so the user gets a clear error
        message instead of a cryptic KeyError deep in a request.

        Returns:
            True if all keys are present.

        Raises:
            ValueError: listing the names of any missing keys.
        """
        missing: list[str] = []

        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.ALPHA_VANTAGE_API_KEY:
            missing.append("ALPHA_VANTAGE_API_KEY")
        if not self.NEWS_API_KEY:
            missing.append("NEWS_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your API keys."
            )

        return True

    def ensure_dirs(self) -> None:
        """Create output directories if they don't already exist."""
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure root logger based on LOG_LEVEL env var."""
        numeric_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# Module-level singleton — import this everywhere:
#   from config import config
config = Config()
