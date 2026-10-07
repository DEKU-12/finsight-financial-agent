

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
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Optional locally; needed on cloud hosts where Yahoo Finance is blocked
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
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
    # Base directory = the folder containing this file (project root)
    BASE_DIR: Path = Path(__file__).parent

    REPORTS_DIR: Path = BASE_DIR / os.getenv("REPORTS_DIR", "data/reports")
    REFERENCE_DATA_DIR: Path = BASE_DIR / os.getenv(
        "REFERENCE_DATA_DIR", "data/reference"
    )

    # ------------------------------------------------------------------
    # LLM Settings
    # ------------------------------------------------------------------
    # The Claude model used to write the final narrative report.
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-opus-5-5")

    # Max output tokens for the report (includes Claude's thinking tokens)
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "16000"))


    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

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

    def validate(self, anthropic_api_key: str = "") -> bool:
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

        if not (anthropic_api_key or self.ANTHROPIC_API_KEY):
            missing.append("ANTHROPIC_API_KEY (enter your own key in the sidebar)")
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
