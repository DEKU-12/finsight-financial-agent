# 📈 FinSight — Autonomous Financial Research Agent

![CI](https://github.com/DEKU-12/finsight-financial-agent/actions/workflows/ci.yml/badge.svg?branch=Ayush)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.13+-lightblue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end MLOps-grade financial research agent that fetches live market data, runs technical and fundamental analysis, detects anomalies, generates AI-powered PDF reports, and tracks every experiment in MLflow — all in under 60 seconds.

---

## 🎯 What It Does

You type a stock ticker like `AAPL` or `NVDA`. FinSight runs an 8-node LangGraph pipeline that:

1. **Fetches** live price history, moving averages, and volume from yfinance
2. **Fetches** fundamentals (P/E, EPS, margins, debt/equity) from Alpha Vantage
3. **Fetches** the latest news headlines and scores sentiment
4. **Analyzes** RSI, Bollinger Bands, momentum, and volatility
5. **Detects** anomalies and assigns a risk level (Low → Critical)
6. **Generates** an AI narrative report using Groq/Llama-3 and saves it as a PDF
7. **Monitors** data quality and statistical drift using Evidently AI
8. **Tracks** every run as a reproducible MLflow experiment

All results are displayed in a live Streamlit dashboard with PDF download.

---

## 🏗️ Architecture

```
User Input (Ticker)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                    │
│                                                        │
│  fetch_price → fetch_fundamentals → fetch_news        │
│       │                                    │           │
│       └──────────────► analyze ◄───────────┘          │
│                            │                           │
│                     detect_anomaly                     │
│                            │                           │
│                     generate_report  ← Groq/Llama-3   │
│                            │                           │
│                        monitor  ← Evidently AI         │
│                            │                           │
│                         track   ← MLflow               │
└───────────────────────────────────────────────────────┘
        │
        ▼
Streamlit Dashboard + PDF Report + MLflow Experiment
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent Framework** | LangGraph + LangChain |
| **LLM** | Groq API (Llama-3.3-70B) |
| **Price Data** | yfinance |
| **Fundamentals** | Alpha Vantage API |
| **News & Sentiment** | NewsAPI |
| **Technical Analysis** | NumPy + Pandas (RSI, BB, Momentum) |
| **PDF Generation** | ReportLab Platypus |
| **Experiment Tracking** | MLflow |
| **Data Monitoring** | Evidently AI |
| **Frontend** | Streamlit |
| **CI/CD** | GitHub Actions |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys for [Groq](https://console.groq.com), [Alpha Vantage](https://www.alphavantage.co/support/#api-key), and [NewsAPI](https://newsapi.org)

### 1. Clone the repo
```bash
git clone https://github.com/DEKU-12/finsight-financial-agent.git
cd finsight-financial-agent
git checkout Ayush
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Start MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5001
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, enter a ticker, and click **Run Analysis**.

---

## 🧪 Running Tests

```bash
pytest tests/test_nodes.py -v
```

**37 tests across 5 test classes:**

| Test Class | What It Checks |
|-----------|----------------|
| `TestFetchPrice` | yfinance data fetching, price validity, MA30/MA200 |
| `TestAnalyze` | RSI range (0–100), Bollinger Band ordering, volatility, signals |
| `TestDetectAnomaly` | Flag count accuracy, risk level values, anomaly triggers |
| `TestFetchNews` | Sentiment label/score validity, API error handling (mocked) |
| `TestConfig` | Config imports, all attributes present, valid defaults |
| `TestOutputValidation` | **Auto cross-checks agent output against yfinance ground truth** |

The `TestOutputValidation` class automatically verifies correctness without manual work:
- Price accuracy within 1% of yfinance's live price
- 52W high always ≥ current price, 52W low always ≤ current price
- MA200 recalculated from raw data and compared
- RSI, volatility, and Bollinger Bands within realistic ranges

---

## 📊 MLflow Dashboard

Start the MLflow server alongside the app to enable the Past Runs tab:

```bash
mlflow server --host 0.0.0.0 --port 5001
```

Open `http://localhost:5001` → click **Model Training** view.

Each run logs:
- **Parameters:** ticker, model, data sources, RSI period
- **Metrics:** price, RSI, volatility, sentiment score, latency, token usage
- **Artifacts:** PDF report, Evidently HTML report, state JSON snapshot

---

## 📁 Project Structure

```
finsight-financial-agent/
├── agent/
│   ├── graph.py              # LangGraph pipeline (8 nodes)
│   ├── prompts.py            # Llama-3 report prompt template
│   └── nodes/
│       ├── fetch_price.py    # yfinance data fetcher
│       ├── fetch_fundamentals.py  # Alpha Vantage fetcher
│       ├── fetch_news.py     # NewsAPI + sentiment classifier
│       ├── analyze.py        # RSI, Bollinger Bands, momentum
│       ├── detect_anomaly.py # Anomaly flagging & risk scoring
│       └── generate_report.py # Groq LLM + ReportLab PDF
├── mlops/
│   ├── tracker.py            # MLflow experiment logging
│   └── monitor.py            # Evidently data quality & drift
├── tests/
│   └── test_nodes.py         # Unit tests (pytest)
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI
├── app.py                    # Streamlit frontend
├── config.py                 # Centralized config singleton
└── requirements.txt          # Python dependencies
```

---

## 📈 Sample Output

**Risk Levels:** Low 🟢 | Medium 🟡 | High 🔴 | Critical 🚨

**Metrics tracked per run:**
- Current price, 52W high/low, market cap
- RSI(14), Bollinger %B, 30-day volatility, 10-day momentum
- P/E ratio, EPS, profit margin, debt/equity, beta
- News sentiment score, article count
- LLM tokens used, latency, agent total runtime

---

## 👤 Author

**Ayush** — MS Data Science, George Washington University
Built as a portfolio project targeting Data Science / MLE roles.

- GitHub: [@DEKU-12](https://github.com/DEKU-12)
- Email: ayush120320@gmail.com

---

## 📄 License

MIT License — free to use and modify.
