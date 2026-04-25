# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — FinSight Streamlit App
#
# Build:  docker build -t finsight-app .
# Run:    docker run -p 8501:8501 --env-file .env finsight-app
#
# In production use docker-compose.yml which also starts the MLflow server.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency installer ────────────────────────────────────────────
# We use a separate stage so the final image doesn't include pip cache or
# build tools — keeps the image lean (~600 MB vs ~1.2 GB).
FROM python:3.11-slim AS builder

WORKDIR /build

# Install OS-level build dependencies needed by some Python packages
# (e.g. numpy, pandas, reportlab need gcc/g++ or libffi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — Docker layer caching means this layer is
# only re-run when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY . .

# Create output directories the app expects at runtime
RUN mkdir -p data/reports data/reference

# Streamlit config: disable the "welcome" page and enable CORS for embedding
RUN mkdir -p /root/.streamlit && cat > /root/.streamlit/config.toml <<'EOF'
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = 8501

[browser]
gatherUsageStats = false
EOF

# Expose Streamlit port
EXPOSE 8501

# Health check — Streamlit returns 200 on its root path when healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
