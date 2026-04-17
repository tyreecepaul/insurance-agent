# ── Insurance Claims Agent — API Service ──────────────────────────────────────
#
# Build:   docker build -t insurance-agent .
# Run:     docker compose up
#
# Ollama is NOT bundled — it must run on the host (or a separate container).
# The default OLLAMA_BASE_URL points to host.docker.internal:11434 which works
# on Docker Desktop (Mac/Windows). On Linux hosts, set it to the host's
# bridge IP (typically 172.17.0.1) or use --network=host.
#
# PySpark (eval harness) requires Java. We install a slim JRE here so the
# full requirements.txt installs cleanly. If you only need the API and want
# a smaller image, remove the openjdk-17-jre-headless line and drop pyspark
# from requirements.txt.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies:
#   - openjdk-17-jre-headless  → PySpark (eval harness)
#   - libglib2.0-0             → PyMuPDF (PDF parsing)
#   - libgomp1                 → PyTorch (OpenMP threading)
#   - curl                     → Docker healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

# Install Python dependencies first (layer-cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/       ./src/
COPY main.py    .
COPY config.json .

# Runtime defaults (overridden via docker-compose environment or -e flags)
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3.2 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    SESSION_EXPIRY_HOURS=24 \
    CLEANUP_INTERVAL_SECS=3600

EXPOSE 8000

# Liveness probe: the /health endpoint must return 200 within 5 s
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py", "--api"]
