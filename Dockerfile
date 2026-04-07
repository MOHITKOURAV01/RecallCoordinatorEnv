FROM python:3.11-slim

# Safe defaults for containerized FastAPI + HF Spaces.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR /app

# 1) System deps first (stable layer). curl is used by HEALTHCHECK.
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# 2) Dependencies layer for caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) Copy application code last (invalidates cache only when code changes).
COPY . /app

# 4) Non-root runtime user (security best practice).
RUN useradd -m -u 10001 appuser \
  && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

