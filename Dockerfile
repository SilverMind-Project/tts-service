# ── Build stage: install dependencies ────────────────────────────
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml .
RUN uv pip install --no-cache ".[all]"

# ── Runtime stage: lean production image ─────────────────────────
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.title="tts-service" \
      org.opencontainers.image.description="Self-hosted TTS service with OpenAI-compatible API" \
      org.opencontainers.image.licenses="AGPL-3.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/data/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ffmpeg \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY app/ app/
COPY config/ config/

RUN mkdir -p data/voice_samples

VOLUME ["/app/data"]

EXPOSE 8600

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8600/health')" || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8600"]
