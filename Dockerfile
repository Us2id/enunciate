# syntax=docker/dockerfile:1.7
#
# CPU-only image with the model baked in, so a container starts without
# network access. Build with --build-arg PRELOAD_MODEL=0 to leave the model
# out and mount a Hugging Face cache at /home/app/.cache/huggingface instead.
#
#   docker build -t enunciate .
#   docker run --rm -p 8000:8000 enunciate

FROM python:3.12-slim AS runtime

ARG PRELOAD_MODEL=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENUNCIATE_DEVICE=cpu \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
# CPU wheels are a fraction of the size of the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server]"

USER app
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from enunciate import Recognizer; Recognizer(device='cpu').load()"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if b'\"model_loaded\": true' in urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).read() else 1)"

CMD ["enunciate", "serve", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
