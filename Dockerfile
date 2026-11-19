FROM python:3.15-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt


# Downloads fastText's Czech vectors (~1.3 GB) and keeps the FREQUENT_WORDS
# most frequent clean words plus rarer dictionary words (about 350,000 in all).
# This layer only rebuilds when the build script, requirements or
# FREQUENT_WORDS change, so normal deploys reuse it from the builder's cache.
FROM base AS vectors
ARG FREQUENT_WORDS=300000
COPY app/__init__.py app/text.py ./app/
COPY scripts/__init__.py scripts/build_vectors.py ./scripts/
RUN python -m scripts.build_vectors \
        --source /tmp/cc.cs.300.vec.gz --output /vectors/cc.cs.300.kv \
        --frequent "$FREQUENT_WORDS" --delete-source


FROM base AS runtime
COPY --from=vectors /vectors ./vectors
COPY app ./app
COPY scripts ./scripts
COPY static ./static
COPY data ./data

# Fail the build, not the deploy, if a secret word has no vector.
RUN python -m scripts.words check --vectors vectors/cc.cs.300.kv --quiet

RUN useradd --system --uid 10001 --no-create-home app
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--timeout-graceful-shutdown", "8"]
