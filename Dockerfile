FROM python:3.11.9-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-built artifacts instead of training at build time.
# The CI pipeline produces models/ and data/device_catalog.json
# before the docker build step runs.
COPY api/ api/
COPY models/ models/
COPY data/device_catalog.json data/device_catalog.json

ENV MODEL_DIR=models \
    CATALOG_PATH=data/device_catalog.json \
    PORT=8080

EXPOSE 8080

# Cloud Run injects $PORT at runtime
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
