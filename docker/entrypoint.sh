#!/bin/bash
set -e

ES_URL="${ES_HOST:-http://elasticsearch:9200}"
ES_IDX="${ES_INDEX:-recipes}"

echo "[startup] Waiting for Elasticsearch at ${ES_URL}..."
until curl -sf "${ES_URL}/_cluster/health" > /dev/null 2>&1; do
  sleep 2
done
echo "[startup] Elasticsearch is ready."

if curl -sf "${ES_URL}/${ES_IDX}" > /dev/null 2>&1; then
  echo "[startup] Index '${ES_IDX}' already exists — skipping indexing."
else
  echo "[startup] Index '${ES_IDX}' not found — indexing recipes (may take a few minutes)..."
  python main.py index
  echo "[startup] Indexing complete."
fi

echo "[startup] Launching Streamlit on http://localhost:8501"
exec streamlit run app/main.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
