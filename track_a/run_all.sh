#!/usr/bin/env bash
# Full pipeline: env -> train/register -> drift report. Add "serve" to also start the API.
set -euo pipefail
cd "$(dirname "$0")"
uv sync
uv run python src/train.py
uv run python src/drift_monitor.py
if [ "${1:-}" = "serve" ]; then
  uv run uvicorn src.serve:app --port 8000
else
  echo "Done. Next: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db"
  echo "      and: uv run uvicorn src.serve:app --port 8000   (or ./run_all.sh serve)"
fi
