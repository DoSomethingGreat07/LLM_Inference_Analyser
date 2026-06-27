#!/usr/bin/env bash
set -euo pipefail

API_URL="${OLLAMA_API_URL:-http://ollama:11434/api/generate}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-180}"
export API_URL
export WAIT_TIMEOUT_SECONDS

echo "Waiting for Ollama API at ${API_URL}..."
python - <<'PY'
import os
import sys
import time
import requests

api_url = os.environ["API_URL"]
deadline = time.time() + int(os.environ["WAIT_TIMEOUT_SECONDS"])

while time.time() < deadline:
    try:
        response = requests.get(api_url.rsplit("/api/generate", 1)[0] + "/api/tags", timeout=2)
        if response.ok:
            sys.exit(0)
    except requests.RequestException:
        pass
    time.sleep(2)

raise SystemExit(
    f"Ollama API did not become ready within {os.environ['WAIT_TIMEOUT_SECONDS']} seconds."
)
PY

exec python src/ollama_real_benchmark.py --api-url "${API_URL}" "$@"
