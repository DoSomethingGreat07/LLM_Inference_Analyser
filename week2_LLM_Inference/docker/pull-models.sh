#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
  echo "Usage: pull-models.sh <model1> [model2 ...]"
  exit 1
fi

export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

echo "Waiting for Ollama at ${OLLAMA_HOST}..."
deadline=$(( $(date +%s) + 180 ))
until curl -fsS "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "Ollama server did not become ready in time."
    exit 1
  fi
  sleep 2
done

for model_name in "$@"; do
  echo "Pulling ${model_name}..."
  ollama pull "${model_name}"
done
