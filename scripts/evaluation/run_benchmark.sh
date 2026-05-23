#!/usr/bin/env bash
# Run the NL→Cypher benchmark across all 11 graphs for one model.
# Usage: bash scripts/evaluation/run_benchmark.sh [model] [mode] [num_samples]
# Examples:
#   bash scripts/evaluation/run_benchmark.sh gemini-3.1-pro-preview both
#   bash scripts/evaluation/run_benchmark.sh openai/gpt-5.2 aware 100

set -euo pipefail

MODEL=${1:-gemini-3.1-pro-preview}
MODE=${2:-both}
NUM_SAMPLES=${3:-}

ARGS=(--model "$MODEL" --mode "$MODE")
if [ -n "$NUM_SAMPLES" ]; then
  ARGS+=(--num-samples "$NUM_SAMPLES")
fi

echo "Model: $MODEL | Mode: $MODE${NUM_SAMPLES:+ | Samples per graph: $NUM_SAMPLES}"
echo ""

uv run python benchmark.py run-all "${ARGS[@]}"

echo ""
echo "=== Done ==="
SLUG=${MODEL//\//_}
SLUG=${SLUG// /_}
for m in naive aware; do
  if [ "$MODE" = "both" ] || [ "$MODE" = "$m" ]; then
    dir="output/benchmark/${SLUG}/${m}"
    echo ""
    echo "[$m] $dir"
    if [ -d "$dir" ]; then
      for f in "$dir"/*.json; do
        [ -f "$f" ] || continue
        graph=$(basename "$f" .json)
        count=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
        echo "  $graph: $count samples"
      done
    fi
  fi
done
