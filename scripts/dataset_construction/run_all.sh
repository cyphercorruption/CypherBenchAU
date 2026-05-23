#!/usr/bin/env bash
# Run full corruption pipeline on all graphs, each saved to its own file.
# Usage: bash scripts/dataset_construction/run_all.sh [model] [num_samples]
# Example: bash scripts/dataset_construction/run_all.sh gemini-3-flash-preview 100

set -euo pipefail

MODEL=${1:-gemini-3-flash-preview}
NUM_SAMPLES=${2:-100}

GRAPHS=(art biology company fictional_character flight_accident geography movie nba politics soccer terrorist_attack)

echo "Model: $MODEL | Samples per type: $NUM_SAMPLES"
echo "Graphs: ${GRAPHS[*]}"
echo ""

for graph in "${GRAPHS[@]}"; do
  echo "=== $graph ==="
  uv run cb-corrupt generate \
    --graphs "$graph" \
    --num-samples "$NUM_SAMPLES" \
    --model "$MODEL" \
    --output "output/final_output/full_${graph}.json"
  echo ""
done

echo "=== All done ==="
echo ""
for graph in "${GRAPHS[@]}"; do
  f="output/full_${graph}.json"
  if [ -f "$f" ]; then
    count=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
    echo "  $graph: $count samples"
  else
    echo "  $graph: MISSING"
  fi
done
