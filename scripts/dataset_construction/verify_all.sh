#!/usr/bin/env bash
# Load each graph into Neo4j and verify its corrupted samples.
# Requires: docker compose up -d (Neo4j running)
# Usage: bash scripts/dataset_construction/verify_all.sh

set -euo pipefail

export NEO4J_URI=${NEO4J_URI:-bolt://localhost:7687}
export NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}
export NEO4J_DATABASE_MAP='{"art":"neo4j","biology":"neo4j","company":"neo4j","fictional_character":"neo4j","flight_accident":"neo4j","geography":"neo4j","movie":"neo4j","nba":"neo4j","politics":"neo4j","soccer":"neo4j","terrorist_attack":"neo4j"}'

GRAPHS=(art biology company fictional_character flight_accident geography movie nba politics soccer terrorist_attack)

INPUT_DIR="output/final_output"
OUTPUT_DIR="output/final_output/verification"
mkdir -p "$OUTPUT_DIR"

echo "Verifying all graphs against Neo4j"
echo "Input: $INPUT_DIR | Output: $OUTPUT_DIR"
echo ""

for graph in "${GRAPHS[@]}"; do
  input="$INPUT_DIR/full_${graph}.json"
  output="$OUTPUT_DIR/verified_${graph}.json"

  if [ ! -f "$input" ]; then
    echo "=== $graph === SKIPPED (no input file)"
    continue
  fi

  echo "=== $graph ==="

  # Step 1: Load graph into Neo4j
  echo "  Loading graph into Neo4j..."
  uv run python scripts/load_graph.py "$graph"

  # Step 2: Verify corrupted samples
  echo "  Verifying..."
  uv run cb-corrupt verify \
    --input "$input" \
    --output "$output"

  echo ""
done

echo "=== All done ==="
echo ""
for graph in "${GRAPHS[@]}"; do
  f="$OUTPUT_DIR/verified_${graph}.json"
  if [ -f "$f" ]; then
    pass=$(python3 -c "import json; data=json.load(open('$f')); print(sum(1 for r in data if r['status']=='pass'))" 2>/dev/null || echo "?")
    fail=$(python3 -c "import json; data=json.load(open('$f')); print(sum(1 for r in data if r['status']=='fail'))" 2>/dev/null || echo "?")
    err=$(python3 -c "import json; data=json.load(open('$f')); print(sum(1 for r in data if r['status']=='error'))" 2>/dev/null || echo "?")
    echo "  $graph: pass=$pass fail=$fail error=$err"
  else
    echo "  $graph: NOT VERIFIED"
  fi
done
