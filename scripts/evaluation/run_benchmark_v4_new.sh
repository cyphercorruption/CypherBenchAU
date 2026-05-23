#!/usr/bin/env bash
# Run benchmark on the 43 NEW samples in curated_200_v4 only, across all 9 models × 2 modes.
# Output → output/benchmark_v4_new/<model_slug>/<mode>/<graph>.json
set -euo pipefail

INPUT_DIR=output/curated_200_v4_new_only
VERIF_DIR=output/curated_200_v4_new_only/verification
OUTPUT_DIR=output/benchmark_v4_new

echo "=== Benchmark v4 NEW (43 sample × 9 models × 2 modes = 774 calls) ==="
echo ""

# Sonnet, Gemma, GPT-5.5, Qwen3.5 family → OpenRouter
for model in \
    "anthropic/claude-sonnet-4.6" \
    "google/gemma-4-31b-it" \
    "openai/gpt-5.5" \
    "qwen/qwen3-coder-30b-a3b-instruct" \
    "qwen/qwen3.5-122b-a10b" \
    "qwen/qwen3.5-27b" \
    "qwen/qwen3.5-9b"
do
  echo "=== $model (OpenRouter) ==="
  set -a; source local.env; set +a
  uv run python benchmark.py run-all \
    --model "$model" --mode both \
    --input-dir "$INPUT_DIR" --verification-dir "$VERIF_DIR" \
    --output-dir "$OUTPUT_DIR" --max-workers 4
  echo ""
done

# qwen3-coder 480B → OpenRouter with Google provider only
echo "=== qwen/qwen3-coder (OpenRouter, provider=Google) ==="
set -a; source local.env; set +a
uv run python benchmark.py run-all \
  --model "qwen/qwen3-coder" --mode both \
  --input-dir "$INPUT_DIR" --verification-dir "$VERIF_DIR" \
  --output-dir "$OUTPUT_DIR" --max-workers 4 \
  --provider-only Google

# Gemini → direct Google endpoint (.env)
echo "=== gemini-3.1-pro-preview (Google direct) ==="
set -a; source .env; set +a
uv run python benchmark.py run-all \
  --model "gemini-3.1-pro-preview" --mode both \
  --input-dir "$INPUT_DIR" --verification-dir "$VERIF_DIR" \
  --output-dir "$OUTPUT_DIR" --max-workers 4

echo ""
echo "=== All 9 models complete ==="
echo "Output: $OUTPUT_DIR"
