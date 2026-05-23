# CypherBench Corruptions

Corruption generator for [CypherBench](https://github.com/megagonlabs/cypherbench): produces ambiguous and unanswerable perturbations of NL-to-Cypher benchmark samples using LLMs.

## Setup

```bash
# Clone the repo
git clone https://github.com/fabriziobattiloro/cypherbench-corruptions.git
cd cypherbench-corruptions

# Install dependencies
uv sync

# Download the benchmark data (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/megagonlabs/cypherbench benchmark

# Set up your API key
cp .env.example .env
# Edit .env with your API key and base URL
```

## Configuration

Create a `.env` file with your LLM provider credentials:

```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

The tool uses the OpenAI-compatible API format. The default model is `gemini-3.1-pro-preview`.

## CLI Commands

All commands are run via `uv run cb-corrupt`.

### `analyze`

Dry-run that shows corruption candidates and matching samples per type per graph. No LLM calls, no cost.

```bash
# All graphs, all corruption types
uv run cb-corrupt analyze

# Single graph
uv run cb-corrupt analyze -g nba

# Specific corruption types
uv run cb-corrupt analyze -g nba -c A1 -c U2

# Verbose logging
uv run cb-corrupt analyze -g nba -v
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--benchmark-path` | Benchmark JSON path relative to project root | `benchmark/test.json` |
| `-g`, `--graphs` | Graphs to analyze (repeatable) | all 11 |
| `-c`, `--corruption-types` | Corruption types (repeatable) | all 9 |
| `-v`, `--verbose` | Debug logging | off |

### `generate`

Runs the full corruption pipeline: analyze schema, select samples, call LLM to produce corrupted questions.

```bash
# Generate 5 corrupted samples per type for NBA
uv run cb-corrupt generate -g nba --num-samples 5

# Specific corruption types
uv run cb-corrupt generate -g nba -c A1 -c U2 --num-samples 10

# Custom output file
uv run cb-corrupt generate -g nba -o output/nba_corrupted.json

# Different model and temperature
uv run cb-corrupt generate -g nba --model gemini-2.5-flash --temperature 0.5

# With cost tracking
uv run cb-corrupt generate -g nba --pricing-input 0.15 --pricing-output 0.60
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--benchmark-path` | Benchmark JSON path relative to project root | `benchmark/test.json` |
| `-g`, `--graphs` | Graphs to corrupt (repeatable) | all 11 |
| `-c`, `--corruption-types` | Corruption types (repeatable) | all 9 |
| `--num-samples` | Max corrupted samples per type per graph | `50` |
| `--model` | LLM model name | `gemini-3.1-pro-preview` |
| `--temperature` | LLM temperature | `0.7` |
| `--pricing-input` | Input price per 1M tokens (USD) | none |
| `--pricing-output` | Output price per 1M tokens (USD) | none |
| `--seed` | Random seed | `42` |
| `-o`, `--output` | Output JSON file | `output/corrupted.json` |
| `-v`, `--verbose` | Debug logging | off |

### `verify`

Verifies corrupted samples by executing Cypher queries against a Neo4j instance. For **ambiguity** samples, it runs each `valid_cyphers` variant and checks they produce different results. For **unanswerability** samples, it substitutes the corrupted element into the gold query and checks it fails with a specific error (e.g., missing property/label/relationship) or returns empty results.

#### Neo4j setup

```bash
# Start Neo4j via Docker
docker compose up -d

# Wait ~15s for startup, then load a graph
uv run python scripts/load_graph.py nba --password password

# Load multiple graphs
uv run python scripts/load_graph.py nba movie soccer --password password

# Load all 11 graphs
uv run python scripts/load_graph.py --all --password password
```

#### Running verification

```bash
# Set connection (Community edition uses a single "neo4j" database)
export NEO4J_PASSWORD=password
export NEO4J_DATABASE_MAP='{"nba":"neo4j"}'

# Verify
uv run cb-corrupt verify -i output/nba/all.json -o output/nba/verification.json -v
```

The `NEO4J_DATABASE_MAP` environment variable maps graph names to Neo4j database names. With Community edition, all graphs share the single `neo4j` database, so load and verify one graph at a time.

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Corrupted samples JSON file | `output/corrupted.json` |
| `--neo4j-uri` | Neo4j URI (or set `NEO4J_URI`) | `bolt://localhost:7687` |
| `--neo4j-username` | Neo4j username (or set `NEO4J_USERNAME`) | `neo4j` |
| `--neo4j-password` | Neo4j password (or set `NEO4J_PASSWORD`) | |
| `-o`, `--output` | Output JSON file | `output/verification.json` |
| `-v`, `--verbose` | Debug logging | off |

**Verification statuses:**

| Status | Meaning |
|--------|---------|
| `pass` | Ambiguity: all Cypher variants return different results. Unanswerability: query fails with specific error or returns empty. |
| `fail` | Ambiguity: some variants return identical results. Unanswerability: query returns actual data (corruption didn't break it). |
| `error` | Query execution failed unexpectedly (connection issue, syntax error, etc.) |

### `evaluate` (scripts/evaluate.py)

Batch precision & recall evaluator. Given a JSON file of samples — each with a list of predicted Cypher queries and a list of target queries — loads each graph into Neo4j once, evaluates all samples for that graph in parallel, and streams results to a JSONL file.

#### Input format

```json
[
  {
    "qid": "abc-123",
    "graph_name": "nba",
    "predictions": ["MATCH (p:Player) RETURN p.name"],
    "targets":     ["MATCH (p:Player) RETURN p.name"]
  }
]
```

`graph_name`, `predictions`, and `targets` are required. Any extra fields (e.g. `qid`, `nl_question`) are passed through unchanged to the output.

#### Output format (JSONL)

One line per sample — the input object merged with all `EvalResult` fields:

```json
{"qid": "abc-123", "graph_name": "nba", "predictions": [...], "targets": [...], "precision": 1.0, "recall": 1.0, "n_predictions": 1, "n_targets": 1, "n_pred_unique": 1, "n_target_unique": 1, "n_matched": 1, "failed_predictions": []}
```

> **Note:** Output line order is non-deterministic due to parallel execution. Use the `qid` (or equivalent passthrough field) to match results back to input samples.

#### Running

```bash
# Basic usage (Community edition — map all graphs to the single neo4j database)
export NEO4J_PASSWORD=password
export NEO4J_DATABASE_MAP='{"nba":"neo4j","geography":"neo4j"}'

uv run python scripts/evaluate.py \
  -i predictions.json \
  -o results.jsonl

# With parallelism control
uv run python scripts/evaluate.py -i predictions.json -o results.jsonl --workers 8

# Resume an interrupted run (skips already-written lines by position)
uv run python scripts/evaluate.py -i predictions.json -o results.jsonl
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--input` | Input JSON file with predictions and targets | required |
| `-o`, `--output` | Output JSONL file | required |
| `--uri` | Neo4j bolt URI (or `NEO4J_URI`) | `bolt://localhost:7687` |
| `--username` | Neo4j username (or `NEO4J_USERNAME`) | `neo4j` |
| `--password` | Neo4j password (or `NEO4J_PASSWORD`) | `password` |
| `--database` | Neo4j target database (or `NEO4J_DATABASE`) | `neo4j` |
| `--workers` | Max parallel `evaluate()` calls per graph group | `4` |
| `-v`, `--verbose` | Debug logging | off |

**Precision & recall definition:**

```
precision = |matched result sets| / |unique predicted result sets|
recall    = |matched result sets| / |unique target result sets|
```

Two Cypher queries are considered equal if and only if their executed result sets are identical after normalization (key ordering, int/float coercion, Neo4j temporal/spatial types). `ORDER BY` queries are compared preserving row order.

### Parallel debugging script

Run all 9 corruption types in parallel on a single graph:

```bash
bash scripts/run_all.sh <graph> [num_samples]

# Examples
bash scripts/run_all.sh nba 5
bash scripts/run_all.sh movie 10
```

Output goes to `output/<graph>/<TYPE>.json` with logs in `output/<graph>/<TYPE>.log`.

## Corruption Types

### Ambiguity (question has 2+ valid Cypher interpretations)

| Type | Name | Description |
|------|------|-------------|
| A1 | Relation Ambiguity | Question uses a term that could refer to 2+ relation types (e.g., "associated with" could mean `playsFor` or `draftedBy`) |
| A2 | Property Ambiguity | Question uses a term that maps to 2+ properties (e.g., "physical characteristic" could mean `height_cm` or `mass_kg`) |
| A3 | Entity Type Ambiguity | Question uses a term that refers to 2+ entity types connected to the same target |
| A5 | Direction Ambiguity | Question doesn't clarify relationship direction for self-referential relations |

### Unanswerability (no valid Cypher can answer the question)

| Type | Name | Description |
|------|------|-------------|
| U1 | Missing Property | Question asks for a property not in the schema (e.g., `wingspan_cm` for Player) |
| U2 | Missing Relation | Question presupposes a relation not in the schema (e.g., `hadJerseyRetiredBy`) |
| U3 | Missing Entity Type | Question refers to an entity type not in the graph (e.g., Coach) |
| U4 | Out-of-Schema Constraint | Question filters on a nonexistent property (e.g., `WHERE jersey_number = 15`) |
| U5 | Temporal Unanswerability | Question asks for temporal info the graph doesn't track (e.g., "what year did they join the division?") |

## Available Graphs

`art`, `biology`, `company`, `fictional_character`, `flight_accident`, `geography`, `movie`, `nba`, `politics`, `soccer`, `terrorist_attack`

## Output Format

Each corrupted sample is a JSON object:

```json
{
  "corruption_id": "A1-nba-45c57262",
  "corruption": {
    "corruption_type": "A1",
    "corruption_category": "ambiguity",
    "original_element": "playsFor",
    "corrupted_element": "associated with",
    "candidate_interpretations": ["playsFor", "draftedBy"],
    "reason_unanswerable": null
  },
  "original_qid": "4c1e37f4-...",
  "original_graph": "nba",
  "original_nl_question": "How many players have played for the Houston Rockets?",
  "original_gold_cypher": "MATCH (n:Player)-[r0:playsFor]->...",
  "corrupted_nl_question": "How many players have been associated with the Houston Rockets?",
  "valid_cyphers": ["MATCH ...playsFor...", "MATCH ...draftedBy..."],
  "expected_answer": "AMBIGUOUS"
}
```

- **Ambiguity samples** have `valid_cyphers` listing all valid interpretations and `expected_answer: "AMBIGUOUS"`
- **Unanswerability samples** have empty `valid_cyphers`, a `reason_unanswerable`, and `expected_answer: "UNANSWERABLE"`

## Project Structure

```
cypherbench-corruptions/
├── benchmark/                 # CypherBench data (downloaded separately, gitignored)
│   ├── test.json
│   └── graphs/simplekg/       # Schema files per graph
├── graph_info.json            # Relation metadata for all 11 graphs
├── docker-compose.yml         # Neo4j for verification
├── benchmark.py               # Model benchmark runner (naive + aware prompt modes)
├── llm_verification.py        # LLM-as-judge verification of corrupted samples
├── src/cb_corruptions/
│   ├── cli.py                 # CLI entry point (cb-corrupt)
│   ├── pipeline.py            # Pipeline orchestration
│   ├── schema.py              # Graph schema data models
│   ├── schema_loader.py       # Data loading from local files
│   ├── cypher_parser.py       # Cypher query parsing
│   ├── cypher_substitutor.py  # Element substitution in Cypher queries
│   ├── graph_analysis.py      # Static schema analysis
│   ├── llm.py                 # LLM client wrapper
│   ├── models.py              # Pydantic models
│   ├── corruptions/           # One module per corruption type (A1, A2, A3, A5, U1–U5)
│   └── verification/          # Query verification and evaluation against Neo4j
│       ├── models.py          # Verification and EvalResult models
│       ├── neo4j_client.py    # Neo4j driver wrapper
│       ├── utils.py           # Result normalization and fingerprinting
│       ├── verifier.py        # Ambiguity & unanswerability checks
│       └── evaluator.py       # Precision/recall evaluation function
├── scripts/
│   ├── load_graph.py                          # Shared utility: load SimpleKG into Neo4j
│   ├── dataset_construction/                  # Phase 1: corruption generation → curation
│   │   ├── run_all.sh                         # Parallel run of all 9 corruption types
│   │   ├── verify_all.sh                      # Load + verify all graphs sequentially
│   │   ├── curate_dataset.py                  # Build the 200-sample subset (v3)
│   │   ├── find_clean_replacements.py         # Find clean A* replacements + cache results
│   │   └── substitute_bad_samples.py          # Build v4: swap 47 empty-result samples
│   └── evaluation/                            # Phases 2–4: cache → benchmark → metrics
│       ├── run_benchmark.sh                   # Wrapper around benchmark.py
│       ├── run_benchmark_v4_new.sh            # Benchmark runner for the v4 dataset
│       ├── cache_query_results.py             # Execute every (graph, cypher) → query_cache
│       ├── evaluate.py                        # Standalone batch precision/recall evaluator
│       ├── run_evaluation.py                  # End-to-end eval of one (model, mode) run
│       ├── run_evaluation_batch.py            # Graph-major batch eval across all runs
│       ├── check_soft_match_batch.py          # Soft-match (forward/inverse/either)
│       ├── value_only_eval.py                 # Value-only (alias-stripped) evaluation
│       ├── eval_v4.py                         # Build v4 evaluation set from cached results
│       └── compute_all_metrics.py             # Paper-grade tables (P/R/F1, CIs, McNemar)
├── output/
│   ├── final_output/                          # Generated corruptions + LLM verification per graph
│   ├── curated_200_v4/                        # 200-sample curated dataset used in the paper
│   └── query_cache/                           # Cached Neo4j results per (graph, cypher)
└── tests/                                     # Unit tests for verification utilities
```

## Query cache

`output/query_cache/` contains pre-executed Neo4j results — every (graph, cypher) pair seen in the v3 and v4 evaluations. It ships as a 44 MB compressed tarball (`output/query_cache.tar.gz`); extract it once before running the evaluation pipeline:

```bash
tar xzf output/query_cache.tar.gz -C output/
```

This produces `output/query_cache/<graph>.jsonl` (one cached row per cypher) plus a `cache_index.json`. The scripts in `scripts/evaluation/` read from this directory.

Without the cache the evaluation scripts still work — they will re-execute every query against Neo4j on first run (~30 min for the full v4 set on a laptop) and write the cache locally.

## Two ways to use this repo

### Just reproduce the paper numbers

The curated dataset and cached query results are already shipped, so you don't need to regenerate corruptions or call the verifier LLM.

```bash
# 1. Start Neo4j (the v4_new_only script loads graphs one at a time)
docker compose up -d

# 2. Run the model benchmark on the curated v4 dataset (will take a while)
bash scripts/evaluation/run_benchmark_v4_new.sh

# 3. Evaluate predictions (graph-major, resumable)
uv run python scripts/evaluation/run_evaluation_batch.py

# 4. Soft-match and value-only variants
uv run python scripts/evaluation/check_soft_match_batch.py
uv run python scripts/evaluation/value_only_eval.py

# 5. Build the v4 evaluation set + final paper tables
uv run python scripts/evaluation/eval_v4.py
uv run python scripts/evaluation/compute_all_metrics.py
```

`output/query_cache/` already contains the executed results for every (graph, cypher) pair seen in v3 and v4, so steps 3–5 do not hit Neo4j again unless new predictions introduce new queries.

### Rebuild the dataset from scratch

If you want to regenerate the corrupted samples — to inspect the corruption methodology, retune prompts, or extend to a different benchmark — run Phase 1 first:

```bash
# 1. Generate corruption candidates for every graph (LLM calls)
bash scripts/dataset_construction/run_all.sh gemini-3.1-pro-preview 100

# 2. Verify candidates against Neo4j (execution-based filter)
docker compose up -d
bash scripts/dataset_construction/verify_all.sh

# 3. LLM-as-judge verification (naturalness, effectiveness, correctness)
uv run python llm_verification.py run-all

# 4. Curate the 200-sample subset
uv run python scripts/dataset_construction/curate_dataset.py

# 5. Replace samples whose gold queries return empty results
uv run python scripts/dataset_construction/find_clean_replacements.py
uv run python scripts/dataset_construction/substitute_bad_samples.py
```

The output of Phase 1 is `output/curated_200_v4/`, which is the input to the reproduction pipeline above.
