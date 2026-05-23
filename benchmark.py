#!/usr/bin/env python3
"""Benchmark NL→Cypher models on corrupted CypherBench samples.

Pipeline:
    1. Load corrupted samples (`full_<graph>.json`) and the LLM verifier
       output (`llm_verified_<graph>.json`).
    2. Filter to samples that passed LLM verification. For ambiguity samples
       additionally require a non-empty `valid_cyphers` list.
    3. For each kept sample, prompt the model with schema + question and ask
       it to return a list of Cypher queries — one query per plausible
       interpretation, or an empty list when the question is unanswerable.
       Two prompt modes:
         - naive : schema + question only (model is expected to return a
                   single query, or empty list if it cannot)
         - aware : schema + question + ambiguity/unanswerability awareness
                   (model is asked to enumerate ALL plausible interpretations
                   for ambiguous questions, and return an empty list for
                   unanswerable ones)
    4. Save predictions to
       `output/benchmark/<model_slug>/<mode>/<graph>.json`.

Usage:
    # Single graph (uses OPENAI_BASE_URL / OPENAI_API_KEY from .env)
    uv run python benchmark.py run --graph nba --mode naive

    # All 11 graphs, both modes
    uv run python benchmark.py run-all --model gemini-3.1-pro-preview --mode both

    # OpenRouter (set env or pass flags)
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \\
    OPENAI_API_KEY=$OPENROUTER_API_KEY \\
    OPENROUTER_REFERER=https://example.com \\
    OPENROUTER_TITLE="cypherbench-corruptions" \\
        uv run python benchmark.py run-all --model openai/gpt-5.2 --mode aware
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from cb_corruptions.llm import LLM
from cb_corruptions.models import AMBIGUITY_TYPES, CorruptedSample, CorruptionType
from cb_corruptions.schema import PropertyGraphSchema
from cb_corruptions.schema_loader import ALL_GRAPHS, load_schema

load_dotenv()
load_dotenv("local.env", override=True)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="bench-eval",
    help="Benchmark NL→Cypher models on corrupted CypherBench samples.",
    no_args_is_help=True,
)


VALID_MODES = ("naive", "aware")


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class CypherPrediction(BaseModel):
    """Model's response: one Cypher query per plausible interpretation.

    - Exactly one query  → the question has a single clear interpretation.
    - Two or more queries → the question is ambiguous; each query corresponds
      to one plausible interpretation.
    - Empty list         → the question is unanswerable given the schema.
    """

    cyphers: list[str] = Field(
        default_factory=list,
        description=(
            "One Cypher query per plausible interpretation of the question. "
            "Return a single-element list when the question has one clear "
            "reading. Return multiple queries when the question is ambiguous "
            "and admits several mutually exclusive interpretations — include "
            "ALL of them. Return an empty list when no valid Cypher query "
            "can answer the question (the question references schema elements "
            "that do not exist)."
        ),
    )


class BenchmarkResult(BaseModel):
    corruption_id: str
    corruption_type: str
    graph: str
    model: str
    mode: str
    corrupted_nl_question: str
    expected_answer: str
    gold_valid_cyphers: list[str] = []
    original_gold_cypher: str
    predicted_cyphers: list[str] = []
    error: Optional[str] = None
    latency_ms: int = 0
    # Token/cost usage from the OpenAI/OpenRouter usage object. None when
    # the provider does not report a given field (e.g. reasoning_tokens on
    # non-reasoning models, cost_openrouter outside OpenRouter).
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cost_openrouter: Optional[float] = None
    # Full reasoning trace text. Excluded from the main JSON output and
    # persisted to a sidecar <graph>.reasoning.jsonl instead.
    reasoning: Optional[str] = Field(default=None, exclude=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_NAIVE = (
    "You are a Cypher query generator for a knowledge graph. Given a graph "
    "schema and a natural language question, produce the Cypher query that "
    "answers the question.\n\n"
    "Return your answer as a structured object with one field, `cyphers`, a "
    "list of Cypher query strings. In normal cases the list contains exactly "
    "one query — the one that answers the question. If you genuinely cannot "
    "produce any valid Cypher query, return an empty list."
)

SYSTEM_AWARE = (
    "You are a Cypher query generator for a knowledge graph. Given a graph "
    "schema and a natural language question, produce the Cypher query (or "
    "queries) that answer the question.\n\n"
    "Return your answer as a structured object with one field, `cyphers`, a "
    "list of Cypher query strings. The length of the list expresses your "
    "judgment about the question:\n\n"
    "  - CLEAR question → return EXACTLY ONE Cypher query in the list.\n"
    "  - AMBIGUOUS question → return MULTIPLE Cypher queries, one for each "
    "plausible interpretation. A question is ambiguous when it admits two or "
    "more mutually exclusive readings that map to different schema elements "
    "(for example, a vague term that could refer to several different "
    "properties, relations, or entity types, or a self-referential relation "
    "with no clear direction). Do NOT pick one and discard the others — "
    "enumerate ALL the plausible interpretations you can identify, and "
    "produce a separate Cypher query for each.\n"
    "  - UNANSWERABLE question → return an EMPTY list. A question is "
    "unanswerable when it references a property, relation, or entity type "
    "that does not exist in the schema, so no valid Cypher query can answer "
    "it.\n\n"
    "Be honest about ambiguity: if you are torn between two readings, return "
    "both rather than guessing. Be honest about unanswerability: if the "
    "schema does not support the question, return an empty list rather than "
    "fabricating a query."
)


def _build_schema_summary(schema: PropertyGraphSchema) -> str:
    lines = [f"Graph: {schema.name}", "", "ENTITY TYPES:"]
    for entity in schema.entities:
        props = {k: v.value for k, v in entity.properties.items() if k != "name"}
        if props:
            prop_str = ", ".join(f"{k} ({v})" for k, v in props.items())
            lines.append(f"  {entity.label}: {prop_str}")
        else:
            lines.append(f"  {entity.label}: (no properties besides name)")
    lines.append("")
    lines.append("RELATIONSHIPS:")
    for rel in schema.relations:
        suffix = ""
        if rel.properties:
            suffix = " [properties: " + ", ".join(rel.properties) + "]"
        lines.append(
            f"  ({rel.subj_label})-[:{rel.label}]->({rel.obj_label}){suffix}"
        )
    return "\n".join(lines)


def _build_user_prompt(sample: CorruptedSample, schema: PropertyGraphSchema) -> str:
    return (
        f"## Knowledge Graph Schema\n{_build_schema_summary(schema)}\n\n"
        f"## Question\n{sample.corrupted_nl_question}\n\n"
        "Return `cyphers` as a list of Cypher queries: one query for a clear "
        "question, one query per interpretation for an ambiguous question, "
        "or an empty list if the question is unanswerable."
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _load_passing_ids(verification_path: Path) -> set[str]:
    """corruption_ids whose LLM verifier verdict was 'pass'."""
    with open(verification_path) as f:
        data = json.load(f)
    return {item["corruption_id"] for item in data if item.get("verdict") == "pass"}


def _filter_samples(
    samples: list[CorruptedSample], passing_ids: set[str]
) -> list[CorruptedSample]:
    out: list[CorruptedSample] = []
    for s in samples:
        if s.corruption_id not in passing_ids:
            continue
        ctype = CorruptionType(s.corruption.corruption_type)
        if ctype in AMBIGUITY_TYPES and not s.valid_cyphers:
            continue
        out.append(s)
    return out


def _cap_per_type(samples: list[CorruptedSample], k: int) -> list[CorruptedSample]:
    """Keep at most `k` samples per corruption type, preserving order."""
    counts: dict[str, int] = {}
    out: list[CorruptedSample] = []
    for s in samples:
        ctype = s.corruption.corruption_type
        if counts.get(ctype, 0) >= k:
            continue
        counts[ctype] = counts.get(ctype, 0) + 1
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    def __init__(
        self,
        llm: LLM,
        mode: str,
        max_retries: int = 3,
        max_workers: int = 10,
    ) -> None:
        self.llm = llm
        self.mode = mode
        self.max_retries = max_retries
        self.max_workers = max_workers
        self._schemas: dict[str, PropertyGraphSchema] = {}

    def _schema(self, graph: str) -> PropertyGraphSchema:
        if graph not in self._schemas:
            self._schemas[graph] = load_schema(graph)
        return self._schemas[graph]

    def predict(self, sample: CorruptedSample) -> BenchmarkResult:
        schema = self._schema(sample.original_graph)
        system = SYSTEM_AWARE if self.mode == "aware" else SYSTEM_NAIVE
        user = _build_user_prompt(sample, schema)

        result = BenchmarkResult(
            corruption_id=sample.corruption_id,
            corruption_type=sample.corruption.corruption_type,
            graph=sample.original_graph,
            model=self.llm.model,
            mode=self.mode,
            corrupted_nl_question=sample.corrupted_nl_question,
            expected_answer=sample.expected_answer,
            gold_valid_cyphers=sample.valid_cyphers,
            original_gold_cypher=sample.original_gold_cypher,
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                resp, usage = self.llm.structured_with_usage(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_model=CypherPrediction,
                    temperature=0.0,
                )
                result.latency_ms = int((time.perf_counter() - t0) * 1000)
                result.predicted_cyphers = resp.cyphers
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
                result.reasoning_tokens = usage.get("reasoning_tokens")
                result.cached_tokens = usage.get("cached_tokens")
                result.cost_openrouter = usage.get("cost_openrouter")
                result.reasoning = usage.get("reasoning")
                return result
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    backoff = 2**attempt
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s (retrying in %ds)",
                        attempt + 1, self.max_retries, sample.corruption_id, e, backoff,
                    )
                    time.sleep(backoff)

        result.error = f"Failed after {self.max_retries} attempts: {last_err}"
        return result

    def predict_batch(self, samples: list[CorruptedSample]) -> list[BenchmarkResult]:
        # Preload schemas once on the main thread to avoid races
        for s in samples:
            self._schema(s.original_graph)

        results: list[BenchmarkResult | None] = [None] * len(samples)

        def _do(idx: int, s: CorruptedSample) -> tuple[int, BenchmarkResult]:
            return idx, self.predict(s)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_do, i, s) for i, s in enumerate(samples)]
            desc = f"{self.llm.model}/{self.mode}"
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx, r = fut.result()
                results[idx] = r

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def _output_path(output_dir: Path, model: str, mode: str, graph: str) -> Path:
    return output_dir / _model_slug(model) / mode / f"{graph}.json"


def _load_corrupted(input_dir: Path, graph: str) -> list[CorruptedSample]:
    path = input_dir / f"full_{graph}.json"
    with open(path) as f:
        return [CorruptedSample(**s) for s in json.load(f)]


def _make_llm(
    model: str,
    temperature: float = 0.0,
    provider_ignore: list[str] | None = None,
    provider_only: list[str] | None = None,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = 1000,
    tool_choice_mode: str | None = None,
) -> LLM:
    """Build an LLM client. Uses OPENAI_API_KEY / OPENAI_BASE_URL from env.
    Optional OpenRouter ranking headers are picked up from env vars.

    OpenRouter-specific routing knobs:
      - provider_ignore: list of provider names to exclude (e.g. ["AtlasCloud"])
      - provider_only:   list of provider names to allow (overrides ignore)
      - reasoning_effort: "low" | "medium" | "high" — force-enable reasoning
      - reasoning_max_tokens: hard cap on reasoning tokens per call. Default
        1000 (~p90 observed on Qwen3.5-27B). Bounds runaway thinking storms
        on hypernym-style A3 questions where the model can spiral into
        80K+ reasoning_tokens. Pass 0 / None to disable the cap.
    """
    headers: dict[str, str] = {}
    if referer := os.getenv("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_TITLE"):
        headers["X-Title"] = title

    extra_body: dict[str, object] = {}
    if provider_only:
        extra_body["provider"] = {"order": list(provider_only), "allow_fallbacks": False}
    elif provider_ignore:
        extra_body["provider"] = {"ignore": list(provider_ignore)}
    reasoning_block: dict[str, object] = {}
    if reasoning_effort:
        reasoning_block["effort"] = reasoning_effort
    if reasoning_max_tokens:
        reasoning_block["max_tokens"] = reasoning_max_tokens
    if reasoning_block:
        extra_body["reasoning"] = reasoning_block

    return LLM(
        model=model,
        temperature=temperature,
        default_headers=headers or None,
        extra_body=extra_body or None,
        tool_choice_mode=tool_choice_mode,
    )


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Core (used by both `run` and `run-all`)
# ---------------------------------------------------------------------------


def _run_graph(
    graph: str,
    model: str,
    mode: str,
    input_dir: Path,
    verification_dir: Path,
    output_dir: Path,
    num_samples: int | None,
    per_type: int | None,
    max_workers: int,
    max_retries: int,
    skip_existing: bool,
    llm: LLM | None = None,
) -> None:
    if mode not in VALID_MODES:
        raise typer.BadParameter(f"Unknown mode: {mode} (use {' or '.join(VALID_MODES)})")

    out = _output_path(output_dir, model, mode, graph)
    if skip_existing and out.exists():
        typer.echo(f"[skip] {out} already exists")
        return

    full_path = input_dir / f"full_{graph}.json"
    verif_path = verification_dir / f"llm_verified_{graph}.json"
    if not full_path.exists():
        typer.echo(f"[skip] {full_path} not found", err=True)
        return
    if not verif_path.exists():
        typer.echo(f"[skip] {verif_path} not found", err=True)
        return

    samples = _load_corrupted(input_dir, graph)
    passing_ids = _load_passing_ids(verif_path)
    filtered = _filter_samples(samples, passing_ids)
    typer.echo(
        f"{graph}: {len(samples)} corrupted, "
        f"{len(passing_ids)} passed verifier, "
        f"{len(filtered)} after ambiguity-non-empty filter"
    )

    if per_type is not None:
        filtered = _cap_per_type(filtered, per_type)
        typer.echo(f"  capped to {per_type} per type -> {len(filtered)}")
    if num_samples is not None:
        filtered = filtered[:num_samples]
        typer.echo(f"  capped to {len(filtered)} total")

    if not filtered:
        typer.echo("  (no samples to benchmark)")
        return

    runner = BenchmarkRunner(
        llm=llm or _make_llm(model),
        mode=mode,
        max_retries=max_retries,
        max_workers=max_workers,
    )
    results = runner.predict_batch(filtered)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    # Sidecar reasoning log: one JSONL per call, only written if at least
    # one record carries a reasoning trace.
    if any(r.reasoning for r in results):
        log_path = out.with_suffix(".reasoning.jsonl")
        with open(log_path, "w") as f:
            for r in results:
                if r.reasoning:
                    f.write(json.dumps({
                        "corruption_id": r.corruption_id,
                        "reasoning": r.reasoning,
                    }, ensure_ascii=False) + "\n")

    empty_count = sum(1 for r in results if r.error is None and not r.predicted_cyphers)
    single_count = sum(1 for r in results if r.error is None and len(r.predicted_cyphers) == 1)
    multi_count = sum(1 for r in results if r.error is None and len(r.predicted_cyphers) >= 2)
    err_count = sum(1 for r in results if r.error is not None)
    typer.echo(
        f"  -> {out} | {len(results)} samples, "
        f"{single_count} single, {multi_count} multi, "
        f"{empty_count} empty, {err_count} errored"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def run(
    graph: Annotated[str, typer.Option("--graph", "-g", help="Graph to benchmark")],
    model: Annotated[str, typer.Option("--model", "-m")] = "gemini-3.1-pro-preview",
    mode: Annotated[str, typer.Option("--mode", help="naive or aware")] = "naive",
    input_dir: Annotated[Path, typer.Option(help="Dir with full_<graph>.json")] = Path("output/curated_200_v3"),
    verification_dir: Annotated[Path, typer.Option(help="Dir with llm_verified_<graph>.json")] = Path("output/curated_200_v3/verification"),
    output_dir: Annotated[Path, typer.Option(help="Output base dir")] = Path("output/benchmark"),
    num_samples: Annotated[Optional[int], typer.Option(help="Cap total samples per graph")] = None,
    per_type: Annotated[Optional[int], typer.Option(help="Cap samples per corruption type")] = None,
    max_workers: Annotated[int, typer.Option(help="Concurrent LLM calls")] = 10,
    max_retries: Annotated[int, typer.Option()] = 3,
    skip_existing: Annotated[bool, typer.Option("--skip-existing/--no-skip-existing")] = True,
    provider_ignore: Annotated[Optional[list[str]], typer.Option("--provider-ignore", help="OpenRouter provider names to exclude (e.g. AtlasCloud)")] = None,
    provider_only: Annotated[Optional[list[str]], typer.Option("--provider-only", help="OpenRouter provider names to allow (overrides --provider-ignore)")] = None,
    reasoning_effort: Annotated[Optional[str], typer.Option("--reasoning-effort", help="Force reasoning on: low/medium/high")] = None,
    reasoning_max_tokens: Annotated[int, typer.Option("--reasoning-max-tokens", help="Hard cap on reasoning tokens per call (default 1000). Set 0 to disable.")] = 1000,
    tool_choice: Annotated[Optional[str], typer.Option("--tool-choice", help="Override tool_choice: auto/required/none (default: explicit function-object)")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the benchmark on a single graph."""
    _setup_logging(verbose)
    llm = _make_llm(
        model,
        provider_ignore=provider_ignore,
        provider_only=provider_only,
        reasoning_effort=reasoning_effort,
        reasoning_max_tokens=reasoning_max_tokens or None,
        tool_choice_mode=tool_choice,
    )
    _run_graph(
        graph=graph, model=model, mode=mode,
        input_dir=input_dir, verification_dir=verification_dir,
        output_dir=output_dir, num_samples=num_samples, per_type=per_type,
        max_workers=max_workers, max_retries=max_retries,
        skip_existing=skip_existing,
        llm=llm,
    )


@app.command("run-all")
def run_all(
    model: Annotated[str, typer.Option("--model", "-m")] = "gemini-3.1-pro-preview",
    mode: Annotated[str, typer.Option("--mode", help="naive, aware, or both")] = "both",
    graphs: Annotated[Optional[list[str]], typer.Option("--graphs", "-g", help="Subset of graphs (default: all 11)")] = None,
    input_dir: Annotated[Path, typer.Option()] = Path("output/curated_200_v3"),
    verification_dir: Annotated[Path, typer.Option()] = Path("output/curated_200_v3/verification"),
    output_dir: Annotated[Path, typer.Option()] = Path("output/benchmark"),
    num_samples: Annotated[Optional[int], typer.Option(help="Cap total samples per graph")] = None,
    per_type: Annotated[Optional[int], typer.Option(help="Cap samples per corruption type")] = None,
    max_workers: Annotated[int, typer.Option()] = 10,
    max_retries: Annotated[int, typer.Option()] = 3,
    skip_existing: Annotated[bool, typer.Option("--skip-existing/--no-skip-existing")] = True,
    provider_ignore: Annotated[Optional[list[str]], typer.Option("--provider-ignore", help="OpenRouter provider names to exclude (e.g. AtlasCloud)")] = None,
    provider_only: Annotated[Optional[list[str]], typer.Option("--provider-only", help="OpenRouter provider names to allow (overrides --provider-ignore)")] = None,
    reasoning_effort: Annotated[Optional[str], typer.Option("--reasoning-effort", help="Force reasoning on: low/medium/high")] = None,
    reasoning_max_tokens: Annotated[int, typer.Option("--reasoning-max-tokens", help="Hard cap on reasoning tokens per call (default 1000). Set 0 to disable.")] = 1000,
    tool_choice: Annotated[Optional[str], typer.Option("--tool-choice", help="Override tool_choice: auto/required/none (default: explicit function-object)")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the benchmark across all graphs (and optionally both modes)."""
    _setup_logging(verbose)

    target_modes = list(VALID_MODES) if mode == "both" else [mode]
    for m in target_modes:
        if m not in VALID_MODES:
            raise typer.BadParameter(f"Unknown mode: {m}")

    target_graphs = graphs or ALL_GRAPHS

    # Reuse a single LLM client across all (graph, mode) combinations so the
    # underlying HTTP client and connection pool are shared.
    llm = _make_llm(
        model,
        provider_ignore=provider_ignore,
        provider_only=provider_only,
        reasoning_effort=reasoning_effort,
        reasoning_max_tokens=reasoning_max_tokens or None,
        tool_choice_mode=tool_choice,
    )

    for m in target_modes:
        for g in target_graphs:
            typer.echo(f"\n=== {g} | mode={m} | model={model} ===")
            try:
                _run_graph(
                    graph=g, model=model, mode=m,
                    input_dir=input_dir, verification_dir=verification_dir,
                    output_dir=output_dir, num_samples=num_samples, per_type=per_type,
                    max_workers=max_workers, max_retries=max_retries,
                    skip_existing=skip_existing,
                    llm=llm,
                )
            except FileNotFoundError as e:
                typer.echo(f"  [skip] {e}", err=True)
            except Exception:
                logger.exception("Failed for %s/%s", g, m)


if __name__ == "__main__":
    app()
