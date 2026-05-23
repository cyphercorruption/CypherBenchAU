#!/usr/bin/env python3
"""LLM-based semantic verification of corrupted samples.

Uses a strong LLM to validate that corrupted questions are genuinely
ambiguous or unanswerable — checking naturalness, effectiveness,
and correctness beyond what query execution can verify.

Usage:
    uv run python llm_verification.py -i output/final_output/full_art.json
    uv run python llm_verification.py -i output/final_output/full_art.json -c A2 -c A3
    uv run python llm_verification.py -i output/final_output/full_art.json --model gemini-3.1-pro-preview --batch-size 10
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from cb_corruptions.llm import LLM
from cb_corruptions.models import AMBIGUITY_TYPES, CorruptedSample, CorruptionType
from cb_corruptions.schema import PropertyGraphSchema
from cb_corruptions.schema_loader import load_schema

load_dotenv()
load_dotenv("local.env", override=True)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="llm-verify",
    help="LLM-based semantic verification of corrupted benchmark samples.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Corruption type descriptions (human-readable, for the prompt)
# ---------------------------------------------------------------------------

CORRUPTION_DESCRIPTIONS: dict[str, str] = {
    "A1": (
        "Relation Ambiguity — the question uses a vague term that could refer "
        "to multiple different relationships in the schema, making it unclear "
        "which relationship is being asked about."
    ),
    "A2": (
        "Property Ambiguity — the question uses a vague term that could refer "
        "to multiple different properties on the same entity type, making it "
        "unclear which property is being queried."
    ),
    "A3": (
        "Entity Type Ambiguity — the question uses a vague term that could "
        "refer to multiple different entity types that share the same "
        "relationship, making it unclear which entity type is meant."
    ),
    "A5": (
        "Direction Ambiguity — the question is unclear about the direction of "
        "a self-referential relationship (A→B vs B→A), making both readings "
        "equally plausible."
    ),
    "U1": (
        "Missing Property — the question asks about a property that does not "
        "exist on any entity type in the schema."
    ),
    "U2": (
        "Missing Relation — the question references a relationship type that "
        "does not exist in the schema."
    ),
    "U3": (
        "Missing Entity Type — the question references an entity type that "
        "does not exist in the schema."
    ),
    "U4": (
        "Out-of-Schema Constraint — the question filters by a property that "
        "does not exist in the schema, making it impossible to apply the filter."
    ),
    "U5": (
        "Temporal Unanswerability — the question asks about temporal aspects "
        "(when, what year, dates) of a relationship that has no temporal "
        "properties in the schema."
    ),
}


# ---------------------------------------------------------------------------
# Response models (structured LLM output)
# ---------------------------------------------------------------------------


class AmbiguityAssessment(BaseModel):
    """LLM assessment of whether a corrupted question is genuinely ambiguous."""

    verdict: Literal["pass", "fail"] = Field(
        description="'pass' if the question is genuinely ambiguous, 'fail' otherwise"
    )
    reasoning: str = Field(
        description=(
            "Step-by-step reasoning: first identify what the ambiguous element "
            "is, then check each interpretation against the schema, then judge "
            "whether a real person would be confused."
        )
    )
    naturalness: int = Field(
        ge=1, le=5,
        description="How natural the question sounds (1=robotic/forced, 5=perfectly natural)",
    )
    ambiguity_strength: int = Field(
        ge=1, le=5,
        description=(
            "How strong the ambiguity is (1=one reading clearly dominates, "
            "5=interpretations are perfectly balanced and irresolvable)"
        ),
    )
    identified_interpretations: list[str] = Field(
        description="The distinct interpretations YOU identified (not the claimed ones)",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found (empty if pass)",
    )


class UnanswerabilityAssessment(BaseModel):
    """LLM assessment of whether a corrupted question is genuinely unanswerable."""

    verdict: Literal["pass", "fail"] = Field(
        description="'pass' if the question is genuinely unanswerable, 'fail' otherwise"
    )
    reasoning: str = Field(
        description=(
            "Step-by-step reasoning: first try to map the question to schema "
            "elements, then check if any path could answer it, then judge "
            "whether the fake element is plausible and distinct."
        )
    )
    naturalness: int = Field(
        ge=1, le=5,
        description="How natural the question sounds (1=robotic/forced, 5=perfectly natural)",
    )
    unanswerability_strength: int = Field(
        ge=1, le=5,
        description=(
            "How clearly unanswerable the question is (1=could be answered "
            "through an alternative path, 5=absolutely no way to answer it)"
        ),
    )
    closest_schema_element: str = Field(
        description=(
            "The real schema element most similar to the fake one "
            "(to check distinctness). Write 'none' if nothing is close."
        ),
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found (empty if pass)",
    )


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class LLMVerificationResult(BaseModel):
    """Final verification result for a single corrupted sample."""

    corruption_id: str
    corruption_type: str
    graph: str
    verdict: str
    reasoning: str
    naturalness: int
    effectiveness: int  # ambiguity_strength or unanswerability_strength
    issues: list[str]


# ---------------------------------------------------------------------------
# Schema summary builder
# ---------------------------------------------------------------------------


def build_schema_summary(schema: PropertyGraphSchema) -> str:
    """Build a structured text summary of the graph schema for the LLM prompt."""
    lines = [f"Graph: {schema.name}", ""]

    lines.append("ENTITY TYPES:")
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
        props_str = ""
        if rel.properties:
            props_str = " [properties: " + ", ".join(rel.properties) + "]"
        lines.append(f"  ({rel.subj_label})-[:{rel.label}]->({rel.obj_label}){props_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class LLMVerifier:
    """Validates corrupted samples using LLM-based semantic assessment."""

    def __init__(self, llm: LLM, batch_size: int = 5, max_retries: int = 3) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._schemas: dict[str, PropertyGraphSchema] = {}

    def load_schema(self, graph_name: str) -> PropertyGraphSchema:
        if graph_name not in self._schemas:
            self._schemas[graph_name] = load_schema(graph_name)
        return self._schemas[graph_name]

    def preload_schemas(self, graph_names: list[str]) -> None:
        """Load all needed schemas upfront to avoid thread contention on first call."""
        for g in set(graph_names):
            if g not in self._schemas:
                self._schemas[g] = load_schema(g)

    def verify(self, sample: CorruptedSample) -> LLMVerificationResult:
        """Verify a single corrupted sample."""
        schema = self.load_schema(sample.original_graph)
        ctype = CorruptionType(sample.corruption.corruption_type)

        if ctype in AMBIGUITY_TYPES:
            return self._verify_ambiguity(sample, schema)
        else:
            return self._verify_unanswerability(sample, schema)

    def verify_batch(
        self, samples: list[CorruptedSample], max_workers: int | None = None,
    ) -> list[LLMVerificationResult]:
        """Verify a batch of samples with parallel LLM calls."""
        workers = max_workers or self.batch_size
        results: list[LLMVerificationResult | None] = [None] * len(samples)

        def _do_verify(idx: int, sample: CorruptedSample):
            last_err: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    return idx, self.verify(sample)
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries - 1:
                        backoff = 2 ** attempt
                        logger.warning(
                            "LLM verification attempt %d/%d failed for %s: %s (retrying in %ds)",
                            attempt + 1, self.max_retries, sample.corruption_id, e, backoff,
                        )
                        time.sleep(backoff)
            logger.error(
                "LLM verification failed for %s after %d attempts: %s",
                sample.corruption_id, self.max_retries, last_err,
            )
            return idx, LLMVerificationResult(
                corruption_id=sample.corruption_id,
                corruption_type=sample.corruption.corruption_type,
                graph=sample.original_graph,
                verdict="error",
                reasoning=f"Verification failed after {self.max_retries} attempts: {last_err}",
                naturalness=0,
                effectiveness=0,
                issues=[str(last_err)],
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_do_verify, i, s)
                for i, s in enumerate(samples)
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="LLM Verifying"
            ):
                idx, result = future.result()
                results[idx] = result

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Ambiguity verification
    # ------------------------------------------------------------------

    def _verify_ambiguity(
        self, sample: CorruptedSample, schema: PropertyGraphSchema,
    ) -> LLMVerificationResult:
        schema_text = build_schema_summary(schema)
        ctype = sample.corruption.corruption_type
        desc = CORRUPTION_DESCRIPTIONS.get(ctype, ctype)

        interpretations_str = "\n".join(
            f"  - {interp}" for interp in sample.corruption.candidate_interpretations
        )

        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous quality assessor for a text-to-Cypher "
                        "benchmark. Your job is to verify whether corrupted questions "
                        "are GENUINELY ambiguous — meaning they admit multiple "
                        "exclusive interpretations that would lead to different "
                        "database queries and different answers.\n\n"
                        "Be strict. A question that SEEMS ambiguous but can be "
                        "resolved through context, common sense, or domain knowledge "
                        "should FAIL. Only questions with genuine, irresolvable "
                        "ambiguity should PASS.\n\n"
                        "Think step by step. First read the question as a naive user "
                        "would. Then check each interpretation against the schema. "
                        "Then make your judgment."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Knowledge Graph Schema\n{schema_text}\n\n"
                        f"## Original Question (before corruption)\n"
                        f"{sample.original_nl_question}\n\n"
                        f"## Corrupted Question (to evaluate)\n"
                        f"{sample.corrupted_nl_question}\n\n"
                        f"## Corruption Details\n"
                        f"- Type: {desc}\n"
                        f"- Ambiguous element used: \"{sample.corruption.corrupted_element}\"\n"
                        f"- Claimed interpretations:\n{interpretations_str}\n\n"
                        f"## Your Task\n"
                        "Evaluate whether the corrupted question is GENUINELY "
                        "ambiguous. Check ALL of the following:\n\n"
                        "1. **Multiple interpretations**: Read ONLY the corrupted "
                        "question. Does it genuinely admit 2+ different readings "
                        "that map to different schema elements? Identify them "
                        "yourself — do not just accept the claimed ones.\n\n"
                        "2. **Exclusivity**: Would each interpretation produce a "
                        "DIFFERENT database query with DIFFERENT results? If two "
                        "interpretations would return the same data, the ambiguity "
                        "is not useful.\n\n"
                        "3. **Balance**: Is each interpretation roughly equally "
                        "plausible? If one reading is obviously the 'correct' one "
                        "and the others are a stretch, it's not truly ambiguous.\n\n"
                        "4. **Naturalness**: Does the question sound like something "
                        "a real person would naturally ask? Flag awkward phrasing, "
                        "schema jargon, or forced constructions.\n\n"
                        "5. **Irresolvability**: Could a knowledgeable person "
                        "resolve the ambiguity using context clues within the "
                        "question itself? If the question contains enough context "
                        "to disambiguate, it should FAIL."
                    ),
                },
            ],
            response_model=AmbiguityAssessment,
            temperature=0.0,
        )

        return LLMVerificationResult(
            corruption_id=sample.corruption_id,
            corruption_type=ctype,
            graph=sample.original_graph,
            verdict=resp.verdict,
            reasoning=resp.reasoning,
            naturalness=resp.naturalness,
            effectiveness=resp.ambiguity_strength,
            issues=resp.issues,
        )

    # ------------------------------------------------------------------
    # Unanswerability verification
    # ------------------------------------------------------------------

    def _verify_unanswerability(
        self, sample: CorruptedSample, schema: PropertyGraphSchema,
    ) -> LLMVerificationResult:
        schema_text = build_schema_summary(schema)
        ctype = sample.corruption.corruption_type
        desc = CORRUPTION_DESCRIPTIONS.get(ctype, ctype)

        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous quality assessor for a text-to-Cypher "
                        "benchmark. Your job is to verify whether corrupted "
                        "questions are GENUINELY unanswerable given a knowledge "
                        "graph schema — meaning the information needed to answer "
                        "them does not exist in the schema and cannot be derived "
                        "from it.\n\n"
                        "Be strict. If there is ANY plausible way to answer the "
                        "question using the schema — even through an indirect "
                        "path, a synonym, or a reinterpretation — it should FAIL.\n\n"
                        "Think step by step. First try to answer the question "
                        "using the schema. Exhaust all possible paths before "
                        "concluding it's unanswerable."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## Knowledge Graph Schema\n{schema_text}\n\n"
                        f"## Original Question (before corruption)\n"
                        f"{sample.original_nl_question}\n\n"
                        f"## Corrupted Question (to evaluate)\n"
                        f"{sample.corrupted_nl_question}\n\n"
                        f"## Corruption Details\n"
                        f"- Type: {desc}\n"
                        f"- Original schema element: \"{sample.corruption.original_element}\"\n"
                        f"- Fake element introduced: \"{sample.corruption.corrupted_element}\"\n"
                        f"- Claimed reason unanswerable: "
                        f"{sample.corruption.reason_unanswerable or 'N/A'}\n\n"
                        f"## Your Task\n"
                        "Evaluate whether the corrupted question is GENUINELY "
                        "unanswerable given the schema. Check ALL of the "
                        "following:\n\n"
                        "1. **Schema gap**: Does the question ask about something "
                        "that genuinely does NOT exist in the schema? Carefully "
                        "check every entity type, property, and relationship. "
                        "The fake element must not match or closely overlap with "
                        "any real element.\n\n"
                        "2. **No alternative path**: Is there NO other way to "
                        "answer this question using existing schema elements? "
                        "Check indirect paths — could the answer be derived by "
                        "combining multiple relationships or using a different "
                        "but related property?\n\n"
                        "3. **Plausibility**: Is the fake element plausible for "
                        "this domain? Would a real user reasonably ask about it? "
                        "If the fake element is absurd or obviously wrong for "
                        "the domain, it's low quality.\n\n"
                        "4. **Naturalness**: Does the question sound like something "
                        "a real person would naturally ask? Flag awkward phrasing, "
                        "schema jargon, placeholders, or forced constructions.\n\n"
                        "5. **Distinctness**: Is the fake element clearly distinct "
                        "from all real schema elements? If it's a synonym, "
                        "abbreviation, or near-duplicate of a real element, "
                        "the question might actually be answerable."
                    ),
                },
            ],
            response_model=UnanswerabilityAssessment,
            temperature=0.0,
        )

        return LLMVerificationResult(
            corruption_id=sample.corruption_id,
            corruption_type=ctype,
            graph=sample.original_graph,
            verdict=resp.verdict,
            reasoning=resp.reasoning,
            naturalness=resp.naturalness,
            effectiveness=resp.unanswerability_strength,
            issues=resp.issues,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.command()
def run(
    input_file: Annotated[Path, typer.Option("-i", "--input", help="Corrupted samples JSON file")],
    corruption_types: Annotated[Optional[list[str]], typer.Option("--corruption-types", "-c", help="Only verify these types (e.g. A1, U1)")] = None,
    model: Annotated[str, typer.Option(help="LLM model for verification")] = "gemini-3.1-pro-preview",
    temperature: Annotated[float, typer.Option(help="LLM temperature (0 recommended)")] = 0.0,
    batch_size: Annotated[int, typer.Option(help="Parallel verification workers")] = 5,
    max_retries: Annotated[int, typer.Option(help="Max retry attempts per failed call")] = 3,
    pricing_input: Annotated[Optional[float], typer.Option(help="Input price per 1M tokens (USD)")] = None,
    pricing_output: Annotated[Optional[float], typer.Option(help="Output price per 1M tokens (USD)")] = None,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output JSON file")] = Path("output/llm_verification.json"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Verify corrupted samples using LLM-based semantic assessment."""
    _setup_logging(verbose)

    if not input_file.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    with open(input_file) as f:
        samples = [CorruptedSample(**s) for s in json.load(f)]

    if corruption_types:
        type_set = {t.upper() for t in corruption_types}
        samples = [s for s in samples if s.corruption.corruption_type in type_set]

    if not samples:
        typer.echo("No samples to verify.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(samples)} samples from {input_file}")

    pricing = None
    if pricing_input is not None and pricing_output is not None:
        pricing = (pricing_input, pricing_output)

    llm = LLM(model=model, temperature=temperature, pricing=pricing, batch_size=batch_size)
    verifier = LLMVerifier(llm, batch_size=batch_size, max_retries=max_retries)
    verifier.preload_schemas([s.original_graph for s in samples])
    typer.echo("Schemas preloaded, starting LLM calls...")
    results = verifier.verify_batch(samples, max_workers=batch_size)

    # Write output
    passed = sum(1 for r in results if r.verdict == "pass")
    failed = sum(1 for r in results if r.verdict == "fail")
    errors = sum(1 for r in results if r.verdict == "error")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([r.model_dump(mode="json") for r in results], f, indent=2)

    typer.echo(f"\nLLM Verification: {passed} pass, {failed} fail, {errors} error -> {output}")
    typer.echo(f"LLM cost: ${llm.total_cost:.4f}")

    # Per-type breakdown
    by_type: dict[str, dict[str, int]] = {}
    avg_nat: dict[str, list[int]] = {}
    avg_eff: dict[str, list[int]] = {}
    for r in results:
        by_type.setdefault(r.corruption_type, {"pass": 0, "fail": 0, "error": 0})[r.verdict] += 1
        avg_nat.setdefault(r.corruption_type, []).append(r.naturalness)
        avg_eff.setdefault(r.corruption_type, []).append(r.effectiveness)

    typer.echo(f"\n{'Type':<6} {'Pass':>6} {'Fail':>6} {'Err':>6} {'Nat':>6} {'Eff':>6}")
    typer.echo("-" * 40)
    for ctype, counters in sorted(by_type.items()):
        nat = sum(avg_nat[ctype]) / len(avg_nat[ctype]) if avg_nat[ctype] else 0
        eff = sum(avg_eff[ctype]) / len(avg_eff[ctype]) if avg_eff[ctype] else 0
        typer.echo(
            f"{ctype:<6} {counters['pass']:>6} {counters['fail']:>6} "
            f"{counters.get('error', 0):>6} {nat:>5.1f} {eff:>5.1f}"
        )


@app.command("run-all")
def run_all(
    input_dir: Annotated[Path, typer.Option("--input-dir", help="Directory with full_<graph>.json files")] = Path("output/final_output"),
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Directory for llm_verified_<graph>.json outputs")] = Path("output/final_output/verification"),
    verification_dir: Annotated[Path, typer.Option("--verification-dir", help="Directory with verified_<graph>.json — required, only query-exec passed samples are verified")] = Path("output/final_output/verification"),
    graphs: Annotated[Optional[list[str]], typer.Option("--graph", "-g", help="Limit to specific graphs (repeat for multiple)")] = None,
    corruption_types: Annotated[Optional[list[str]], typer.Option("--corruption-types", "-c", help="Only verify these types")] = None,
    model: Annotated[str, typer.Option(help="LLM model")] = "gemini-3.1-pro-preview",
    temperature: Annotated[float, typer.Option(help="LLM temperature")] = 0.0,
    batch_size: Annotated[int, typer.Option(help="Parallel workers")] = 50,
    max_retries: Annotated[int, typer.Option(help="Max retry attempts per failed call")] = 3,
    pricing_input: Annotated[Optional[float], typer.Option(help="Input price per 1M tokens (USD)")] = None,
    pricing_output: Annotated[Optional[float], typer.Option(help="Output price per 1M tokens (USD)")] = None,
    skip_existing: Annotated[bool, typer.Option("--skip-existing/--no-skip-existing", help="Skip graphs whose output already exists")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run LLM verification across all graphs. Writes llm_verified_<graph>.json per graph."""
    _setup_logging(verbose)

    full_files = sorted(input_dir.glob("full_*.json"))
    if not full_files:
        typer.echo(f"No full_*.json files in {input_dir}", err=True)
        raise typer.Exit(1)

    if graphs:
        wanted = {g.lower() for g in graphs}
        full_files = [f for f in full_files if f.stem.replace("full_", "").lower() in wanted]

    output_dir.mkdir(parents=True, exist_ok=True)

    pricing = None
    if pricing_input is not None and pricing_output is not None:
        pricing = (pricing_input, pricing_output)

    llm = LLM(model=model, temperature=temperature, pricing=pricing, batch_size=batch_size)
    verifier = LLMVerifier(llm, batch_size=batch_size, max_retries=max_retries)

    type_set = {t.upper() for t in corruption_types} if corruption_types else None
    global_totals: dict[str, dict[str, int]] = {}
    per_graph_summary: list[tuple[str, int, int, int, int]] = []

    for full_file in full_files:
        graph_name = full_file.stem.replace("full_", "")
        out_path = output_dir / f"llm_verified_{graph_name}.json"

        if skip_existing and out_path.exists():
            typer.echo(f"[{graph_name}] skip — output exists: {out_path}")
            continue

        with open(full_file) as f:
            samples = [CorruptedSample(**s) for s in json.load(f)]

        if type_set:
            samples = [s for s in samples if s.corruption.corruption_type in type_set]

        verif_path = verification_dir / f"verified_{graph_name}.json"
        if not verif_path.exists():
            typer.echo(f"[{graph_name}] missing required verification file: {verif_path}", err=True)
            raise typer.Exit(1)
        with open(verif_path) as f:
            verified = json.load(f)
        passed_ids = {v["corruption_id"] for v in verified if v.get("status") == "pass"}
        before = len(samples)
        samples = [s for s in samples if s.corruption_id in passed_ids]
        typer.echo(f"[{graph_name}] filtered {before} -> {len(samples)} via query-exec pass")

        if not samples:
            typer.echo(f"[{graph_name}] no samples to verify, skipping")
            continue

        verifier.preload_schemas([graph_name])
        typer.echo(f"[{graph_name}] verifying {len(samples)} samples -> {out_path}")
        results = verifier.verify_batch(samples, max_workers=batch_size)

        with open(out_path, "w") as f:
            json.dump([r.model_dump(mode="json") for r in results], f, indent=2)

        p = sum(1 for r in results if r.verdict == "pass")
        fa = sum(1 for r in results if r.verdict == "fail")
        er = sum(1 for r in results if r.verdict == "error")
        per_graph_summary.append((graph_name, len(results), p, fa, er))

        for r in results:
            bucket = global_totals.setdefault(r.corruption_type, {"pass": 0, "fail": 0, "error": 0})
            bucket[r.verdict] += 1

        typer.echo(f"[{graph_name}] {p} pass, {fa} fail, {er} error (cost so far: ${llm.total_cost:.4f})")

    typer.echo("\n=== Per-graph summary ===")
    typer.echo(f"{'Graph':<22} {'N':>5} {'Pass':>6} {'Fail':>6} {'Err':>6}")
    typer.echo("-" * 50)
    for g, n, p, fa, er in per_graph_summary:
        typer.echo(f"{g:<22} {n:>5} {p:>6} {fa:>6} {er:>6}")

    typer.echo("\n=== Aggregate per-type ===")
    typer.echo(f"{'Type':<6} {'Pass':>6} {'Fail':>6} {'Err':>6}")
    typer.echo("-" * 30)
    for t, c in sorted(global_totals.items()):
        typer.echo(f"{t:<6} {c['pass']:>6} {c['fail']:>6} {c['error']:>6}")

    typer.echo(f"\nTotal LLM cost: ${llm.total_cost:.4f}")


if __name__ == "__main__":
    app()
