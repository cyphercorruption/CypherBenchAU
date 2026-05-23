"""CLI entry point for cypherbench-corruptions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv

from cb_corruptions.llm import LLM
from cb_corruptions.models import ALL_CORRUPTION_TYPES, CorruptionType
from cb_corruptions.pipeline import PipelineConfig, run_analysis, run_pipeline
from cb_corruptions.schema_loader import ALL_GRAPHS

load_dotenv()
load_dotenv("local.env", override=True)

app = typer.Typer(
    name="cb-corrupt",
    help="Generate corrupted CypherBench samples: ambiguity and unanswerability perturbations.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # Suppress noisy Neo4j driver logs
    logging.getLogger("neo4j").setLevel(logging.WARNING)


def _parse_corruption_types(types: list[str] | None) -> list[CorruptionType]:
    if types is None:
        return ALL_CORRUPTION_TYPES
    return [CorruptionType(t) for t in types]


def _parse_graphs(graphs: list[str] | None) -> list[str]:
    if graphs is None:
        return ALL_GRAPHS
    for g in graphs:
        if g not in ALL_GRAPHS:
            raise typer.BadParameter(f"Unknown graph: {g}. Valid: {', '.join(ALL_GRAPHS)}")
    return graphs


@app.command()
def generate(
    benchmark_path: Annotated[str, typer.Option(help="Benchmark JSON path or directory relative to project root")] = "benchmark",
    graphs: Annotated[Optional[list[str]], typer.Option("--graphs", "-g", help="Graphs to corrupt")] = None,
    corruption_types: Annotated[Optional[list[str]], typer.Option("--corruption-types", "-c", help="Corruption types (A1,A2,A3,A5,U1-U5)")] = None,
    num_samples: Annotated[int, typer.Option(help="Max corrupted samples per type per graph")] = 50,
    model: Annotated[str, typer.Option(help="LLM model name")] = "gemini-3.1-pro-preview",
    temperature: Annotated[float, typer.Option(help="LLM temperature")] = 0.7,
    pricing_input: Annotated[Optional[float], typer.Option(help="Input price per 1M tokens (USD)")] = None,
    pricing_output: Annotated[Optional[float], typer.Option(help="Output price per 1M tokens (USD)")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output JSON file")] = Path("output/corrupted.json"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the corruption pipeline and produce corrupted samples."""
    _setup_logging(verbose)

    config = PipelineConfig(
        benchmark_path=benchmark_path,
        graphs=_parse_graphs(graphs),
        corruption_types=_parse_corruption_types(corruption_types),
        num_samples=num_samples,
        seed=seed,
    )

    pricing = None
    if pricing_input is not None and pricing_output is not None:
        pricing = (pricing_input, pricing_output)

    llm = LLM(model=model, temperature=temperature, pricing=pricing)

    results = run_pipeline(config, llm)

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([r.model_dump(mode="json") for r in results], f, indent=2)

    typer.echo(f"\nProduced {len(results)} corrupted samples -> {output}")
    typer.echo(f"LLM cost: ${llm.total_cost:.4f}")


@app.command()
def analyze(
    benchmark_path: Annotated[str, typer.Option(help="Benchmark JSON path or directory relative to project root")] = "benchmark",
    graphs: Annotated[Optional[list[str]], typer.Option("--graphs", "-g", help="Graphs to analyze")] = None,
    corruption_types: Annotated[Optional[list[str]], typer.Option("--corruption-types", "-c", help="Corruption types")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Dry-run: show corruption candidates per type per graph (no LLM calls)."""
    _setup_logging(verbose)

    config = PipelineConfig(
        benchmark_path=benchmark_path,
        graphs=_parse_graphs(graphs),
        corruption_types=_parse_corruption_types(corruption_types),
        num_samples=0,  # not used in analysis
    )

    results = run_analysis(config)

    # Print a table
    typer.echo(f"\n{'Type':<6} {'Graph':<22} {'Candidates':>12} {'Matching Samples':>18}")
    typer.echo("-" * 62)
    for r in results:
        typer.echo(f"{r.corruption_type:<6} {r.graph:<22} {r.num_candidates:>12} {r.num_matching_samples:>18}")


@app.command()
def verify(
    input_file: Annotated[Path, typer.Option("-i", "--input", help="Corrupted samples JSON file")] = Path("output/corrupted.json"),
    neo4j_uri: Annotated[Optional[str], typer.Option(help="Neo4j URI (or set NEO4J_URI env var)")] = None,
    neo4j_username: Annotated[Optional[str], typer.Option(help="Neo4j username (or set NEO4J_USERNAME)")] = None,
    neo4j_password: Annotated[Optional[str], typer.Option(help="Neo4j password (or set NEO4J_PASSWORD)")] = None,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output JSON file for verification results")] = Path("output/verification.json"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Verify corrupted samples by executing queries against Neo4j."""
    _setup_logging(verbose)

    try:
        from cb_corruptions.verification import Verifier, VerificationStatus
        from cb_corruptions.verification.neo4j_client import Neo4jClient
    except ImportError:
        typer.echo("Neo4j driver not installed. Run: uv pip install 'cypherbench-corruptions[verify]'", err=True)
        raise typer.Exit(1)

    from cb_corruptions.models import CorruptedSample

    # Load samples
    if not input_file.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    with open(input_file) as f:
        samples = [CorruptedSample(**s) for s in json.load(f)]

    typer.echo(f"Loaded {len(samples)} corrupted samples from {input_file}")

    # Verify
    with Neo4jClient(uri=neo4j_uri, username=neo4j_username, password=neo4j_password) as client:
        try:
            client.verify_connectivity()
        except Exception as e:
            typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
            raise typer.Exit(1)

        verifier = Verifier(client)
        results = verifier.verify_batch(samples)

    # Summary
    passed = sum(1 for r in results if r.status == VerificationStatus.PASS)
    failed = sum(1 for r in results if r.status == VerificationStatus.FAIL)
    errors = sum(1 for r in results if r.status == VerificationStatus.ERROR)

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump([r.model_dump(mode="json") for r in results], f, indent=2)

    typer.echo(f"\nVerification: {passed} pass, {failed} fail, {errors} error -> {output}")

    # Per-type breakdown
    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        counters = by_type.setdefault(r.corruption_type, {"pass": 0, "fail": 0, "error": 0})
        counters[r.status.value] += 1

    typer.echo(f"\n{'Type':<6} {'Pass':>6} {'Fail':>6} {'Error':>6}")
    typer.echo("-" * 28)
    for ctype, counters in sorted(by_type.items()):
        typer.echo(f"{ctype:<6} {counters['pass']:>6} {counters['fail']:>6} {counters['error']:>6}")



if __name__ == "__main__":
    app()
