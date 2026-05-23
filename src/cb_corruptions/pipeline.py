"""Pipeline orchestration: run corruption generators over benchmark samples."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import cycle

from tqdm import tqdm

from cb_corruptions.corruptions import CORRUPTION_REGISTRY, BaseCorruption
from cb_corruptions.llm import LLM
from cb_corruptions.models import CorruptedSample, CorruptionType
from cb_corruptions.schema_loader import (
    load_all_schemas,
    load_benchmark,
    load_graph_info,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    benchmark_path: str
    graphs: list[str]
    corruption_types: list[CorruptionType]
    num_samples: int
    seed: int = 42


def run_pipeline(config: PipelineConfig, llm: LLM) -> list[CorruptedSample]:
    """Run all requested corruption generators and return corrupted samples."""

    # 1. Load data
    logger.info("Loading schemas...")
    schemas = load_all_schemas(config.graphs)
    graph_infos = load_graph_info()

    logger.info("Loading benchmark samples from '%s'...", config.benchmark_path)
    all_samples = load_benchmark(config.benchmark_path)

    rng = random.Random(config.seed)
    results: list[CorruptedSample] = []

    # 2. Run each corruption type
    for corruption_type in config.corruption_types:
        type_code = corruption_type.value
        cls = CORRUPTION_REGISTRY.get(type_code)
        if cls is None:
            logger.warning("Unknown corruption type: %s (skipping)", type_code)
            continue

        corruption: BaseCorruption = cls(llm=llm)
        logger.info("Running corruption %s...", type_code)

        for graph_name in config.graphs:
            if graph_name not in schemas:
                continue

            schema = schemas[graph_name]
            graph_info = graph_infos.get(graph_name)
            if graph_info is None:
                logger.warning("No graph_info for '%s', skipping", graph_name)
                continue

            corruption.set_schema(schema)

            # Stage 1: Analyze schema
            candidates = corruption.analyze(schema, graph_info)

            # Stage 2: Select samples
            graph_samples = [s for s in all_samples if s.graph == graph_name]
            if not graph_samples:
                logger.info("  %s: no benchmark samples for graph '%s'", type_code, graph_name)
                continue

            pairs = corruption.select_samples(graph_samples, candidates)
            if not pairs:
                logger.info("  %s/%s: no matching samples found", type_code, graph_name)
                continue

            # Diversify: interleave samples across candidate groups so we don't
            # draw all samples from the same pool
            pairs = _interleave_by_candidate(pairs, rng)
            limit = min(config.num_samples, len(pairs))

            # Stage 3: Corrupt with LLM (in parallel batches)
            batch_size = llm.batch_size
            to_process = pairs[:limit]
            count = 0
            desc = f"{type_code}/{graph_name}"

            for i in tqdm(range(0, len(to_process), batch_size), desc=desc):
                if count >= limit:
                    break
                batch = to_process[i : i + batch_size]
                batch_results = _corrupt_batch(corruption, batch, graph_name)
                for result in batch_results:
                    if result is not None and count < limit:
                        results.append(result)
                        count += 1

            logger.info("  %s/%s: produced %d corrupted samples", type_code, graph_name, count)

    return results


@dataclass
class AnalysisResult:
    """Summary of corruption candidates found per type per graph."""

    corruption_type: str
    graph: str
    num_candidates: int
    num_matching_samples: int


def _corrupt_batch(
    corruption: BaseCorruption,
    batch: list[tuple],
    graph_name: str,
) -> list[CorruptedSample | None]:
    """Run a batch of corrupt() calls in parallel using threads."""
    results: list[CorruptedSample | None] = [None] * len(batch)

    def _do_corrupt(idx: int, sample, candidate):
        try:
            return idx, corruption.corrupt(sample, candidate, graph_name)
        except Exception as e:
            logger.error("  Failed to corrupt sample %s: %s", sample.qid, e)
            return idx, None

    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
        futures = [
            pool.submit(_do_corrupt, i, sample, candidate)
            for i, (sample, candidate) in enumerate(batch)
        ]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


def _interleave_by_candidate(
    pairs: list[tuple], rng: random.Random
) -> list[tuple]:
    """Interleave (sample, candidate) pairs across candidate groups.

    Groups pairs by candidate identity, shuffles within each group,
    then round-robins across groups so the output draws evenly from
    all pools. For U-types where candidate is None, all pairs are
    in one group and this is equivalent to a plain shuffle.
    """
    groups: dict[str, list[tuple]] = defaultdict(list)
    for sample, candidate in pairs:
        # Use candidate's string repr as group key (None for U-types)
        key = str(id(candidate)) if candidate is not None else "default"
        groups[key].append((sample, candidate))

    # Shuffle within each group
    for group in groups.values():
        rng.shuffle(group)

    if len(groups) <= 1:
        # Single group — just return shuffled
        return list(groups.values())[0] if groups else []

    # Round-robin across groups
    iterators = [iter(g) for g in groups.values()]
    rng.shuffle(iterators)
    result: list[tuple] = []
    for it in cycle(iterators):
        try:
            result.append(next(it))
        except StopIteration:
            # Remove exhausted iterators
            iterators = [i for i in iterators if i is not it]
            if not iterators:
                break
    return result


def run_analysis(config: PipelineConfig) -> list[AnalysisResult]:
    """Dry-run: analyze schemas and report candidate counts without LLM calls."""

    schemas = load_all_schemas(config.graphs)
    graph_infos = load_graph_info()

    # Try to load benchmark samples (may not exist)
    try:
        all_samples = load_benchmark(config.benchmark_path)
    except FileNotFoundError:
        logger.warning("Benchmark file not found, showing candidates only (no sample counts)")
        all_samples = []

    results: list[AnalysisResult] = []

    for corruption_type in config.corruption_types:
        type_code = corruption_type.value
        cls = CORRUPTION_REGISTRY.get(type_code)
        if cls is None:
            continue

        corruption: BaseCorruption = cls(llm=None)  # type: ignore[arg-type]  # LLM not used in analysis

        for graph_name in config.graphs:
            if graph_name not in schemas:
                continue

            schema = schemas[graph_name]
            graph_info = graph_infos.get(graph_name)
            if graph_info is None:
                continue

            corruption.set_schema(schema)

            candidates = corruption.analyze(schema, graph_info)
            graph_samples = [s for s in all_samples if s.graph == graph_name]
            num_matching = len(corruption.select_samples(graph_samples, candidates)) if graph_samples else 0

            results.append(
                AnalysisResult(
                    corruption_type=type_code,
                    graph=graph_name,
                    num_candidates=len(candidates),
                    num_matching_samples=num_matching,
                )
            )

    return results
