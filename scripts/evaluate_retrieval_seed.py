"""Run an explicitly configured, offline retrieval-evidence experiment.

Import and --help are inert: corpus, database and model access occur only in
run_command. No generation, parameter sweep or interpretation of unjudged IDs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_seed import (  # noqa: E402
    EvaluationError, evaluate_cases, parse_seed, validate_seed_against_chunks,
)


def _git_state(root: Path) -> dict:
    """Record source provenance, including uncommitted code; do not stage it."""
    result = {}
    for key, args in (("head", ["rev-parse", "HEAD"]),
                      ("status", ["status", "--short", "--untracked-files=all"]),
                      ("diff", ["diff", "HEAD", "--", "src", "scripts"])):
        run = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                             capture_output=True, check=True)
        result[key] = run.stdout.decode("utf-8", errors="replace").strip()
    return result


def run_command(args: argparse.Namespace) -> int:
    """Validate all seed evidence before creating any retrieval backend."""
    root = args.project_root.resolve()
    seed_path = (root / args.seed).resolve()
    output = args.output.resolve()
    if output.exists():
        raise EvaluationError(f"Output already exists; choose a new path: {output}")
    if not output.parent.is_dir():
        raise EvaluationError("Output parent directory must already exist.")
    if not (root / "data/indexes/chroma").is_dir():
        raise EvaluationError("Existing data/indexes/chroma directory is required; no index will be built.")
    if not args.dense_candidates > 0 or not args.lexical_candidates > 0 or not args.cutoff > 0:
        raise EvaluationError("Candidate counts and cutoff must be positive.")

    # Set before importing HF/Transformers; the existing adapter is lazy.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
    from src.ingestion.chunk_set_builder import build_chunk_set
    from src.ingestion.corpus_loader import load_corpus
    from src.ingestion.manifest import load_manifest
    from src.retrieval.dense_index import dense_collection_name
    from src.retrieval.dense_query import DenseRetriever
    from src.retrieval.embeddings import GTE_MODERNBERT_BASE, TransformerEmbedder
    from src.retrieval.hybrid_query import retrieve_hybrid
    from src.retrieval.lexical_query import BM25Config, BM25Retriever

    seed_bytes = seed_path.read_bytes()
    seed = parse_seed(seed_bytes)
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    bm25_config = BM25Config(k1=args.k1, b=args.b)
    if not (0 <= args.rank_constant < float("inf")):
        raise EvaluationError("rank_constant must be finite and nonnegative.")
    manifest_path = (root / seed["frozen_corpus"]["manifest_path"]).resolve()
    if not manifest_path.is_relative_to(root):
        raise EvaluationError("Manifest path must stay inside project root.")
    print("Validating frozen corpus and approved evidence...", flush=True)
    manifest = load_manifest(manifest_path, project_root=root, verify_files=True)
    chunk_set = build_chunk_set(manifest, load_corpus(manifest, project_root=root),
                                BASELINE_CHUNKING_CONFIG)
    validate_seed_against_chunks(seed, chunk_set)
    lexical = BM25Retriever(chunk_set=chunk_set, config=bm25_config)
    git_before = _git_state(root)

    # No get_or_create, add, update, delete or build operation is used here.
    import chromadb
    client = chromadb.PersistentClient(path=str(root / "data/indexes/chroma"))
    embedder = TransformerEmbedder(GTE_MODERNBERT_BASE)
    dense = DenseRetriever(client=client, chunk_set=chunk_set, embedder=embedder)
    started = datetime.now(timezone.utc).isoformat()

    def retrieve(query: str) -> tuple:
        print(f"Retrieving case {retrieve.calls + 1}/{len(seed['cases'])}...", flush=True)
        retrieve.calls += 1
        return retrieve_hybrid(
            query, chunk_set=chunk_set, dense_retriever=dense, lexical_retriever=lexical,
            dense_candidate_count=args.dense_candidates,
            lexical_candidate_count=args.lexical_candidates, rank_constant=args.rank_constant,
        )

    retrieve.calls = 0
    results = evaluate_cases(seed, retrieve=retrieve, cutoff=args.cutoff)
    packages = {}
    for name in ("numpy", "chromadb", "torch", "transformers", "huggingface-hub"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {
        "format_version": "retrieval-evidence-run/v1",
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if all(r["status"] == "ok" for r in results) else "completed-with-errors",
        "git_before": git_before, "git_after": _git_state(root),
        "seed": {"artifact_id": seed["artifact_id"], "path": str(seed_path),
                 "sha256": seed_sha256},
        "frozen_corpus": seed["frozen_corpus"],
        "chunking_config_id": chunk_set.chunking_config_id,
        "parameters": {"dense_candidate_count": args.dense_candidates,
                       "lexical_candidate_count": args.lexical_candidates,
                       "bm25": asdict(bm25_config), "rank_constant": args.rank_constant,
                       "cutoff": args.cutoff},
        "embedding": asdict(GTE_MODERNBERT_BASE),
        "dense_collection": dense_collection_name(chunk_set, GTE_MODERNBERT_BASE),
        "lexical_index_id": lexical.index_id,
        "runtime": {"python": sys.version, "platform": sys.platform, "packages": packages},
        "code_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (root / "src", root / "scripts")
            for path in sorted(directory.rglob("*.py"))
        },
        "offline_model_loading": True, "hybrid_calls": retrieve.calls,
        "interpretation": "Recovery of approved evidence only; all other chunks are unjudged. No overall quality claim.",
        "cases": results,
    }
    # Serialize before opening, and never overwrite an earlier experiment.
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    print("Case  Branch   Approved found  Any  Complete")
    for result in results:
        if result["status"] == "error":
            print(f"{result['case_id']}: ERROR ({result['error_stage']}); see report")
            continue
        for branch, assessment in result["assessment"].items():
            print(f"{result['case_id']:4}  {branch:7}  "
                  f"{assessment['approved_evidence_found']}/{assessment['approved_evidence_total']}             "
                  f"{assessment['any_approved_support_found']!s:5}"
                  f"{assessment['complete_approved_combination_found']}")
    print(f"Report: {output}")
    return 0 if report["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-candidates", type=int, required=True)
    parser.add_argument("--lexical-candidates", type=int, required=True)
    parser.add_argument("--k1", type=float, required=True)
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--rank-constant", type=float, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        return run_command(args)
    except Exception as exc:
        print(f"Evaluation setup/report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
