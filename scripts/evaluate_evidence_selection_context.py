"""Offline prefix-5/7/9 evidence-context comparison from frozen rankings.

The four input artifacts are fixed and hash-pinned.  The script reconstructs
the authoritative ChunkSet from prepared local sources, never opens Chroma,
loads an embedding model, retrieves, prepares a Gemini request, reads
credentials, or contacts a provider.  Judgments are applied only after the
real selector and evidence-context builder have produced their outputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.prospective_cases import (  # noqa: E402
    parse_prospective,
    validate_prospective_against_chunks,
)
from src.evaluation.retrieval_seed import (  # noqa: E402
    parse_seed,
    validate_seed_against_chunks,
)
from src.generation.evidence_context import build_evidence_context  # noqa: E402
from src.ingestion.chunk_set_builder import build_chunk_set  # noqa: E402
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG  # noqa: E402
from src.ingestion.corpus_loader import load_corpus  # noqa: E402
from src.ingestion.manifest import load_manifest  # noqa: E402
from src.retrieval.dense_index import chunk_metadata  # noqa: E402
from src.retrieval.dense_results import (  # noqa: E402
    DenseCandidate, deserialize_chunk_metadata,
)
from src.retrieval.evidence_selection import select_evidence  # noqa: E402
from src.retrieval.fusion import fuse_candidates  # noqa: E402
from src.retrieval.lexical_query import LexicalCandidate  # noqa: E402


FORMAT_VERSION = "evidence-selection-context-comparison/v1"
PREFIXES = (5, 7, 9)
CASE_IDS = {
    "original": tuple(f"C{i}" for i in range(1, 7)),
    "prospective": tuple(f"P{i}" for i in range(1, 5)),
}
FIXED_INPUTS = {
    "original_report": (
        "data/evaluation/runs/mdr-retrieval-selected-depth10-v2-bcc0cdb-attempt01.json",
        "08d853838d3cc6fa493b84ac0645048b411c4b9da16cbf3ac5c8c428922f5bf0",
    ),
    "prospective_report": (
        "data/evaluation/runs/mdr-retrieval-prospective-v1-cbcdb25-attempt01.json",
        "3c6adf4769535223e6326128209f2f89ba2eb92a06d24de759bf70f31f832b7e",
    ),
    "original_judgments": (
        "data/evaluation/retrieval_seed_v1.json",
        "b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258",
    ),
    "prospective_judgments": (
        "data/evaluation/prospective_cases_v1.json",
        "b24050f555c31507a4f9521fbb8d2cf50a9235a08375f1fccbb5111dc2b6b758",
    ),
}

_REPORT_KEYS = {
    "original": {
        "format_version", "status", "started_at_utc", "finished_at_utc",
        "provenance", "seed", "configuration", "expected_identities",
        "observed_identities", "operations", "provider_calls", "cases",
        "primary_failure", "cleanup_failures", "cleanup", "interpretation",
    },
    "prospective": {
        "format_version", "status", "started_at_utc", "finished_at_utc",
        "provenance", "configuration", "expected_identities",
        "observed_identities", "operations", "provider_calls", "cases",
        "primary_failure", "cleanup_failures", "cleanup", "interpretation",
        "artifact",
    },
}
_CASE_KEYS = {
    "original": {
        "case_id", "topic_id", "question", "status", "rankings",
        "assessment", "approved_evidence_positions", "failure",
    },
    "prospective": {
        "case_id", "question", "status", "rankings", "assessment",
        "answer_behavior", "semantic_review", "failure",
    },
}
_CONFIG_KEYS = {
    "bm25", "rank_constant", "assessment_cutoff", "max_selected_chunks",
    "tenant", "database", "migration_hash_algorithm", "dense_candidate_count",
    "lexical_candidate_count", "project_root", "manifest_path",
    "original_index_path", "temp_parent", "output_path", "chunking",
    "embedding", "writers_stopped_acknowledged", "disposable_index_path",
}
_IDENTITY_KEYS = {
    "source_count", "chunk_count", "corpus_identity", "chunk_set_identity",
    "chunking_config_id", "dense_collection", "lexical_index_id",
}


class ComparisonError(ValueError):
    """A fixed input, reconstruction, comparison, or output invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError("Duplicate JSON key.")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ComparisonError(f"Nonstandard JSON constant: {value}")


def _json(content: bytes, name: str) -> dict:
    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{name} is not strict UTF-8 JSON.") from exc
    if type(value) is not dict:
        raise ComparisonError(f"{name} must be a JSON object.")
    return value


def _exact(value: object, keys: set[str], name: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ComparisonError(f"{name} has unexpected or missing fields.")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ComparisonError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonError(f"{name} must be finite.")
    return result


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ComparisonError(f"{name} must be a nonblank string.")
    return value


def _validate_branch(rows: object, branch: str, case_id: str) -> list[dict]:
    if type(rows) is not list or len(rows) != 10:
        raise ComparisonError(f"{case_id} {branch} must contain ranks 1..10.")
    score_name = "cosine_distance" if branch == "dense" else "bm25_score"
    expected = {"rank", "chunk_id", score_name}
    seen: set[str] = set()
    for position, row in enumerate(rows, 1):
        _exact(row, expected, f"{case_id} {branch} row")
        if _integer(row["rank"], "branch rank", minimum=1) != position:
            raise ComparisonError(f"{case_id} {branch} ranks are not contiguous.")
        chunk_id = _text(row["chunk_id"], "chunk_id")
        if chunk_id in seen:
            raise ComparisonError(f"{case_id} {branch} contains a duplicate candidate.")
        seen.add(chunk_id)
        _number(row[score_name], score_name)
    return rows


def _optional_rank(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name, minimum=1)


def _validate_rankings(value: object, case_id: str) -> dict:
    rankings = _exact(value, {"dense", "lexical", "fused", "selected"},
                      f"{case_id} rankings")
    dense = _validate_branch(rankings["dense"], "dense", case_id)
    lexical = _validate_branch(rankings["lexical"], "lexical", case_id)
    dense_ranks = {row["chunk_id"]: row["rank"] for row in dense}
    lexical_ranks = {row["chunk_id"]: row["rank"] for row in lexical}
    union_ids = set(dense_ranks) | set(lexical_ranks)
    fused = rankings["fused"]
    if type(fused) is not list or len(fused) != len(union_ids):
        raise ComparisonError(f"{case_id} fused rows do not cover the branch union.")
    seen: set[str] = set()
    for position, row in enumerate(fused, 1):
        _exact(row, {"rank", "chunk_id", "rrf_score", "dense_rank", "lexical_rank"},
               f"{case_id} fused row")
        if _integer(row["rank"], "fused rank", minimum=1) != position:
            raise ComparisonError(f"{case_id} fused ranks are not contiguous.")
        chunk_id = _text(row["chunk_id"], "chunk_id")
        if chunk_id in seen:
            raise ComparisonError(f"{case_id} fused rows contain a duplicate candidate.")
        seen.add(chunk_id)
        if _number(row["rrf_score"], "rrf_score") <= 0:
            raise ComparisonError("RRF scores must be positive.")
        dense_rank = _optional_rank(row["dense_rank"], "dense_rank")
        lexical_rank = _optional_rank(row["lexical_rank"], "lexical_rank")
        if dense_rank != dense_ranks.get(chunk_id) or lexical_rank != lexical_ranks.get(chunk_id):
            raise ComparisonError(f"{case_id} fused branch provenance does not match.")
    if seen != union_ids:
        raise ComparisonError(f"{case_id} fused IDs differ from the branch union.")
    selected = rankings["selected"]
    if type(selected) is not list or len(selected) != 5:
        raise ComparisonError(f"{case_id} selected rows must contain five records.")
    for position, row in enumerate(selected, 1):
        _exact(row, {"rank", "chunk_id", "fused_rank"}, f"{case_id} selected row")
        if row != {"rank": position, "chunk_id": fused[position - 1]["chunk_id"],
                   "fused_rank": position}:
            raise ComparisonError(f"{case_id} recorded selection is not fused prefix five.")
    return rankings


def _validate_configuration(config: object, name: str) -> dict:
    value = _exact(config, _CONFIG_KEYS, f"{name} configuration")
    if value["bm25"] != {"k1": 1.2, "b": 0.75}:
        raise ComparisonError(f"{name} BM25 settings differ from the review.")
    expected = {
        "rank_constant": 2.0, "assessment_cutoff": 5,
        "max_selected_chunks": 5, "dense_candidate_count": 10,
        "lexical_candidate_count": 10, "writers_stopped_acknowledged": True,
    }
    for field, required in expected.items():
        if value[field] != required or type(value[field]) is not type(required):
            raise ComparisonError(f"{name} has unexpected {field}.")
    for field in ("tenant", "database", "migration_hash_algorithm", "project_root",
                  "manifest_path", "original_index_path", "temp_parent", "output_path",
                  "disposable_index_path"):
        _text(value[field], f"configuration.{field}")
    if type(value["chunking"]) is not dict or type(value["embedding"]) is not dict:
        raise ComparisonError(f"{name} has invalid profile records.")
    return value


def _validate_report(report: dict, group: str) -> tuple[dict, dict[str, dict]]:
    _exact(report, _REPORT_KEYS[group], f"{group} report")
    expected_format = "retrieval-evidence-run/v2" if group == "original" else "prospective-retrieval-run/v1"
    if report["format_version"] != expected_format or report["status"] != "completed":
        raise ComparisonError(f"{group} report is not the reviewed completed format.")
    if report["provider_calls"] != 0 or type(report["provider_calls"]) is not int:
        raise ComparisonError(f"{group} report contains provider operations.")
    if report["primary_failure"] is not None or report["cleanup_failures"] != []:
        raise ComparisonError(f"{group} report contains an execution or cleanup failure.")
    config = _validate_configuration(report["configuration"], group)
    expected = _exact(report["expected_identities"], _IDENTITY_KEYS,
                      f"{group} expected identities")
    observed = _exact(report["observed_identities"], _IDENTITY_KEYS,
                      f"{group} observed identities")
    if observed != expected:
        raise ComparisonError(f"{group} expected and observed identities differ.")
    cases = report["cases"]
    required_all = CASE_IDS[group] if group == "original" else tuple(f"P{i}" for i in range(1, 9))
    if type(cases) is not list or [case.get("case_id") for case in cases] != list(required_all):
        raise ComparisonError(f"{group} case membership or order differs from approval.")
    result: dict[str, dict] = {}
    for case in cases:
        _exact(case, _CASE_KEYS[group], f"{group} case")
        case_id = case["case_id"]
        _text(case["question"], f"{case_id} question")
        if case_id in CASE_IDS[group]:
            if case["status"] != "ok" or case["failure"] is not None:
                raise ComparisonError(f"{case_id} is not a successful retrieval case.")
            if group == "prospective" and (
                case["answer_behavior"] != "not_evaluated"
                or case["semantic_review"] != "not_evaluated"
            ):
                raise ComparisonError(f"{case_id} contains an unauthorized answer review.")
            _validate_rankings(case["rankings"], case_id)
            result[case_id] = case
        else:
            if (case["status"] != "not_executed" or case["rankings"] is not None
                    or case["assessment"] != {"applicability": "not_applicable"}
                    or case["failure"] is not None
                    or case["answer_behavior"] != "not_evaluated"
                    or case["semantic_review"] != "not_evaluated"):
                raise ComparisonError(f"{case_id} must remain unexecuted and unevaluated.")
    return config, result


def _load_fixed_inputs(project_root: Path) -> tuple[dict, dict[str, dict]]:
    records: dict[str, dict] = {}
    parsed: dict[str, dict] = {}
    for name, (relative, approved_hash) in FIXED_INPUTS.items():
        path = (project_root / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(project_root):
            raise ComparisonError(f"Fixed input path is invalid: {name}.")
        content = path.read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        records[name] = {"path": relative.replace("\\", "/"), "sha256": observed}
        if observed != approved_hash:
            raise ComparisonError(f"Fixed input SHA-256 mismatch: {name}.")
        if name == "original_judgments":
            parsed[name] = parse_seed(content)
        elif name == "prospective_judgments":
            parsed[name] = parse_prospective(content)
        else:
            parsed[name] = _json(content, name)
    return records, parsed


def _matches_fixed_path(value: object, project_root: Path, relative: str) -> bool:
    if type(value) is not str or not value.strip():
        return False
    reported = Path(value)
    if not reported.is_absolute():
        reported = project_root / reported
    return reported.resolve(strict=False) == (project_root / relative).resolve(strict=False)


def _validate_inputs(
    parsed: dict[str, dict], project_root: Path
) -> tuple[dict, dict, dict, dict]:
    original_report = parsed["original_report"]
    prospective_report = parsed["prospective_report"]
    seed = parsed["original_judgments"]
    prospective = parsed["prospective_judgments"]
    original_config, original_cases = _validate_report(original_report, "original")
    prospective_config, prospective_cases = _validate_report(prospective_report, "prospective")
    compatible = (
        "bm25", "rank_constant", "assessment_cutoff", "max_selected_chunks",
        "tenant", "database", "migration_hash_algorithm", "dense_candidate_count",
        "lexical_candidate_count", "chunking", "embedding",
    )
    if any(original_config[field] != prospective_config[field] for field in compatible):
        raise ComparisonError("The two report configurations are not compatible.")
    if original_report["expected_identities"] != prospective_report["expected_identities"]:
        raise ComparisonError("The two reports do not bind the same corpus and indexes.")
    original_binding = original_report["seed"]
    if (
        type(original_binding) is not dict
        or set(original_binding) != {"path", "sha256", "artifact_id"}
        or not _matches_fixed_path(
            original_binding["path"],
            project_root,
            FIXED_INPUTS["original_judgments"][0],
        )
        or original_binding["sha256"] != FIXED_INPUTS["original_judgments"][1]
        or original_binding["artifact_id"] != seed["artifact_id"]
    ):
        raise ComparisonError("Original report is not bound to the fixed seed.")
    prospective_binding = prospective_report["artifact"]
    if (
        type(prospective_binding) is not dict
        or set(prospective_binding) != {"path", "sha256", "artifact_id"}
        or not _matches_fixed_path(
            prospective_binding["path"],
            project_root,
            FIXED_INPUTS["prospective_judgments"][0],
        )
        or prospective_binding["sha256"] != FIXED_INPUTS["prospective_judgments"][1]
        or prospective_binding["artifact_id"] != prospective["artifact"]
    ):
        raise ComparisonError("Prospective report is not bound to the fixed artifact.")
    seed_cases = {case["case_id"]: case for case in seed["cases"]}
    prospective_cases_by_id = {case["id"]: case for case in prospective["cases"]}
    for case_id, record in original_cases.items():
        if record["question"] != seed_cases[case_id]["question"]:
            raise ComparisonError(f"{case_id} question differs from the fixed seed.")
    for case_id, record in prospective_cases.items():
        if record["question"] != prospective_cases_by_id[case_id]["question"]:
            raise ComparisonError(f"{case_id} question differs from the fixed artifact.")
    return original_report, prospective_report, seed, prospective


def _load_authoritative_chunk_set(project_root: Path, manifest_path: Path):
    manifest = load_manifest(manifest_path, project_root=project_root, verify_files=True)
    sources = load_corpus(manifest, project_root=project_root)
    return build_chunk_set(manifest, sources, BASELINE_CHUNKING_CONFIG)


def _validate_chunk_set(chunk_set, identities: dict) -> None:
    actual = {
        "source_count": len(chunk_set.manifest.sources),
        "chunk_count": len(chunk_set.chunks),
        "corpus_identity": chunk_set.corpus_identity,
        "chunk_set_identity": chunk_set.chunk_set_id,
        "chunking_config_id": chunk_set.chunking_config_id,
    }
    if any(actual[name] != identities[name] for name in actual):
        raise ComparisonError("Authoritative ChunkSet does not match report identities.")


def _reconstruct_case(case: dict, chunk_set, rank_constant: float):
    originals = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
    metadata = {}

    def resolved(chunk_id: str):
        if chunk_id not in originals:
            raise ComparisonError(f"Recorded candidate is absent from ChunkSet: {chunk_id}")
        if chunk_id not in metadata:
            metadata[chunk_id] = deserialize_chunk_metadata(chunk_metadata(originals[chunk_id]))
        return originals[chunk_id], metadata[chunk_id]

    dense = []
    for row in case["rankings"]["dense"]:
        chunk, trusted = resolved(row["chunk_id"])
        dense.append(DenseCandidate(row["rank"], chunk.chunk_id, chunk.text, trusted,
                                    row["cosine_distance"]))
    lexical = []
    for row in case["rankings"]["lexical"]:
        chunk, trusted = resolved(row["chunk_id"])
        lexical.append(LexicalCandidate(row["rank"], chunk.chunk_id, chunk.text, trusted,
                                        row["bm25_score"]))
    fused = fuse_candidates(
        chunk_set=chunk_set, dense_candidates=tuple(dense),
        lexical_candidates=tuple(lexical), rank_constant=rank_constant,
    )
    recorded = case["rankings"]["fused"]
    if len(fused) != len(recorded):
        raise ComparisonError(f"{case['case_id']} recomputed union length differs.")
    for actual, expected in zip(fused, recorded):
        row = {
            "rank": actual.rank, "chunk_id": actual.chunk_id,
            "rrf_score": actual.rrf_score,
            "dense_rank": actual.dense_candidate.rank if actual.dense_candidate else None,
            "lexical_rank": actual.lexical_candidate.rank if actual.lexical_candidate else None,
        }
        if row != expected:
            raise ComparisonError(
                f"{case['case_id']} recomputed fusion differs at rank {actual.rank}."
            )
    baseline = select_evidence(chunk_set=chunk_set, fused_candidates=fused, max_chunks=5)
    recorded_selected = [row["chunk_id"] for row in case["rankings"]["selected"]]
    if [item.chunk.chunk_id for item in baseline.selected] != recorded_selected:
        raise ComparisonError(f"{case['case_id']} real prefix-five selection differs.")
    return fused


def _original_judgments(seed: dict) -> dict[str, dict]:
    evidence = {row["evidence_id"]: row["chunk_id"] for row in seed["evidence"]}
    result = {}
    for case in seed["cases"]:
        nominated = tuple(dict.fromkeys(
            evidence[row["evidence_id"]] for row in case["supporting_judgments"]
        ))
        combinations = tuple(
            tuple(evidence[item] for item in combo["members"])
            for combo in case["support_combinations"] if combo["approved"] is True
        )
        result[case["case_id"]] = {
            "nominated": nominated, "combinations": combinations,
            "separate": {},
        }
    return result


def _prospective_judgments(artifact: dict) -> dict[str, dict]:
    result = {}
    for case in artifact["cases"][:4]:
        nominated = tuple(dict.fromkeys(
            excerpt["chunk_id"]
            for element in case["mandatory_element_evidence"]
            for excerpt in element["supporting_excerpts"]
        ))
        separate = {}
        if case["id"] == "P3":
            separate["supporting_context"] = case["supporting_context"]["chunk_id"]
        if case["id"] == "P4":
            separate["answer_review_qualification"] = (
                case["answer_review_qualification"]["element"]
                ["supporting_excerpts"][0]["chunk_id"]
            )
        result[case["id"]] = {
            "nominated": nominated,
            "combinations": tuple(tuple(combo) for combo in
                                  case["accepted_substantive_complete_combinations"]),
            "separate": separate,
        }
    return result


def _score(selected_ids: tuple[str, ...], judgment: dict) -> dict:
    found = tuple(chunk_id for chunk_id in judgment["nominated"] if chunk_id in selected_ids)
    completed = tuple(
        index for index, combo in enumerate(judgment["combinations"], 1)
        if all(chunk_id in selected_ids for chunk_id in combo)
    )
    return {
        "nominated_found": len(found),
        "nominated_total": len(judgment["nominated"]),
        "found_chunk_ids": list(found),
        "missing_chunk_ids": [chunk_id for chunk_id in judgment["nominated"]
                              if chunk_id not in selected_ids],
        "complete_combination_found": bool(completed),
        "completed_combination_indexes": list(completed),
        "other_chunks": "unjudged",
    }


def _context_size(selection, context) -> dict:
    chunks = [item.chunk for item in selection.selected]
    raw_chars = sum(len(chunk.text) for chunk in chunks)
    raw_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
    serialized = context.model_context_json
    serialized_chars = len(serialized)
    serialized_bytes = len(serialized.encode("utf-8"))
    return {
        "selected_chunk_count": len(chunks),
        "raw_text_unicode_characters": raw_chars,
        "raw_text_utf8_bytes": raw_bytes,
        "serialized_context_unicode_characters": serialized_chars,
        "serialized_context_utf8_bytes": serialized_bytes,
        "serialization_expansion_unicode_characters": serialized_chars - raw_chars,
        "serialization_expansion_utf8_bytes": serialized_bytes - raw_bytes,
        "table_segment_count": sum(len(chunk.table_segments) for chunk in chunks),
        "chunks_with_tables": sum(bool(chunk.table_segments) for chunk in chunks),
        "annotation_count": sum(len(chunk.annotations) for chunk in chunks),
        "chunks_with_annotations": sum(bool(chunk.annotations) for chunk in chunks),
    }


_SIZE_FIELDS = (
    "selected_chunk_count", "raw_text_unicode_characters", "raw_text_utf8_bytes",
    "serialized_context_unicode_characters", "serialized_context_utf8_bytes",
    "serialization_expansion_unicode_characters", "serialization_expansion_utf8_bytes",
    "table_segment_count", "chunks_with_tables", "annotation_count",
    "chunks_with_annotations",
)


def _aggregate(cases: list[dict], prefix: int) -> dict:
    options = [case["options"][f"prefix_{prefix}"] for case in cases]
    aggregate_sizes = {}
    for field in _SIZE_FIELDS:
        values = [option["context_size"][field] for option in options]
        aggregate_sizes[field] = {
            "minimum": min(values), "median": statistics.median(values),
            "maximum": max(values), "sum": sum(values),
        }
    return {
        "case_count_denominator": len(cases),
        "complete_combination_cases": sum(
            option["recovery"]["complete_combination_found"] for option in options
        ),
        "nominated_evidence_occurrence_denominator": sum(
            option["recovery"]["nominated_total"] for option in options
        ),
        "nominated_evidence_occurrences_found": sum(
            option["recovery"]["nominated_found"] for option in options
        ),
        "context_size_statistics": aggregate_sizes,
    }


def _initial_groups(original_report: dict, prospective_report: dict) -> dict:
    reports = {"original": original_report, "prospective": prospective_report}
    groups = {}
    for group, report in reports.items():
        case_records = {case["case_id"]: case for case in report["cases"]}
        groups[group] = {
            "status": "not_attempted",
            "case_ids": list(CASE_IDS[group]),
            "cases": [
                {
                    "case_id": case_id,
                    "question": case_records[case_id]["question"],
                    "status": "not_attempted",
                    "candidate_reconstruction": {"status": "not_attempted"},
                    "options": {
                        f"prefix_{prefix}": {
                            "status": "not_attempted",
                            "max_chunks": prefix,
                        }
                        for prefix in PREFIXES
                    },
                }
                for case_id in CASE_IDS[group]
            ],
            "aggregates_status": "not_computed_group_incomplete",
            "aggregates": None,
        }
    return groups


def evaluate_comparison(*, original_report: dict, prospective_report: dict,
                        seed: dict, prospective: dict, chunk_set,
                        groups: dict | None = None,
                        progress: dict | None = None) -> dict:
    """Evaluate fixed prefixes while preserving each completed measurement."""

    reports = {"original": original_report, "prospective": prospective_report}
    judgments = {
        "original": _original_judgments(seed),
        "prospective": _prospective_judgments(prospective),
    }
    if groups is None:
        groups = _initial_groups(original_report, prospective_report)
    if progress is None:
        progress = {}
    for group, report in reports.items():
        case_records = {case["case_id"]: case for case in report["cases"]}
        group_result = groups[group]
        group_result["status"] = "in_progress"
        for case_result in group_result["cases"]:
            case_id = case_result["case_id"]
            record = case_records[case_id]
            case_result["status"] = "in_progress"
            progress.update(
                group=group, case_id=case_id,
                phase="candidate_reconstruction", prefix=None,
            )
            try:
                fused = _reconstruct_case(
                    record, chunk_set, report["configuration"]["rank_constant"]
                )
            except Exception:
                case_result["candidate_reconstruction"]["status"] = "failed"
                case_result["status"] = "failed"
                group_result["status"] = "failed"
                raise
            case_result["candidate_reconstruction"]["status"] = "completed"
            for prefix in PREFIXES:
                option_key = f"prefix_{prefix}"
                progress.update(
                    group=group, case_id=case_id,
                    phase="prefix_measurement", prefix=prefix,
                )
                case_result["options"][option_key] = {
                    "status": "in_progress", "max_chunks": prefix,
                }
                try:
                    selection = select_evidence(
                        chunk_set=chunk_set,
                        fused_candidates=fused,
                        max_chunks=prefix,
                    )
                    context = build_evidence_context(
                        selection=selection, chunk_set=chunk_set
                    )
                    selected_ids = tuple(
                        item.chunk.chunk_id for item in selection.selected
                    )
                    separate = {
                        role: {
                            "chunk_id": chunk_id,
                            "selected": chunk_id in selected_ids,
                        }
                        for role, chunk_id in judgments[group][case_id][
                            "separate"
                        ].items()
                    }
                    completed_option = {
                        "status": "completed",
                        "max_chunks": prefix,
                        "selected": [
                            {
                                "citation_id": str(index),
                                "chunk_id": item.chunk.chunk_id,
                                "fused_rank": item.candidate.rank,
                            }
                            for index, item in enumerate(selection.selected, 1)
                        ],
                        "recovery": _score(
                            selected_ids, judgments[group][case_id]
                        ),
                        "separate_evidence": separate,
                        "context_size": _context_size(selection, context),
                        "provider_token_count": "not_performed",
                        "answer_behavior": "not_evaluated",
                        "semantic_review": "not_evaluated",
                    }
                except Exception:
                    case_result["options"][option_key] = {
                        "status": "failed", "max_chunks": prefix,
                    }
                    case_result["status"] = "failed"
                    group_result["status"] = "failed"
                    raise
                case_result["options"][option_key] = completed_option
            case_result["status"] = "completed"
        progress.update(
            group=group, case_id=None, phase="aggregation", prefix=None
        )
        try:
            group_result["aggregates"] = {
                f"prefix_{prefix}": _aggregate(group_result["cases"], prefix)
                for prefix in PREFIXES
            }
        except Exception:
            group_result["status"] = "failed"
            raise
        group_result["aggregates_status"] = "completed"
        group_result["status"] = "completed"
    return groups


def _provenance(project_root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(project_root), *args],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout.rstrip("\r\n")

    paths = [
        Path(__file__).resolve(),
        project_root / "src/ingestion/corpus_loader.py",
        project_root / "src/ingestion/chunk_set_builder.py",
        project_root / "src/retrieval/fusion.py",
        project_root / "src/retrieval/evidence_selection.py",
        project_root / "src/generation/evidence_context.py",
    ]
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short", "--untracked-files=all"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            str(path.relative_to(project_root)).replace("\\", "/"): _sha256(path)
            for path in paths
        },
    }


@contextmanager
def _block_network():
    def forbidden(*_args, **_kwargs):
        raise ComparisonError("Network access is forbidden during this comparison.")

    with (patch.object(socket.socket, "connect", forbidden),
          patch.object(socket.socket, "connect_ex", forbidden),
          patch.object(socket, "create_connection", forbidden)):
        yield


def _failure(stage: str, exc: BaseException,
             location: dict | None = None) -> dict:
    return {
        "stage": stage, "error_type": type(exc).__name__,
        "message": "Comparison failed; inspect the stage and local diagnostics.",
        "location": dict(location) if location else None,
    }


def run_comparison(*, project_root: Path, output_path: Path,
                   chunk_set_loader=_load_authoritative_chunk_set,
                   provenance=_provenance) -> dict:
    """Reserve one report, run locally once, and preserve failure diagnostics."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_path).resolve(strict=False)
    if not root.is_dir() or not output.parent.is_dir():
        raise ComparisonError("Project root and output parent must exist.")
    fixed_paths = {(root / relative).resolve(strict=False) for relative, _ in FIXED_INPUTS.values()}
    if output in fixed_paths:
        raise ComparisonError("Output path cannot replace a fixed input.")
    try:
        handle = output.open("x", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ComparisonError("Unable to reserve the non-overwriting output path.") from exc

    report = {
        "format_version": FORMAT_VERSION,
        "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "execution": {
            "project_root": str(root),
            "output_path": str(output),
            "output_must_be_absent": True,
        },
        "provenance": None,
        "inputs": None,
        "comparison": {
            "prefixes": list(PREFIXES),
            "selection_policy": "validated_full_fused_prefix",
            "judgments_influence_selection": False,
            "aggregate_definitions": {
                "case_count_denominator": "executed positive cases in the named group",
                "nominated_evidence_occurrence_denominator": (
                    "sum of distinct nominated chunk IDs per case; a chunk nominated in two cases counts twice"
                ),
                "complete_combination_case": (
                    "selected IDs contain every member of at least one accepted combination"
                ),
                "size_statistics": "minimum, median, maximum, and sum across cases within one group/prefix",
            },
        },
        "identities": None,
        "groups": None,
        "provider_token_count": "not_performed",
        "gemini_request_preparation": "not_performed",
        "answer_behavior": "not_evaluated",
        "semantic_review": "not_evaluated",
        "failure": None,
    }
    stage = "provenance"
    progress: dict = {}
    try:
        with _block_network():
            report["provenance"] = provenance(root)
            stage = "fixed_inputs"
            records, parsed = _load_fixed_inputs(root)
            report["inputs"] = records
            stage = "input_validation"
            original_report, prospective_report, seed, prospective = _validate_inputs(
                parsed, root
            )
            report["groups"] = _initial_groups(
                original_report, prospective_report
            )
            identities = original_report["expected_identities"]
            report["identities"] = {"expected": identities, "observed": None}
            stage = "chunk_set_reconstruction"
            manifest_path = root / "data/corpus_manifest.json"
            chunk_set = chunk_set_loader(root, manifest_path)
            _validate_chunk_set(chunk_set, identities)
            report["identities"]["observed"] = {
                "source_count": len(chunk_set.manifest.sources),
                "chunk_count": len(chunk_set.chunks),
                "corpus_identity": chunk_set.corpus_identity,
                "chunk_set_identity": chunk_set.chunk_set_id,
                "chunking_config_id": chunk_set.chunking_config_id,
            }
            stage = "judgment_binding"
            validate_seed_against_chunks(seed, chunk_set)
            validate_prospective_against_chunks(prospective, chunk_set)
            stage = "candidate_reconstruction_and_context"
            evaluate_comparison(
                original_report=original_report,
                prospective_report=prospective_report,
                seed=seed,
                prospective=prospective,
                chunk_set=chunk_set,
                groups=report["groups"],
                progress=progress,
            )
            report["status"] = "completed"
    except Exception as exc:
        report["failure"] = _failure(stage, exc, progress)
    except BaseException as exc:
        report["failure"] = _failure(stage, exc, progress)
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            with handle:
                handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
        except Exception as exc:
            raise ComparisonError("Comparison report write failed; reserved output may be incomplete.") from exc
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_comparison(**vars(args))
    except Exception as exc:
        print(f"Comparison reservation/report failure: {type(exc).__name__}.", file=sys.stderr)
        return 1
    print(f"Comparison status: {result['status']}; report: {args.output_path}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
