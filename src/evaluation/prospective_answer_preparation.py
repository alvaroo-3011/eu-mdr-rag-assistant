"""Provider-free preparation of prospective answer-evaluation requests.

Inputs are caller-owned; this module does not load a corpus, retrieve, open an
index, access credentials, count tokens, or contact a provider. The report is
an audit artifact, not an authorization or a budget approval.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from scripts import evaluate_evidence_selection_context as comparison
from src.application.answer_policy import local_empty_evidence_result
from src.evaluation.prospective_cases import parse_prospective, validate_prospective_against_chunks
from src.evaluation.retrieval_seed import ranking_records
from src.generation.evidence_context import build_evidence_context
from src.generation.gemini_request import (
    GEMINI_PROMPT_VERSION, GeminiGenerationConfig, prepare_gemini_request,
)
from src.retrieval.evidence_selection import EvidenceSelection, select_evidence
from src.retrieval.fusion import fuse_candidates


FORMAT = "prospective-answer-preparation/v1"
ARTIFACT_SHA256 = "b24050f555c31507a4f9521fbb8d2cf50a9235a08375f1fccbb5111dc2b6b758"
RANKING_SHA256 = "3c6adf4769535223e6326128209f2f89ba2eb92a06d24de759bf70f31f832b7e"
COMPARISON_SHA256 = "d405f01154aea835ed13047d7d5e327011f2c442db6e7b6373b144662dba7b1e"
CASE_IDS = tuple(f"P{i}" for i in range(1, 9))


class PreparationError(ValueError):
    """The supplied inputs cannot be faithfully prepared."""


def _checked_bytes(content: bytes, expected: str, label: str) -> str:
    if type(content) is not bytes:
        raise PreparationError(f"{label} must be exact bytes.")
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected:
        raise PreparationError(f"{label} SHA-256 mismatch.")
    return observed


def _inputs(artifact_bytes, ranking_bytes, comparison_bytes, chunk_set):
    hashes = {
        "artifact_sha256": _checked_bytes(artifact_bytes, ARTIFACT_SHA256, "artifact"),
        "ranking_report_sha256": _checked_bytes(ranking_bytes, RANKING_SHA256, "ranking report"),
        "comparison_report_sha256": _checked_bytes(comparison_bytes, COMPARISON_SHA256, "comparison report"),
    }
    artifact = parse_prospective(artifact_bytes)
    ranking = comparison._json(ranking_bytes, "prospective ranking report")
    comp = comparison._json(comparison_bytes, "context comparison report")
    _, ranked_cases = comparison._validate_report(ranking, "prospective")
    validate_prospective_against_chunks(artifact, chunk_set)
    comparison._validate_chunk_set(chunk_set, ranking["expected_identities"])
    if (ranking["artifact"]["sha256"] != hashes["artifact_sha256"]
            or ranking["artifact"]["artifact_id"] != artifact["artifact"]):
        raise PreparationError("Ranking report artifact binding differs.")
    if (comp.get("format_version") != comparison.FORMAT_VERSION
            or comp.get("status") != "completed"
            or comp.get("failure") is not None
            or comp.get("inputs", {}).get("prospective_report", {}).get("sha256") != hashes["ranking_report_sha256"]
            or comp.get("inputs", {}).get("prospective_judgments", {}).get("sha256") != hashes["artifact_sha256"]
            or comp.get("identities", {}).get("expected") != ranking["expected_identities"]):
        raise PreparationError("Comparison report identity or status differs.")
    group = comp["groups"]["prospective"]
    if group["status"] != "completed" or group["case_ids"] != list(CASE_IDS[:4]):
        raise PreparationError("Comparison case membership differs.")
    compared = {row["case_id"]: row for row in group["cases"]}
    if list(compared) != list(CASE_IDS[:4]):
        raise PreparationError("Comparison cases differ.")
    cases = {case["id"]: case for case in artifact["cases"]}
    for case_id, row in ranked_cases.items():
        if row["question"] != cases[case_id]["question"] or compared[case_id]["question"] != cases[case_id]["question"]:
            raise PreparationError(f"{case_id} question mismatch.")
    return hashes, artifact, ranking, ranked_cases, compared, cases


def _supplied_union(value, case, chunk_set):
    if type(value) is not dict or set(value) != {"question", "fused_candidates"}:
        raise PreparationError("Supplied hybrid result must carry its exact question and fused candidates.")
    if value["question"] != case["question"]:
        raise PreparationError("Supplied hybrid question mismatch.")
    fused = value["fused_candidates"]
    rankings = ranking_records(fused)  # rejects empty dense/hybrid output
    if len(rankings["dense"]) > 10 or len(rankings["lexical"]) > 10:
        raise PreparationError("Hybrid branch exceeds the 10/10 ceiling.")
    dense = tuple(sorted((x.dense_candidate for x in fused if x.dense_candidate is not None), key=lambda x: x.rank))
    lexical = tuple(sorted((x.lexical_candidate for x in fused if x.lexical_candidate is not None), key=lambda x: x.rank))
    rebuilt = fuse_candidates(chunk_set=chunk_set, dense_candidates=dense,
                              lexical_candidates=lexical, rank_constant=2.0)
    if rebuilt != fused:
        raise PreparationError("Supplied fusion differs from authoritative recomputation.")
    return fused, rankings


def _prepare_case(case, chunk_set, config, ranked_cases, compared, supplied):
    case_id = case["id"]
    if case_id in CASE_IDS[:4]:
        fused = comparison._reconstruct_case(ranked_cases[case_id], chunk_set, 2.0)
        rankings = ranked_cases[case_id]["rankings"]
        selection = select_evidence(chunk_set=chunk_set, fused_candidates=fused, max_chunks=9)
        source = "recorded_rankings"
    else:
        value = supplied[case_id]
        if type(value) is dict and set(value) == {"question", "empty_selection"}:
            if value["question"] != case["question"] or type(value["empty_selection"]) is not EvidenceSelection:
                raise PreparationError("Invalid local selection or question.")
            selection = replace(value["empty_selection"])
            if selection.max_chunks != 9 or selection.selected or selection.omitted_chunk_ids:
                raise PreparationError("Local selection must be a valid empty prefix-nine selection.")
            rankings = None
            source = "caller_validated_empty_selection"
        else:
            fused, rankings = _supplied_union(value, case, chunk_set)
            selection = select_evidence(chunk_set=chunk_set, fused_candidates=fused, max_chunks=9)
            source = "caller_supplied_hybrid"
    context = build_evidence_context(selection=selection, chunk_set=chunk_set)
    selected = [{"citation_id": citation.citation_id, "chunk_id": citation.chunk.chunk_id,
                 "fused_rank": item.candidate.rank}
                for item, citation in zip(selection.selected, context.citations)]
    if case_id in CASE_IDS[:4]:
        row = compared[case_id]
        option = row["options"]["prefix_9"]
        if row["status"] != "completed" or option["status"] != "completed" or option["max_chunks"] != 9 or selected != option["selected"]:
            raise PreparationError(f"{case_id} prefix-nine selection or citation mapping differs.")
    local = local_empty_evidence_result(selection)
    record = {
        "case_id": case_id, "question": case["question"], "status": "completed",
        "selection_source": source, "selected": selected,
        "omitted_chunk_ids": list(selection.omitted_chunk_ids), "rankings": rankings,
        "model_context_json": context.model_context_json,
        "request": None, "local_result": None,
        "token_count": "not_performed", "generation": "not_performed",
        "structural_validation": "not_performed", "status_match": "not_evaluated",
        "semantic_review": "pending",
    }
    if local is not None:
        record["local_result"] = {"status": local.status, "limitation": local.limitation,
                                  "answer_blocks": [], "clarification_question": None}
    else:
        request = prepare_gemini_request(question=case["question"], context=context,
                                         chunk_set=chunk_set, config=config)
        record["request"] = {
            "request_json": request.request_json,
            "count_tokens_body_json": request.count_tokens_body_json,
            "request_id": request.request_id, "model": request.model,
            "prompt_version": GEMINI_PROMPT_VERSION,
        }
    return record


def write_preparation_report(*, output_path: Path, chunk_set, artifact_bytes: bytes,
                             ranking_bytes: bytes, comparison_bytes: bytes,
                             generation_config: GeminiGenerationConfig,
                             case_ids: tuple[str, ...], supplied_results: dict) -> dict:
    """Reserve output exclusively, then write completed records and first failure.

    The final report is written once in a finally block. This preserves completed
    records on ordinary failures, but does not guarantee survival of process or
    filesystem failure. A failed write may leave an incomplete reserved file.
    """
    output = Path(output_path)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        report = {"format_version": FORMAT, "status": "failed", "inputs": None,
                  "settings": None, "cases": [], "failure": None,
                  "token_count": "not_performed", "generation": "not_performed",
                  "semantic_review": "pending"}
        stage, active = "inputs", None
        try:
            if (type(case_ids) is not tuple or not case_ids or len(set(case_ids)) != len(case_ids)
                    or any(type(x) is not str or x not in CASE_IDS for x in case_ids)):
                raise PreparationError("case_ids must be a nonempty unique tuple of P1–P8 IDs.")
            report["cases"] = [{"case_id": x, "status": "not_attempted"} for x in case_ids]
            if type(supplied_results) is not dict or set(supplied_results) != set(case_ids) - set(CASE_IDS[:4]):
                raise PreparationError("Supplied results must exactly match requested P5–P8 cases.")
            if type(generation_config) is not GeminiGenerationConfig:
                raise PreparationError("Explicit GeminiGenerationConfig is required.")
            config = replace(generation_config)
            hashes, artifact, ranking, ranked, compared, cases = _inputs(
                artifact_bytes, ranking_bytes, comparison_bytes, chunk_set)
            report["inputs"] = {**hashes, "artifact_id": artifact["artifact"],
                                "corpus_identity": chunk_set.corpus_identity,
                                "chunk_set_identity": chunk_set.chunk_set_id,
                                "chunking_config_id": chunk_set.chunking_config_id}
            report["settings"] = {"dense_candidates": 10, "lexical_candidates": 10,
                                  "rank_constant": 2.0, "max_chunks": 9,
                                  "generation": asdict(config), "prompt_version": GEMINI_PROMPT_VERSION}
            stage = "case_preparation"
            for index, case_id in enumerate(case_ids):
                active = case_id
                report["cases"][index] = {"case_id": case_id, "status": "in_progress"}
                report["cases"][index] = _prepare_case(cases[case_id], chunk_set, config, ranked, compared, supplied_results)
            report["status"] = "completed"
        except Exception as exc:
            if active is not None:
                report["cases"][index] = {"case_id": active, "status": "failed"}
            report["failure"] = {"stage": stage, "case_id": active, "error_type": type(exc).__name__,
                                 "message": str(exc)}
        finally:
            handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
    return report


__all__ = ["PreparationError", "write_preparation_report"]
