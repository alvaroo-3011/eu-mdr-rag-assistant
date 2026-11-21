"""Validate reviewed seed data and assess recovery of approved evidence.

No model, database, corpus loader, or retrieval is executed on import. Scores
describe the nominated evidence only: an unjudged result is not irrelevant.
The input seed remains ordinary JSON data; callers must not mutate it between
validation and evaluation. This module does not decide regulatory sufficiency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import re
from typing import Any

from src.ingestion.chunk_models import ChunkSet
from src.ingestion.models import RegulatoryRegime


class EvaluationError(ValueError):
    """Invalid seed, corpus binding, or supplied ranking."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _rows(value: object, name: str, *, nonempty: bool = True) -> list:
    _require(type(value) is list, f"{name} must be an array.")
    _require(not nonempty or bool(value), f"{name} must not be empty.")
    return value


def _indexed(rows: object, key: str, name: str) -> dict[str, dict]:
    result = {}
    for row in _rows(rows, name):
        _require(type(row) is dict and _text(row.get(key)), f"Invalid {name} record.")
        _require(row[key] not in result, f"Duplicate {name} {key}: {row[key]}.")
        result[row[key]] = row
    return result


def _strings(value: object, name: str, *, nonempty: bool = True) -> list[str]:
    items = _rows(value, name, nonempty=nonempty)
    _require(all(_text(x) for x in items), f"{name} requires nonblank strings.")
    _require(len(items) == len(set(items)), f"Duplicate values in {name}.")
    return items


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise EvaluationError(f"Nonstandard JSON constant: {value}.")


def load_seed(path: Path) -> dict:
    """Parse v1 JSON, rejecting duplicate keys and nonstandard constants."""
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Unable to read seed: {path}.") from exc
    return parse_seed(content)


def parse_seed(content: bytes) -> dict:
    """Validate exactly the bytes the runner hashes for run provenance."""
    try:
        seed = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("Unable to parse UTF-8 seed JSON.") from exc
    validate_seed_structure(seed)
    return seed


def validate_seed_structure(seed: object) -> None:
    """Validate the existing seed format without requiring corpus files."""
    _require(type(seed) is dict, "Seed must be an object.")
    _require(seed.get("format_version") == "retrieval-evaluation-seed/v1",
             "Unsupported seed format_version.")
    _require(_text(seed.get("artifact_id")), "Missing artifact_id.")
    frozen = seed.get("frozen_corpus")
    _require(type(frozen) is dict, "Missing frozen_corpus.")
    for name, prefix in (("corpus_identity", "corpus"),
                         ("chunk_set_identity", "chunk-set")):
        value = frozen.get(name)
        _require(type(value) is str and bool(re.fullmatch(
            rf"{prefix}:sha256:[0-9a-f]{{64}}", value)), f"Invalid {name}.")
    for name in ("source_count", "chunk_count"):
        _require(_positive_integer(frozen.get(name)), f"Invalid {name}.")
    _require(_text(frozen.get("manifest_path")), "Missing manifest_path.")
    policy = seed.get("judgment_policy")
    _require(type(policy) is dict, "Missing judgment_policy.")
    _require(policy.get("judgments_are_question_specific") is True
             and policy.get("all_other_chunks") == "unjudged"
             and policy.get("approved_labels_in_this_seed") == ["supporting"],
             "Only question-specific supporting judgments are supported in v1.")
    provenance = seed.get("review_provenance")
    _require(type(provenance) is dict and all(_text(provenance.get(k)) for k in
             ("approved_by", "approved_in", "approved_on", "basis", "scope")),
             "Missing review provenance.")
    evidence = _indexed(seed.get("evidence"), "evidence_id", "evidence")
    chunk_ids, excerpt_ids = set(), set()
    for ev in evidence.values():
        chunk_id = ev.get("chunk_id")
        _require(type(chunk_id) is str and bool(re.fullmatch(
            r"chunk:sha256:[0-9a-f]{64}", chunk_id)), "Invalid chunk_id.")
        _require(chunk_id not in chunk_ids, "Duplicate evidence chunk_id.")
        chunk_ids.add(chunk_id)
        source = ev.get("source")
        _require(type(source) is dict and all(_text(source.get(k)) for k in (
            "document_id", "title", "version", "source_version_identity", "sha256", "path"
        )), "Invalid evidence source.")
        for field in ("regulatory_regimes", "permitted_regimes"):
            _strings(source.get(field), field)
        _require("MDR" in source["permitted_regimes"], "Evidence must permit MDR.")
        _strings(ev.get("source_block_ids"), "source_block_ids")
        for field in ("hierarchy", "legal_locators", "physical_pdf_pages"):
            _rows(ev.get(field), field, nonempty=False)
        pages = ev["physical_pdf_pages"]
        _require(all(_positive_integer(p) for p in pages)
                 and pages == sorted(set(pages)), "Invalid physical PDF pages.")
        for entry in ev["hierarchy"]:
            _require(type(entry) is dict and set(entry) == {"kind", "title"}
                     and _text(entry["kind"]) and type(entry["title"]) is str,
                     "Invalid hierarchy entry.")
        for locator in ev["legal_locators"]:
            for part in _rows(locator, "legal locator"):
                _require(type(part) is dict and set(part) == {"kind", "value"}
                         and _text(part["kind"]) and _text(part["value"]),
                         "Invalid legal locator part.")
        if "structured_table" in ev:
            table = ev["structured_table"]
            _require(type(table) is dict and _text(table.get("block_id"))
                     and _positive_integer(table.get("physical_pdf_page"))
                     and type(table.get("source_row_start")) is int
                     and type(table.get("source_row_end")) is int
                     and 0 <= table["source_row_start"] < table["source_row_end"]
                     and type(table.get("is_complete_table")) is bool,
                     "Invalid structured table metadata.")
            headers = _rows(table.get("headers"), "table headers", nonempty=False)
            rows = _rows(table.get("rows"), "table rows")
            _require(len(rows) == table["source_row_end"] - table["source_row_start"],
                     "Table row bounds/count mismatch.")
            for row in rows:
                _rows(row, "table row")
            _require(all(cell is None or type(cell) is str
                         for row in [headers, *rows] for cell in row),
                     "Invalid table cell.")
        _strings(ev.get("interpretation_limitations"), "interpretation_limitations",
                 nonempty=False)
        for excerpt in _indexed(ev.get("exact_excerpts"), "excerpt_id", "excerpts").values():
            _require(excerpt["excerpt_id"] not in excerpt_ids, "Duplicate excerpt_id.")
            _require(_text(excerpt.get("text")), "Blank excerpt.")
            excerpt_ids.add(excerpt["excerpt_id"])
    combinations_seen = set()
    alternative_count = 0
    for case in _indexed(seed.get("cases"), "case_id", "cases").values():
        for field in ("question", "topic_id"):
            _require(_text(case.get(field)), f"Missing case {field}.")
        _require(case.get("review_status") == "approved-for-seed-v1", "Unapproved case.")
        _strings(case.get("coverage_tags"), "coverage_tags")
        elements = _indexed(case.get("required_answer_elements"), "element_id", "elements")
        _require(all(_text(e.get("text")) for e in elements.values()), "Blank element.")
        judgments = _indexed(case.get("supporting_judgments"), "evidence_id", "judgments")
        _require(set(judgments) <= evidence.keys(), "Unknown judgment evidence reference.")
        _require(all(j.get("label") == "supporting" and _text(j.get("support_extent"))
                     for j in judgments.values()), "Invalid supporting judgment.")
        mapping = case.get("element_to_evidence")
        _require(type(mapping) is dict and mapping.keys() == elements.keys(),
                 "Answer-element mapping must have exact element coverage.")
        mapped = set()
        for refs in mapping.values():
            mapped.update(_strings(refs, "element evidence"))
        _require(mapped == judgments.keys(), "Element mappings must cover judged evidence only.")
        combos = _indexed(case.get("support_combinations"), "combination_id", "combinations")
        alternative_count += len(combos) - 1
        member_sets = set()
        for combo in combos.values():
            _require(combo["combination_id"] not in combinations_seen,
                     "Duplicate combination_id across cases.")
            combinations_seen.add(combo["combination_id"])
            members = frozenset(_strings(combo.get("members"), "combination members"))
            _require(combo.get("approved") is True
                     and combo.get("members_jointly_required") is True,
                     "Combination must be approved with jointly required members.")
            _require(members <= judgments.keys(), "Unknown combination evidence reference.")
            _require(members not in member_sets, "Duplicate support combination.")
            member_sets.add(members)
            _require(all(members.intersection(refs) for refs in mapping.values()),
                     "Combination does not cover every mapped answer element.")
        _require(set().union(*member_sets) == judgments.keys(),
                 "Judged evidence must belong to an approved combination.")
    _require(type(policy.get("approved_alternative_combinations")) is int
             and policy["approved_alternative_combinations"] == alternative_count,
             "Alternative-combination count disagrees with cases.")


def validate_seed_against_chunks(seed: dict, chunk_set: ChunkSet) -> None:
    """Bind judgments and stored evidence details to original corpus records.

    The caller must checksum-verify and load the manifest before building the
    ChunkSet. This function cannot prove file checksums from in-memory records.
    Legal locators and block IDs in the seed may be relevant subsets of a chunk.
    """
    validate_seed_structure(seed)
    _require(isinstance(chunk_set, ChunkSet), "Expected a validated ChunkSet.")
    frozen = seed["frozen_corpus"]
    actual = {
        "corpus_identity": chunk_set.corpus_identity,
        "chunk_set_identity": chunk_set.chunk_set_id,
        "source_count": len(chunk_set.manifest.sources),
        "chunk_count": len(chunk_set.chunks),
    }
    _require(all(frozen[k] == v for k, v in actual.items()), "Frozen corpus identity/count mismatch.")
    chunks = {c.chunk_id: c for c in chunk_set.chunks}
    for ev in seed["evidence"]:
        chunk = chunks.get(ev["chunk_id"])
        _require(chunk is not None, f"Unknown evidence chunk: {ev['chunk_id']}.")
        source = chunk.source
        expected_source = {
            "document_id": source.document_id, "title": source.official_title,
            "version": source.version, "source_version_identity": source.source_identity,
            "sha256": source.sha256, "path": source.local_path,
            "regulatory_regimes": [r.value for r in source.regulatory_regimes],
            "permitted_regimes": [r.value for r in source.permitted_regimes],
        }
        _require(ev["source"] == expected_source, "Evidence source/version metadata mismatch.")
        _require(RegulatoryRegime.MDR in source.permitted_regimes, "Chunk does not permit MDR.")
        paths = [[{"kind": h.kind, "title": h.title} for h in path]
                 for path in chunk.hierarchies]
        _require(ev["hierarchy"] in paths or (not paths and not ev["hierarchy"]),
                 "Evidence hierarchy mismatch.")
        locators = [[{"kind": p.kind, "value": p.value} for p in path]
                    for path in chunk.legal_locators]
        _require(all(loc in locators for loc in ev["legal_locators"]), "Legal locator mismatch.")
        _require(ev["physical_pdf_pages"] == [p.pdf_page_number for p in chunk.pages],
                 "Physical PDF page mismatch.")
        selections = {s.block_id for s in chunk.block_selections}
        selections.update(s.source_block_id for s in chunk.block_selections
                          if s.source_block_id is not None)
        _require(set(ev["source_block_ids"]) <= selections, "Source block reference mismatch.")
        _require(all(x["text"] in chunk.text for x in ev["exact_excerpts"]),
                 "Evidence excerpt not found verbatim in original chunk.")
        if "structured_table" in ev:
            table = ev["structured_table"]
            _require(type(table) is dict, "Invalid structured table.")
            segments = [s for s in chunk.table_segments if s.block_id == table.get("block_id")]
            _require(len(segments) == 1, "Table block mismatch.")
            segment = segments[0]
            expected_table = {
                "block_id": segment.block_id,
                "physical_pdf_page": segment.table.source_page,
                "headers": list(segment.table.headers),
                "rows": [list(row) for row in segment.table.rows],
                "source_row_start": segment.source_row_start,
                "source_row_end": segment.source_row_end,
                "is_complete_table": segment.is_complete_table,
            }
            _require(table == expected_table
                     and type(table.get("is_complete_table")) is bool,
                     "Structured table snapshot mismatch.")


def assess_ranking(case: dict, evidence: list[dict], ranked_chunk_ids: Sequence[str],
                   *, cutoff: int) -> dict:
    """Count nominated evidence within a cutoff; never judge other IDs."""
    _require(_positive_integer(cutoff), "cutoff must be a positive nonboolean integer.")
    _require(isinstance(ranked_chunk_ids, (list, tuple)), "Ranking must be a list or tuple.")
    _require(all(_text(x) for x in ranked_chunk_ids), "Invalid ranking chunk ID.")
    _require(len(ranked_chunk_ids) == len(set(ranked_chunk_ids)), "Duplicate ranking chunk ID.")
    by_evidence = {e["evidence_id"]: e["chunk_id"] for e in evidence}
    approved = [by_evidence[j["evidence_id"]] for j in case["supporting_judgments"]]
    top = ranked_chunk_ids[:cutoff]
    found = [cid for cid in top if cid in approved]
    completed = [combo["combination_id"] for combo in case["support_combinations"]
                 if all(by_evidence[eid] in top for eid in combo["members"])]
    return {
        "cutoff": cutoff, "available_count": len(ranked_chunk_ids),
        "examined_count": len(top), "approved_evidence_total": len(approved),
        "approved_evidence_found": len(found), "found_chunk_ids": found,
        "missing_approved_chunk_ids": [cid for cid in approved if cid not in top],
        "any_approved_support_found": bool(found),
        "complete_approved_combination_found": bool(completed),
        "completed_combination_ids": completed,
    }


def _ranking_records(fused: tuple) -> dict[str, list[dict]]:
    """Recover original branch ranks/scores from the complete hybrid union."""
    from src.retrieval.fusion import FusedCandidate

    _require(type(fused) is tuple and bool(fused), "Hybrid output must be a nonempty tuple.")
    _require(all(isinstance(c, FusedCandidate) for c in fused), "Invalid hybrid candidate.")
    _require([c.rank for c in fused] == list(range(1, len(fused) + 1)), "Invalid fused ranks.")
    _require(len({c.chunk_id for c in fused}) == len(fused), "Duplicate fused IDs.")
    rankings = {}
    for name, attribute, score in (
        ("dense", "dense_candidate", "cosine_distance"),
        ("lexical", "lexical_candidate", "bm25_score"),
    ):
        candidates = sorted((getattr(c, attribute) for c in fused
                             if getattr(c, attribute) is not None), key=lambda c: c.rank)
        _require([c.rank for c in candidates] == list(range(1, len(candidates) + 1)),
                 f"Invalid {name} provenance ranks.")
        rankings[name] = [{"rank": c.rank, "chunk_id": c.chunk_id,
                           score: getattr(c, score)} for c in candidates]
    _require(bool(rankings["dense"]), "Hybrid provenance must contain dense results.")
    rankings["fused"] = [{"rank": c.rank, "chunk_id": c.chunk_id,
                           "rrf_score": c.rrf_score,
                           "dense_rank": c.dense_candidate.rank if c.dense_candidate else None,
                           "lexical_rank": c.lexical_candidate.rank if c.lexical_candidate else None}
                          for c in fused]
    return rankings


def ranking_records(fused: tuple) -> dict[str, list[dict]]:
    """Validate and recover branch and fused ranks from one complete union."""
    return _ranking_records(fused)


def _error_record(exc: Exception) -> list[dict]:
    """Preserve ordinary error causes without turning a failure into a miss."""
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append({"type": type(exc).__name__, "message": str(exc)})
        exc = exc.__cause__
    return chain


def evaluate_cases(seed: dict, *, retrieve: Callable[[str], tuple], cutoff: int) -> list[dict]:
    """Call the supplied hybrid function once per case; no retries/fallback.

    An ordinary failure is recorded without assessment values, then the next
    case is attempted. BaseException propagates. The callable must use the
    identity-bound hybrid entry point; this function does not create backends.
    """
    validate_seed_structure(seed)
    _require(_positive_integer(cutoff), "Invalid cutoff.")
    results = []
    for case in seed["cases"]:
        record = {"case_id": case["case_id"], "topic_id": case["topic_id"],
                  "question": case["question"]}
        stage = "retrieval"
        try:
            fused = retrieve(case["question"])
            stage = "assessment"
            rankings = _ranking_records(fused)
            assessment = {name: assess_ranking(
                case, seed["evidence"], [c["chunk_id"] for c in rows], cutoff=cutoff,
            ) for name, rows in rankings.items()}
        except Exception as exc:
            record.update(status="error", error_stage=stage, errors=_error_record(exc))
        else:
            record.update(status="ok", rankings=rankings, assessment=assessment)
        results.append(record)
    return results
