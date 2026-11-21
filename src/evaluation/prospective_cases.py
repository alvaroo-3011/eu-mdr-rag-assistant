"""Offline assessment of the approved prospective development cases.

Callers own the authoritative ChunkSet and any execution. This module creates
no resources, makes no routing decision, and does not infer semantic support.
Unnominated chunks are unjudged, not irrelevant.
For each element, all chunks carrying its excerpts are treated as jointly
required. Alternative per-element evidence groups are not represented.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from src.evaluation.retrieval_seed import EvaluationError
from src.generation.gemini_request import PreparedGeminiRequest, prepare_gemini_request
from src.generation.gemini_response import ParsedGeminiResponse, validate_response_binding
from src.ingestion.chunk_models import ChunkSet
from src.ingestion.models import RegulatoryRegime


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key.")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise EvaluationError("Nonstandard JSON constant: " + value)


def _excerpt(value: object) -> None:
    _require(type(value) is dict and set(value) ==
             {"chunk_id", "source_identity", "locator", "exact_quote"}, "Invalid evidence excerpt.")
    _require(type(value.get("chunk_id")) is str and re.fullmatch(
        r"chunk:sha256:[0-9a-f]{64}", value["chunk_id"]) is not None,
        "Invalid evidence chunk ID.")
    _require(type(value.get("source_identity")) is str and re.fullmatch(
        r"[a-z0-9-]+@sha256:[0-9a-f]{64}", value["source_identity"]) is not None,
        "Invalid source identity.")
    _require(_text(value.get("locator")) and _text(value.get("exact_quote")),
             "Missing evidence locator or exact quotation.")


def _element(value: object, prefix: str) -> list[dict]:
    _require(type(value) is dict and set(value) == {"summary", "element_id", "supporting_excerpts"} and
             _text(value.get("summary")) and
             type(value.get("element_id")) is str and
             re.fullmatch(re.escape(prefix) + r"\.[1-9][0-9]*", value["element_id"]) is not None,
             "Invalid answer element.")
    excerpts = value.get("supporting_excerpts")
    _require(type(excerpts) is list and bool(excerpts), "Missing element evidence.")
    for entry in excerpts:
        _excerpt(entry)
    return excerpts


def validate_prospective_structure(artifact: object) -> None:
    """Check the frozen artifact's relevant structure without loading a corpus."""
    _require(type(artifact) is dict and artifact.get("artifact") == "prospective-cases/v1"
             and artifact.get("status") == "frozen_user_approved_development_judgments",
             "Unsupported or unapproved prospective artifact.")
    _require(set(artifact) == {"artifact", "status", "analysis_head", "corpus_identity",
             "chunk_set_identity", "source_count", "chunk_count", "note", "cases", "approval", "scoring"},
             "Unexpected artifact fields.")
    for key, prefix in (("corpus_identity", "corpus"), ("chunk_set_identity", "chunk-set")):
        _require(type(artifact.get(key)) is str and re.fullmatch(
            prefix + r":sha256:[0-9a-f]{64}", artifact[key]) is not None, "Invalid identity.")
    for key in ("source_count", "chunk_count"):
        _require(type(artifact.get(key)) is int and artifact[key] > 0, "Invalid count.")
    for key in ("analysis_head", "note", "approval", "scoring"):
        _require(_text(artifact.get(key)), "Missing artifact metadata.")
    cases = artifact.get("cases")
    _require(type(cases) is list and len(cases) == 8, "Expected eight cases.")
    by_id = {}
    for case in cases:
        _require(type(case) is dict and _text(case.get("id")) and case["id"] not in by_id,
                 "Invalid or duplicate case ID.")
        by_id[case["id"]] = case
    _require(set(by_id) == {f"P{i}" for i in range(1, 9)}, "Unexpected case IDs.")
    for i, case in enumerate((by_id[f"P{i}"] for i in range(1, 9)), 1):
        status = "answered" if i <= 4 else "insufficient_evidence" if i <= 6 else (
            "needs_clarification" if i == 7 else "out_of_scope")
        _require(case.get("expected_behavior") == status and _text(case.get("question")),
                 "Invalid case behavior or question.")
        common = {"id", "question", "expected_behavior", "accepted_substantive_complete_combinations"}
        if i <= 4:
            fields = common | {"single_chunk_assessment", "mandatory_element_evidence", "combination_status",
                               "complete_combination_coverage"}
            if i in (2, 3):
                fields.add("alternative_adjudication")
            if i == 3:
                fields.add("supporting_context")
            if i == 4:
                fields.add("answer_review_qualification")
        elif i in (5, 6):
            fields = common | {"expected_elements", "evidence_review", "refusal_rubric"}
        elif i == 7:
            fields = common | {"acceptable_clarification", "unresolved"}
        else:
            fields = common | {"scope_evidence", "qualification", "oracle"}
        _require(set(case) == fields, "Unexpected or missing case fields.")
        combos = case.get("accepted_substantive_complete_combinations")
        _require(type(combos) is list, "Missing combinations.")
        if i <= 4:
            _require(_text(case.get("single_chunk_assessment")) and
                     _text(case.get("complete_combination_coverage")) and
                     case.get("combination_status") ==
                     "user_approved_assistant_assisted_development_judgment_not_independent_regulatory_validation",
                     "Missing approved combination metadata.")
            elements = case.get("mandatory_element_evidence")
            _require(type(elements) is list and bool(elements) and bool(combos), "Missing substantive evidence.")
            ids, evidence, mappings = set(), set(), []
            for element in elements:
                refs = _element(element, case["id"])
                _require(element["element_id"] not in ids, "Duplicate element ID.")
                ids.add(element["element_id"])
                mappings.append({x["chunk_id"] for x in refs})
                evidence.update(mappings[-1])
            seen = set()
            for combo in combos:
                _require(type(combo) is list and bool(combo) and
                         all(type(x) is str for x in combo) and len(combo) == len(set(combo)),
                         "Invalid combination.")
                members = frozenset(combo)
                _require(members <= evidence and members not in seen and
                         all(refs <= members for refs in mappings), "Broken combination coverage.")
                seen.add(members)
        else:
            _require(not combos and "mandatory_element_evidence" not in case,
                     "Non-answer case cannot have positive retrieval judgments.")
            if i in (5, 6):
                rubric = case.get("refusal_rubric")
                _require(type(rubric) is dict and set(rubric) == {"acceptable", "prohibited", "limit"} and
                         all(_text(rubric.get(k)) for k in
                         ("acceptable", "prohibited", "limit")) and
                         _text(case.get("evidence_review")) and
                         type(case.get("expected_elements")) is list and
                         bool(case["expected_elements"]) and all(_text(x) for x in case["expected_elements"]),
                         "Invalid refusal rubric.")
            elif i == 7:
                rubric = case.get("acceptable_clarification")
                _require(type(rubric) is dict and set(rubric) == {"status", "answer_blocks",
                         "clarification_question", "limitation", "forbidden", "schema_reference"} and
                         _text(case.get("unresolved")) and rubric.get("status") == status and
                         type(rubric.get("answer_blocks")) is list and rubric["answer_blocks"] == [] and all(_text(rubric.get(k)) for k in
                         ("clarification_question", "limitation", "forbidden", "schema_reference")),
                         "Invalid clarification rubric.")
            else:
                oracle = case.get("oracle")
                _require(type(oracle) is dict and set(oracle) == {"product_scope", "optional_regulatory_evidence"} and
                         _text(case.get("qualification")) and all(_text(oracle.get(k)) for k in
                         ("product_scope", "optional_regulatory_evidence")), "Invalid scope oracle.")
                scope = case.get("scope_evidence")
                _require(type(scope) is dict and set(scope) == {"source_identity", "locator", "chunk_id", "exact_quotes"} and
                         type(scope.get("exact_quotes")) is list and
                         bool(scope["exact_quotes"]), "Invalid optional scope evidence.")
                for quote in scope["exact_quotes"]:
                    _excerpt({k: scope[k] for k in ("source_identity", "locator", "chunk_id")} | {"exact_quote": quote})
    context = by_id["P3"].get("supporting_context")
    _require(type(context) is dict and set(context) == {"role", "chunk_id", "evidence"} and _text(context.get("role")) and
             type(context.get("evidence")) is list and bool(context["evidence"]), "Invalid P3 context.")
    for excerpt in context["evidence"]:
        _excerpt(excerpt)
        _require(excerpt["chunk_id"] == context.get("chunk_id"), "Broken P3 context reference.")
    p3_alt = by_id["P3"].get("alternative_adjudication")
    _require(type(p3_alt) is dict and set(p3_alt) == {"decision", "structured_table"} and _text(p3_alt.get("decision")) and
             _text(p3_alt.get("structured_table")), "Missing P3 table adjudication.")
    qual = by_id["P4"].get("answer_review_qualification")
    _require(type(qual) is dict and set(qual) == {"element", "scoring"} and _text(qual.get("scoring")), "Missing P4 answer review.")
    extra = _element(qual.get("element"), "P4")
    _require(qual["element"]["element_id"] not in
             {e["element_id"] for e in by_id["P4"]["mandatory_element_evidence"]} and
             all(x["chunk_id"] not in combo for x in extra for combo in
                 by_id["P4"]["accepted_substantive_complete_combinations"]),
             "P4 qualification must be separate from substantive evidence.")
    alt = by_id["P2"].get("alternative_adjudication")
    _require(type(alt) is dict and set(alt) == {"finding", "guidance_excerpt"} and _text(alt.get("finding")), "Missing alternative adjudication.")
    _excerpt(alt.get("guidance_excerpt"))


def parse_prospective(content: bytes) -> dict:
    """Parse exact UTF-8 bytes, rejecting duplicate keys and JSON constants."""
    try:
        artifact = json.loads(content.decode("utf-8"), object_pairs_hook=_unique,
                              parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, AttributeError) as exc:
        raise EvaluationError("Invalid prospective JSON.") from exc
    validate_prospective_structure(artifact)
    return artifact


def load_prospective(path: Path) -> dict:
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise EvaluationError("Unable to read prospective artifact.") from exc
    return parse_prospective(content)


def validate_prospective_against_chunks(artifact: dict, chunk_set: ChunkSet) -> None:
    """Bind every recorded excerpt to caller-supplied authoritative chunks.

    This verifies in-memory identities and text, not source files or indexes.
    The caller must validate the manifest and files before construction.
    """
    validate_prospective_structure(artifact)
    _require(isinstance(chunk_set, ChunkSet), "Expected authoritative ChunkSet.")
    _require(artifact["corpus_identity"] == chunk_set.corpus_identity and
             artifact["chunk_set_identity"] == chunk_set.chunk_set_id and
             artifact["source_count"] == len(chunk_set.manifest.sources) and
             artifact["chunk_count"] == len(chunk_set.chunks), "Corpus binding mismatch.")
    chunks = {c.chunk_id: c for c in chunk_set.chunks}
    for case in artifact["cases"]:
        refs = [x for e in case.get("mandatory_element_evidence", [])
                for x in e["supporting_excerpts"]]
        refs += case.get("supporting_context", {}).get("evidence", [])
        qual = case.get("answer_review_qualification")
        if qual:
            refs += qual["element"]["supporting_excerpts"]
        if case["id"] == "P2":
            refs.append(case["alternative_adjudication"]["guidance_excerpt"])
        if case["id"] == "P8":
            scope = case["scope_evidence"]
            refs += [{k: scope[k] for k in ("source_identity", "locator", "chunk_id")} |
                     {"exact_quote": q} for q in scope["exact_quotes"]]
        for ref in refs:
            chunk = chunks.get(ref["chunk_id"])
            _require(chunk is not None and chunk.source.source_identity == ref["source_identity"] and
                     RegulatoryRegime.MDR in chunk.source.permitted_regimes and
                     ref["exact_quote"] in chunk.text, "Evidence chunk, source, regime, or excerpt mismatch.")


def assess_prospective_ranking(case: dict, ranked_chunk_ids: list[str] | tuple[str, ...], *, cutoff: int) -> dict:
    """Assess nominated substantive chunks; context and qualifications are separate."""
    _require(type(cutoff) is int and cutoff > 0, "Invalid cutoff.")
    _require(type(ranked_chunk_ids) in (list, tuple) and all(_text(x) for x in ranked_chunk_ids)
             and len(ranked_chunk_ids) == len(set(ranked_chunk_ids)), "Invalid ranking.")
    if case["expected_behavior"] != "answered":
        return {"applicability": "not_applicable", "reason": "No positive retrieval judgment."}
    nominated = list(dict.fromkeys(x["chunk_id"] for e in case["mandatory_element_evidence"]
                                   for x in e["supporting_excerpts"]))
    top = ranked_chunk_ids[:cutoff]
    found = [cid for cid in top if cid in nominated]
    combos = case["accepted_substantive_complete_combinations"]
    completed = [i + 1 for i, members in enumerate(combos) if all(cid in top for cid in members)]
    separate = {}
    if "supporting_context" in case:
        separate["supporting_context"] = case["supporting_context"]["chunk_id"]
    if "answer_review_qualification" in case:
        separate["answer_review_qualification"] = case["answer_review_qualification"]["element"]["supporting_excerpts"][0]["chunk_id"]
    separate = {role: {"chunk_id": cid, "rank": ranked_chunk_ids.index(cid) + 1
                       if cid in ranked_chunk_ids else None,
                       "within_cutoff": cid in top} for role, cid in separate.items()}
    return {"applicability": "applicable", "cutoff": cutoff, "available_count": len(ranked_chunk_ids),
            "examined_count": len(top), "nominated_total": len(nominated), "nominated_found": len(found),
            "found_chunk_ids": found, "missing_chunk_ids": [x for x in nominated if x not in top],
            "complete_combination_found": bool(completed), "completed_combination_indexes": completed,
            "separate_evidence": separate,
            "other_chunk_status": "unjudged_except_separately_identified_evidence"}


def project_retrieval_report(*, artifact: dict, case_id: str, rankings: dict, cutoff: int) -> dict:
    """Project caller-recorded branch/union/selected rankings; executes nothing."""
    validate_prospective_structure(artifact)
    case = next((c for c in artifact["cases"] if c["id"] == case_id), None)
    _require(case is not None and type(rankings) is dict and
             set(rankings) == {"dense", "lexical", "fused", "selected"}, "Invalid case or rankings.")
    assessment = {name: assess_prospective_ranking(case, ids, cutoff=cutoff)
                  for name, ids in rankings.items()}
    assessment["fused_union"] = assess_prospective_ranking(
        case, rankings["fused"], cutoff=max(1, len(rankings["fused"])))
    return {"case_id": case_id, "retrieval": assessment, "answer_behavior": "not_evaluated",
            "semantic_review": "not_evaluated"}


def assess_response(*, artifact: dict, case_id: str, chunk_set: ChunkSet,
                    response: ParsedGeminiResponse | None, request: PreparedGeminiRequest | None,
                    execution_error: bool, reviewed_judgment: dict | None) -> dict:
    """Compare a request-bound validated response; semantic review remains separate.

    Directly constructed ParsedGeminiResponse objects establish structure only;
    callers must use the normal authoritative parser at initial ingress.
    """
    validate_prospective_against_chunks(artifact, chunk_set)
    case = next((c for c in artifact["cases"] if c["id"] == case_id), None)
    _require(case is not None, "Unknown case ID.")
    _require(type(execution_error) is bool, "execution_error must be boolean.")
    _require(not (execution_error and response is not None), "Execution failure cannot have a response.")
    if request is not None:
        _require(type(request) is PreparedGeminiRequest and request.question == case["question"],
                 "Request question does not match the approved case.")
        try:
            checked = prepare_gemini_request(question=request.question, context=request.context,
                                             chunk_set=chunk_set, config=request.config)
            _require(checked.request_id == request.request_id, "Request differs from authoritative preparation.")
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError("Request evidence does not match the approved ChunkSet.") from exc
    _require(response is None or request is not None, "Response requires its prepared request.")
    if reviewed_judgment is not None:
        _require(type(reviewed_judgment) is dict and set(reviewed_judgment) == {"reviewer", "decision"}
                 and _text(reviewed_judgment["reviewer"]) and
                 reviewed_judgment["decision"] in ("pass", "fail"), "Invalid reviewed judgment.")
        _require(response is not None and not execution_error, "Review requires a response.")
    result = {"expected_status": case["expected_behavior"], "observed_status": None,
              "status_matches": None, "semantic_review": reviewed_judgment or "pending_human_review"}
    if execution_error:
        result["response_state"] = "execution_error"
    elif response is None:
        result["response_state"] = "unavailable"
    else:
        _require(type(response) is ParsedGeminiResponse, "Expected parsed response.")
        validate_response_binding(response=response, request=request)
        result.update(response_state="available", observed_status=response.status,
                      status_matches=response.status == case["expected_behavior"])
    return result
