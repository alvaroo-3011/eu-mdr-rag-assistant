"""Offline prospective-artifact assessment using typed synthetic records."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from src.evaluation.prospective_cases import (
    EvaluationError, assess_prospective_ranking, assess_response, load_prospective,
    parse_prospective, project_retrieval_report, validate_prospective_against_chunks,
    validate_prospective_structure,
)
from src.generation.gemini_response import parse_gemini_response
from src.generation.gemini_request import prepare_gemini_request
from src.generation.evidence_context import build_evidence_context
from src.retrieval.evidence_selection import select_evidence
from tests.unit.evaluation.test_retrieval_seed import fixture, synthetic_union
from tests.unit.generation.test_gemini_request import _config, _fixture

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_heavy_modules():
    with patch.dict(sys.modules, {"chromadb": None, "torch": None, "transformers": None}):
        yield


def synthetic():
    artifact = load_prospective(ROOT / "data/evaluation/prospective_cases_v1.json")
    _, chunk_set = fixture()
    artifact.update(corpus_identity=chunk_set.corpus_identity, chunk_set_identity=chunk_set.chunk_set_id,
                    source_count=len(chunk_set.manifest.sources), chunk_count=len(chunk_set.chunks))
    chunks = chunk_set.chunks
    for case in artifact["cases"]:
        for i, element in enumerate(case.get("mandatory_element_evidence", [])):
            chunk = chunks[i % 2]
            for ref in element["supporting_excerpts"]:
                ref.update(chunk_id=chunk.chunk_id, source_identity=chunk.source.source_identity,
                           exact_quote=chunk.text)
        if case["expected_behavior"] == "answered":
            refs = [x["chunk_id"] for e in case["mandatory_element_evidence"]
                    for x in e["supporting_excerpts"]]
            case["accepted_substantive_complete_combinations"] = [list(dict.fromkeys(refs))]
    for case in artifact["cases"]:
        if case["id"] == "P2":
            ref = case["alternative_adjudication"]["guidance_excerpt"]
            ref.update(chunk_id=chunks[2].chunk_id, source_identity=chunks[2].source.source_identity,
                       exact_quote=chunks[2].text)
        if case["id"] == "P3":
            context = case["supporting_context"]
            context["chunk_id"] = chunks[2].chunk_id
            for ref in context["evidence"]:
                ref.update(chunk_id=chunks[2].chunk_id, source_identity=chunks[2].source.source_identity,
                           exact_quote=chunks[2].text)
        if case["id"] == "P4":
            ref = case["answer_review_qualification"]["element"]["supporting_excerpts"][0]
            ref.update(chunk_id=chunks[2].chunk_id, source_identity=chunks[2].source.source_identity,
                       exact_quote=chunks[2].text)
        if case["id"] == "P8":
            case["scope_evidence"].update(chunk_id=chunks[2].chunk_id,
                source_identity=chunks[2].source.source_identity, exact_quotes=[chunks[2].text])
    return artifact, chunk_set


def test_committed_artifact_and_json_strictness():
    artifact = load_prospective(ROOT / "data/evaluation/prospective_cases_v1.json")
    assert len(artifact["cases"]) == 8
    for content in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'[]'):
        with pytest.raises(EvaluationError):
            parse_prospective(content)


@pytest.mark.parametrize("change", [
    lambda a: a["cases"][1].update(id="P1"),
    lambda a: a.update(chunk_count=True),
    lambda a: a["cases"][4].update(expected_behavior="answered"),
    lambda a: a["cases"][0]["accepted_substantive_complete_combinations"][0].append("unknown"),
    lambda a: a["cases"][0]["mandatory_element_evidence"][0].update(element_id="bad"),
    lambda a: a["cases"][3]["accepted_substantive_complete_combinations"][0].append(
        a["cases"][3]["answer_review_qualification"]["element"]["supporting_excerpts"][0]["chunk_id"]),
    lambda a: a["cases"][6]["acceptable_clarification"].update(answer_blocks=["answer"]),
    lambda a: a["cases"][6].update(unresolved=None),
    lambda a: a["cases"][7].update(qualification=[]),
    lambda a: a["cases"][0]["mandatory_element_evidence"][0].update(extra="unexpected"),
    lambda a: a["cases"][0]["mandatory_element_evidence"][0]["supporting_excerpts"][0].update(extra=1),
    lambda a: a["cases"][2]["supporting_context"].update(extra=1),
    lambda a: a["cases"][3]["answer_review_qualification"].update(extra=1),
    lambda a: a["cases"][4]["refusal_rubric"].update(extra=1),
    lambda a: a["cases"][6].update(acceptable_clarification=[]),
    lambda a: a["cases"][7]["scope_evidence"].update(exact_quotes=[None]),
    lambda a: a["cases"][7]["oracle"].update(extra=1),
])
def test_invalid_structure(change):
    artifact, _ = synthetic()
    change(artifact)
    with pytest.raises(EvaluationError):
        validate_prospective_structure(artifact)


def test_binding_and_identity_source_excerpt_failures():
    artifact, chunks = synthetic()
    validate_prospective_against_chunks(artifact, chunks)
    for field, value in (("corpus_identity", "corpus:sha256:" + "0" * 64),
                         ("chunk_set_identity", "chunk-set:sha256:" + "0" * 64)):
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(EvaluationError):
            validate_prospective_against_chunks(bad, chunks)
    for field, value in (("source_identity", "other@sha256:" + "0" * 64),
                         ("exact_quote", "Not present in source.")):
        bad = deepcopy(artifact)
        bad["cases"][0]["mandatory_element_evidence"][0]["supporting_excerpts"][0][field] = value
        with pytest.raises(EvaluationError):
            validate_prospective_against_chunks(bad, chunks)


def test_rankings_combinations_unjudged_and_separate_context():
    artifact, chunks = synthetic()
    ids = [c.chunk_id for c in chunks.chunks]
    for case in artifact["cases"][:4]:
        result = assess_prospective_ranking(case, [ids[2], ids[0], ids[1]], cutoff=2)
        assert result["nominated_found"] == 1 and not result["complete_combination_found"]
        assert result["other_chunk_status"] == "unjudged_except_separately_identified_evidence"
        result = assess_prospective_ranking(case, [ids[2], ids[0], ids[1]], cutoff=3)
        assert result["complete_combination_found"] and result["nominated_total"] == 2
    assert artifact["cases"][2]["supporting_context"]["chunk_id"] == ids[2]
    assert artifact["cases"][3]["answer_review_qualification"]["element"]["supporting_excerpts"][0]["chunk_id"] == ids[2]
    for case, role in ((artifact["cases"][2], "supporting_context"),
                       (artifact["cases"][3], "answer_review_qualification")):
        result = assess_prospective_ranking(case, [ids[2], ids[0], ids[1]], cutoff=2)
        assert result["separate_evidence"][role]["rank"] == 1
        assert result["separate_evidence"][role]["within_cutoff"] is True
        assert result["nominated_found"] == 1


def test_projection_marks_negative_cases_and_response_not_evaluated():
    artifact, chunks = synthetic()
    ids = [c.chunk_id for c in chunks.chunks]
    rankings = {name: ids for name in ("dense", "lexical", "fused", "selected")}
    for case in artifact["cases"]:
        report = project_retrieval_report(artifact=artifact, case_id=case["id"], rankings=rankings, cutoff=2)
        assert report["answer_behavior"] == report["semantic_review"] == "not_evaluated"
        assert report["retrieval"]["dense"]["applicability"] == (
            "applicable" if case["id"] in ("P1", "P2", "P3", "P4") else "not_applicable")
        if case["id"] == "P1":
            assert report["retrieval"]["fused_union"]["complete_combination_found"]


def _case_request(chunks, question):
    selection = select_evidence(chunk_set=chunks, fused_candidates=synthetic_union(chunks), max_chunks=2)
    context = build_evidence_context(selection=selection, chunk_set=chunks)
    return prepare_gemini_request(question=question, context=context, chunk_set=chunks, config=_config())


def _response(chunks, request):
    return parse_gemini_response(response_json=json.dumps({"status": "insufficient_evidence",
        "answer_blocks": [], "limitation": "Not enough evidence.", "clarification_question": None}),
        request_id=request.request_id, request=request, chunk_set=chunks)


def test_response_status_and_semantic_review_are_independent():
    artifact, chunks = synthetic()
    request = _case_request(chunks, artifact["cases"][4]["question"])
    response = _response(chunks, request)
    args = dict(artifact=artifact, chunk_set=chunks, response=response, request=request,
                execution_error=False, reviewed_judgment=None)
    match = assess_response(case_id="P5", **args)
    other_request = _case_request(chunks, artifact["cases"][0]["question"])
    mismatch = assess_response(case_id="P1", **{**args, "request": other_request,
                                "response": _response(chunks, other_request)})
    assert match["status_matches"] is True and mismatch["status_matches"] is False
    assert match["semantic_review"] == mismatch["semantic_review"] == "pending_human_review"
    reviewed = assess_response(case_id="P5", **{**args,
        "reviewed_judgment": {"reviewer": "reviewer", "decision": "pass"}})
    assert reviewed["semantic_review"]["decision"] == "pass"
    for error, state in ((True, "execution_error"), (False, "unavailable")):
        result = assess_response(artifact=artifact, case_id="P5", chunk_set=chunks, response=None, request=None,
                                 execution_error=error, reviewed_judgment=None)
        assert result["response_state"] == state and result["status_matches"] is None
    with pytest.raises(EvaluationError):
        assess_response(case_id="P5", **{**args, "execution_error": True})


def test_response_rejects_unrelated_question_and_corpus_or_request_chunk_set():
    artifact, chunks = synthetic()
    request = _case_request(chunks, artifact["cases"][4]["question"])
    response = _response(chunks, request)
    args = dict(artifact=artifact, case_id="P5", chunk_set=chunks, response=response,
                request=request, execution_error=False, reviewed_judgment=None)
    unrelated = _case_request(chunks, "An unrelated question?")
    with pytest.raises(EvaluationError):
        assess_response(**{**args, "request": unrelated, "response": _response(chunks, unrelated)})
    for field, value in (("corpus_identity", "corpus:sha256:" + "0" * 64),
                         ("chunk_set_identity", "chunk-set:sha256:" + "0" * 64)):
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(EvaluationError):
            assess_response(**{**args, "artifact": bad})
    foreign_chunks, foreign_context, config, _ = _fixture()
    foreign_request = prepare_gemini_request(question=artifact["cases"][4]["question"],
        context=foreign_context, chunk_set=foreign_chunks, config=config)
    with pytest.raises(EvaluationError):
        assess_response(**{**args, "request": foreign_request,
                           "response": _response(foreign_chunks, foreign_request)})
