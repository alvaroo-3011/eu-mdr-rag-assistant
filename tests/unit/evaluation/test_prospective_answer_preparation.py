"""Synthetic, provider-free tests for prospective request preparation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import socket
import sys
from unittest.mock import patch

import pytest

from scripts import evaluate_evidence_selection_context as comparison
from src.evaluation import prospective_answer_preparation as preparation
from src.generation.gemini_request import GeminiGenerationConfig
from src.retrieval.evidence_selection import select_evidence
from tests.unit.evaluation.test_evidence_selection_context import synthetic_run


@pytest.fixture(autouse=True)
def forbid_resources():
    with (patch.dict(sys.modules, {"chromadb": None, "torch": None, "transformers": None}),
          patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")),
          patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden")),
          patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden"))):
        yield


@pytest.fixture
def bundle(synthetic_run, monkeypatch):
    root = synthetic_run["root"]
    result = comparison.run_comparison(
        project_root=root, output_path=root / "comparison.json",
        chunk_set_loader=synthetic_run["loader"], provenance=synthetic_run["provenance"],
    )
    assert result["status"] == "completed"
    artifact = (root / synthetic_run["names"]["prospective_judgments"]).read_bytes()
    ranking = (root / synthetic_run["names"]["prospective_report"]).read_bytes()
    compared = (root / "comparison.json").read_bytes()
    for name, content in (("ARTIFACT_SHA256", artifact), ("RANKING_SHA256", ranking),
                          ("COMPARISON_SHA256", compared)):
        monkeypatch.setattr(preparation, name, hashlib.sha256(content).hexdigest())
    config = GeminiGenerationConfig(model="models/synthetic", max_output_tokens=100,
                                    temperature=0.0, thinking_budget=0, thinking_level=None)
    return {"root": root, "chunk_set": synthetic_run["chunk_set"],
            "artifact_bytes": artifact, "ranking_bytes": ranking,
            "comparison_bytes": compared, "generation_config": config}


def run(bundle, ids=("P1",), supplied=None, output="prepared.json"):
    return preparation.write_preparation_report(
        **{k: v for k, v in bundle.items() if k != "root"},
        output_path=bundle["root"] / output, case_ids=ids,
        supplied_results={} if supplied is None else supplied,
    )


def supplied_result(bundle, case_id):
    artifact = json.loads(bundle["artifact_bytes"])
    ranking = json.loads(bundle["ranking_bytes"])
    case = next(c for c in artifact["cases"] if c["id"] == case_id)
    reference = ranking["cases"][0]
    fused = comparison._reconstruct_case(reference, bundle["chunk_set"], 2.0)
    return {"question": case["question"], "fused_candidates": fused}


def test_recorded_and_supplied_results_reproduce_exact_request(bundle):
    supplied = {"P5": supplied_result(bundle, "P5")}
    first = run(bundle, ("P1", "P5"), supplied, "first.json")
    second = run(bundle, ("P1", "P5"), supplied, "second.json")
    assert first == second
    assert first["status"] == "completed"
    assert json.loads((bundle["root"] / "first.json").read_text(encoding="utf-8")) == first
    for case in first["cases"]:
        assert len(case["selected"]) == 9
        assert [x["citation_id"] for x in case["selected"]] == [str(i) for i in range(1, 10)]
        request = case["request"]
        logical = json.loads(request["request_json"])
        assert json.loads(request["count_tokens_body_json"])["generateContentRequest"] == logical
        user = json.loads(logical["contents"][0]["parts"][0]["text"])
        assert user["question"] == case["question"]
        assert user["evidence_context"] == json.loads(case["model_context_json"])
        assert set(user) == {"question", "evidence_context"}
        assert case["token_count"] == case["generation"] == "not_performed"
        assert case["semantic_review"] == "pending"
        assert "approval" not in case


def test_invalid_hash_or_evidence_binding_fails_before_cases(bundle, monkeypatch):
    bad = dict(bundle, artifact_bytes=bundle["artifact_bytes"] + b" ")
    report = run(bad, output="hash.json")
    assert report["status"] == "failed" and report["cases"][0]["status"] == "not_attempted"
    artifact = json.loads(bundle["artifact_bytes"])
    artifact["cases"][0]["mandatory_element_evidence"][0]["supporting_excerpts"][0]["exact_quote"] = "not present"
    content = json.dumps(artifact).encode()
    monkeypatch.setattr(preparation, "ARTIFACT_SHA256", hashlib.sha256(content).hexdigest())
    bad = dict(bundle, artifact_bytes=content)
    report = run(bad, output="binding.json")
    assert report["status"] == "failed" and report["failure"]["stage"] == "inputs"


def test_changed_question_fails_after_preserving_completed_case(bundle):
    supplied = {"P5": supplied_result(bundle, "P5")}
    supplied["P5"]["question"] += " changed"
    report = run(bundle, ("P1", "P5", "P6"),
                 {**supplied, "P6": supplied_result(bundle, "P6")})
    assert report["status"] == "failed"
    assert [x["status"] for x in report["cases"]] == ["completed", "failed", "not_attempted"]
    assert report["failure"]["case_id"] == "P5"
    assert json.loads((bundle["root"] / "prepared.json").read_text(encoding="utf-8")) == report


def test_local_empty_selection_is_not_a_provider_request(bundle):
    artifact = json.loads(bundle["artifact_bytes"])
    empty = select_evidence(chunk_set=bundle["chunk_set"], fused_candidates=(), max_chunks=9)
    value = {"question": artifact["cases"][4]["question"], "empty_selection": empty}
    report = run(bundle, ("P5",), {"P5": value})
    case = report["cases"][0]
    assert report["status"] == "completed" and case["request"] is None
    assert case["local_result"]["status"] == "insufficient_evidence"
    assert case["selected"] == [] and case["rankings"] is None
    value = {"question": artifact["cases"][4]["question"], "fused_candidates": ()}
    failed = run(bundle, ("P5",), {"P5": value}, "empty-hybrid.json")
    assert failed["status"] == "failed" and failed["cases"][0]["status"] == "failed"


def test_wrong_fused_score_and_output_collision_stop(bundle):
    supplied = supplied_result(bundle, "P5")
    from dataclasses import replace
    fused = list(supplied["fused_candidates"])
    fused[0] = replace(fused[0], rrf_score=fused[0].rrf_score + 0.01)
    supplied["fused_candidates"] = tuple(fused)
    report = run(bundle, ("P5",), {"P5": supplied})
    assert report["status"] == "failed"
    with pytest.raises(FileExistsError):
        run(bundle)
    assert json.loads((bundle["root"] / "prepared.json").read_text(encoding="utf-8")) == report


def test_comparison_mapping_mismatch_fails(bundle, monkeypatch):
    data = json.loads(bundle["comparison_bytes"])
    data["groups"]["prospective"]["cases"][0]["options"]["prefix_9"]["selected"][0]["citation_id"] = "99"
    content = json.dumps(data).encode()
    monkeypatch.setattr(preparation, "COMPARISON_SHA256", hashlib.sha256(content).hexdigest())
    report = run(dict(bundle, comparison_bytes=content))
    assert report["status"] == "failed" and report["failure"]["case_id"] == "P1"
