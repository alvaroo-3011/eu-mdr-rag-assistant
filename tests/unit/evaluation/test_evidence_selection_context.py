"""Synthetic offline checks for the evidence-selection/context comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

from scripts import evaluate_evidence_selection_context as runner
from src.evaluation.prospective_cases import parse_prospective
from src.evaluation.retrieval_seed import parse_seed
from src.ingestion.chunk_models import ChunkTableSegment
from src.ingestion.models import HierarchyEntry, LegalLocatorPart, PageIdentity
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.fusion import fuse_candidates
from src.retrieval.lexical_query import LexicalCandidate
from tests.unit.evaluation.test_prospective_cases import synthetic as prospective_fixture
from tests.unit.evaluation.test_retrieval_seed import fixture as seed_fixture
from tests.unit.ingestion.test_chunk_models import (
    _chunk,
    _chunk_set,
    _selection,
    _table,
)


@pytest.fixture(autouse=True)
def no_heavy_modules():
    """Make accidental heavyweight/runtime imports fail during every test."""

    with patch.dict(
        sys.modules,
        {"chromadb": None, "torch": None, "transformers": None},
    ):
        yield


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _ten_chunk_set():
    _, base = seed_fixture()
    source = base.manifest.sources[0]
    identity = base.chunk_set_id
    chunks = []
    for order in range(10):
        selection = _selection(source, f"block-{order}", block_order=order)
        chunks.append(
            _chunk(
                identity,
                source,
                order=order,
                text=f"Café € synthetic evidence {order}.",
                block_selections=(selection,),
                hierarchies=(
                    (HierarchyEntry(1, "section", selection.block_id, "Scope"),),
                ),
                pages=(PageIdentity(0, 1),),
                legal_locators=((LegalLocatorPart("section", str(order + 1)),),),
                table_segments=(
                    (
                        ChunkTableSegment(
                            selection.block_id,
                            _table((("X", "Y"),)),
                            0,
                            1,
                            True,
                        ),
                    )
                    if order == 0
                    else ()
                ),
            )
        )
    linked = tuple(
        replace(
            chunk,
            previous_chunk_id=chunks[index - 1].chunk_id if index else None,
            next_chunk_id=(
                chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
            ),
        )
        for index, chunk in enumerate(chunks)
    )
    return _chunk_set(base.manifest, linked)


def _synthetic_seed(chunk_set) -> dict:
    seed, _ = seed_fixture()
    chunks = chunk_set.chunks
    seed["artifact_id"] = "synthetic-selection-context-seed"
    seed["frozen_corpus"].update(
        source_count=len(chunk_set.manifest.sources),
        chunk_count=len(chunks),
        corpus_identity=chunk_set.corpus_identity,
        chunk_set_identity=chunk_set.chunk_set_id,
    )
    for index, evidence in enumerate(seed["evidence"]):
        chunk = chunks[index]
        evidence.update(
            chunk_id=chunk.chunk_id,
            physical_pdf_pages=[1],
            source_block_ids=[f"block-{index}"],
        )
        evidence["hierarchy"] = [{"kind": "section", "title": "Scope"}]
        evidence["legal_locators"] = [
            [{"kind": "section", "value": str(index + 1)}]
        ]
        evidence["exact_excerpts"][0]["text"] = chunk.text
        if index == 0:
            evidence["structured_table"].update(
                block_id=chunk.block_selections[0].block_id,
                physical_pdf_page=1,
            )
    template = seed["cases"][0]
    seed["cases"] = []
    for number in range(1, 7):
        case = deepcopy(template)
        case.update(
            case_id=f"C{number}",
            question=f"Synthetic original question {number}?",
            topic_id=f"topic-{number}",
        )
        case["support_combinations"][0]["combination_id"] = f"S{number}"
        seed["cases"].append(case)
    return seed


def _synthetic_prospective(chunk_set) -> dict:
    artifact, _ = prospective_fixture()
    chunks = chunk_set.chunks
    artifact.update(
        artifact="prospective-cases/v1",
        corpus_identity=chunk_set.corpus_identity,
        chunk_set_identity=chunk_set.chunk_set_id,
        source_count=len(chunk_set.manifest.sources),
        chunk_count=len(chunks),
    )
    for case in artifact["cases"]:
        for index, element in enumerate(case.get("mandatory_element_evidence", [])):
            chunk = chunks[index % 2]
            for excerpt in element["supporting_excerpts"]:
                excerpt.update(
                    chunk_id=chunk.chunk_id,
                    source_identity=chunk.source.source_identity,
                    exact_quote=chunk.text,
                )
        if case["expected_behavior"] == "answered":
            nominated = [
                excerpt["chunk_id"]
                for element in case["mandatory_element_evidence"]
                for excerpt in element["supporting_excerpts"]
            ]
            case["accepted_substantive_complete_combinations"] = [
                list(dict.fromkeys(nominated))
            ]
    p2 = artifact["cases"][1]["alternative_adjudication"]["guidance_excerpt"]
    p2.update(
        chunk_id=chunks[2].chunk_id,
        source_identity=chunks[2].source.source_identity,
        exact_quote=chunks[2].text,
    )
    context = artifact["cases"][2]["supporting_context"]
    context["chunk_id"] = chunks[9].chunk_id
    for excerpt in context["evidence"]:
        excerpt.update(
            chunk_id=chunks[9].chunk_id,
            source_identity=chunks[9].source.source_identity,
            exact_quote=chunks[9].text,
        )
    qualification = artifact["cases"][3]["answer_review_qualification"]
    qualification = qualification["element"]["supporting_excerpts"][0]
    qualification.update(
        chunk_id=chunks[8].chunk_id,
        source_identity=chunks[8].source.source_identity,
        exact_quote=chunks[8].text,
    )
    scope = artifact["cases"][7]["scope_evidence"]
    scope.update(
        chunk_id=chunks[2].chunk_id,
        source_identity=chunks[2].source.source_identity,
        exact_quotes=[chunks[2].text],
    )
    return artifact


def _rankings(chunk_set) -> dict:
    dense_order = tuple(range(10))
    lexical_order = (1, 0, 2, 3, 4, 5, 6, 7, 8, 9)

    def branch(order, kind):
        result = []
        for rank, index in enumerate(order, 1):
            chunk = chunk_set.chunks[index]
            metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
            score = 0.05 * rank
            result.append(kind(rank, chunk.chunk_id, chunk.text, metadata, score))
        return tuple(result)

    dense = branch(dense_order, DenseCandidate)
    lexical = branch(lexical_order, LexicalCandidate)
    fused = fuse_candidates(
        chunk_set=chunk_set,
        dense_candidates=dense,
        lexical_candidates=lexical,
        rank_constant=2.0,
    )
    return {
        "dense": [
            {
                "rank": row.rank,
                "chunk_id": row.chunk_id,
                "cosine_distance": row.cosine_distance,
            }
            for row in dense
        ],
        "lexical": [
            {
                "rank": row.rank,
                "chunk_id": row.chunk_id,
                "bm25_score": row.bm25_score,
            }
            for row in lexical
        ],
        "fused": [
            {
                "rank": row.rank,
                "chunk_id": row.chunk_id,
                "rrf_score": row.rrf_score,
                "dense_rank": (
                    row.dense_candidate.rank if row.dense_candidate else None
                ),
                "lexical_rank": (
                    row.lexical_candidate.rank if row.lexical_candidate else None
                ),
            }
            for row in fused
        ],
        "selected": [
            {"rank": rank, "chunk_id": row.chunk_id, "fused_rank": rank}
            for rank, row in enumerate(fused[:5], 1)
        ],
    }


def _configuration() -> dict:
    return {
        "bm25": {"k1": 1.2, "b": 0.75},
        "rank_constant": 2.0,
        "assessment_cutoff": 5,
        "max_selected_chunks": 5,
        "tenant": "synthetic-tenant",
        "database": "synthetic-database",
        "migration_hash_algorithm": "sha256",
        "dense_candidate_count": 10,
        "lexical_candidate_count": 10,
        "project_root": "synthetic-root",
        "manifest_path": "synthetic-manifest.json",
        "original_index_path": "synthetic-index",
        "temp_parent": "synthetic-temp",
        "output_path": "synthetic-output.json",
        "chunking": {"profile": "synthetic"},
        "embedding": {"model": "synthetic"},
        "writers_stopped_acknowledged": True,
        "disposable_index_path": "synthetic-copy",
    }


def _identities(chunk_set) -> dict:
    return {
        "source_count": len(chunk_set.manifest.sources),
        "chunk_count": len(chunk_set.chunks),
        "corpus_identity": chunk_set.corpus_identity,
        "chunk_set_identity": chunk_set.chunk_set_id,
        "chunking_config_id": chunk_set.chunking_config_id,
        "dense_collection": "synthetic-dense",
        "lexical_index_id": "synthetic-lexical",
    }


def _reports(chunk_set, seed, artifact, paths, hashes):
    rankings = _rankings(chunk_set)
    identities = _identities(chunk_set)
    common = {
        "status": "completed",
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:01:00+00:00",
        "provenance": {"synthetic": True},
        "configuration": _configuration(),
        "expected_identities": identities,
        "observed_identities": identities,
        "operations": {"synthetic": True},
        "provider_calls": 0,
        "primary_failure": None,
        "cleanup_failures": [],
        "cleanup": {"synthetic": True},
        "interpretation": {"synthetic": True},
    }
    original_cases = []
    for case in seed["cases"]:
        original_cases.append(
            {
                "case_id": case["case_id"],
                "topic_id": case["topic_id"],
                "question": case["question"],
                "status": "ok",
                "rankings": deepcopy(rankings),
                "assessment": {"synthetic": True},
                "approved_evidence_positions": [],
                "failure": None,
            }
        )
    original = {
        "format_version": "retrieval-evidence-run/v2",
        **common,
        "seed": {
            "path": paths["original_judgments"],
            "sha256": hashes["original_judgments"],
            "artifact_id": seed["artifact_id"],
        },
        "cases": original_cases,
    }
    prospective_cases = []
    for case in artifact["cases"]:
        executed = case["id"] in {"P1", "P2", "P3", "P4"}
        prospective_cases.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "status": "ok" if executed else "not_executed",
                "rankings": deepcopy(rankings) if executed else None,
                "assessment": (
                    {"synthetic": True}
                    if executed
                    else {"applicability": "not_applicable"}
                ),
                "answer_behavior": "not_evaluated",
                "semantic_review": "not_evaluated",
                "failure": None,
            }
        )
    prospective = {
        "format_version": "prospective-retrieval-run/v1",
        **common,
        "artifact": {
            "path": paths["prospective_judgments"],
            "sha256": hashes["prospective_judgments"],
            "artifact_id": artifact["artifact"],
        },
        "cases": prospective_cases,
    }
    return original, prospective


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    chunk_set = _ten_chunk_set()
    seed = _synthetic_seed(chunk_set)
    artifact = _synthetic_prospective(chunk_set)
    names = {
        "original_report": "original-report.json",
        "prospective_report": "prospective-report.json",
        "original_judgments": "original-judgments.json",
        "prospective_judgments": "prospective-judgments.json",
    }
    _write_json(tmp_path / names["original_judgments"], seed)
    _write_json(tmp_path / names["prospective_judgments"], artifact)
    hashes = {
        name: _sha256(tmp_path / relative)
        for name, relative in names.items()
        if "judgments" in name
    }
    parsed_seed = parse_seed((tmp_path / names["original_judgments"]).read_bytes())
    parsed_artifact = parse_prospective(
        (tmp_path / names["prospective_judgments"]).read_bytes()
    )
    original, prospective = _reports(
        chunk_set, parsed_seed, parsed_artifact, names, hashes
    )
    _write_json(tmp_path / names["original_report"], original)
    _write_json(tmp_path / names["prospective_report"], prospective)
    hashes.update(
        {
            name: _sha256(tmp_path / relative)
            for name, relative in names.items()
            if "report" in name
        }
    )
    fixed = {name: (relative, hashes[name]) for name, relative in names.items()}
    monkeypatch.setattr(runner, "FIXED_INPUTS", fixed)
    loader = Mock(return_value=chunk_set)
    provenance = Mock(
        return_value={
            "git_head": "0" * 40,
            "git_status": "synthetic",
            "python_version": "synthetic",
            "platform": "synthetic",
            "code_sha256": {"synthetic.py": "1" * 64},
        }
    )
    return {
        "root": tmp_path,
        "chunk_set": chunk_set,
        "loader": loader,
        "provenance": provenance,
        "fixed": fixed,
        "names": names,
    }


def _run(bundle, name="result.json"):
    output = bundle["root"] / name
    report = runner.run_comparison(
        project_root=bundle["root"],
        output_path=output,
        chunk_set_loader=bundle["loader"],
        provenance=bundle["provenance"],
    )
    return report, output


def test_success_uses_real_prefix_selector_and_records_sizes(synthetic_run):
    report, output = _run(synthetic_run)

    assert report["status"] == "completed"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["execution"]["output_path"] == str(output.resolve())
    assert report["execution"]["output_must_be_absent"] is True
    synthetic_run["loader"].assert_called_once()
    assert report["comparison"]["prefixes"] == [5, 7, 9]
    assert report["comparison"]["judgments_influence_selection"] is False
    assert report["provider_token_count"] == "not_performed"
    assert report["gemini_request_preparation"] == "not_performed"
    assert report["answer_behavior"] == "not_evaluated"
    assert report["semantic_review"] == "not_evaluated"
    assert report["groups"]["original"]["case_ids"] == [
        "C1", "C2", "C3", "C4", "C5", "C6"
    ]
    assert report["groups"]["prospective"]["case_ids"] == [
        "P1", "P2", "P3", "P4"
    ]

    case = report["groups"]["original"]["cases"][0]
    previous = []
    for prefix in (5, 7, 9):
        option = case["options"][f"prefix_{prefix}"]
        assert option["status"] == "completed"
        selected = [row["chunk_id"] for row in option["selected"]]
        assert len(selected) == prefix
        assert selected[: len(previous)] == previous
        assert [row["citation_id"] for row in option["selected"]] == [
            str(index) for index in range(1, prefix + 1)
        ]
        previous = selected
    size = case["options"]["prefix_5"]["context_size"]
    assert size["raw_text_utf8_bytes"] > size["raw_text_unicode_characters"]
    assert size["serialized_context_utf8_bytes"] > size[
        "serialized_context_unicode_characters"
    ]
    assert size["serialization_expansion_utf8_bytes"] == (
        size["serialized_context_utf8_bytes"] - size["raw_text_utf8_bytes"]
    )
    assert size["table_segment_count"] == 1
    assert size["chunks_with_tables"] == 1
    assert size["annotation_count"] == 0

    aggregate = report["groups"]["original"]["aggregates"]["prefix_5"]
    assert report["groups"]["original"]["aggregates_status"] == "completed"
    assert aggregate["case_count_denominator"] == 6
    assert aggregate["nominated_evidence_occurrence_denominator"] == 18
    assert set(aggregate["context_size_statistics"]["raw_text_utf8_bytes"]) == {
        "minimum", "median", "maximum", "sum"
    }


def test_p3_context_and_p4_qualification_stay_separate_from_scoring(synthetic_run):
    report, _ = _run(synthetic_run)
    cases = {
        case["case_id"]: case
        for case in report["groups"]["prospective"]["cases"]
    }
    p3 = cases["P3"]["options"]["prefix_9"]
    p4 = cases["P4"]["options"]["prefix_9"]
    assert set(p3["separate_evidence"]) == {"supporting_context"}
    assert set(p4["separate_evidence"]) == {"answer_review_qualification"}
    assert p3["recovery"]["nominated_total"] == 2
    assert p4["recovery"]["nominated_total"] == 2
    assert p3["separate_evidence"]["supporting_context"]["chunk_id"] not in (
        p3["recovery"]["found_chunk_ids"] + p3["recovery"]["missing_chunk_ids"]
    )
    assert p4["separate_evidence"]["answer_review_qualification"][
        "chunk_id"
    ] not in (
        p4["recovery"]["found_chunk_ids"] + p4["recovery"]["missing_chunk_ids"]
    )
    # Judgments differ, but selection is rank-only and therefore identical.
    assert [row["chunk_id"] for row in p3["selected"]] == [
        row["chunk_id"] for row in p4["selected"]
    ]


def test_fixed_input_hash_mismatch_preserves_failure_report(synthetic_run):
    path = synthetic_run["root"] / synthetic_run["names"]["original_report"]
    path.write_bytes(path.read_bytes() + b" ")

    report, output = _run(synthetic_run)

    assert report["status"] == "failed"
    assert report["failure"]["stage"] == "fixed_inputs"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    synthetic_run["loader"].assert_not_called()


def test_structurally_valid_excerpt_mismatch_fails_binding_before_measurement(
    synthetic_run, monkeypatch
):
    judgment_path = synthetic_run["root"] / synthetic_run["names"][
        "original_judgments"
    ]
    judgment = json.loads(judgment_path.read_text(encoding="utf-8"))
    judgment["evidence"][0]["exact_excerpts"][0]["text"] = (
        "Structurally valid but absent excerpt."
    )
    _write_json(judgment_path, judgment)
    judgment_hash = _sha256(judgment_path)

    report_path = synthetic_run["root"] / synthetic_run["names"][
        "original_report"
    ]
    input_report = json.loads(report_path.read_text(encoding="utf-8"))
    input_report["seed"]["sha256"] = judgment_hash
    _write_json(report_path, input_report)

    fixed = deepcopy(synthetic_run["fixed"])
    fixed["original_judgments"] = (
        fixed["original_judgments"][0], judgment_hash,
    )
    fixed["original_report"] = (
        fixed["original_report"][0], _sha256(report_path),
    )
    monkeypatch.setattr(runner, "FIXED_INPUTS", fixed)
    context_builder = Mock(side_effect=AssertionError("context must not be built"))
    monkeypatch.setattr(runner, "build_evidence_context", context_builder)

    result, output = _run(synthetic_run)

    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "judgment_binding"
    assert result["failure"]["location"] is None
    context_builder.assert_not_called()
    for group in result["groups"].values():
        assert group["status"] == "not_attempted"
        assert group["aggregates"] is None
        assert group["aggregates_status"] == "not_computed_group_incomplete"
        assert all(case["status"] == "not_attempted" for case in group["cases"])
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize("damage", ["duplicate", "missing"])
def test_duplicate_or_missing_branch_candidate_fails_before_loading(
    synthetic_run, monkeypatch, damage
):
    report_path = synthetic_run["root"] / synthetic_run["names"]["original_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["cases"][0]["rankings"]["dense"]
    if damage == "duplicate":
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
    else:
        rows.pop()
    _write_json(report_path, report)
    fixed = deepcopy(synthetic_run["fixed"])
    fixed["original_report"] = (
        fixed["original_report"][0],
        _sha256(report_path),
    )
    monkeypatch.setattr(runner, "FIXED_INPUTS", fixed)

    result, _ = _run(synthetic_run)

    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "input_validation"
    synthetic_run["loader"].assert_not_called()


@pytest.mark.parametrize("damage", ["configuration", "identity"])
def test_configuration_and_identity_incompatibility_fail_before_loading(
    synthetic_run, monkeypatch, damage
):
    report_path = synthetic_run["root"] / synthetic_run["names"][
        "prospective_report"
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if damage == "configuration":
        report["configuration"]["rank_constant"] = 3.0
    else:
        report["expected_identities"]["dense_collection"] = "different"
        report["observed_identities"]["dense_collection"] = "different"
    _write_json(report_path, report)
    fixed = deepcopy(synthetic_run["fixed"])
    fixed["prospective_report"] = (
        fixed["prospective_report"][0],
        _sha256(report_path),
    )
    monkeypatch.setattr(runner, "FIXED_INPUTS", fixed)

    result, _ = _run(synthetic_run)

    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "input_validation"
    synthetic_run["loader"].assert_not_called()


@pytest.mark.parametrize("damage", ["score", "tie_order"])
def test_recomputed_fusion_detects_score_and_order_mismatch(
    synthetic_run, monkeypatch, damage
):
    report_path = synthetic_run["root"] / synthetic_run["names"]["original_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fused = report["cases"][0]["rankings"]["fused"]
    if damage == "score":
        fused[0]["rrf_score"] += 0.01
    else:
        # The first two records have equal RRF scores; ChunkSet order breaks
        # that tie. Reverse only those records while keeping the report's
        # internal fused-prefix bookkeeping self-consistent.
        fused[0], fused[1] = fused[1], fused[0]
        fused[0]["rank"], fused[1]["rank"] = 1, 2
        selected = report["cases"][0]["rankings"]["selected"]
        selected[0]["chunk_id"] = fused[0]["chunk_id"]
        selected[1]["chunk_id"] = fused[1]["chunk_id"]
    _write_json(report_path, report)
    fixed = deepcopy(synthetic_run["fixed"])
    fixed["original_report"] = (
        fixed["original_report"][0],
        _sha256(report_path),
    )
    monkeypatch.setattr(runner, "FIXED_INPUTS", fixed)

    result, _ = _run(synthetic_run)

    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "candidate_reconstruction_and_context"
    assert result["failure"]["location"] == {
        "group": "original",
        "case_id": "C1",
        "phase": "candidate_reconstruction",
        "prefix": None,
    }
    first_case = result["groups"]["original"]["cases"][0]
    assert first_case["candidate_reconstruction"]["status"] == "failed"
    assert all(
        option["status"] == "not_attempted"
        for option in first_case["options"].values()
    )
    synthetic_run["loader"].assert_called_once()


def test_context_failure_preserves_completed_work_and_failure_location(
    synthetic_run, monkeypatch
):
    real_builder = runner.build_evidence_context
    calls = 0

    def fail_on_prospective_prefix_seven(*, selection, chunk_set):
        nonlocal calls
        calls += 1
        if calls == 20:
            raise RuntimeError("synthetic context failure")
        return real_builder(selection=selection, chunk_set=chunk_set)

    monkeypatch.setattr(
        runner, "build_evidence_context", fail_on_prospective_prefix_seven
    )

    result, output = _run(synthetic_run)

    assert result["status"] == "failed"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    assert result["failure"]["stage"] == "candidate_reconstruction_and_context"
    assert result["failure"]["location"] == {
        "group": "prospective",
        "case_id": "P1",
        "phase": "prefix_measurement",
        "prefix": 7,
    }

    original = result["groups"]["original"]
    assert original["status"] == "completed"
    assert original["aggregates_status"] == "completed"
    assert original["aggregates"] is not None
    assert all(case["status"] == "completed" for case in original["cases"])

    prospective = result["groups"]["prospective"]
    assert prospective["status"] == "failed"
    assert prospective["aggregates"] is None
    assert prospective["aggregates_status"] == "not_computed_group_incomplete"
    p1 = prospective["cases"][0]
    assert p1["status"] == "failed"
    assert p1["candidate_reconstruction"]["status"] == "completed"
    assert p1["options"]["prefix_5"]["status"] == "completed"
    assert p1["options"]["prefix_7"] == {
        "status": "failed", "max_chunks": 7,
    }
    assert p1["options"]["prefix_9"] == {
        "status": "not_attempted", "max_chunks": 9,
    }
    assert all(
        case["status"] == "not_attempted"
        for case in prospective["cases"][1:]
    )


def test_output_collision_is_non_overwriting(synthetic_run):
    output = synthetic_run["root"] / "existing.json"
    output.write_text("keep me", encoding="utf-8")

    with pytest.raises(runner.ComparisonError):
        runner.run_comparison(
            project_root=synthetic_run["root"],
            output_path=output,
            chunk_set_loader=synthetic_run["loader"],
            provenance=synthetic_run["provenance"],
        )

    assert output.read_text(encoding="utf-8") == "keep me"
    synthetic_run["loader"].assert_not_called()


def test_chunk_reconstruction_failure_is_reported_without_success(synthetic_run):
    synthetic_run["loader"].side_effect = RuntimeError("synthetic loader failure")

    report, output = _run(synthetic_run)

    assert report["status"] == "failed"
    assert report["failure"]["stage"] == "chunk_set_reconstruction"
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert all(
        group["status"] == "not_attempted"
        for group in persisted["groups"].values()
    )
    assert all(
        group["aggregates"] is None
        for group in persisted["groups"].values()
    )
