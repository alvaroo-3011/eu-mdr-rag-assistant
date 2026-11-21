"""Synthetic checks for reviewed evidence recovery and runner orchestration."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.evaluation.retrieval_seed import (
    EvaluationError, assess_ranking, evaluate_cases, load_seed, parse_seed,
    validate_seed_against_chunks, validate_seed_structure,
)
from src.ingestion.chunk_models import ChunkTableSegment
from src.ingestion.models import HierarchyEntry, LegalLocatorPart, PageIdentity, RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.fusion import fuse_candidates
from src.retrieval.lexical_query import LexicalCandidate
from tests.unit.ingestion.test_chunk_models import (
    _chunk, _chunk_set, _chunk_set_id, _guidance_source, _manifest, _selection, _table,
)


ROOT = Path(__file__).resolve().parents[3]


def fixture():
    """Three real typed synthetic chunks, including dual permission and table."""
    source = replace(_guidance_source(),
                     regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
                     permitted_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR))
    manifest = _manifest(source)
    identity = _chunk_set_id(manifest)
    chunks = []
    for order in range(3):
        selection = _selection(source, f"block-{order}", block_order=order)
        chunks.append(_chunk(
            identity, source, order=order, text=f"Synthetic evidence {order}.",
            block_selections=(selection,),
            hierarchies=((HierarchyEntry(1, "section", selection.block_id, "Scope"),),),
            pages=(PageIdentity(0, 1),),
            legal_locators=((LegalLocatorPart("section", "1"),),),
            table_segments=(ChunkTableSegment(selection.block_id, _table((("X", "Y"),)),
                                             0, 1, True),) if order == 0 else (),
        ))
    chunks = tuple(replace(c, previous_chunk_id=chunks[i-1].chunk_id if i else None,
                           next_chunk_id=chunks[i+1].chunk_id if i+1 < len(chunks) else None)
                   for i, c in enumerate(chunks))
    chunk_set = _chunk_set(manifest, chunks)
    evidence = []
    for i, chunk in enumerate(chunks):
        evidence.append({
            "evidence_id": f"E{i}", "chunk_id": chunk.chunk_id,
            "source": {"document_id": source.document_id, "title": source.official_title,
                       "version": source.version, "source_version_identity": source.source_identity,
                       "sha256": source.sha256, "path": source.local_path,
                       "regulatory_regimes": ["MDR", "IVDR"],
                       "permitted_regimes": ["MDR", "IVDR"]},
            "hierarchy": [{"kind": "section", "title": "Scope"}],
            "legal_locators": [[{"kind": "section", "value": "1"}]],
            "physical_pdf_pages": [1], "source_block_ids": [f"block-{i}"],
            "exact_excerpts": [{"excerpt_id": f"X{i}", "text": chunk.text}],
            "interpretation_limitations": [],
        })
    evidence[0]["structured_table"] = {
        "block_id": chunks[0].block_selections[0].block_id, "physical_pdf_page": 1,
        "headers": ["Device", "Class"], "rows": [["X", "Y"]],
        "source_row_start": 0, "source_row_end": 1, "is_complete_table": True,
    }
    case = {
        "case_id": "C1", "question": "  Synthetic question?\n", "topic_id": "test-topic",
        "coverage_tags": ["multi-chunk"],
        "required_answer_elements": [{"element_id": f"A{i}", "text": f"Need {i}"}
                                     for i in range(3)],
        "supporting_judgments": [{"evidence_id": f"E{i}", "label": "supporting",
                                  "support_extent": "partial"} for i in range(3)],
        "element_to_evidence": {f"A{i}": [f"E{i}"] for i in range(3)},
        "support_combinations": [{"combination_id": "S1", "members": ["E0", "E1", "E2"],
                                   "members_jointly_required": True, "approved": True}],
        "review_status": "approved-for-seed-v1",
    }
    seed = {
        "format_version": "retrieval-evaluation-seed/v1", "artifact_id": "synthetic",
        "frozen_corpus": {"manifest_path": "data/corpus_manifest.json", "source_count": 1,
                          "chunk_count": 3, "corpus_identity": chunk_set.corpus_identity,
                          "chunk_set_identity": identity},
        "review_provenance": {k: "synthetic" for k in (
            "approved_by", "approved_in", "approved_on", "basis", "scope")},
        "judgment_policy": {"judgments_are_question_specific": True,
                            "approved_labels_in_this_seed": ["supporting"],
                            "all_other_chunks": "unjudged", "approved_alternative_combinations": 0},
        "evidence": evidence, "cases": [case],
    }
    return seed, chunk_set


def synthetic_union(chunk_set, dense_order=(2, 0), lexical_order=(1, 0)):
    dense, lexical = [], []
    for target, order, cls in ((dense, dense_order, DenseCandidate),
                               (lexical, lexical_order, LexicalCandidate)):
        for rank, index in enumerate(order, 1):
            chunk = chunk_set.chunks[index]
            metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
            target.append(cls(rank, chunk.chunk_id, chunk.text, metadata, 0.25 * rank))
    return fuse_candidates(chunk_set=chunk_set, dense_candidates=tuple(dense),
                           lexical_candidates=tuple(lexical), rank_constant=2.0)


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.seed, self.chunk_set = fixture()
        # Runtime access to heavy components is forbidden throughout the tests.
        self.enterContext(patch.dict(sys.modules, {
            "chromadb": None, "torch": None, "transformers": None,
        }))

    def test_load_committed_seed_structure_without_corpus(self):
        seed = load_seed(ROOT / "data/evaluation/retrieval_seed_v1.json")
        self.assertEqual(len(seed["cases"]), 6)
        self.assertEqual(len(seed["evidence"]), 7)
        self.assertEqual(len({c["topic_id"] for c in seed["cases"]}), 5)

    def test_full_binding_accepts_dual_permitted_source_and_table(self):
        validate_seed_against_chunks(self.seed, self.chunk_set)

    def test_malformed_json_duplicate_keys_constants_and_format(self):
        for content in (b"{", b'{"a":1,"a":2}', b'{"a":NaN}',
                        b'{"a":Infinity}', b'[]', b'\xff'):
            with self.subTest(content=content), self.assertRaises(EvaluationError):
                parse_seed(content)
        self.seed["format_version"] = "future"
        with self.assertRaises(EvaluationError):
            validate_seed_structure(self.seed)

    def test_damaged_references_and_unapproved_judgments_fail(self):
        changes = [
            lambda s: s["cases"][0]["element_to_evidence"].update(A0=["unknown"]),
            lambda s: s["cases"][0]["support_combinations"][0].update(members=["unknown"]),
            lambda s: s["cases"][0]["supporting_judgments"][0].update(label="irrelevant"),
            lambda s: s["cases"][0].update(review_status="draft"),
            lambda s: s["cases"][0]["support_combinations"][0].update(approved=False),
            lambda s: s["judgment_policy"].update(all_other_chunks="irrelevant"),
            lambda s: s["evidence"].append(deepcopy(s["evidence"][0])),
            lambda s: s["frozen_corpus"].update(source_count=True),
            lambda s: s["cases"][0]["element_to_evidence"].pop("A1"),
            lambda s: s["cases"][0]["support_combinations"][0].update(members=["E0"]),
        ]
        for i, change in enumerate(changes):
            seed = deepcopy(self.seed)
            change(seed)
            with self.subTest(change=i), self.assertRaises(EvaluationError):
                validate_seed_structure(seed)

    def test_original_evidence_mismatch_fails(self):
        changes = [
            lambda s: s["frozen_corpus"].update(chunk_set_identity="chunk-set:sha256:" + "0"*64),
            lambda s: s["frozen_corpus"].update(chunk_count=99),
            lambda s: s["evidence"][0].update(chunk_id="chunk:sha256:" + "0"*64),
            lambda s: s["evidence"][0]["source"].update(version="wrong"),
            lambda s: s["evidence"][0].update(physical_pdf_pages=[2]),
            lambda s: s["evidence"][0].update(physical_pdf_pages=[True]),
            lambda s: s["evidence"][0].update(hierarchy=[]),
            lambda s: s["evidence"][0].update(source_block_ids=["invented"]),
            lambda s: s["evidence"][0]["exact_excerpts"][0].update(text="invented"),
            lambda s: s["evidence"][0]["structured_table"].update(rows=[["wrong", "Y"]]),
            lambda s: s["evidence"][0]["structured_table"].update(physical_pdf_page=True),
            lambda s: s["evidence"][0]["structured_table"].update(source_row_start=False),
        ]
        for i, change in enumerate(changes):
            seed = deepcopy(self.seed)
            change(seed)
            with self.subTest(change=i), self.assertRaises(EvaluationError):
                validate_seed_against_chunks(seed, self.chunk_set)

    def test_partial_complete_and_cutoff_boundary(self):
        ids = [c.chunk_id for c in self.chunk_set.chunks]
        for cutoff, found, complete in ((1, 1, False), (2, 2, False), (3, 3, True), (5, 3, True)):
            with self.subTest(cutoff=cutoff):
                result = assess_ranking(self.seed["cases"][0], self.seed["evidence"], ids,
                                        cutoff=cutoff)
                self.assertEqual(result["approved_evidence_found"], found)
                self.assertEqual(result["complete_approved_combination_found"], complete)
                self.assertEqual(result["examined_count"], min(cutoff, 3))

    def test_empty_and_unjudged_results_are_not_errors_or_irrelevance_labels(self):
        for ids in ([], ["unjudged-result"]):
            result = assess_ranking(self.seed["cases"][0], self.seed["evidence"], ids, cutoff=5)
            self.assertEqual(result["approved_evidence_found"], 0)
            self.assertFalse(result["any_approved_support_found"])
            self.assertFalse(result["complete_approved_combination_found"])
            self.assertNotIn("precision", result)

    def test_invalid_cutoff_and_duplicate_ranking_rejected(self):
        for cutoff in (True, 0, -1, 1.0):
            with self.subTest(cutoff=cutoff), self.assertRaises(EvaluationError):
                assess_ranking(self.seed["cases"][0], self.seed["evidence"], [], cutoff=cutoff)
        with self.assertRaises(EvaluationError):
            assess_ranking(self.seed["cases"][0], self.seed["evidence"], ["x", "x"], cutoff=5)

    def test_alternative_combination_or_semantics(self):
        case = self.seed["cases"][0]
        case["support_combinations"] = [
            {"combination_id": "S1", "members": ["E0", "E1"]},
            {"combination_id": "S2", "members": ["E2"]},
        ]
        result = assess_ranking(case, self.seed["evidence"],
                                [self.chunk_set.chunks[2].chunk_id], cutoff=5)
        self.assertTrue(result["complete_approved_combination_found"])
        self.assertEqual(result["completed_combination_ids"], ["S2"])
        self.assertEqual(result["approved_evidence_found"], 1)

    def test_one_call_preserves_query_and_reconstructs_branch_ranks(self):
        union = synthetic_union(self.chunk_set)
        retrieve = Mock(return_value=union)
        result = evaluate_cases(self.seed, retrieve=retrieve, cutoff=2)[0]
        retrieve.assert_called_once_with("  Synthetic question?\n")
        self.assertEqual(result["status"], "ok")
        ids = [c.chunk_id for c in self.chunk_set.chunks]
        self.assertEqual([c["chunk_id"] for c in result["rankings"]["dense"]], [ids[2], ids[0]])
        self.assertEqual([c["chunk_id"] for c in result["rankings"]["lexical"]], [ids[1], ids[0]])
        self.assertEqual(len(result["rankings"]["fused"]), 3)  # union not truncated
        self.assertEqual(result["rankings"]["dense"][1]["cosine_distance"], 0.5)
        self.assertEqual(result["rankings"]["lexical"][1]["bm25_score"], 0.5)
        self.assertFalse(result["assessment"]["fused"]["complete_approved_combination_found"])

    def test_empty_lexical_branch_survives_provenance_reconstruction(self):
        union = synthetic_union(self.chunk_set, lexical_order=())
        result = evaluate_cases(self.seed, retrieve=lambda q: union, cutoff=5)[0]
        self.assertEqual(result["rankings"]["lexical"], [])
        self.assertEqual(result["assessment"]["lexical"]["approved_evidence_found"], 0)

    def test_real_hybrid_orchestration_with_fake_branches_calls_each_once(self):
        from src.retrieval.hybrid_query import retrieve_hybrid

        union = synthetic_union(self.chunk_set)
        dense = Mock()
        dense.retrieve.return_value = tuple(sorted(
            (c.dense_candidate for c in union if c.dense_candidate), key=lambda c: c.rank))
        lexical = Mock(chunk_set_id=self.chunk_set.chunk_set_id)
        lexical.retrieve.return_value = tuple(sorted(
            (c.lexical_candidate for c in union if c.lexical_candidate), key=lambda c: c.rank))

        def retrieve(query):
            return retrieve_hybrid(query, chunk_set=self.chunk_set,
                dense_retriever=dense, lexical_retriever=lexical,
                dense_candidate_count=5, lexical_candidate_count=5, rank_constant=2.0)

        result = evaluate_cases(self.seed, retrieve=retrieve, cutoff=5)[0]
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["assessment"]["fused"]["complete_approved_combination_found"])
        dense.retrieve.assert_called_once_with(self.seed["cases"][0]["question"], candidate_count=5)
        lexical.retrieve.assert_called_once_with(self.seed["cases"][0]["question"], candidate_count=5)

    def test_errors_are_unscored_next_case_continues_no_retry(self):
        second = deepcopy(self.seed["cases"][0])
        second.update(case_id="C2", question="Second question")
        second["support_combinations"][0]["combination_id"] = "S2"
        self.seed["cases"].append(second)
        error = ValueError("dense stage")
        error.__cause__ = RuntimeError("synthetic backend failure")
        retrieve = Mock(side_effect=[error, synthetic_union(self.chunk_set)])
        results = evaluate_cases(self.seed, retrieve=retrieve, cutoff=5)
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(results[0]["status"], "error")
        self.assertNotIn("assessment", results[0])
        self.assertNotIn("rankings", results[0])
        self.assertEqual(len(results[0]["errors"]), 2)
        self.assertEqual(results[1]["status"], "ok")

    def test_baseexception_propagates_and_malformed_result_is_unscored(self):
        with self.assertRaises(KeyboardInterrupt):
            evaluate_cases(self.seed, retrieve=Mock(side_effect=KeyboardInterrupt), cutoff=5)
        result = evaluate_cases(self.seed, retrieve=lambda q: (), cutoff=5)[0]
        self.assertEqual(result["error_stage"], "assessment")
        self.assertNotIn("assessment", result)


def runner_module():
    spec = importlib.util.spec_from_file_location("evaluation_runner", ROOT / "scripts/evaluate_retrieval_seed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.seed, self.chunk_set = fixture()
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        (self.root / "data/indexes/chroma").mkdir(parents=True)
        (self.root / "seed.json").write_text(json.dumps(self.seed), encoding="utf-8")
        self.runner = runner_module()
        self.args = argparse.Namespace(project_root=self.root, seed=Path("seed.json"),
            output=self.root / "run.json", dense_candidates=5, lexical_candidates=5,
            k1=1.2, b=0.75, rank_constant=2.0, cutoff=5)
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.dict(os.environ))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")))
        self.enterContext(patch.dict(sys.modules, {"torch": None, "transformers": None}))

    def backends(self):
        self.enterContext(patch("src.ingestion.manifest.load_manifest", return_value=self.chunk_set.manifest))
        self.enterContext(patch("src.ingestion.corpus_loader.load_corpus", return_value=()))
        self.enterContext(patch("src.ingestion.chunk_set_builder.build_chunk_set", return_value=self.chunk_set))
        self.enterContext(patch.object(self.runner, "_git_state", return_value={"head": "synthetic", "status": "", "diff": ""}))
        self.client = Mock()
        self.enterContext(patch.dict(sys.modules, {"chromadb": SimpleNamespace(PersistentClient=self.client)}))
        self.dense = self.enterContext(patch("src.retrieval.dense_query.DenseRetriever"))
        self.embedder = self.enterContext(patch("src.retrieval.embeddings.TransformerEmbedder"))
        self.hybrid = self.enterContext(patch("src.retrieval.hybrid_query.retrieve_hybrid",
                                              return_value=synthetic_union(self.chunk_set)))

    def test_success_writes_complete_report_with_explicit_settings(self):
        self.backends()
        self.assertEqual(self.runner.run_command(self.args), 0)
        report = json.loads(self.args.output.read_text())
        self.assertEqual(report["parameters"]["bm25"], {"k1": 1.2, "b": 0.75})
        self.assertEqual(report["parameters"]["cutoff"], 5)
        self.assertEqual(report["seed"]["sha256"], hashlib.sha256((self.root / "seed.json").read_bytes()).hexdigest())
        self.assertEqual(report["hybrid_calls"], 1)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.hybrid.assert_called_once()
        self.client.assert_called_once_with(path=str(self.root / "data/indexes/chroma"))
        self.assertEqual(self.hybrid.call_args.args[0], self.seed["cases"][0]["question"])

    def test_seed_mismatch_stops_before_backend_construction(self):
        self.backends()
        self.seed["frozen_corpus"]["chunk_count"] = 99
        (self.root / "seed.json").write_text(json.dumps(self.seed))
        with self.assertRaises(EvaluationError):
            self.runner.run_command(self.args)
        self.client.assert_not_called()
        self.dense.assert_not_called()
        self.hybrid.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_failed_case_is_persisted_and_returns_nonzero(self):
        self.backends()
        self.hybrid.side_effect = ValueError("synthetic retrieval failure")
        self.assertEqual(self.runner.run_command(self.args), 1)
        report = json.loads(self.args.output.read_text())
        self.assertEqual(report["status"], "completed-with-errors")
        self.assertNotIn("assessment", report["cases"][0])
        self.hybrid.assert_called_once()

    def test_existing_output_and_missing_index_are_not_overwritten_or_created(self):
        self.args.output.write_text("previous experiment")
        with self.assertRaisesRegex(EvaluationError, "already exists"):
            self.runner.run_command(self.args)
        self.assertEqual(self.args.output.read_text(), "previous experiment")
        self.args.output.unlink()
        (self.root / "data/indexes/chroma").rmdir()
        with self.assertRaisesRegex(EvaluationError, "Existing"):
            self.runner.run_command(self.args)
        self.assertFalse((self.root / "data/indexes/chroma").exists())

    def test_help_and_missing_arguments_do_not_execute(self):
        with patch.object(self.runner, "run_command") as execute:
            with self.assertRaises(SystemExit) as caught:
                self.runner.main(["--help"])
            self.assertEqual(caught.exception.code, 0)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.runner.main([])
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
