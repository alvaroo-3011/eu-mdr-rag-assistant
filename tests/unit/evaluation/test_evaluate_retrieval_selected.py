"""Offline tests for the selected-evidence experiment runner."""

from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_retrieval_selected as runner
from src.evaluation.retrieval_seed import ranking_records, evaluate_cases
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
from src.retrieval.embeddings import GTE_MODERNBERT_BASE
from src.retrieval.evidence_selection import select_evidence
from src.retrieval.hybrid_query import retrieve_hybrid
from src.retrieval.lexical_query import BM25Config
from tests.unit.evaluation.test_retrieval_seed import fixture, synthetic_union
from tests.unit.ingestion.test_chunk_models import _chunk, _chunk_set, _selection


class SelectedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.index = self.root / "index"
        self.index.mkdir()
        (self.index / "chroma.sqlite3").write_bytes(b"synthetic only")
        (self.index / "segment").mkdir()
        (self.index / "segment" / "data").write_bytes(b"not an index")
        self.output = self.root / "report.json"
        self.seed, self.chunks = fixture()
        first = self.seed["cases"][0]
        self.seed["cases"] = []
        for number in range(1, 7):
            case = deepcopy(first)
            case["case_id"] = f"C{number}"
            case["question"] = f"Synthetic question {number}?"
            case["support_combinations"][0]["combination_id"] = f"S{number}"
            self.seed["cases"].append(case)
        self.seed["artifact_id"] = "mdr-retrieval-seed-v1"
        self.seed_path = self.root / "seed.json"
        self.seed_path.write_text(json.dumps(self.seed), encoding="utf-8")
        self.union = synthetic_union(self.chunks)
        self.dense = Mock()
        self.dense.retrieve.return_value = tuple(sorted(
            (x.dense_candidate for x in self.union if x.dense_candidate), key=lambda x: x.rank,
        ))
        self.lexical = Mock(chunk_set_id=self.chunks.chunk_set_id, index_id="synthetic-lexical")
        self.lexical.retrieve.return_value = tuple(sorted(
            (x.lexical_candidate for x in self.union if x.lexical_candidate), key=lambda x: x.rank,
        ))
        self.client = Mock()
        self.open_client = Mock(return_value=self.client)
        self.initialize = Mock(return_value=(self.chunks, self.dense, self.lexical))
        self.collection_name = Mock(return_value="synthetic-collection")
        self.select = Mock(wraps=select_evidence)
        self.returned_unions = []

        def track_hybrid(*args, **kwargs):
            union = retrieve_hybrid(*args, **kwargs)
            self.returned_unions.append(union)
            return union

        self.hybrid = Mock(side_effect=track_hybrid)
        self.dependencies = (self.open_client, self.initialize, BASELINE_CHUNKING_CONFIG,
            GTE_MODERNBERT_BASE, self.collection_name, self.select, self.hybrid, BM25Config)
        self.provenance = Mock(return_value={"git_head": "synthetic-head", "git_status": "",
            "code_sha256": {}, "package_versions": {"chromadb": "1.5.9"}})
        self.temp_privacy = self.enterContext(
            patch.object(runner, "_require_private_windows_temp_parent")
        )
        self.enterContext(patch.multiple(runner,
            SEED_HASH=hashlib.sha256(self.seed_path.read_bytes()).hexdigest(),
            SOURCE_COUNT=1, CHUNK_COUNT=3,
            CORPUS_ID=self.chunks.corpus_identity, CHUNK_SET_ID=self.chunks.chunk_set_id,
            CHUNKING_ID=self.chunks.chunking_config_id, COLLECTION="synthetic-collection",
            LEXICAL_ID="synthetic-lexical"))
        # No real model, Chroma, network, or credential boundary can be used.
        self.enterContext(patch.dict("sys.modules", {"chromadb": None, "torch": None,
                                                      "transformers": None}))
        self.enterContext(patch.object(runner.os, "environ", {}))

    def run_case(self, *, dense_candidate_count=5, lexical_candidate_count=5):
        return runner.run_experiment(project_root=self.root, seed_path=self.seed_path,
            original_index=self.index, temp_parent=self.root, output_path=self.output,
            acknowledge_writers_stopped=True, dense_candidate_count=dense_candidate_count,
            lexical_candidate_count=lexical_candidate_count, dependencies=self.dependencies,
            provenance=self.provenance)

    def test_success_forwards_parameters_and_assesses_actual_selected_order(self):
        result = self.run_case()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), result)
        self.assertEqual(result["provenance"]["git_head"], "synthetic-head")
        self.assertEqual(result["configuration"]["dense_candidate_count"], 5)
        self.assertEqual(result["configuration"]["lexical_candidate_count"], 5)
        self.assertEqual([r["status"] for r in result["cases"]], ["ok"] * 6)
        self.assertEqual(self.open_client.call_count, 1)
        self.assertEqual(self.initialize.call_count, 1)
        self.assertEqual(self.hybrid.call_count, 6)
        self.assertEqual(self.dense.retrieve.call_count, 6)
        self.assertEqual(self.lexical.retrieve.call_count, 6)
        self.assertEqual(self.select.call_count, 6)
        self.assertEqual(self.open_client.call_args.kwargs["tenant"], "default_tenant")
        self.assertEqual(self.open_client.call_args.kwargs["database"], "default_database")
        self.assertEqual(self.open_client.call_args.kwargs["migration_hash_algorithm"], "md5")
        self.assertEqual(self.initialize.call_args.kwargs["bm25_config"], BM25Config(k1=1.2, b=0.75))
        self.assertIs(self.initialize.call_args.kwargs["dense_client"], self.client)
        self.assertEqual(self.hybrid.call_args.kwargs["dense_candidate_count"], 5)
        self.assertEqual(self.hybrid.call_args.kwargs["lexical_candidate_count"], 5)
        self.assertEqual(self.hybrid.call_args.kwargs["rank_constant"], 2.0)
        self.assertEqual(self.select.call_args.kwargs["max_chunks"], 5)
        for call, union in zip(self.select.call_args_list, self.returned_unions):
            self.assertIs(call.kwargs["fused_candidates"], union)
        case = result["cases"][0]
        self.assertEqual([r["chunk_id"] for r in case["rankings"]["selected"]],
                         [c.chunk_id for c in self.union])
        self.assertEqual(case["rankings"]["dense"][1]["cosine_distance"], 0.5)
        self.assertEqual(case["rankings"]["lexical"][1]["bm25_score"], 0.5)
        self.assertTrue(case["assessment"]["selected"]["complete_approved_combination_found"])
        self.assertTrue(case["assessment"]["fused_union"]["complete_approved_combination_found"])
        self.assertEqual([p["evidence_id"] for p in case["approved_evidence_positions"]],
                         ["E0", "E1", "E2"])
        for position in case["approved_evidence_positions"]:
            selected = next(row for row in case["rankings"]["selected"]
                            if row["chunk_id"] == position["chunk_id"])
            self.assertEqual(position["ranks"]["selected"], selected["rank"])
        self.assertEqual(result["operations"]["dense"], {"attempted": 6, "completed": 6})
        self.assertEqual(result["provider_calls"], 0)
        self.assertTrue(result["cleanup"]["original_fingerprint_match"])
        self.assertEqual(result["cleanup"]["disposable_copy_removal"], "passed")
        self.client.close.assert_called_once_with()

    def test_explicit_ten_counts_are_forwarded_and_reported_without_changing_limits(self):
        parameters_before = deepcopy(runner.PARAMETERS)
        result = self.run_case(dense_candidate_count=10, lexical_candidate_count=10)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["configuration"]["dense_candidate_count"], 10)
        self.assertEqual(result["configuration"]["lexical_candidate_count"], 10)
        self.assertEqual(result["configuration"]["assessment_cutoff"], 5)
        self.assertEqual(result["configuration"]["max_selected_chunks"], 5)
        self.assertEqual(result["cases"][0]["assessment"]["fused"]["cutoff"], 5)
        for call in self.hybrid.call_args_list:
            self.assertEqual(call.kwargs["dense_candidate_count"], 10)
            self.assertEqual(call.kwargs["lexical_candidate_count"], 10)
        for call in self.dense.retrieve.call_args_list + self.lexical.retrieve.call_args_list:
            self.assertEqual(call.kwargs["candidate_count"], 10)
        self.assertTrue(all(call.kwargs["max_chunks"] == 5 for call in self.select.call_args_list))
        self.assertEqual(runner.PARAMETERS, parameters_before)

    def test_invalid_counts_fail_before_report_or_resource_access(self):
        for field in ("dense_candidate_count", "lexical_candidate_count"):
            for value in (0, -1, True, False, 1.5, "5", None):
                with self.subTest(field=field, value=value):
                    with patch.object(Path, "open") as open_file, \
                         patch.object(runner, "fingerprint") as fingerprint:
                        with self.assertRaisesRegex(runner.ExperimentError, "Candidate count"):
                            self.run_case(**{field: value})
                        open_file.assert_not_called()
                        fingerprint.assert_not_called()
                    self.assertFalse(self.output.exists())
                    self.provenance.assert_not_called()
                    self.open_client.assert_not_called()
                    self.initialize.assert_not_called()

    def test_cli_maps_counts_and_rejects_missing_or_invalid_values(self):
        common = ["--project-root", str(self.root), "--seed-path", str(self.seed_path),
                  "--original-index", str(self.index), "--temp-parent", str(self.root),
                  "--output-path", str(self.output), "--acknowledge-writers-stopped"]
        with patch.object(runner, "run_experiment", return_value={"status": "completed"}) as run, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(common + ["--dense-candidates", "10",
                                                   "--lexical-candidates", "5"]), 0)
            self.assertEqual(run.call_args.kwargs["dense_candidate_count"], 10)
            self.assertEqual(run.call_args.kwargs["lexical_candidate_count"], 5)
            run.reset_mock()
            for options in ([], ["--dense-candidates", "5"],
                            ["--dense-candidates", "0", "--lexical-candidates", "5"],
                            ["--dense-candidates", "5", "--lexical-candidates", "-1"],
                            ["--dense-candidates", "abc", "--lexical-candidates", "5"]):
                with self.subTest(options=options), self.assertRaises(SystemExit) as caught:
                    runner.main(common + options)
                self.assertEqual(caught.exception.code, 2)
                run.assert_not_called()

    def test_incomplete_coverage_is_success_not_operational_failure(self):
        partial = synthetic_union(self.chunks, dense_order=(2,), lexical_order=())
        self.dense.retrieve.return_value = (partial[0].dense_candidate,)
        self.lexical.retrieve.return_value = ()
        result = self.run_case()
        self.assertEqual(result["status"], "completed")
        assessment = result["cases"][0]["assessment"]["selected"]
        self.assertEqual(assessment["approved_evidence_found"], 1)
        self.assertEqual(assessment["approved_evidence_total"], 3)
        self.assertFalse(assessment["complete_approved_combination_found"])

    def test_operational_failure_stops_and_preserves_primary_and_cleanup_errors(self):
        self.dense.retrieve.side_effect = [self.dense.retrieve.return_value,
                                           RuntimeError("private backend detail")]
        self.client.close.side_effect = RuntimeError("private cleanup detail")
        result = self.run_case()
        self.assertEqual(result["status"], "cleanup_failed")
        self.assertEqual([r["status"] for r in result["cases"]],
                         ["ok", "failed", "not_run", "not_run", "not_run", "not_run"])
        self.assertEqual(result["primary_failure"]["stage"], "retrieval")
        self.assertEqual(result["cleanup_failures"][0]["stage"], "client_close")
        self.assertEqual(result["operations"]["dense"], {"attempted": 2, "completed": 1})
        self.assertEqual(result["operations"]["lexical"], {"attempted": 1, "completed": 1})
        self.assertEqual(result["cleanup"]["disposable_copy_removal"], "passed")
        self.assertTrue(result["cleanup"]["original_fingerprint_match"])
        self.assertNotIn("private", self.output.read_text(encoding="utf-8"))

    def test_selection_failure_stops_without_another_retrieval(self):
        self.select.side_effect = RuntimeError("synthetic selection failure")
        result = self.run_case()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["primary_failure"]["stage"], "selection")
        self.assertEqual(result["operations"]["selection"], {"attempted": 1, "completed": 0})
        self.assertEqual(self.hybrid.call_count, 1)
        self.assertEqual(result["cases"][1]["status"], "not_run")
        self.client.close.assert_called_once_with()

    def test_existing_output_is_never_overwritten_or_followed_by_resources(self):
        self.output.write_text("existing", encoding="utf-8")
        with patch.object(runner, "fingerprint") as fingerprint:
            with self.assertRaisesRegex(runner.ExperimentError, "reserve"):
                self.run_case()
            fingerprint.assert_not_called()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing")
        self.open_client.assert_not_called()
        self.initialize.assert_not_called()

    def test_nonprivate_temp_fails_before_report_reservation_or_copy(self):
        self.temp_privacy.side_effect = runner.ExperimentError(
            "Temporary parent Windows permissions allow replacement."
        )
        with patch.object(runner.tempfile, "mkdtemp") as make_scratch, \
             patch.object(Path, "open") as open_file:
            with self.assertRaisesRegex(runner.ExperimentError, "permissions"):
                self.run_case()
        self.temp_privacy.assert_called_once_with(self.root.resolve())
        make_scratch.assert_not_called()
        open_file.assert_not_called()
        self.provenance.assert_not_called()
        self.open_client.assert_not_called()

    def test_windows_acl_policy_rejects_untrusted_mutation_and_wrong_owner(self):
        current = "S-1-5-21-1-2-3-1001"
        private = runner._WindowsDirectorySecurity(current, True, (
            runner._WindowsAccessGrant(current, 0x1F01FF, 0x03),
            runner._WindowsAccessGrant("S-1-5-32-545", 0x1200A9, 0x03),
        ))
        shared = runner._WindowsDirectorySecurity(current, True, (
            runner._WindowsAccessGrant("S-1-5-32-545", 0x1301BF, 0x03),
        ))
        wrong_owner = runner._WindowsDirectorySecurity("S-1-5-21-9", True, ())
        null_dacl = runner._WindowsDirectorySecurity(current, False, ())

        self.assertTrue(runner._windows_security_is_private(
            private, current, entry_parent=False))
        self.assertFalse(runner._windows_security_is_private(
            shared, current, entry_parent=False))
        self.assertFalse(runner._windows_security_is_private(
            wrong_owner, current, entry_parent=False))
        self.assertFalse(runner._windows_security_is_private(
            null_dacl, current, entry_parent=False))

    def test_windows_acl_policy_rejects_entry_parent_delete_child(self):
        current = "S-1-5-21-1-2-3-1001"
        unsafe_parent = runner._WindowsDirectorySecurity(current, True, (
            runner._WindowsAccessGrant("S-1-5-32-545", 0x00000040, 0),
        ))
        inherited_only = runner._WindowsDirectorySecurity(current, True, (
            runner._WindowsAccessGrant("S-1-5-32-545", 0x00000040, 0x0B),
        ))
        self.assertFalse(runner._windows_security_is_private(
            unsafe_parent, current, entry_parent=True))
        self.assertTrue(runner._windows_security_is_private(
            inherited_only, current, entry_parent=True))

    def test_cleanup_refuses_replaced_scratch_without_calling_rmtree(self):
        scratch = self.root / "mdr-selected-eval-replaced"
        scratch.mkdir()
        with patch.object(Path, "is_junction", autospec=True,
                          side_effect=lambda path: path == scratch), \
             patch.object(runner.shutil, "rmtree") as remove:
            with self.assertRaisesRegex(runner.ExperimentError, "unsafe"):
                runner._remove_disposable_directory(scratch, self.root)
        remove.assert_not_called()

    def test_initialization_failure_closes_client_and_verifies_original(self):
        self.initialize.side_effect = RuntimeError("synthetic initialization failure")
        result = self.run_case()
        self.assertEqual(result["primary_failure"]["stage"], "initialization")
        self.assertEqual(result["operations"]["initialization"],
                         {"attempted": 1, "completed": 0})
        self.assertEqual([c["status"] for c in result["cases"]], ["not_run"] * 6)
        self.hybrid.assert_not_called()
        self.client.close.assert_called_once_with()
        self.assertTrue(result["cleanup"]["original_fingerprint_match"])

    def test_original_change_after_copy_prevents_success_and_is_reported(self):
        def mutate_after_open(**_kwargs):
            (self.index / "segment" / "data").write_bytes(b"changed synthetic input")
            return self.client

        self.open_client.side_effect = mutate_after_open
        result = self.run_case()
        self.assertEqual(result["status"], "cleanup_failed")
        self.assertIsNone(result["primary_failure"])
        self.assertEqual(result["cleanup_failures"][0]["stage"], "original_fingerprint_after")
        self.assertFalse(result["cleanup"]["original_fingerprint_match"])
        self.client.close.assert_called_once_with()

    def test_disposable_cleanup_failure_is_recorded_separately(self):
        with patch.object(runner.shutil, "rmtree", side_effect=OSError("synthetic cleanup")):
            result = self.run_case()
        self.assertEqual(result["status"], "cleanup_failed")
        self.assertEqual(result["cleanup"]["disposable_copy_removal"], "failed")
        self.assertEqual(result["cleanup_failures"][0]["stage"], "disposable_copy_removal")
        self.assertTrue(result["cleanup"]["original_fingerprint_match"])

    def test_fingerprint_propagates_directory_enumeration_errors(self):
        original_walk = runner.os.walk

        def fail_on_segment(top, *, followlinks, onerror):
            for base, dirs, files in original_walk(top, followlinks=followlinks):
                if Path(base).name == "segment":
                    onerror(PermissionError("synthetic directory enumeration failure"))
                yield base, dirs, files

        with patch.object(runner.os, "walk", side_effect=fail_on_segment):
            with self.assertRaisesRegex(runner.ExperimentError, "enumerate") as caught:
                runner.fingerprint(self.index)
        self.assertIsInstance(caught.exception.__cause__, PermissionError)

    def test_fingerprint_rejects_mocked_junction_before_traversing(self):
        # Mock the Python 3.12 junction predicate; no actual junction is created.
        junction = self.index / "segment"
        original_walk = runner.os.walk
        visited = []

        def track_walk(*args, **kwargs):
            for base, dirs, files in original_walk(*args, **kwargs):
                visited.append(Path(base))
                yield base, dirs, files

        with patch.object(Path, "is_junction", autospec=True,
                          side_effect=lambda path: path == junction), \
             patch.object(runner.os, "walk", side_effect=track_walk):
            with self.assertRaisesRegex(runner.ExperimentError, "junction"):
                runner.fingerprint(self.index)
        self.assertNotIn(junction, visited)

    def test_mocked_index_root_junction_rejected_before_resource_open(self):
        with patch.object(Path, "is_junction", autospec=True,
                          side_effect=lambda path: path == self.index):
            with self.assertRaisesRegex(runner.ExperimentError, "junction"):
                self.run_case()
        self.open_client.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_python_socket_attempt_during_close_is_blocked_and_cleanup_continues(self):
        def attempt_connection():
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)

        self.client.close.side_effect = attempt_connection
        result = self.run_case()
        self.assertEqual(result["status"], "cleanup_failed")
        self.assertEqual(result["cleanup"]["client_close"], "failed")
        self.assertEqual(result["cleanup_failures"][0]["stage"], "client_close")
        self.assertEqual(result["cleanup_failures"][0]["error_type"], "ExperimentError")
        self.assertEqual(result["cleanup"]["disposable_copy_removal"], "passed")
        self.assertTrue(result["cleanup"]["original_fingerprint_match"])

    def test_full_union_and_selected_cutoff_distinguish_complete_combination(self):
        source = self.chunks.chunks[0].source
        chunks = list(self.chunks.chunks)
        for order in range(3, 6):
            chunks.append(_chunk(self.chunks.chunk_set_id, source, order=order,
                text=f"Additional unjudged synthetic evidence {order}.",
                block_selections=(_selection(source, f"extra-{order}", block_order=order),)))
        chunks = tuple(replace(chunk,
            previous_chunk_id=chunks[i - 1].chunk_id if i else None,
            next_chunk_id=chunks[i + 1].chunk_id if i + 1 < len(chunks) else None)
            for i, chunk in enumerate(chunks))
        self.chunks = _chunk_set(self.chunks.manifest, chunks)
        self.seed["frozen_corpus"]["chunk_count"] = 6
        self.seed_path.write_text(json.dumps(self.seed), encoding="utf-8")
        union = synthetic_union(self.chunks, dense_order=(0, 3, 4, 5, 1),
                                lexical_order=(3, 4, 5, 1, 2))
        self.assertEqual(len(union), 6)
        self.assertEqual(union[5].chunk_id, self.chunks.chunks[2].chunk_id)
        self.dense.retrieve.return_value = tuple(sorted(
            (item.dense_candidate for item in union if item.dense_candidate),
            key=lambda item: item.rank))
        self.lexical.retrieve.return_value = tuple(sorted(
            (item.lexical_candidate for item in union if item.lexical_candidate),
            key=lambda item: item.rank))
        self.initialize.return_value = (self.chunks, self.dense, self.lexical)
        with patch.multiple(runner, CHUNK_COUNT=6,
            SEED_HASH=hashlib.sha256(self.seed_path.read_bytes()).hexdigest()):
            result = self.run_case()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.hybrid.call_count, 6)
        self.assertEqual(self.dense.retrieve.call_count, 6)
        self.assertEqual(self.lexical.retrieve.call_count, 6)
        self.assertEqual(self.select.call_count, 6)
        for call, returned in zip(self.select.call_args_list, self.returned_unions):
            self.assertIs(call.kwargs["fused_candidates"], returned)
            self.assertEqual(len(returned), 6)
        case = result["cases"][0]
        self.assertEqual([row["chunk_id"] for row in case["rankings"]["selected"]],
                         [item.chunk_id for item in self.returned_unions[0][:5]])
        self.assertTrue(case["assessment"]["fused_union"]["complete_approved_combination_found"])
        self.assertFalse(case["assessment"]["selected"]["complete_approved_combination_found"])
        self.assertEqual(case["assessment"]["selected"]["approved_evidence_found"], 2)

    def test_report_write_failure_is_explicit_after_cleanup_and_original_check(self):
        existing = self.root / "existing.json"
        existing.write_text("preserve existing output", encoding="utf-8")
        real_open = Path.open
        original_fingerprint = runner.fingerprint
        fingerprints = []

        def track_fingerprint(path):
            fingerprints.append(path)
            return original_fingerprint(path)

        class FailingWriter:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()

            def write(self, _text):
                self_test.assertEqual(self_test.client.close.call_count, 1)
                self_test.assertFalse(any(self_test.root.glob("mdr-selected-eval-*")))
                self_test.assertEqual(fingerprints[-1], self_test.index)
                raise OSError("synthetic report write failure")

        self_test = self

        def open_with_failure(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            if path == self.output and args and args[0] == "x":
                return FailingWriter(handle)
            return handle

        with patch.object(Path, "open", autospec=True, side_effect=open_with_failure), \
             patch.object(runner, "fingerprint", side_effect=track_fingerprint):
            with self.assertRaisesRegex(runner.ExperimentError, "Report write failed") as caught:
                self.run_case()
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(existing.read_text(encoding="utf-8"), "preserve existing output")
        self.assertTrue(self.output.exists())
        self.assertEqual(self.output.stat().st_size, 0)

    def test_public_ranking_helper_matches_existing_evaluation_and_validates(self):
        records = ranking_records(self.union)
        historical = evaluate_cases(self.seed, retrieve=lambda _q: self.union, cutoff=5)
        self.assertEqual(records, historical[0]["rankings"])
        self.assertEqual(len(historical), 6)
        with self.assertRaises(ValueError):
            ranking_records(())


@unittest.skipUnless(os.name == "nt", "Windows ACL integration tests")
class WindowsTempPrivacyIntegrationTests(unittest.TestCase):
    def _set_private_acl(self, path: Path) -> None:
        current_sid = runner._windows_current_user_sid()
        command = ["icacls.exe", str(path), "/inheritance:r", "/grant:r",
                   f"*{current_sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F",
                   "*S-1-5-32-544:(OI)(CI)F"]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            self.fail("Unable to prepare synthetic private Windows ACL fixture.")

    def test_real_private_and_shared_synthetic_directories(self):
        with tempfile.TemporaryDirectory(prefix="mdr-private-acl-") as temporary:
            root = Path(temporary)
            self._set_private_acl(root)
            private = root / "private"
            private.mkdir()
            runner._require_private_windows_temp_parent(private)

            shared = root / "shared"
            shared.mkdir()
            completed = subprocess.run(
                ["icacls.exe", str(shared), "/grant",
                 "*S-1-5-32-545:(OI)(CI)M"],
                check=False, capture_output=True, text=True,
            )
            if completed.returncode:
                self.fail("Unable to prepare synthetic shared Windows ACL fixture.")
            with self.assertRaisesRegex(runner.ExperimentError, "permissions"):
                runner._require_private_windows_temp_parent(shared)


if __name__ == "__main__":
    unittest.main()
