"""Prospective mode through the existing guarded lifecycle, with fake resources."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from unittest.mock import patch

import pytest

from scripts import evaluate_retrieval_selected as runner
from tests.unit.evaluation import test_evaluate_retrieval_selected as helpers
from tests.unit.evaluation.test_prospective_cases import synthetic


@pytest.fixture
def environment():
    env = helpers.SelectedRunnerTests(methodName="runTest")
    env.setUp()
    try:
        artifact, chunks = synthetic()
        assert chunks == env.chunks
        env.seed_path.write_text(json.dumps(artifact), encoding="utf-8")
        with patch.object(runner, "PROSPECTIVE_HASH",
                          hashlib.sha256(env.seed_path.read_bytes()).hexdigest()):
            yield env, artifact
    finally:
        env.doCleanups()


def run(env):
    return runner.run_experiment(project_root=env.root, seed_path=env.seed_path,
        original_index=env.index, temp_parent=env.root, output_path=env.output,
        acknowledge_writers_stopped=True, dense_candidate_count=10, lexical_candidate_count=10,
        mode="prospective", dependencies=env.dependencies, provenance=env.provenance)


def test_prospective_mode_binds_then_executes_four_full_unions(environment):
    env, artifact = environment
    original = runner.validate_prospective_against_chunks
    checked = []

    def bind(data, chunks):
        assert env.hybrid.call_count == 0
        original(data, chunks)
        checked.append(True)

    with patch.object(runner, "validate_prospective_against_chunks", side_effect=bind):
        report = run(env)
    assert checked == [True]
    assert report["format_version"] == "prospective-retrieval-run/v1"
    assert report["status"] == "completed" and report["provider_calls"] == 0
    assert report["artifact"]["sha256"] == hashlib.sha256(env.seed_path.read_bytes()).hexdigest()
    assert report["provenance"]["git_head"] == "synthetic-head"
    assert report["observed_identities"] == report["expected_identities"]
    assert report["configuration"]["dense_candidate_count"] == 10
    assert report["configuration"]["lexical_candidate_count"] == 10
    assert report["configuration"]["assessment_cutoff"] == 5
    assert report["configuration"]["max_selected_chunks"] == 5
    assert env.hybrid.call_count == env.select.call_count == 4
    assert env.dense.retrieve.call_count == env.lexical.retrieve.call_count == 4
    assert [call.args[0] for call in env.hybrid.call_args_list] == [c["question"] for c in artifact["cases"][:4]]
    for call, fused in zip(env.select.call_args_list, env.returned_unions):
        assert call.kwargs["fused_candidates"] is fused
    assert [c["status"] for c in report["cases"]] == ["ok"] * 4 + ["not_executed"] * 4
    for case in report["cases"]:
        assert case["answer_behavior"] == case["semantic_review"] == "not_evaluated"
    for case in report["cases"][4:]:
        assert case["assessment"]["applicability"] == "not_applicable" and case["rankings"] is None
    for case in report["cases"][:4]:
        assert [r["chunk_id"] for r in case["rankings"]["fused"]] == [x.chunk_id for x in env.returned_unions[0]]
        assert [r["chunk_id"] for r in case["rankings"]["selected"]] == [x.chunk_id for x in env.returned_unions[0][:5]]
        assert case["assessment"]["fused_union"]["complete_combination_found"]
    assert "supporting_context" in report["cases"][2]["assessment"]["fused"]["separate_evidence"]
    assert "answer_review_qualification" in report["cases"][3]["assessment"]["fused"]["separate_evidence"]
    assert report["operations"]["hybrid"] == {"attempted": 4, "completed": 4}
    assert report["operations"]["dense"] == {"attempted": 4, "completed": 4}
    assert report["cleanup"]["client_close"] == report["cleanup"]["disposable_copy_removal"] == "passed"
    assert report["cleanup"]["original_fingerprint_match"] is True
    assert json.loads(env.output.read_text(encoding="utf-8")) == report


def test_binding_failure_stops_before_any_retrieval_and_cleans_up(environment):
    env, _ = environment
    with patch.object(runner, "validate_prospective_against_chunks", side_effect=ValueError("failure")):
        report = run(env)
    assert report["primary_failure"]["stage"] == "identity_validation"
    assert env.hybrid.call_count == env.select.call_count == 0
    assert [c["status"] for c in report["cases"]] == ["not_run"] * 4 + ["not_executed"] * 4
    env.client.close.assert_called_once_with()
    assert report["cleanup"]["disposable_copy_removal"] == "passed"
    assert report["cleanup"]["original_fingerprint_match"] is True


def test_actual_excerpt_binding_failure_stops_before_retrieval(environment):
    env, artifact = environment
    artifact["cases"][0]["mandatory_element_evidence"][0]["supporting_excerpts"][0]["exact_quote"] = "missing excerpt"
    env.seed_path.write_text(json.dumps(artifact), encoding="utf-8")
    with patch.object(runner, "PROSPECTIVE_HASH",
                      hashlib.sha256(env.seed_path.read_bytes()).hexdigest()):
        report = run(env)
    assert report["primary_failure"]["stage"] == "identity_validation"
    assert report["primary_failure"]["error_type"] == "EvaluationError"
    env.hybrid.assert_not_called()
    env.client.close.assert_called_once_with()
    assert report["cleanup"]["disposable_copy_removal"] == "passed"
    assert report["cleanup"]["original_fingerprint_match"] is True


def test_retrieval_failure_and_cleanup_failure_are_separate(environment):
    env, _ = environment
    env.dense.retrieve.side_effect = [env.dense.retrieve.return_value, RuntimeError("private")]
    env.client.close.side_effect = RuntimeError("private")
    report = run(env)
    assert report["status"] == "cleanup_failed"
    assert report["primary_failure"]["stage"] == "retrieval"
    assert report["cleanup_failures"][0]["stage"] == "client_close"
    assert [c["status"] for c in report["cases"]] == ["ok", "failed", "not_run", "not_run"] + ["not_executed"] * 4
    assert env.hybrid.call_count == 2 and env.select.call_count == 1
    assert report["cleanup"]["disposable_copy_removal"] == "passed"
    assert report["cleanup"]["original_fingerprint_match"] is True
    assert "private" not in env.output.read_text(encoding="utf-8")


def test_collision_is_rejected_before_resources(environment):
    env, _ = environment
    env.output.write_text("preserve", encoding="utf-8")
    with pytest.raises(runner.ExperimentError, match="reserve"):
        run(env)
    assert env.output.read_text(encoding="utf-8") == "preserve"
    env.open_client.assert_not_called()


def test_cli_explicit_mode_and_invalid_mode(environment):
    env, _ = environment
    args = ["--project-root", str(env.root), "--seed-path", str(env.seed_path),
            "--original-index", str(env.index), "--temp-parent", str(env.root),
            "--output-path", str(env.output), "--acknowledge-writers-stopped",
            "--dense-candidates", "10", "--lexical-candidates", "10"]
    with patch.object(runner, "run_experiment", return_value={"status": "completed"}) as execute, \
         redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        assert runner.main(args + ["--mode", "prospective"]) == 0
        assert execute.call_args.kwargs["mode"] == "prospective"
        execute.reset_mock()
        assert runner.main(args) == 0
        assert execute.call_args.kwargs["mode"] == "seed"
        execute.reset_mock()
        with pytest.raises(SystemExit):
            runner.main(args + ["--mode", "other"])
        execute.assert_not_called()
