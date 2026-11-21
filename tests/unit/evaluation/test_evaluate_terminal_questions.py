"""Offline tests for the terminal evaluation runner."""
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from scripts.evaluate_terminal_questions import failure_details, run_batch
from scripts import evaluate_terminal_questions
from src.generation.gemini_transport import GeminiRESTTransport, GeminiTransportConfig, run_gemini_request
from src.generation.request_budget import RequestBudget, check_request_budget
from tests.unit.application.test_local_interface import _run
from tests.unit.generation.test_gemini_request import _fixture


def test_budget_rejection_is_flushed_before_next_question_and_batch_continues(tmp_path, capsys):
    _, _, _, request = _fixture()
    budget = RequestBudget(input_token_limit=100, output_token_limit=256)
    config = SimpleNamespace(budget=budget, generation_config=SimpleNamespace(model=request.model))
    path = tmp_path / "report.md"
    questions = ["First synthetic question?", "Second synthetic question?"]
    calls = []
    secret = "FAKE_SECRET_NOT_A_CREDENTIAL"

    def submit(*, config, question, api_key):
        calls.append(question)
        if len(calls) == 1:
            adapter = GeminiRESTTransport(
                api_key=api_key,
                config=GeminiTransportConfig(api_version="v1beta", timeout_seconds=1, max_response_bytes=1024),
                transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            )
            try:
                with patch.object(GeminiRESTTransport, "_post", return_value='{"totalTokens":101}'):
                    check_request_budget(request=request, budget=budget, token_counter=adapter)
            finally:
                adapter.close()
        assert "Input tokens / limit: 101 / 100" in path.read_text(encoding="utf-8")
        return _run("out_of_scope")

    with path.open("x", encoding="utf-8") as report:
        run_batch(questions=questions, config=config, api_key=secret, report=report, submit=submit)

    output = path.read_text(encoding="utf-8")
    assert calls == questions
    assert "Stage: token_budget" in output
    assert "Exception: RequestBudgetExceededError" in output
    assert "Input tokens / limit: 101 / 100" in output
    assert "Outcome: out_of_scope" in output
    assert "Input tokens / limit: unknown / 100" in output
    assert secret not in output
    assert secret not in capsys.readouterr().out


def test_provider_http_status_and_count_stage_are_reported_without_error_body():
    _, _, _, request = _fixture()
    adapter = GeminiRESTTransport(
        api_key="FAKE_KEY",
        config=GeminiTransportConfig(api_version="v1beta", timeout_seconds=1, max_response_bytes=1024),
        transport=httpx.MockTransport(lambda request: httpx.Response(429, content=b"private body")),
    )
    try:
        try:
            check_request_budget(
                request=request, budget=RequestBudget(input_token_limit=100, output_token_limit=256),
                token_counter=adapter,
            )
        except Exception as exc:
            assert failure_details(exc) == ("countTokens", "GeminiHTTPStatusError", 429)
        else:
            raise AssertionError("Expected HTTP failure")
    finally:
        adapter.close()


def test_validated_answer_and_citation_details_are_written(tmp_path):
    config = SimpleNamespace(
        budget=SimpleNamespace(input_token_limit=100, output_token_limit=256),
        generation_config=SimpleNamespace(model="synthetic-model"),
    )
    path = tmp_path / "answer.md"
    with path.open("x", encoding="utf-8") as report:
        run_batch(questions=["Synthetic question?"], config=config, api_key="FAKE_KEY",
                  report=report, submit=lambda **kwargs: _run(rich=True))
    output = path.read_text(encoding="utf-8")
    assert "Outcome: answered" in output
    assert "<script>model text remains text</script>" in output
    assert "Citation [1]:" in output
    assert "pages=3, 5" in output
    assert "article: 15 > paragraph: 3" in output


def test_cli_accepts_the_five_question_list_without_live_calls(tmp_path, monkeypatch):
    config = SimpleNamespace(
        budget=SimpleNamespace(input_token_limit=100, output_token_limit=256),
        generation_config=SimpleNamespace(model="synthetic-model"),
    )
    report = tmp_path / "five.md"
    monkeypatch.setattr(evaluate_terminal_questions.local_config, "load_local_app_config", lambda _: config)
    monkeypatch.setattr(evaluate_terminal_questions.local_runtime, "load_gemini_api_key", lambda _: "FAKE_KEY")
    calls = []

    def batch(**kwargs):
        calls.extend(kwargs["questions"])

    monkeypatch.setattr(evaluate_terminal_questions, "run_batch", batch)
    monkeypatch.setattr("sys.argv", ["runner", "--report", str(report), "--questions",
                                    str(evaluate_terminal_questions.ROOT / "data/evaluation/broad_mdr_questions_5.json")])
    assert evaluate_terminal_questions.main() == 0
    assert len(calls) == 5
    assert calls[0] == "What is the EU Medical Device Regulation, and who does it apply to?"


def test_report_records_safe_503_metadata_and_continues(tmp_path, capsys):
    chunk_set, _, _, request = _fixture()
    budget = RequestBudget(input_token_limit=200, output_token_limit=256)
    config = SimpleNamespace(budget=budget, generation_config=SimpleNamespace(model=request.model))
    secret, body_secret = "FAKE_KEY_PRIVATE", "PRIVATE_PROVIDER_PAYLOAD"
    calls = []
    def handler(wire):
        calls.append(wire.url.path)
        if wire.url.path.endswith(":countTokens"):
            return httpx.Response(200, json={"totalTokens": 123})
        return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE",
            "message": body_secret, "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                                  "reason": "BACKEND_ERROR", "metadata": {"key": secret}}]}},
            headers={"Retry-After": "10", "x-request-id": "0123456789abcdef0123456789abcdef"})
    def submit(**kwargs):
        if kwargs["question"] == "second":
            return _run("out_of_scope")
        adapter = GeminiRESTTransport(api_key=secret,
            config=GeminiTransportConfig(api_version="v1beta", timeout_seconds=1, max_response_bytes=1024),
            transport=httpx.MockTransport(handler))
        try:
            return run_gemini_request(request=request, chunk_set=chunk_set, budget=budget, adapter=adapter)
        finally:
            adapter.close()
    path = tmp_path / "error.md"
    with path.open("x", encoding="utf-8") as report:
        run_batch(questions=["first", "second"], config=config, api_key=secret, report=report, submit=submit)
    output = path.read_text(encoding="utf-8")
    assert "Stage: generateContent" in output
    assert "Provider error code: 503" in output and "Provider status: UNAVAILABLE" in output
    assert "Provider reasons: BACKEND_ERROR" in output and "Retry-After: 10 seconds" in output
    assert "Provider request ID: 0123456789abcdef0123456789abcdef" in output
    assert "Input tokens / limit: 123 / 200" in output and "Outcome: out_of_scope" in output
    assert f"Application request ID: {request.request_id}" in output
    console = capsys.readouterr()
    assert secret not in output + console.out + console.err
    assert body_secret not in output + console.out + console.err
    assert [x.split(":")[-1] for x in calls] == ["countTokens", "generateContent"]
