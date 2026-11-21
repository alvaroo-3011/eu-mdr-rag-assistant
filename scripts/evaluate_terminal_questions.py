"""Run the local assistant sequentially and persist validated results per question.

No execution at import; no retries, extra provider calls, or raw response logging.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application import local_config, local_runtime, presentation
from src.generation import gemini_provider_response, gemini_response, gemini_transport
from src.generation.request_budget import RequestBudgetExceededError, RequestTokenCount

QUESTIONS = ROOT / "data" / "evaluation" / "terminal_questions_20.json"
CONFIG = ROOT / "config" / "local_app_config.json"


@contextmanager
def capture_count():
    """Observe the existing counter's validated return, including before rejection."""
    captured = {"input_tokens": None, "request_id": None}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if (event == "return" and frame.f_code is gemini_transport.GeminiRESTTransport.count_tokens.__code__
                and type(arg) is RequestTokenCount):
            captured["input_tokens"] = arg.input_tokens
            if re.fullmatch(r"gemini-request:sha256:[0-9a-f]{64}", arg.request_id):
                captured["request_id"] = arg.request_id
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield captured
    finally:
        sys.setprofile(previous)


def failure_details(exc):
    """Extract only bounded metadata; never retain messages or provider bodies."""
    stage, name, status = "unknown", "unknown", None
    seen = set()
    while isinstance(exc, BaseException) and id(exc) not in seen:
        seen.add(id(exc))
        candidate = type(exc).__name__
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", candidate):
            name = candidate
        if isinstance(exc, RequestBudgetExceededError):
            stage = "token_budget"
        if isinstance(exc, gemini_provider_response.GeminiHTTPStatusError):
            value = exc.status_code
            if type(value) is int and 100 <= value <= 599:
                status = value
        tb = exc.__traceback__
        while tb:
            code = tb.tb_frame.f_code
            if code is gemini_transport.GeminiRESTTransport._post.__code__:
                operation = tb.tb_frame.f_locals.get("operation")
                if operation in ("countTokens", "generateContent"):
                    stage = operation
            elif code is gemini_provider_response.parse_count_tokens_response.__code__:
                stage = "count_response_validation"
            elif code in (gemini_provider_response.parse_generate_content_response.__code__,
                          gemini_response.parse_gemini_response.__code__):
                stage = "generation_response_validation"
            tb = tb.tb_next
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return stage, name, status


def provider_diagnostics(exc):
    """Return only metadata produced by the bounded transport diagnostic."""
    seen = set()
    while isinstance(exc, BaseException) and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, gemini_provider_response.GeminiHTTPStatusError):
            value = exc.diagnostic
            if type(value) is dict:
                return value
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return {}


def _safe(value, key):
    return str(value).replace(key, "[REDACTED]") if key else str(value)


def _result_text(run):
    view = presentation.build_answer_view(run)
    lines = [f"Outcome: {view.status}"]
    for block in view.answer_blocks:
        lines.append(f"Answer: {block.text}")
        for c in block.citations:
            lines.append(
                f"Citation [{c.citation_id}]: {c.official_title}; document={c.document_id}; "
                f"version={c.version}; source={c.source_identity}; URL={c.official_url}; "
                f"pages={', '.join(map(str, c.pages)) or 'none'}; "
                f"locators={'; '.join(c.legal_locators) or 'none'}; "
                f"hierarchies={'; '.join(c.hierarchies) or 'none'}; "
                f"permitted regimes={', '.join(c.permitted_regimes)}"
            )
    if view.limitation is not None:
        lines.append(f"Limitation: {view.limitation}")
    if view.clarification_question is not None:
        lines.append(f"Clarification: {view.clarification_question}")
    if not any(block.citations for block in view.answer_blocks):
        lines.append("Citations: none")
    return lines


def run_batch(*, questions, config, api_key, report, submit=local_runtime.run_local_question):
    """Write and flush each result before attempting the next question."""
    report.write("# Terminal EU MDR evaluation\n\n")
    report.write(f"Model: {_safe(config.generation_config.model, api_key)}\n\n")
    report.write(f"Input token limit: {config.budget.input_token_limit}; output token limit: {config.budget.output_token_limit}\n\n")
    report.flush()
    os.fsync(report.fileno())
    for number, question in enumerate(questions, 1):
        lines = [f"## {number}", f"Question: {question}"]
        with capture_count() as count:
            try:
                run = submit(config=config, question=question, api_key=api_key)
                lines.extend(_result_text(run))
            except Exception as exc:
                stage, name, status = failure_details(exc)
                lines.extend(["Outcome: error", f"Stage: {stage}", f"Exception: {name}",
                              f"HTTP status: {status if status is not None else 'unknown'}"])
                diagnostic = provider_diagnostics(exc)
                lines.extend([
                    f"Provider error code: {diagnostic.get('provider_code') or 'unknown'}",
                    f"Provider status: {diagnostic.get('provider_status') or 'unknown'}",
                    f"Provider reasons: {', '.join(diagnostic.get('provider_reasons') or ()) or 'unknown'}",
                    f"Retry-After: {diagnostic.get('retry_after') or 'unknown'}",
                    f"Provider request ID: {diagnostic.get('request_id') or 'unknown'}",
                ])
        tokens = count["input_tokens"]
        lines.append(f"Input tokens / limit: {tokens if tokens is not None else 'unknown'} / {config.budget.input_token_limit}")
        lines.append(f"Application request ID: {count['request_id'] or 'unknown'}")
        report.write(_safe("\n\n".join(lines), api_key) + "\n\n")
        report.flush()
        os.fsync(report.fileno())
        print(f"Question {number}: recorded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="New report path; existing files are never overwritten")
    parser.add_argument("--questions", type=Path, default=QUESTIONS,
                        help="JSON array of questions (defaults to the 20-question set)")
    args = parser.parse_args()
    try:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        if type(questions) is not list or not questions or any(type(q) is not str or not q.strip() for q in questions):
            raise ValueError("Invalid question list")
        config = local_config.load_local_app_config(CONFIG)
        api_key = local_runtime.load_gemini_api_key(os.environ)
        with args.report.open("x", encoding="utf-8", newline="\n") as report:
            run_batch(questions=questions, config=config, api_key=api_key, report=report)
    except Exception as exc:
        # Never print exception text: external errors can contain secrets or payloads.
        name = type(exc).__name__
        print(f"Evaluation stopped ({name if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,79}', name) else 'error'}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
