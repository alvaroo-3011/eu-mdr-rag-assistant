"""Native Streamlit AppTest coverage for the local MDR entry point."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.application.local_config import LocalConfigError
from src.application.local_runtime import LocalCredentialError, LocalRuntimeError
from src.application.presentation import build_answer_view
from src.generation.gemini_transport import GeminiTransportError
from tests.unit.application.test_local_interface import _run


_MAIN = Path(__file__).resolve().parents[2] / "main.py"
_FAKE_KEY = "FAKE_KEY_NOT_A_CREDENTIAL"
_MODEL_TEXT = "<script>model text remains text</script>"
_ANSWER_LINE = _MODEL_TEXT + " [1]"


def _values(app: AppTest) -> list[str]:
    values: list[str] = []
    for collection_name in (
        "title", "info", "warning", "caption", "subheader", "text", "error",
    ):
        values.extend(str(element.value) for element in getattr(app, collection_name))
    return values


@contextmanager
def _native_app(*, run_result):
    """Run main.py with every external execution boundary intercepted."""

    config_sentinel = object()
    with ExitStack() as stack:
        load_config = stack.enter_context(patch(
            "src.application.local_config.load_local_app_config",
            return_value=config_sentinel,
        ))
        load_key = stack.enter_context(patch(
            "src.application.local_runtime.load_gemini_api_key",
            return_value=_FAKE_KEY,
        ))
        run_question = stack.enter_context(patch(
            "src.application.local_runtime.run_local_question",
            return_value=run_result,
        ))
        resource_guards = (
            stack.enter_context(patch(
                "src.application.local_runtime.open_existing_chroma_client",
                side_effect=AssertionError("real Chroma construction is forbidden"),
            )),
            stack.enter_context(patch(
                "src.application.local_runtime.initialize_retrieval",
                side_effect=AssertionError("real retrieval initialization is forbidden"),
            )),
            stack.enter_context(patch(
                "src.application.local_runtime.httpx.HTTPTransport",
                side_effect=AssertionError("real HTTP transport construction is forbidden"),
            )),
            stack.enter_context(patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("network access is forbidden"),
            )),
        )
        app = AppTest.from_file(_MAIN, default_timeout=5).run()
        try:
            yield app, config_sentinel, load_config, load_key, run_question
        finally:
            for guard in resource_guards:
                guard.assert_not_called()


class NativeStreamlitInterfaceTests(unittest.TestCase):
    def assert_no_app_exception(self, app: AppTest) -> None:
        self.assertEqual(list(app.exception), [])

    def test_initial_render_has_required_controls_and_makes_no_execution_call(self):
        with _native_app(run_result=_run()) as (
            app, _config, load_config, load_key, run_question,
        ):
            self.assert_no_app_exception(app)
            self.assertEqual(app.title[0].value, "EU MDR Assistant")
            self.assertIn("Ask a question about the EU Medical Device Regulation.", [x.value for x in app.markdown])
            self.assertEqual([x.value for x in app.caption], [
                "Uses local retrieval and may call an LLM.",
                "Informational use only. Not legal advice.",
                "Do not enter confidential, personal, patient or client-identifying information.",
                "Answers use selected EU MDR documents only. Coverage is limited. IVDR is out of scope.",
            ])
            self.assertEqual(len(app.text_area), 1)
            self.assertEqual(app.text_area[0].label, "Question")
            self.assertEqual(len(app.form_submit_button), 1)
            self.assertEqual(app.form_submit_button[0].label, "Submit question")
            load_config.assert_not_called()
            load_key.assert_not_called()
            run_question.assert_not_called()

    def test_blank_submission_shows_fixed_error_without_execution(self):
        with _native_app(run_result=_run()) as (
            app, _config, load_config, load_key, run_question,
        ):
            app.text_area[0].set_value("  \t ")
            app.form_submit_button[0].click().run()

            self.assert_no_app_exception(app)
            self.assertEqual([item.value for item in app.error], [
                "Enter a nonblank EU MDR question.",
            ])
            load_config.assert_not_called()
            load_key.assert_not_called()
            run_question.assert_not_called()

    def test_submission_rerun_and_later_submission_have_exact_call_counts(self):
        run = _run(rich=True)
        with _native_app(run_result=run) as (
            app, config, load_config, load_key, run_question,
        ):
            app.text_area[0].set_value("First MDR question?")
            app.form_submit_button[0].click().run()

            self.assert_no_app_exception(app)
            run_question.assert_called_once_with(
                config=config,
                question="First MDR question?",
                api_key=_FAKE_KEY,
            )
            self.assertIn("Validated answer", [item.value for item in app.subheader])
            self.assertIn(_ANSWER_LINE, [item.value for item in app.text])

            app.run()
            self.assert_no_app_exception(app)
            self.assertEqual(run_question.call_count, 1)
            self.assertIn(_ANSWER_LINE, [item.value for item in app.text])

            app.text_area[0].set_value("Second MDR question?")
            app.form_submit_button[0].click().run()
            self.assert_no_app_exception(app)
            self.assertEqual(run_question.call_count, 2)
            self.assertEqual(
                run_question.call_args.kwargs["question"],
                "Second MDR question?",
            )
            self.assertEqual(load_config.call_count, 2)
            self.assertEqual(load_key.call_count, 2)

    def test_all_four_valid_statuses_render_limitations_and_clarification(self):
        expected = {
            "answered": "Validated answer",
            "insufficient_evidence": "Insufficient selected evidence",
            "needs_clarification": "Clarification needed",
            "out_of_scope": "Outside the EU MDR MVP scope",
        }
        for status, heading in expected.items():
            with self.subTest(status=status), _native_app(run_result=_run(status)) as (
                app, _config, _load_config, _load_key, run_question,
            ):
                app.text_area[0].set_value("Status question?")
                app.form_submit_button[0].click().run()

                self.assert_no_app_exception(app)
                run_question.assert_called_once()
                self.assertIn(heading, [item.value for item in app.subheader])
                if status == "answered":
                    self.assertIn(_ANSWER_LINE, [item.value for item in app.text])
                elif status == "needs_clarification":
                    self.assertEqual(len(app.info), 0)
                    self.assertIn(
                        "Which device scope do you mean?",
                        [item.value for item in app.text],
                    )
                else:
                    self.assertIn(
                        "The validated response reports a limitation.",
                        [item.value for item in app.text],
                    )

    def test_p8_replaces_p7_clarification_in_the_same_session(self):
        p7 = "What material must my device be made from to comply with the MDR?"
        p8 = "Under the IVDR, what performance evaluation requirements apply to my in vitro diagnostic test?"
        with _native_app(run_result=_run("needs_clarification")) as (
            app, config, _load_config, _load_key, run_question,
        ):
            run_question.side_effect = [_run("needs_clarification"), _run("out_of_scope")]
            app.text_area[0].set_value(p7)
            app.form_submit_button[0].click().run()
            self.assert_no_app_exception(app)
            self.assertEqual([item.value for item in app.subheader], ["Clarification needed"])
            self.assertIn("Which device scope do you mean?", [item.value for item in app.text])

            app.text_area[0].set_value(p8)
            app.form_submit_button[0].click().run()
            self.assert_no_app_exception(app)
            self.assertEqual([item.value for item in app.subheader], ["Outside the EU MDR MVP scope"])
            self.assertNotIn("Which device scope do you mean?", [item.value for item in app.text])
            self.assertIn("The validated response reports a limitation.", [item.value for item in app.text])
            self.assertEqual(run_question.call_count, 2)
            self.assertEqual([call.kwargs for call in run_question.call_args_list], [
                {"config": config, "question": p7, "api_key": _FAKE_KEY},
                {"config": config, "question": p8, "api_key": _FAKE_KEY},
            ])

    def test_citations_use_authoritative_metadata_and_omit_absent_locators(self):
        rich_run = _run(rich=True)
        citation = build_answer_view(rich_run).answer_blocks[0].citations[0]
        with _native_app(run_result=rich_run) as (app, *_boundaries):
            app.text_area[0].set_value("Cited question?")
            app.form_submit_button[0].click().run()

            text = [item.value for item in app.text]
            markdown = [item.value for item in app.markdown]
            self.assertIn("##### Answer", markdown)
            self.assertIn("##### Sources", markdown)
            self.assertIn(_ANSWER_LINE, text)
            self.assertIn("[1] Guidance, 1.0 — Article 15; Annex I", text)
            self.assertEqual(text.count("[1] Guidance, 1.0 — Article 15; Annex I"), 1)
            self.assertEqual([item.label for item in app.expander], ["Technical details [1]"])
            self.assertIn(f"Title: {citation.official_title}", text)
            self.assertFalse(any(
                str(item.value).startswith("Citations:") for item in app.caption
            ))
            self.assertFalse(any(citation.official_title in str(value) for value in markdown))
            self.assertIn(f"Document: {citation.document_id}", text)
            self.assertIn(f"Version: {citation.version}", text)
            self.assertIn(f"Source identity: {citation.source_identity}", text)
            self.assertIn("PDF page(s): 3, 5", text)
            self.assertIn("Legal locator: article: 15 > paragraph: 3", text)
            self.assertIn("Hierarchy: Scope", text)
            self.assertIn("Permitted regimes: MDR, IVDR", text)
            self.assertIn(
                "Validated citation identity does not by itself prove semantic support or sufficiency.",
                [item.value for item in app.caption],
            )

        with _native_app(run_result=_run()) as (app, *_boundaries):
            app.text_area[0].set_value("Minimal citation question?")
            app.form_submit_button[0].click().run()

            text = [item.value for item in app.text]
            self.assertFalse(any(item.startswith("PDF page(s):") for item in text))
            self.assertFalse(any(item.startswith("Legal locator:") for item in text))
            self.assertFalse(any(item.startswith("Hierarchy:") for item in text))

    def test_failures_are_sanitized_and_never_show_partial_answers(self):
        secret = "SECRET_VALUE_MUST_NOT_LEAK"
        cases = (
            (
                "credential",
                LocalCredentialError("raw credential failure " + secret),
                "The LLM API key is missing or invalid.",
            ),
            (
                "configuration",
                LocalConfigError("raw configuration failure " + secret),
                "The local application configuration is missing or invalid.",
            ),
            (
                "startup",
                LocalRuntimeError("raw startup failure " + secret),
                "The local application could not complete startup safely.",
            ),
            (
                "provider",
                GeminiTransportError("raw provider failure " + secret),
                "The provider request or validated response failed; no partial answer is shown.",
            ),
        )
        for owner, failure, expected in cases:
            with self.subTest(owner=owner), _native_app(run_result=_run()) as (
                app, _config, load_config, load_key, run_question,
            ):
                if owner == "configuration":
                    load_config.side_effect = failure
                elif owner == "credential":
                    load_key.side_effect = failure
                else:
                    run_question.side_effect = failure
                app.text_area[0].set_value("Fail safely?")
                app.form_submit_button[0].click().run()

                self.assert_no_app_exception(app)
                self.assertEqual([item.value for item in app.error], [expected])
                displayed = "\n".join(_values(app))
                self.assertNotIn(secret, displayed)
                self.assertNotIn("raw ", displayed)
                self.assertNotIn(_MODEL_TEXT, displayed)
                self.assertNotIn("Validated answer", displayed)
                if owner in ("configuration", "credential"):
                    run_question.assert_not_called()
                else:
                    run_question.assert_called_once()

    def test_failed_submission_can_recover_on_next_explicit_submission(self):
        secret = "SECRET_VALUE_MUST_NOT_LEAK"
        with _native_app(run_result=_run()) as (
            app, _config, _load_config, _load_key, run_question,
        ):
            run_question.side_effect = (
                GeminiTransportError("raw provider failure " + secret),
                _run(),
            )
            app.text_area[0].set_value("First attempt?")
            app.form_submit_button[0].click().run()
            self.assertEqual(len(app.error), 1)
            self.assertNotIn(_MODEL_TEXT, _values(app))

            app.text_area[0].set_value("Second attempt?")
            app.form_submit_button[0].click().run()

            self.assert_no_app_exception(app)
            self.assertEqual(run_question.call_count, 2)
            self.assertEqual(len(app.error), 0)
            self.assertIn(_ANSWER_LINE, [item.value for item in app.text])
            self.assertNotIn(secret, "\n".join(_values(app)))


if __name__ == "__main__":
    unittest.main()
