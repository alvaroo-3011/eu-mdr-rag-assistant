"""Synthetic, offline coverage of the temporary provider failure diagnostic."""

from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import httpx
import main

from src.generation.gemini_transport import (
    GeminiRESTTransport, GeminiTransportConfig, run_gemini_request,
)
from src.generation.request_budget import RequestBudget
from tests.unit.generation.test_gemini_request import _fixture


class FailureDiagnosticTests(unittest.TestCase):
    def test_mocked_failures_show_only_safe_stage_class_and_status(self):
        chunk_set, _context, _settings, request = _fixture()
        budget = RequestBudget(input_token_limit=123, output_token_limit=256)
        fake_key = "FAKE_API_KEY_NOT_A_SECRET"
        scenarios = (
            (lambda wire: httpx.Response(429, text="SECRET_BODY"),
             "stage=countTokens exception=GeminiHTTPStatusError http_status=429"),
            (lambda wire: httpx.Response(200, json={"unexpected": "SECRET_BODY"}),
             "stage=countTokens_response_validation exception=GeminiProviderResponseError"),
            (lambda wire: httpx.Response(200, json={"totalTokens": 123})
             if wire.url.path.endswith(":countTokens")
             else httpx.Response(503, text="SECRET_BODY"),
             "stage=generateContent exception=GeminiHTTPStatusError http_status=503"),
            (lambda wire: httpx.Response(200, json={"totalTokens": 123})
             if wire.url.path.endswith(":countTokens")
             else httpx.Response(200, json={"unexpected": "SECRET_BODY"}),
             "stage=generateContent_response_validation exception=GeminiProviderResponseError"),
        )
        for handler, expected in scenarios:
            with self.subTest(expected=expected):
                output = io.StringIO()
                state = {}
                with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), \
                     patch.dict("os.environ", {"MDR_FAILURE_DIAGNOSTICS": "1"}), \
                     redirect_stderr(output), \
                     GeminiRESTTransport(
                         api_key=fake_key,
                         config=GeminiTransportConfig(api_version="v1beta", timeout_seconds=1,
                                                      max_response_bytes=20000),
                         transport=httpx.MockTransport(handler),
                     ) as adapter:
                    main._apply_submission(
                        submitted=True, question="Synthetic question", session_state=state,
                        submitter=lambda _question: run_gemini_request(
                            request=request, chunk_set=chunk_set, budget=budget, adapter=adapter),
                    )
                self.assertEqual(output.getvalue().strip(), "MDR_FAILURE " + expected)
                self.assertEqual(state[main._ERROR_KEY],
                    "The provider request or validated response failed; no partial answer is shown.")
                self.assertNotIn("SECRET_BODY", output.getvalue())
                self.assertNotIn(fake_key, output.getvalue())

    def test_diagnostic_is_disabled_by_default(self):
        output = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), redirect_stderr(output):
            main._apply_submission(
                submitted=True, question="Synthetic question", session_state={},
                submitter=lambda _question: (_ for _ in ()).throw(ValueError("SECRET_BODY")),
            )
        self.assertEqual(output.getvalue(), "")
