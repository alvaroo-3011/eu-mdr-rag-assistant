"""HTTPX MockTransport tests only: fake key, synthetic evidence, blocked sockets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import traceback
import unittest
from unittest.mock import patch

import httpx

from tests.unit.retrieval.test_lexical_query import _forbid_production_imports

with _forbid_production_imports():
    from src.generation.gemini_transport import (
        GeminiRESTTransport, GeminiTransportConfig, GeminiTransportError,
        GeminiTransportRun, run_gemini_request,
    )

from src.generation.gemini_provider_response import GeminiHTTPStatusError, GeminiProviderResponseError
from src.generation.request_budget import BudgetApproval, RequestBudget, RequestBudgetError, RequestBudgetExceededError
from tests.unit.generation.test_gemini_request import _fixture
from tests.unit.generation.test_gemini_provider_response import _envelope


_FAKE_KEY = "FAKE_API_KEY_NOT_A_SECRET"


def _config(**changes):
    return GeminiTransportConfig(**{"api_version": "v1beta", "timeout_seconds": 1.25,
                                    "max_response_bytes": 20000, **changes})


class _Stream(httpx.SyncByteStream):
    def __init__(self, chunks, *, error=None, close_error=None):
        self.chunks, self.error, self.close_error = chunks, error, close_error
        self.closed = False

    def __iter__(self):
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class GeminiTransportTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Real network forbidden")))
        self.enterContext(patch("httpx.HTTPTransport", side_effect=AssertionError("Implicit real transport forbidden")))
        self.chunk_set, self.context, self.settings, self.request = _fixture()
        self.budget = RequestBudget(input_token_limit=123, output_token_limit=256)
        self.calls = []

    def _adapter(self, handler=None, **config):
        def handle(wire):
            self.calls.append(wire)
            if handler is not None:
                return handler(wire)
            if wire.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 123}, headers={"Set-Cookie": "untrusted=1"})
            return httpx.Response(200, json=_envelope())
        adapter = GeminiRESTTransport(api_key=_FAKE_KEY, config=_config(**config), transport=httpx.MockTransport(handle))
        self.addCleanup(adapter.close)
        return adapter

    def _approval(self, **changes):
        return BudgetApproval(**{"request_id": self.request.request_id, "model": self.request.model,
            "input_tokens": 123, "max_output_tokens": 256, "budget": self.budget, **changes})

    def _run(self, adapter, **changes):
        return run_gemini_request(**{"request": self.request, "chunk_set": self.chunk_set,
                                   "budget": self.budget, "adapter": adapter, **changes})

    def test_success_counts_then_generates_exactly_once_and_retains_audit(self):
        result = self._run(self._adapter())
        self.assertIsInstance(result, GeminiTransportRun)
        self.assertEqual([wire.url.path.split(":")[-1] for wire in self.calls], ["countTokens", "generateContent"])
        self.assertEqual(result.approval.input_tokens, 123)
        self.assertEqual(result.generation.response.request_id, self.request.request_id)
        self.assertIs(result.generation.response.answer_blocks[0].citations[0].chunk, self.chunk_set.chunks[0])
        self.assertEqual(json.loads(result.generation.provider_body_json), _envelope())

    def test_exact_urls_bodies_headers_and_timeout_settings(self):
        self._run(self._adapter())
        for wire, expected in zip(self.calls, (self.request.count_tokens_body_json, self.request.generate_content_body_json)):
            self.assertEqual(wire.method, "POST")
            self.assertEqual(wire.url.scheme, "https")
            self.assertEqual(wire.url.host, "generativelanguage.googleapis.com")
            self.assertEqual(wire.url.query, b"")
            self.assertEqual(wire.content, expected.encode("utf-8"))
            self.assertEqual(wire.headers["x-goog-api-key"], _FAKE_KEY)
            self.assertEqual(wire.headers["accept-encoding"], "identity")
            self.assertEqual(wire.headers["content-type"], "application/json")
            self.assertEqual(wire.extensions["timeout"], {"connect": 1.25, "read": 1.25, "write": 1.25, "pool": 1.25})
            self.assertNotIn(_FAKE_KEY, str(wire.url))
            self.assertNotIn(_FAKE_KEY.encode(), wire.content)
            self.assertNotIn("cookie", wire.headers)
        logical = json.loads(self.request.request_json)
        self.assertEqual(json.loads(self.calls[0].content), {"generateContentRequest": logical})
        self.assertEqual({"model": self.request.model, **json.loads(self.calls[1].content)}, logical)

    def test_explicit_api_version_is_used_without_an_implicit_default(self):
        adapter = self._adapter(api_version="v1")
        adapter.count_tokens(request=self.request)
        self.assertEqual(self.calls[0].url.path, "/v1/models/gemini-synthetic:countTokens")

    def test_output_overflow_blocks_all_http_calls(self):
        with self.assertRaises(RequestBudgetExceededError):
            self._run(self._adapter(), budget=replace(self.budget, output_token_limit=255))
        self.assertEqual(self.calls, [])

    def test_input_overflow_makes_one_count_and_no_generation(self):
        with self.assertRaises(RequestBudgetExceededError):
            self._run(self._adapter(), budget=replace(self.budget, input_token_limit=122))
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0].url.path.endswith(":countTokens"))

    def test_count_quota_and_server_errors_stop_without_retry(self):
        for status in (429, 500, 503):
            self.calls.clear()
            adapter = self._adapter(lambda wire: httpx.Response(status, content=b"private error body"))
            with self.subTest(status=status), self.assertRaises(RequestBudgetError) as caught:
                self._run(adapter)
            self.assertEqual(len(self.calls), 1)
            self.assertIsInstance(caught.exception.__cause__, GeminiHTTPStatusError)
            self.assertNotIn("private error body", str(caught.exception.__cause__))

    def test_generation_error_stops_after_one_count_and_one_generation(self):
        def handler(wire):
            return httpx.Response(200, json={"totalTokens": 123}) if wire.url.path.endswith(":countTokens") else httpx.Response(429, content=b"quota")
        with self.assertRaises(GeminiHTTPStatusError):
            self._run(self._adapter(handler))
        self.assertEqual(len(self.calls), 2)

    def test_503_metadata_is_bounded_and_excludes_untrusted_error_content(self):
        request_id = "0123456789abcdef0123456789abcdef"
        sensitive = "SENSITIVE_BODY_MUST_NOT_LEAK"
        payload = {"error": {"code": 503, "status": "UNAVAILABLE", "message": sensitive,
                   "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                "reason": "SERVICE_UNAVAILABLE", "metadata": {"secret": _FAKE_KEY}},
                               {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": sensitive}]}}
        def handler(wire):
            if wire.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 123})
            return httpx.Response(503, json=payload, headers={"Retry-After": "45", "x-request-id": request_id})
        with self.assertRaises(GeminiHTTPStatusError) as caught:
            self._run(self._adapter(handler))
        self.assertEqual(caught.exception.diagnostic, {"provider_code": 503, "provider_status": "UNAVAILABLE",
            "provider_reasons": ("SERVICE_UNAVAILABLE",), "retry_after": "45 seconds", "request_id": request_id})
        rendered = repr(caught.exception.diagnostic) + "".join(traceback.format_exception(caught.exception))
        self.assertNotIn(sensitive, rendered)
        self.assertNotIn(_FAKE_KEY, rendered)
        self.assertEqual(len(self.calls), 2)

    def test_error_diagnostics_ignore_oversize_malformed_and_untrusted_fields(self):
        cases = [
            (b"x" * 8193, {"content-type": "application/json"}),
            (b'{"error":', {"content-type": "application/json"}),
            (json.dumps({"error": {"code": True, "status": _FAKE_KEY,
                 "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": _FAKE_KEY}]}}).encode(),
             {"content-type": "application/json", "retry-after": _FAKE_KEY, "x-request-id": _FAKE_KEY}),
        ]
        for body, headers in cases:
            with self.subTest(length=len(body)):
                self.calls.clear()
                adapter = self._adapter(lambda wire: httpx.Response(503, content=body, headers=headers))
                with self.assertRaises(GeminiHTTPStatusError) as caught:
                    adapter.count_tokens(request=self.request)
                self.assertEqual(caught.exception.status_code, 503)
                self.assertEqual(caught.exception.diagnostic, {"provider_code": None, "provider_status": None,
                    "provider_reasons": (), "retry_after": None, "request_id": None})
                self.assertEqual(len(self.calls), 1)

    def test_diagnostic_read_failure_does_not_mask_http_error(self):
        stream = _Stream([], error=RuntimeError(_FAKE_KEY))
        adapter = self._adapter(lambda wire: httpx.Response(503, stream=stream, headers={"Content-Type": "application/json"}))
        with self.assertRaises(GeminiHTTPStatusError) as caught:
            adapter.count_tokens(request=self.request)
        self.assertTrue(stream.closed)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertNotIn(_FAKE_KEY, "".join(traceback.format_exception(caught.exception)))

    def test_retry_after_http_date_and_trace_identifier_are_sanitized(self):
        adapter = self._adapter(lambda wire: httpx.Response(503, content=b"unparsed",
            headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT",
                     "x-cloud-trace-context": "0123456789abcdef0123456789abcdef/123;o=1"}))
        with self.assertRaises(GeminiHTTPStatusError) as caught:
            adapter.count_tokens(request=self.request)
        self.assertEqual(caught.exception.diagnostic["retry_after"], "Wed, 21 Oct 2026 07:28:00 GMT")
        self.assertEqual(caught.exception.diagnostic["request_id"], "0123456789abcdef0123456789abcdef")

    def test_redirect_is_not_followed_or_forwarded_with_credentials(self):
        adapter = self._adapter(lambda wire: httpx.Response(307, headers={"Location": "https://elsewhere.example/"}))
        with self.assertRaises(GeminiHTTPStatusError) as caught:
            adapter.count_tokens(request=self.request)
        self.assertEqual(caught.exception.status_code, 307)
        self.assertEqual(len(self.calls), 1)

    def test_timeouts_and_connection_errors_are_sanitized_and_not_retried(self):
        for kind in (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectError):
            self.calls.clear()
            def handler(wire):
                raise kind(_FAKE_KEY + " sensitive backend detail", request=wire)
            with self.subTest(kind=kind), self.assertRaises(GeminiTransportError) as caught:
                self._adapter(handler).count_tokens(request=self.request)
            rendered = "".join(traceback.format_exception(caught.exception))
            self.assertNotIn(_FAKE_KEY, rendered)
            self.assertNotIn("sensitive backend detail", rendered)
            self.assertEqual(len(self.calls), 1)

    def test_non_200_body_is_not_read_and_response_is_closed(self):
        stream = _Stream([], error=AssertionError("Error body must not be consumed"))
        adapter = self._adapter(lambda wire: httpx.Response(429, stream=stream))
        with self.assertRaises(GeminiHTTPStatusError):
            adapter.count_tokens(request=self.request)
        self.assertTrue(stream.closed)

    def test_wrong_missing_or_unsupported_json_content_type_fails(self):
        for content_type in (None, "text/html", "text/event-stream", "application/problem+json",
                             "application/json; charset=latin-1", "application/json, text/html"):
            headers = {} if content_type is None else {"Content-Type": content_type}
            adapter = self._adapter(lambda wire: httpx.Response(200, content=b'{"totalTokens":123}', headers=headers))
            with self.subTest(content_type=content_type), self.assertRaises(GeminiTransportError):
                adapter.count_tokens(request=self.request)

    def test_json_charset_utf8_variants_are_accepted(self):
        for content_type in ("application/json", "Application/JSON; charset=UTF-8", 'application/json; charset="utf-8"'):
            adapter = self._adapter(lambda wire: httpx.Response(200, content=b'{"totalTokens":123}', headers={"Content-Type": content_type}))
            with self.subTest(content_type=content_type):
                self.assertEqual(adapter.count_tokens(request=self.request).input_tokens, 123)

    def test_compressed_response_is_rejected_without_reading(self):
        stream = _Stream([], error=AssertionError("Compressed data must not be read"))
        adapter = self._adapter(lambda wire: httpx.Response(200, stream=stream,
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"}))
        with self.assertRaises(GeminiTransportError):
            adapter.count_tokens(request=self.request)
        self.assertTrue(stream.closed)

    def test_response_limit_exact_boundary_accepts_and_one_byte_less_rejects(self):
        raw = b'{"totalTokens":123}'
        for limit, accepted in ((len(raw), True), (len(raw)-1, False)):
            adapter = self._adapter(lambda wire: httpx.Response(200, content=raw, headers={"Content-Type": "application/json"}), max_response_bytes=limit)
            with self.subTest(limit=limit):
                if accepted:
                    self.assertEqual(adapter.count_tokens(request=self.request).input_tokens, 123)
                else:
                    with self.assertRaises(GeminiTransportError):
                        adapter.count_tokens(request=self.request)

    def test_streamed_oversize_without_content_length_is_bounded_and_closed(self):
        stream = _Stream([b"x" * 8, b"y" * 8, b"z" * 8])
        adapter = self._adapter(lambda wire: httpx.Response(200, stream=stream, headers={"Content-Type": "application/json"}), max_response_bytes=16)
        with self.assertRaisesRegex(GeminiTransportError, "exceeds"):
            adapter.count_tokens(request=self.request)
        self.assertTrue(stream.closed)

    def test_content_length_syntax_and_actual_length_are_checked(self):
        for length in ("-1", "abc", "1,1", "99", "1"):
            stream = _Stream([b'{"totalTokens":123}'])
            adapter = self._adapter(lambda wire: httpx.Response(200, stream=stream,
                headers={"Content-Type": "application/json", "Content-Length": length}))
            with self.subTest(length=length), self.assertRaises(GeminiTransportError):
                adapter.count_tokens(request=self.request)
            self.assertTrue(stream.closed)

    def test_declared_oversize_is_rejected_before_body_is_read(self):
        stream = _Stream([], error=AssertionError("Oversize body must not be read"))
        adapter = self._adapter(lambda wire: httpx.Response(200, stream=stream,
            headers={"Content-Type": "application/json", "Content-Length": "100"}), max_response_bytes=20)
        with self.assertRaises(GeminiTransportError):
            adapter.count_tokens(request=self.request)
        self.assertTrue(stream.closed)

    def test_invalid_utf8_is_rejected_without_replacement(self):
        adapter = self._adapter(lambda wire: httpx.Response(200, content=b'\xff', headers={"Content-Type": "application/json"}))
        with self.assertRaises(GeminiTransportError):
            adapter.count_tokens(request=self.request)

    def test_invalid_count_and_generation_payloads_reach_existing_validators(self):
        with self.assertRaises(GeminiProviderResponseError):
            self._adapter(lambda wire: httpx.Response(200, json={"totalTokens": True})).count_tokens(request=self.request)
        value = _envelope()
        value["candidates"][0]["finishReason"] = "CONTINUATION"
        adapter = self._adapter(lambda wire: httpx.Response(200, json=value))
        with self.assertRaises(GeminiProviderResponseError):
            adapter.generate_content(request=self.request, approval=self._approval(), chunk_set=self.chunk_set)

    def test_generation_requires_matching_approval_before_http(self):
        adapter = self._adapter()
        for approval in (None, {}, self._approval(request_id="other"), self._approval(model="models/other"),
                         self._approval(max_output_tokens=255)):
            with self.subTest(approval=type(approval)), self.assertRaises(RequestBudgetError):
                adapter.generate_content(request=self.request, approval=approval, chunk_set=self.chunk_set)
        self.assertEqual(self.calls, [])

    def test_authoritative_evidence_is_checked_before_counting_or_generation(self):
        adapter = self._adapter()
        with self.assertRaises(ValueError):
            self._run(adapter, chunk_set=None)
        with self.assertRaises(ValueError):
            adapter.generate_content(request=self.request, approval=self._approval(), chunk_set=None)
        self.assertEqual(self.calls, [])

    def test_request_mutation_during_count_blocks_generation(self):
        request = deepcopy(self.request)
        def handler(wire):
            object.__setattr__(request, "question", "Changed in mock callback")
            return httpx.Response(200, json={"totalTokens": 123})
        with self.assertRaises(RequestBudgetError):
            self._run(self._adapter(handler), request=request)
        self.assertEqual(len(self.calls), 1)

    def test_request_mutation_during_generation_rejects_returned_answer(self):
        request = deepcopy(self.request)
        def handler(wire):
            object.__setattr__(request, "question", "Changed in generation callback")
            return httpx.Response(200, json=_envelope())
        with self.assertRaises(GeminiTransportError):
            self._adapter(handler).generate_content(request=request, approval=self._approval(), chunk_set=self.chunk_set)
        self.assertEqual(len(self.calls), 1)

    def test_transport_and_key_are_explicit_without_environment_lookup(self):
        with self.assertRaises(TypeError):
            GeminiRESTTransport(config=_config(), transport=httpx.MockTransport(lambda wire: httpx.Response(200)))
        for key in (None, "", " ", "line\nbreak", "tab\tkey", "nonascii-é", 1, True):
            with self.subTest(key=repr(key)), self.assertRaises(GeminiTransportError):
                GeminiRESTTransport(api_key=key, config=_config(), transport=httpx.MockTransport(lambda wire: httpx.Response(200)))
        for transport in (None, {}, object()):
            with self.subTest(transport=type(transport)), self.assertRaises(GeminiTransportError):
                GeminiRESTTransport(api_key=_FAKE_KEY, config=_config(), transport=transport)
        self.assertNotIn(_FAKE_KEY, repr(self._adapter()))

    def test_config_rejects_missing_or_unsafe_settings_without_defaults(self):
        for version in (None, "", "v1/other", "https://other", "v1?key=x", "../v1", True):
            with self.subTest(version=version), self.assertRaises(GeminiTransportError):
                _config(api_version=version)
        for timeout in (None, True, 0, -1, float("nan"), float("inf"), "1", 1e-999):
            with self.subTest(timeout=timeout), self.assertRaises(GeminiTransportError):
                _config(timeout_seconds=timeout)
        for size in (None, True, 0, -1, 1.0, "10"):
            with self.subTest(size=size), self.assertRaises(GeminiTransportError):
                _config(max_response_bytes=size)
        with self.assertRaises(TypeError):
            GeminiTransportConfig(api_version="v1beta", timeout_seconds=1)

    def test_config_is_frozen_and_run_results_must_match(self):
        config = _config()
        with self.assertRaises(FrozenInstanceError):
            config.api_version = "v1"
        result = self._run(self._adapter())
        with self.assertRaises(FrozenInstanceError):
            result.approval = self._approval()
        with self.assertRaises(GeminiTransportError):
            replace(result, approval=self._approval(request_id="other"))
        with self.assertRaises(GeminiTransportError):
            replace(result, generation=None)

    def test_context_manager_closes_owned_transport_and_reuse_fails(self):
        adapter = self._adapter()
        with adapter as entered:
            self.assertIs(entered, adapter)
            adapter.count_tokens(request=self.request)
        with self.assertRaises(GeminiTransportError):
            adapter.count_tokens(request=self.request)
        with self.assertRaises(GeminiTransportError):
            adapter.__enter__()
        adapter.close()
        self.assertEqual(len(self.calls), 1)

    def test_stream_read_failure_closes_response_without_partial_receipt(self):
        stream = _Stream([b'{"totalTokens":'], error=httpx.ReadTimeout(_FAKE_KEY))
        adapter = self._adapter(lambda wire: httpx.Response(200, stream=stream, headers={"Content-Type": "application/json"}))
        with self.assertRaises(GeminiTransportError):
            adapter.count_tokens(request=self.request)
        self.assertTrue(stream.closed)
        self.assertEqual(len(self.calls), 1)

    def test_baseexception_propagates_even_when_cleanup_has_ordinary_failure(self):
        for error in (KeyboardInterrupt(), SystemExit()):
            stream = _Stream([], error=error, close_error=RuntimeError("cleanup failed"))
            adapter = self._adapter(lambda wire: httpx.Response(200, stream=stream, headers={"Content-Type": "application/json"}))
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                adapter.count_tokens(request=self.request)
            self.assertTrue(stream.closed)

    def test_injected_transport_does_not_create_implicit_real_backend(self):
        # The setUp guard makes implicit HTTPTransport construction fail.
        self._run(self._adapter())
        self.assertEqual(len(self.calls), 2)

    def test_success_does_not_mutate_request_or_evidence(self):
        before = deepcopy((self.request, self.chunk_set, self.budget))
        self._run(self._adapter())
        self.assertEqual((self.request, self.chunk_set, self.budget), before)

    def test_invalid_adapter_or_request_does_not_send(self):
        with self.assertRaises(GeminiTransportError):
            self._run(None)
        adapter = self._adapter()
        with self.assertRaises(GeminiTransportError):
            adapter.count_tokens(request=None)
        self.assertEqual(self.calls, [])

    def test_public_run_arguments_are_required_and_keyword_only(self):
        values = {"request": self.request, "chunk_set": self.chunk_set, "budget": self.budget, "adapter": self._adapter()}
        for key in values:
            with self.subTest(missing=key), self.assertRaises(TypeError):
                run_gemini_request(**{name: value for name, value in values.items() if name != key})
        with self.assertRaises(TypeError):
            run_gemini_request(*values.values())


if __name__ == "__main__":
    unittest.main()
