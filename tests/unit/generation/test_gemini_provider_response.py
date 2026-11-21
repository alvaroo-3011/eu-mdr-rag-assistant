"""Synthetic REST-envelope checks; no real provider, corpus, model, or database."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

from tests.unit.retrieval.test_lexical_query import _forbid_production_imports

with _forbid_production_imports():
    from src.generation.gemini_provider_response import (
        GeminiHTTPStatusError, GeminiProviderGeneration, GeminiProviderResponseError,
        parse_count_tokens_response, parse_generate_content_response,
    )

from src.generation.gemini_response import GeminiResponseError, parse_gemini_response
from src.generation.request_budget import RequestBudget, check_request_budget
from tests.unit.generation.test_gemini_request import _fixture
from tests.unit.generation.test_gemini_response import _payload


_MODULE = "src.generation.gemini_provider_response"


def _envelope(text=None):
    return {"candidates": [{
        "content": {"role": "model", "parts": [{"text": json.dumps(_payload()) if text is None else text}]},
        "finishReason": "STOP", "index": 0,
    }]}


class GeminiProviderResponseTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.chunk_set, self.context, self.config, self.request = _fixture()

    def _count(self, payload=None, **changes):
        return parse_count_tokens_response(**{
            "status_code": 200, "body_json": json.dumps({"totalTokens": 123} if payload is None else payload),
            "request_id": self.request.request_id, "request": self.request, **changes,
        })

    def _generate(self, payload=None, **changes):
        return parse_generate_content_response(**{
            "status_code": 200, "body_json": json.dumps(_envelope() if payload is None else payload),
            "request_id": self.request.request_id, "request": self.request,
            "chunk_set": self.chunk_set, **changes,
        })

    def test_count_is_bound_to_exact_request_and_accepts_explicit_zero(self):
        for count in (0, 123):
            with self.subTest(count=count):
                receipt = self._count({"totalTokens": count})
                self.assertEqual(receipt.input_tokens, count)
                self.assertEqual(receipt.request_id, self.request.request_id)
                self.assertEqual(receipt.model, self.request.model)

    def test_count_receipt_connects_to_existing_injected_budget_boundary(self):
        outer = self
        calls = []
        class Counter:
            def count_tokens(self, *, request):
                calls.append(request)
                return outer._count(request=request, request_id=request.request_id)
        approval = check_request_budget(request=self.request,
            budget=RequestBudget(input_token_limit=123, output_token_limit=256), token_counter=Counter())
        self.assertEqual(calls, [self.request])
        self.assertIs(calls[0], self.request)
        self.assertEqual(approval.input_tokens, 123)

    def test_count_does_not_coerce_or_infer_total(self):
        for value in (None, -1, True, False, 1.0, "123", [], {}):
            with self.subTest(value=value), self.assertRaises(GeminiProviderResponseError):
                self._count({"totalTokens": value})
        for payload in ({}, {"cachedContentTokenCount": 10}, {"usageMetadata": {"promptTokenCount": 123}}):
            with self.subTest(payload=payload), self.assertRaises(GeminiProviderResponseError):
                self._count(payload)

    def test_count_metadata_does_not_replace_or_subtract_from_total(self):
        receipt = self._count({"totalTokens": 123, "cachedContentTokenCount": 17,
                               "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 123}],
                               "cacheTokensDetails": []})
        self.assertEqual(receipt.input_tokens, 123)
        for payload in ({"totalTokens": 1, "cachedContentTokenCount": True},
                        {"totalTokens": 1, "promptTokensDetails": {}},
                        {"totalTokens": 1, "cacheTokensDetails": None}):
            with self.subTest(payload=payload), self.assertRaises(GeminiProviderResponseError):
                self._count(payload)

    def test_valid_stop_output_calls_application_parser_once_and_retains_raw_body(self):
        raw = " \n" + json.dumps(_envelope(), indent=2) + "\n"
        with patch(_MODULE + ".parse_gemini_response", wraps=parse_gemini_response) as parser:
            result = self._generate(body_json=raw)
        self.assertEqual(parser.call_count, 1)
        self.assertIs(parser.call_args.kwargs["request"], self.request)
        self.assertIs(parser.call_args.kwargs["chunk_set"], self.chunk_set)
        self.assertEqual(result.provider_body_json, raw)
        self.assertEqual(result.response.status, "answered")
        self.assertIs(result.response.answer_blocks[0].citations[0].chunk, self.chunk_set.chunks[0])

    def test_all_four_application_statuses_still_use_existing_validator(self):
        for status in ("answered", "insufficient_evidence", "needs_clarification", "out_of_scope"):
            payload = _payload(status=status)
            if status != "answered":
                payload.update(answer_blocks=[], limitation="Selected evidence limitation.")
            if status == "needs_clarification":
                payload["clarification_question"] = "Which scope?"
            with self.subTest(status=status):
                result = self._generate(_envelope(json.dumps(payload)))
                self.assertEqual(result.response.status, status)

    def test_text_parts_are_joined_exactly_without_inserted_whitespace(self):
        app = json.dumps(_payload(answer_blocks=[{"text": "Évidence α", "citation_ids": ["1"]}]), ensure_ascii=False)
        position = app.index("Évidence") + 3
        value = _envelope()
        value["candidates"][0]["content"]["parts"] = [
            {"text": app[:position]}, {"text": "", "thought": False},
            {"text": app[position:], "thoughtSignature": "opaque-provider-signature"},
        ]
        result = self._generate(value)
        self.assertEqual(result.response.answer_blocks[0].text, "Évidence α")
        self.assertIn("opaque-provider-signature", result.provider_body_json)

    def test_optional_index_and_role_may_be_absent(self):
        value = _envelope()
        del value["candidates"][0]["index"]
        del value["candidates"][0]["content"]["role"]
        self.assertEqual(self._generate(value).response.status, "answered")

    def test_non_200_status_stops_without_parsing_or_echoing_body(self):
        for status in (201, 204, 301, 400, 401, 403, 408, 429, 500, 503):
            for method in (self._count, self._generate):
                with self.subTest(status=status, method=method.__name__), self.assertRaises(GeminiHTTPStatusError) as caught:
                    method(status_code=status, body_json="SENSITIVE BODY THAT IS NOT JSON")
                self.assertEqual(caught.exception.status_code, status)
                self.assertNotIn("SENSITIVE", str(caught.exception))
                self.assertIsNone(caught.exception.__cause__)

    def test_status_code_requires_integer_in_http_range(self):
        for status in (None, True, False, 200.0, "200", 99, 600):
            for method in (self._count, self._generate):
                with self.subTest(status=status, method=method.__name__), self.assertRaises(GeminiProviderResponseError):
                    method(status_code=status)

    def test_http_200_error_envelope_is_not_success(self):
        for method in (self._count, self._generate):
            with self.subTest(method=method.__name__), self.assertRaises(GeminiProviderResponseError):
                method({"error": {"code": 429, "message": "Quota exhausted"}})

    def test_provider_body_must_be_one_exact_json_object_string(self):
        for raw in (None, {}, b"{}", 1, True, "", " ", "{", "[]", "null", "true",
                    "{}{}", "```json\n{}\n```", "\ufeff{}", '{"x":/* comment */1}'):
            for method in (self._count, self._generate):
                with self.subTest(raw=repr(raw), method=method.__name__), self.assertRaises(GeminiProviderResponseError):
                    method(body_json=raw)

    def test_duplicate_keys_fail_at_root_and_nested_depth(self):
        raw = json.dumps(_envelope())
        for method, body in ((self._count, '{"totalTokens":1,"totalTokens":1}'),
                             (self._generate, raw[:-1] + ',"candidates":[]}'),
                             (self._generate, raw.replace('"finishReason": "STOP"', '"finishReason": "STOP", "finishReason": "STOP"'))):
            with self.subTest(body=body), self.assertRaisesRegex(GeminiProviderResponseError, "Duplicate"):
                method(body_json=body)

    def test_invalid_numbers_and_unicode_are_rejected_even_in_audit_fields(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e999", '"\\ud800"'):
            body = json.dumps(_envelope())[:-1] + ',"usageMetadata":{"extra":' + value + '}}'
            with self.subTest(value=value), self.assertRaises(GeminiProviderResponseError):
                self._generate(body_json=body)
        with self.assertRaises(GeminiProviderResponseError):
            self._count(body_json='{"totalTokens":1,"promptTokensDetails":["\ud800"]}')

    def test_missing_empty_or_multiple_candidates_fail_before_application_parser(self):
        for value in ({}, {"candidates": None}, {"candidates": {}}, {"candidates": []},
                      {"candidates": _envelope()["candidates"] * 2}):
            with self.subTest(value=value), patch(_MODULE + ".parse_gemini_response") as parser:
                with self.assertRaises(GeminiProviderResponseError):
                    self._generate(value)
                parser.assert_not_called()

    def test_missing_unknown_or_incomplete_finish_reason_blocks_valid_answer_json(self):
        for reason in (None, "", "MAX_TOKENS", "SAFETY", "RECITATION", "CONTINUATION",
                       "OTHER", "FINISH_REASON_UNSPECIFIED", "NEW_REASON", True, 1, []):
            value = _envelope()
            value["candidates"][0]["finishReason"] = reason
            with self.subTest(reason=reason), patch(_MODULE + ".parse_gemini_response") as parser:
                with self.assertRaises(GeminiProviderResponseError):
                    self._generate(value)
                parser.assert_not_called()
        value = _envelope()
        del value["candidates"][0]["finishReason"]
        with self.assertRaises(GeminiProviderResponseError):
            self._generate(value)

    def test_any_continuation_token_or_external_grounding_field_is_rejected(self):
        for field, extra in (("continuationToken", "opaque"), ("groundingMetadata", {}),
                             ("groundingAttributions", []), ("urlContextMetadata", {})):
            value = _envelope()
            value["candidates"][0][field] = extra
            with self.subTest(field=field), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_prompt_block_reason_rejects_even_a_contradictory_valid_candidate(self):
        for reason in ("SAFETY", "OTHER", "BLOCK_REASON_UNSPECIFIED", "NEW_REASON", None, "", 0):
            value = _envelope()
            value["promptFeedback"] = {"blockReason": reason}
            with self.subTest(reason=reason), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_safety_blocked_at_prompt_or_candidate_is_rejected(self):
        for owner in ("prompt", "candidate"):
            for blocked in (True, None, 1, "false", []):
                value = _envelope()
                container = value.setdefault("promptFeedback", {}) if owner == "prompt" else value["candidates"][0]
                container["safetyRatings"] = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "LOW", "blocked": blocked}]
                with self.subTest(owner=owner, blocked=blocked), self.assertRaises(GeminiProviderResponseError):
                    self._generate(value)

    def test_optional_unblocked_safety_ratings_are_retained_without_new_threshold(self):
        value = _envelope()
        value["promptFeedback"] = {"safetyRatings": [{"category": "CAT1", "probability": "LOW", "blocked": False}]}
        value["candidates"][0]["safetyRatings"] = [{"category": "CAT2", "probability": "HIGH"}]
        result = self._generate(value)
        self.assertEqual(json.loads(result.provider_body_json), value)

    def test_malformed_safety_collections_and_duplicate_categories_fail(self):
        rating = {"category": "CAT1", "probability": "LOW"}
        for ratings in (None, {}, [None], [{}], [rating, rating],
                        [{"category": [], "probability": "LOW"}],
                        [{"category": "CAT", "probability": 1}],
                        [{**rating, "futureField": True}]):
            value = _envelope()
            value["candidates"][0]["safetyRatings"] = ratings
            with self.subTest(ratings=ratings), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_role_and_index_types_are_checked_without_coercion(self):
        for field, values in (("index", (None, True, 0.0, "0", -1, 1)),
                              ("role", (None, "", "user", "MODEL", True, []))):
            for extra in values:
                value = _envelope()
                container = value["candidates"][0] if field == "index" else value["candidates"][0]["content"]
                container[field] = extra
                with self.subTest(field=field, extra=extra), self.assertRaises(GeminiProviderResponseError):
                    self._generate(value)

    def test_content_and_parts_must_have_supported_structure(self):
        for content in (None, [], {}, {"parts": []}, {"parts": None}, {"parts": "text"},
                        {"parts": [None]}, {"parts": [{}]}, {"parts": [{"text": None}]},
                        {"parts": [{"text": 1}]}, {"parts": [{"text": " \n"}]},
                        {"parts": [{"text": ""}]}):
            value = _envelope()
            value["candidates"][0]["content"] = content
            with self.subTest(content=content), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_tools_multimodal_parts_and_mixed_parts_are_not_silently_dropped(self):
        for extra in ({"functionCall": {}}, {"inlineData": {}}, {"fileData": {}},
                      {"executableCode": {}}, {"text": "text", "functionCall": {}},
                      {"text": "text", "partMetadata": {}}):
            value = _envelope()
            value["candidates"][0]["content"]["parts"].append(extra)
            with self.subTest(extra=extra), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_thought_output_or_wrong_metadata_types_are_rejected(self):
        for extra in ({"thought": True}, {"thought": 0}, {"thought": None}, {"thoughtSignature": 1}):
            value = _envelope()
            value["candidates"][0]["content"]["parts"][0].update(extra)
            with self.subTest(extra=extra), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_audit_metadata_is_preserved_without_model_or_accounting_inference(self):
        value = _envelope()
        value.update(modelVersion="provider-reported-revision", responseId="provider-response-id",
                     modelStatus={"modelStage": "STABLE"},
                     usageMetadata={"promptTokenCount": 120, "thoughtsTokenCount": 7, "candidatesTokenCount": 20, "totalTokenCount": 147})
        value["candidates"][0].update(tokenCount=20, avgLogprobs=-0.5,
            logprobsResult={}, finishMessage="", citationMetadata={"citationSources": []})
        result = self._generate(value)
        self.assertEqual(json.loads(result.provider_body_json), value)
        self.assertEqual(result.response.model, self.request.model)
        self.assertFalse(hasattr(result, "charges"))

    def test_audit_metadata_container_and_counter_types_are_checked(self):
        for changes in ({"modelVersion": ""}, {"responseId": 1}, {"usageMetadata": None},
                        {"modelStatus": []}, {"usageMetadata": {"totalTokenCount": True}},
                        {"usageMetadata": {"thoughtsTokenCount": -1}}):
            value = _envelope()
            value.update(changes)
            with self.subTest(changes=changes), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)
        for changes in ({"tokenCount": True}, {"avgLogprobs": True}, {"finishMessage": []},
                        {"citationMetadata": None}, {"logprobsResult": []}):
            value = _envelope()
            value["candidates"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(GeminiProviderResponseError):
                self._generate(value)

    def test_unrecognized_root_or_candidate_fields_fail_closed(self):
        for method, value in ((self._count, {"totalTokens": 1, "future": 1}),
                              (self._generate, {**_envelope(), "future": 1})):
            with self.subTest(method=method.__name__), self.assertRaises(GeminiProviderResponseError):
                method(value)
        value = _envelope()
        value["candidates"][0]["future"] = 1
        with self.assertRaises(GeminiProviderResponseError):
            self._generate(value)

    def test_invalid_application_json_or_citations_fail_with_cause(self):
        for text in ("not JSON", json.dumps(_payload(answer_blocks=[{"text": "A", "citation_ids": ["unknown"]}]))):
            with self.subTest(text=text), self.assertRaises(GeminiProviderResponseError) as caught:
                self._generate(_envelope(text))
            self.assertIsInstance(caught.exception.__cause__, GeminiResponseError)

    def test_stale_request_tag_and_wrong_request_type_fail(self):
        other = replace(self.request, question="Another question")
        for method in (self._count, self._generate):
            for tag in (None, 1, "", " ", other.request_id):
                with self.subTest(method=method.__name__, tag=tag), self.assertRaises(GeminiProviderResponseError):
                    method(request_id=tag)
            for request in (None, {}, self.context):
                with self.subTest(request=type(request)), self.assertRaises(GeminiProviderResponseError):
                    method(request=request)

    def test_generation_still_checks_authoritative_chunk_set(self):
        with self.assertRaises(GeminiProviderResponseError) as caught:
            self._generate(chunk_set=None)
        self.assertIsInstance(caught.exception.__cause__, GeminiResponseError)

    def test_request_mutation_during_body_decoding_is_rejected(self):
        for method in (self._count, self._generate):
            request = deepcopy(self.request)
            decode = json.loads
            def mutate(value, *args, **kwargs):
                if "object_pairs_hook" in kwargs:
                    object.__setattr__(request, "question", "Changed during response decoding")
                return decode(value, *args, **kwargs)
            with self.subTest(method=method.__name__), patch(_MODULE + ".json.loads", side_effect=mutate):
                with self.assertRaises(GeminiProviderResponseError):
                    method(request=request)

    def test_result_is_frozen_and_retained_body_must_match_application_answer(self):
        result = self._generate()
        with self.assertRaises(FrozenInstanceError):
            result.provider_body_json = "{}"
        with self.assertRaises(GeminiProviderResponseError):
            replace(result, provider_body_json=json.dumps(_envelope(json.dumps(_payload(limitation="Changed")))))
        with self.assertRaises(GeminiProviderResponseError):
            replace(result, response={})
        with self.assertRaises(GeminiProviderResponseError):
            replace(result, provider_body_json="[]")

    def test_parsing_does_not_mutate_request_evidence_or_provider_objects(self):
        envelope = _envelope()
        before = deepcopy((self.request, self.chunk_set, envelope))
        self._generate(envelope)
        self._count()
        self.assertEqual((self.request, self.chunk_set, envelope), before)

    def test_ordinary_errors_preserve_causes_and_baseexceptions_propagate(self):
        with self.assertRaises(GeminiProviderResponseError) as caught:
            self._count(body_json="{")
        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)
        for method in (self._count, self._generate):
            for error in (KeyboardInterrupt(), SystemExit()):
                with self.subTest(method=method.__name__, error=type(error)), patch(_MODULE + "._http_body", side_effect=error):
                    with self.assertRaises(type(error)):
                        method()

    def test_arguments_are_explicit_and_keyword_only(self):
        shared = {"status_code": 200, "body_json": "{}", "request_id": self.request.request_id, "request": self.request}
        for function, values in ((parse_count_tokens_response, shared),
                                 (parse_generate_content_response, {**shared, "chunk_set": self.chunk_set})):
            for key in values:
                with self.subTest(function=function.__name__, missing=key), self.assertRaises(TypeError):
                    function(**{name: value for name, value in values.items() if name != key})
            with self.assertRaises(TypeError):
                function(*values.values())


if __name__ == "__main__":
    unittest.main()
