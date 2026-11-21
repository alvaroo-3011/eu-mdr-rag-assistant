"""Offline integration tests for the importable question-answering flow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
import json
import unittest
from unittest.mock import patch

import httpx

from tests.unit.retrieval.test_lexical_query import (
    _forbid_production_imports,
    _synthetic_chunk_set,
)

with _forbid_production_imports():
    import src.application.question_answering as question_answering

from src.application.answer_policy import LocalInsufficientEvidence, LocalNeedsClarification
from src.generation.evidence_context import build_evidence_context
from src.generation.gemini_provider_response import (
    GeminiHTTPStatusError,
    GeminiProviderResponseError,
)
from src.generation.gemini_request import GeminiGenerationConfig, prepare_gemini_request
from src.generation.gemini_transport import (
    GeminiRESTTransport,
    GeminiTransportConfig,
    GeminiTransportRun,
    run_gemini_request,
)
from src.generation.request_budget import (
    RequestBudget,
    RequestBudgetError,
    RequestBudgetExceededError,
)
from src.ingestion.models import RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.evidence_selection import select_evidence
from src.retrieval.hybrid_query import HybridError, retrieve_hybrid
from src.retrieval.lexical_query import LexicalCandidate
from tests.unit.generation.test_gemini_provider_response import _envelope
from tests.unit.ingestion.test_chunk_models import _guidance_source


_DUMMY_KEY = "DUMMY_KEY_NOT_A_CREDENTIAL"


class _FakeRetriever:
    def __init__(self, name, result, events):
        self.name = name
        self.result = result
        self.events = events
        self.calls = []

    def retrieve(self, query, *, candidate_count):
        self.events.append(self.name)
        self.calls.append((query, candidate_count))
        return self.result


def _application_payload(status="answered"):
    payload = {
        "status": status,
        "answer_blocks": [],
        "limitation": None,
        "clarification_question": None,
    }
    if status == "answered":
        payload["answer_blocks"] = [{
            "text": "The supplied synthetic evidence supports this answer.",
            "citation_ids": ["1"],
        }]
    elif status == "needs_clarification":
        payload["clarification_question"] = "Which MDR device scope do you mean?"
    else:
        payload["limitation"] = "The selected evidence does not support an answer."
    return payload


class QuestionAnsweringTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.enterContext(patch(
            "socket.socket.connect",
            side_effect=AssertionError("Real network forbidden."),
        ))
        self.enterContext(patch(
            "httpx.HTTPTransport",
            side_effect=AssertionError("Implicit real transport forbidden."),
        ))

        dual_source = replace(
            _guidance_source(),
            regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
            permitted_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
        )
        self.chunk_set = _synthetic_chunk_set((
            dual_source,
            ("Synthetic evidence one.", "Synthetic evidence two.", "Synthetic evidence three."),
        ))
        metadata = tuple(
            deserialize_chunk_metadata(chunk_metadata(chunk))
            for chunk in self.chunk_set.chunks
        )
        dense = tuple(
            DenseCandidate(
                rank,
                self.chunk_set.chunks[index].chunk_id,
                self.chunk_set.chunks[index].text,
                metadata[index],
                0.9 - rank / 10,
            )
            for rank, index in enumerate((2, 0, 1), 1)
        )
        lexical = tuple(
            LexicalCandidate(
                rank,
                self.chunk_set.chunks[index].chunk_id,
                self.chunk_set.chunks[index].text,
                metadata[index],
                10.0 / rank,
            )
            for rank, index in enumerate((1, 0), 1)
        )
        self.events = []
        self.dense = _FakeRetriever("dense", dense, self.events)
        self.lexical = _FakeRetriever("lexical", lexical, self.events)
        self.lexical.chunk_set_id = self.chunk_set.chunk_set_id
        self.question = "  What does the supplied MDR evidence say?  "
        self.config = GeminiGenerationConfig(
            model="models/gemini-synthetic",
            max_output_tokens=256,
            temperature=0.25,
            thinking_budget=32,
            thinking_level=None,
        )
        self.budget = RequestBudget(input_token_limit=123, output_token_limit=256)
        self.http_calls = []

    def _adapter(self, *, payload=None, handler=None):
        application_payload = _application_payload() if payload is None else payload

        def handle(wire):
            operation = wire.url.path.split(":")[-1]
            self.http_calls.append(operation)
            self.events.append(operation)
            if handler is not None:
                return handler(wire)
            if operation == "countTokens":
                return httpx.Response(200, json={"totalTokens": 123})
            return httpx.Response(200, json=_envelope(json.dumps(application_payload)))

        adapter = GeminiRESTTransport(
            api_key=_DUMMY_KEY,
            config=GeminiTransportConfig(
                api_version="v1beta",
                timeout_seconds=1.25,
                max_response_bytes=20000,
            ),
            transport=httpx.MockTransport(handle),
        )
        self.addCleanup(adapter.close)
        return adapter

    def _answer(self, adapter, **changes):
        return question_answering.answer_question(**{
            "question": self.question,
            "chunk_set": self.chunk_set,
            "dense_retriever": self.dense,
            "lexical_retriever": self.lexical,
            "dense_candidate_count": 3,
            "lexical_candidate_count": 2,
            "rank_constant": 2.0,
            "max_chunks": 2,
            "generation_config": self.config,
            "budget": self.budget,
            "adapter": adapter,
            **changes,
        })

    def test_successful_stage_order_full_union_and_trusted_citation(self):
        observed = {}

        def hybrid(*args, **kwargs):
            self.events.append("retrieve_hybrid")
            observed["hybrid_question"] = args[0]
            observed["hybrid_chunk_set"] = kwargs["chunk_set"]
            result = retrieve_hybrid(*args, **kwargs)
            observed["fused"] = result
            return result

        def selection(**kwargs):
            self.events.append("select_evidence")
            observed["selection_input"] = kwargs["fused_candidates"]
            result = select_evidence(**kwargs)
            observed["selection"] = result
            return result

        def context(**kwargs):
            self.events.append("build_evidence_context")
            return build_evidence_context(**kwargs)

        def request(**kwargs):
            self.events.append("prepare_gemini_request")
            observed["request_question"] = kwargs["question"]
            observed["request_chunk_set"] = kwargs["chunk_set"]
            return prepare_gemini_request(**kwargs)

        def run(**kwargs):
            self.events.append("run_gemini_request")
            observed["run_chunk_set"] = kwargs["chunk_set"]
            return run_gemini_request(**kwargs)

        with (
            patch.object(question_answering, "retrieve_hybrid", side_effect=hybrid),
            patch.object(question_answering, "select_evidence", side_effect=selection),
            patch.object(question_answering, "build_evidence_context", side_effect=context),
            patch.object(question_answering, "prepare_gemini_request", side_effect=request),
            patch.object(question_answering, "run_gemini_request", side_effect=run),
        ):
            result = self._answer(self._adapter())

        self.assertIsInstance(result, GeminiTransportRun)
        self.assertEqual(self.events, [
            "retrieve_hybrid", "dense", "lexical", "select_evidence",
            "build_evidence_context", "prepare_gemini_request",
            "run_gemini_request", "countTokens", "generateContent",
        ])
        self.assertEqual(self.dense.calls, [(self.question, 3)])
        self.assertEqual(self.lexical.calls, [(self.question, 2)])
        self.assertIs(observed["selection_input"], observed["fused"])
        self.assertEqual(len(observed["fused"]), 3)
        self.assertIs(observed["hybrid_question"], self.question)
        self.assertIs(observed["request_question"], self.question)
        for name in ("hybrid_chunk_set", "request_chunk_set", "run_chunk_set"):
            self.assertIs(observed[name], self.chunk_set)
        citation = result.generation.response.answer_blocks[0].citations[0]
        self.assertIs(citation.chunk, observed["selection"].selected[0].chunk)
        self.assertIn(RegulatoryRegime.MDR, citation.chunk.source.permitted_regimes)
        self.assertIn(RegulatoryRegime.IVDR, citation.chunk.source.permitted_regimes)
        parameters = inspect.signature(question_answering.answer_question).parameters
        self.assertTrue(all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values()))
        self.assertTrue(all(p.default is inspect.Parameter.empty for p in parameters.values()))

    def test_empty_dense_output_stops_before_http(self):
        self.dense.result = ()
        adapter = self._adapter()
        with self.assertRaises(HybridError):
            self._answer(adapter)
        self.assertEqual(self.http_calls, [])
        self.assertEqual(len(self.dense.calls), 1)
        self.assertEqual(self.lexical.calls, [])

    def test_valid_empty_selection_returns_locally_before_context_or_provider(self):
        empty = select_evidence(
            chunk_set=self.chunk_set,
            fused_candidates=(),
            max_chunks=2,
        )
        with (
            patch.object(question_answering, "select_evidence", return_value=empty),
            patch.object(
                question_answering,
                "build_evidence_context",
                side_effect=AssertionError("context must not be built"),
            ) as context,
            patch.object(
                question_answering,
                "prepare_gemini_request",
                side_effect=AssertionError("request must not be prepared"),
            ) as request,
            patch.object(
                question_answering,
                "run_gemini_request",
                side_effect=AssertionError("provider boundary must not run"),
            ) as provider,
        ):
            result = self._answer(self._adapter())

        self.assertIs(type(result), LocalInsufficientEvidence)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.limitation,
            "No MDR evidence was selected and no provider request was made.",
        )
        context.assert_not_called()
        request.assert_not_called()
        provider.assert_not_called()
        self.assertEqual(self.http_calls, [])

    def test_undescribed_mdr_device_requires_one_local_clarification(self):
        for question in (
            "What material must my device be made from to comply with the MDR?",
            "Which materials should our device be made of to meet EU MDR?",
            "What risk class is my device under the MDR?",
        ):
            with self.subTest(question=question):
                self.events.clear()
                self.http_calls.clear()
                result = self._answer(self._adapter(), question=question)
                self.assertIs(type(result), LocalNeedsClarification)
                self.assertEqual(result.status, "needs_clarification")
                self.assertEqual(result.clarification_question,
                                 "What is your device and what is its intended purpose?")
                self.assertEqual(self.events, [])
                self.assertEqual(self.http_calls, [])

    def test_general_mdr_question_keeps_normal_evidence_and_provider_path(self):
        result = self._answer(self._adapter(), question="What does the MDR require for manufacturers?")
        self.assertIsInstance(result, GeminiTransportRun)
        self.assertEqual(result.generation.response.status, "answered")
        self.assertEqual(self.http_calls, ["countTokens", "generateContent"])
        self.assertEqual(self.events[:2], ["dense", "lexical"])

    def test_clear_question_with_insufficient_evidence_keeps_existing_result(self):
        result = self._answer(
            self._adapter(payload=_application_payload("insufficient_evidence")),
            question="What does the unpublished report for my specific device conclude?",
        )
        self.assertEqual(result.generation.response.status, "insufficient_evidence")
        self.assertEqual(self.http_calls, ["countTokens", "generateContent"])

    def test_explicit_ivdr_question_keeps_existing_scope_path(self):
        result = self._answer(
            self._adapter(payload=_application_payload("out_of_scope")),
            question="What material must my device be made from to comply with the IVDR?",
        )
        self.assertEqual(result.generation.response.status, "out_of_scope")
        self.assertEqual(self.http_calls, ["countTokens", "generateContent"])

    def test_mocked_comparison_and_separable_scope_responses_use_existing_statuses(self):
        cases = (
            ("Compare MDR and IVDR requirements", "out_of_scope"),
            ("Explain MDR duties and separately explain IVDR duties", "needs_clarification"),
        )
        for question, status in cases:
            with self.subTest(question=question):
                self.http_calls.clear()
                result = self._answer(self._adapter(payload=_application_payload(status)), question=question)
                self.assertEqual(result.generation.response.status, status)
                self.assertEqual(self.http_calls, ["countTokens", "generateContent"])

    def test_mixed_scope_or_added_device_description_does_not_short_circuit(self):
        for question in (
            "What material must my device be made from to comply with the MDR and IVDR?",
            "What material must my device be made from to comply with the MDR? It is a contact lens.",
        ):
            with self.subTest(question=question):
                self.http_calls.clear()
                result = self._answer(self._adapter(), question=question)
                self.assertIsInstance(result, GeminiTransportRun)
                self.assertEqual(self.http_calls, ["countTokens", "generateContent"])

    def test_output_budget_rejection_makes_no_http_call(self):
        adapter = self._adapter()
        budget = RequestBudget(input_token_limit=123, output_token_limit=255)
        with self.assertRaises(RequestBudgetExceededError):
            self._answer(adapter, budget=budget)
        self.assertEqual(self.http_calls, [])

    def test_input_budget_rejection_counts_once_and_never_generates(self):
        adapter = self._adapter()
        budget = RequestBudget(input_token_limit=122, output_token_limit=256)
        with self.assertRaises(RequestBudgetExceededError):
            self._answer(adapter, budget=budget)
        self.assertEqual(self.http_calls, ["countTokens"])

    def test_count_and_generation_failures_stop_without_retry(self):
        def count_failure(_wire):
            return httpx.Response(429, content=b"private quota body")

        with self.assertRaises(RequestBudgetError) as count_error:
            self._answer(self._adapter(handler=count_failure))
        self.assertIsInstance(count_error.exception.__cause__, GeminiHTTPStatusError)
        self.assertNotIn("private quota body", str(count_error.exception))
        self.assertEqual(self.http_calls, ["countTokens"])

        self.http_calls.clear()

        def generation_failure(wire):
            if wire.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 123})
            return httpx.Response(503, content=b"private server body")

        with self.assertRaises(GeminiHTTPStatusError) as generation_error:
            self._answer(self._adapter(handler=generation_failure))
        self.assertEqual(generation_error.exception.status_code, 503)
        self.assertNotIn("private server body", str(generation_error.exception))
        self.assertEqual(self.http_calls, ["countTokens", "generateContent"])

    def test_all_four_application_statuses_pass_through(self):
        for status in (
            "answered", "insufficient_evidence", "needs_clarification", "out_of_scope",
        ):
            self.http_calls.clear()
            with self.subTest(status=status):
                result = self._answer(self._adapter(payload=_application_payload(status)))
                self.assertEqual(result.generation.response.status, status)
                self.assertEqual(self.http_calls, ["countTokens", "generateContent"])

    def test_invalid_citation_rejects_without_partial_answer(self):
        payload = _application_payload()
        payload["answer_blocks"][0]["citation_ids"] = ["999"]
        with self.assertRaises(GeminiProviderResponseError) as caught:
            self._answer(self._adapter(payload=payload))
        self.assertEqual(self.http_calls, ["countTokens", "generateContent"])
        self.assertIsNotNone(caught.exception.__cause__)

    def test_caller_owns_open_adapter_and_inputs_remain_unchanged(self):
        before = deepcopy((
            self.question,
            self.chunk_set,
            self.dense.result,
            self.lexical.result,
            self.config,
            self.budget,
        ))
        adapter = self._adapter()
        self._answer(adapter)
        self.assertFalse(adapter._client.is_closed)
        self.assertEqual((
            self.question,
            self.chunk_set,
            self.dense.result,
            self.lexical.result,
            self.config,
            self.budget,
        ), before)


if __name__ == "__main__":
    unittest.main()
