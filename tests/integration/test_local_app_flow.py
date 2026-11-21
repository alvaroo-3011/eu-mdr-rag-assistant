"""Offline local-runtime integration through real answer and response validators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.application.local_config import (
    HTTPBackendConfig,
    LocalAppConfig,
    RetrievalConfig,
)
from src.application.local_runtime import run_local_question
from src.application.presentation import build_answer_view
from src.generation.gemini_provider_response import (
    GeminiHTTPStatusError,
    GeminiProviderResponseError,
)
from src.generation.gemini_request import GeminiGenerationConfig
from src.generation.gemini_transport import GeminiTransportConfig, GeminiTransportRun
from src.generation.request_budget import RequestBudget, RequestBudgetExceededError
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.embeddings import GTE_MODERNBERT_BASE
from src.retrieval.lexical_query import BM25Config
from tests.unit.generation.test_gemini_provider_response import _envelope
from tests.unit.ingestion.test_chunk_models import _chunk_set, _manifest
from tests.unit.retrieval.test_dense_results import _representative_chunk


_FAKE_KEY = "FAKE_KEY_NOT_A_CREDENTIAL"


class _Client:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class _Retriever:
    def __init__(self, result, chunk_set_id=None):
        self.result = result
        self.chunk_set_id = chunk_set_id
        self.calls = []

    def retrieve(self, question, *, candidate_count):
        self.calls.append((question, candidate_count))
        return self.result


class _TrackingMockTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()


def _fingerprints(root: Path):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


class LocalAppFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / "data"
        data.mkdir()
        self.manifest_path = data / "corpus_manifest.json"
        self.manifest_path.write_text("{}", encoding="utf-8")
        self.index = data / "indexes" / "chroma"
        self.index.mkdir(parents=True)
        (self.index / "chroma.sqlite3").write_bytes(b"synthetic")

        original = _representative_chunk()
        chunk = type(original)(
            chunk_id=original.chunk_id,
            chunk_set_id=original.chunk_set_id,
            source=original.source,
            order=0,
            text=original.text,
            block_selections=original.block_selections,
            hierarchies=original.hierarchies,
            pages=original.pages,
            legal_locators=original.legal_locators,
            table_segments=original.table_segments,
            annotations=original.annotations,
            previous_chunk_id=None,
            next_chunk_id=None,
        )
        self.chunk_set = _chunk_set(_manifest(chunk.source), (chunk,))
        metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
        dense = (DenseCandidate(1, chunk.chunk_id, chunk.text, metadata, 0.1),)
        self.dense = _Retriever(dense)
        self.lexical = _Retriever((), self.chunk_set.chunk_set_id)
        self.client = _Client()
        self.operations = []
        self.transports = []
        self.payload = {
            "status": "answered",
            "answer_blocks": [{"text": "Synthetic validated answer.", "citation_ids": ["1"]}],
            "limitation": None,
            "clarification_question": None,
        }
        self.config = LocalAppConfig(
            profile_label="Synthetic integration",
            project_root=self.root.resolve(),
            manifest_path=self.manifest_path.resolve(),
            persistence_directory=self.index.resolve(),
            chroma_tenant="default_tenant",
            chroma_database="default_database",
            migration_hash_algorithm="md5",
            chunking_profile_name="baseline_frozen_18_source_v1",
            chunking_config=BASELINE_CHUNKING_CONFIG,
            embedding_profile_name="gte_modernbert_base_pinned_cpu",
            embedding_config=GTE_MODERNBERT_BASE,
            bm25_config=BM25Config(k1=1.2, b=0.75),
            retrieval_config=RetrievalConfig(
                dense_candidate_count=1,
                lexical_candidate_count=1,
                rank_constant=2.0,
                max_chunks=1,
            ),
            generation_config=GeminiGenerationConfig(
                model="models/gemini-synthetic",
                max_output_tokens=256,
                temperature=0.25,
                thinking_budget=32,
                thinking_level=None,
            ),
            budget=RequestBudget(input_token_limit=123, output_token_limit=256),
            transport_config=GeminiTransportConfig(
                api_version="v1beta", timeout_seconds=1.25, max_response_bytes=20000,
            ),
            http_backend_config=HTTPBackendConfig(
                verify_tls=True, trust_env=False, follow_redirects=False, retries=0,
            ),
        )
        self.enterContext(patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("Real sockets are forbidden."),
        ))

    def _chroma(self, **_kwargs):
        return self.client

    def _initialize(self, **kwargs):
        self.assertIs(kwargs["dense_client"], self.client)
        return self.chunk_set, self.dense, self.lexical

    def _handler(self, request):
        operation = request.url.path.split(":")[-1]
        self.operations.append(operation)
        if operation == "countTokens":
            return httpx.Response(200, json={"totalTokens": 123})
        return httpx.Response(200, json=_envelope(json.dumps(self.payload)))

    def _http(self, **kwargs):
        self.assertEqual(kwargs, {"verify": True, "retries": 0})
        transport = _TrackingMockTransport(self._handler)
        self.transports.append(transport)
        return transport

    def _run(self, **changes):
        return run_local_question(**{
            "config": self.config,
            "question": "What does the synthetic MDR evidence say?",
            "api_key": _FAKE_KEY,
            "chroma_client_factory": self._chroma,
            "retrieval_initializer": self._initialize,
            "http_transport_factory": self._http,
            **changes,
        })

    def test_real_orchestration_validators_mock_http_and_authoritative_citation(self):
        before = _fingerprints(self.root)
        result = self._run()
        view = build_answer_view(result)

        self.assertIsInstance(result, GeminiTransportRun)
        self.assertEqual(self.operations, ["countTokens", "generateContent"])
        self.assertEqual(self.dense.calls, [("What does the synthetic MDR evidence say?", 1)])
        self.assertEqual(self.lexical.calls, [("What does the synthetic MDR evidence say?", 1)])
        self.assertEqual(self.client.close_count, 1)
        self.assertEqual(self.transports[0].close_count, 1)
        citation = view.answer_blocks[0].citations[0]
        self.assertEqual(citation.source_identity, self.chunk_set.chunks[0].source.source_identity)
        self.assertIn("MDR", citation.permitted_regimes)
        self.assertIn("IVDR", citation.permitted_regimes)
        self.assertEqual(_fingerprints(self.root), before)

    def test_budget_provider_and_validation_failures_return_no_partial_or_retry(self):
        self.config = type(self.config)(**{
            field: getattr(self.config, field)
            for field in self.config.__dataclass_fields__
            if field != "budget"
        }, budget=RequestBudget(input_token_limit=122, output_token_limit=256))
        with self.assertRaises(RequestBudgetExceededError):
            self._run()
        self.assertEqual(self.operations, ["countTokens"])
        self.assertEqual(self.client.close_count, 1)

        self.setUp()

        def provider_failure(request):
            operation = request.url.path.split(":")[-1]
            self.operations.append(operation)
            if operation == "countTokens":
                return httpx.Response(200, json={"totalTokens": 123})
            return httpx.Response(503, content=b"private provider body")

        self._handler = provider_failure
        with self.assertRaises(GeminiHTTPStatusError) as caught:
            self._run()
        self.assertNotIn("private provider body", str(caught.exception))
        self.assertEqual(self.operations, ["countTokens", "generateContent"])

        self.setUp()
        self.payload["answer_blocks"][0]["citation_ids"] = ["999"]
        with self.assertRaises(GeminiProviderResponseError):
            self._run()
        self.assertEqual(self.operations, ["countTokens", "generateContent"])
        self.assertEqual(self.client.close_count, 1)


if __name__ == "__main__":
    unittest.main()
