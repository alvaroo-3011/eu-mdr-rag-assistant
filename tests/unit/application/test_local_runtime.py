"""Resource-lifetime tests for submission-scoped local execution."""

from __future__ import annotations

import httpx
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import src.application.local_runtime as local_runtime
from src.application.answer_policy import LocalInsufficientEvidence
from src.application.local_config import (
    HTTPBackendConfig,
    LocalAppConfig,
    LocalConfigError,
    RetrievalConfig,
)
from src.application.local_runtime import (
    LocalCredentialError,
    LocalRuntimeError,
    load_gemini_api_key,
    run_local_question,
)
from src.generation.gemini_request import GeminiGenerationConfig
from src.generation.gemini_transport import GeminiTransportConfig
from src.generation.request_budget import RequestBudget
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
from src.retrieval.embeddings import GTE_MODERNBERT_BASE
from src.retrieval.lexical_query import BM25Config


_FAKE_KEY = "FAKE_KEY_NOT_A_CREDENTIAL"


class _Resource:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self.close_count = 0

    def close(self):
        self.close_count += 1
        self.events.append(self.name + "_close")


class _TrackingMockTransport(httpx.MockTransport):
    def __init__(self, events):
        super().__init__(lambda _request: httpx.Response(500))
        self.events = events
        self.close_count = 0

    def close(self):
        self.close_count += 1
        self.events.append("http_close")
        super().close()


class _Adapter(_Resource):
    def __init__(self, transport, events):
        super().__init__("adapter", events)
        self.transport = transport

    def close(self):
        super().close()
        self.transport.close()


def _config(root: Path) -> LocalAppConfig:
    data = root / "data"
    data.mkdir(exist_ok=True)
    manifest = data / "corpus_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    index = data / "indexes" / "chroma"
    index.mkdir(parents=True, exist_ok=True)
    (index / "chroma.sqlite3").write_bytes(b"synthetic")
    return LocalAppConfig(
        profile_label="Synthetic",
        project_root=root.resolve(),
        manifest_path=manifest.resolve(),
        persistence_directory=index.resolve(),
        chroma_tenant="default_tenant",
        chroma_database="default_database",
        migration_hash_algorithm="md5",
        chunking_profile_name="baseline_frozen_18_source_v1",
        chunking_config=BASELINE_CHUNKING_CONFIG,
        embedding_profile_name="gte_modernbert_base_pinned_cpu",
        embedding_config=GTE_MODERNBERT_BASE,
        bm25_config=BM25Config(k1=1.2, b=0.75),
        retrieval_config=RetrievalConfig(
            dense_candidate_count=3,
            lexical_candidate_count=2,
            rank_constant=2.0,
            max_chunks=2,
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


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = _config(self.root)
        self.events: list[str] = []
        self.client = _Resource("chroma", self.events)
        self.transport = _TrackingMockTransport(self.events)
        self.adapter = _Adapter(self.transport, self.events)
        self.answer_calls = []

    def _chroma(self, **kwargs):
        self.events.append("chroma")
        self.chroma_kwargs = kwargs
        return self.client

    def _initialize(self, **kwargs):
        self.events.append("initialize")
        self.initialize_kwargs = kwargs
        return "chunk-set", "dense", "lexical"

    def _http(self, **kwargs):
        self.events.append("http")
        self.http_kwargs = kwargs
        return self.transport

    def _adapter(self, **kwargs):
        self.events.append("adapter")
        self.adapter_kwargs = kwargs
        return self.adapter

    def _answer(self, **kwargs):
        self.events.append("answer")
        self.answer_calls.append(kwargs)
        return "validated-run"

    def _run(self, **changes):
        values = {
            "config": self.config,
            "question": "  What does MDR require?  ",
            "api_key": _FAKE_KEY,
            "chroma_client_factory": self._chroma,
            "retrieval_initializer": self._initialize,
            "http_transport_factory": self._http,
            "adapter_factory": self._adapter,
            "answerer": self._answer,
            **changes,
        }
        return run_local_question(**values)

    def test_success_constructs_once_forwards_explicit_settings_and_closes(self):
        result = self._run()

        self.assertEqual(result, "validated-run")
        self.assertEqual(self.events, [
            "chroma", "initialize", "http", "adapter", "answer",
            "adapter_close", "http_close", "chroma_close",
        ])
        self.assertEqual(len(self.answer_calls), 1)
        call = self.answer_calls[0]
        self.assertEqual(call["question"], "  What does MDR require?  ")
        self.assertEqual(call["dense_candidate_count"], 3)
        self.assertEqual(call["lexical_candidate_count"], 2)
        self.assertEqual(call["rank_constant"], 2.0)
        self.assertEqual(call["max_chunks"], 2)
        self.assertEqual(self.http_kwargs, {"verify": True, "retries": 0})
        self.assertEqual(self.client.close_count, 1)
        self.assertEqual(self.adapter.close_count, 1)
        self.assertEqual(self.transport.close_count, 1)

    def test_local_non_answer_is_returned_while_all_resources_are_closed(self):
        result = self._run(answerer=lambda **_kwargs: LocalInsufficientEvidence())

        self.assertIs(type(result), LocalInsufficientEvidence)
        self.assertEqual(self.events, [
            "chroma", "initialize", "http", "adapter",
            "adapter_close", "http_close", "chroma_close",
        ])
        self.assertEqual(self.client.close_count, 1)
        self.assertEqual(self.adapter.close_count, 1)
        self.assertEqual(self.transport.close_count, 1)

    def test_default_http_transport_disables_environment_trust(self):
        sentinel = object()
        calls = []

        def sentinel_factory(**kwargs):
            calls.append(kwargs)
            return sentinel

        with patch.object(httpx, "HTTPTransport", sentinel_factory):
            result = local_runtime._default_http_transport_factory(
                verify=True,
                retries=0,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(
            calls,
            [{"verify": True, "trust_env": False, "retries": 0}],
        )

    def test_blank_question_invalid_paths_and_bad_key_stop_before_resources(self):
        for changes, expected in (
            ({"question": "  \t"}, LocalRuntimeError),
            ({"api_key": "bad key"}, LocalCredentialError),
        ):
            with self.subTest(changes=changes), self.assertRaises(expected):
                self._run(**changes)
            self.assertEqual(self.events, [])

        (self.config.persistence_directory / "chroma.sqlite3").unlink()
        with self.assertRaises(LocalConfigError):
            self._run()
        self.assertEqual(self.events, [])

    def test_partial_startup_and_answer_failures_close_every_successful_resource(self):
        def initialize_failure(**_kwargs):
            self.events.append("initialize")
            raise RuntimeError("synthetic init failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic init failure"):
            self._run(retrieval_initializer=initialize_failure)
        self.assertEqual(self.events, ["chroma", "initialize", "chroma_close"])
        self.assertEqual(self.client.close_count, 1)

        self.setUp()

        def adapter_failure(**_kwargs):
            self.events.append("adapter")
            raise RuntimeError("synthetic adapter failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic adapter failure"):
            self._run(adapter_factory=adapter_failure)
        self.assertEqual(self.events, [
            "chroma", "initialize", "http", "adapter", "http_close", "chroma_close",
        ])

        self.setUp()

        def answer_failure(**_kwargs):
            self.events.append("answer")
            raise RuntimeError("synthetic answer failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic answer failure"):
            self._run(answerer=answer_failure)
        self.assertEqual(self.events, [
            "chroma", "initialize", "http", "adapter", "answer",
            "adapter_close", "http_close", "chroma_close",
        ])

    def test_environment_key_lookup_is_explicit_and_secret_safe(self):
        self.assertEqual(load_gemini_api_key({"GEMINI_API_KEY": _FAKE_KEY}), _FAKE_KEY)
        for environment in ({}, {"GEMINI_API_KEY": "private value with spaces"}):
            with self.assertRaises(LocalCredentialError) as caught:
                load_gemini_api_key(environment)
            self.assertNotIn("private", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
