"""Native Streamlit through real runtime orchestration with synthetic resources.

Mock responses verify integration contracts, not model routing or semantic support.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import socket
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

import src.application.local_runtime as local_runtime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from tests.unit.application.test_local_runtime import _config
from tests.unit.generation.test_gemini_provider_response import _envelope
from tests.unit.ingestion.test_chunk_models import _chunk_set, _manifest
from tests.unit.retrieval.test_dense_results import _representative_chunk


MAIN = Path(__file__).resolve().parents[2] / "main.py"
FAKE_KEY = "FAKE_KEY_NOT_A_CREDENTIAL"
ANSWER = "<script>synthetic answer remains text</script>"
PRIVATE = "SYNTHETIC_PRIVATE_FAILURE_MUST_NOT_APPEAR"


class FakeClient:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FakeRetriever:
    def __init__(self, name, result, events, chunk_set_id=None):
        self.name = name
        self.result = result
        self.events = events
        self.chunk_set_id = chunk_set_id

    def retrieve(self, question, *, candidate_count):
        self.events.append((self.name, question, candidate_count))
        return self.result


class TrackingTransport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()


def test_runtime_to_streamlit_submission_failure_and_recovery(tmp_path):
    config = _config(tmp_path)
    original = _representative_chunk()
    chunk = replace(original, order=0, previous_chunk_id=None, next_chunk_id=None)
    chunks = _chunk_set(_manifest(chunk.source), (chunk,))
    metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
    dense = (DenseCandidate(1, chunk.chunk_id, chunk.text, metadata, 0.1),)
    events = []
    clients = []
    transports = []
    actual_run = local_runtime.run_local_question

    def forward(*, config, question, api_key):
        events.append(("runtime", question))
        attempt = len(clients) + 1

        def open_client(**_kwargs):
            events.append(("client_open", attempt))
            client = FakeClient()
            clients.append(client)
            return client

        def initialize(**kwargs):
            events.append(("initialize", attempt))
            assert kwargs["dense_client"] is clients[-1]
            return (
                chunks,
                FakeRetriever("dense", dense, events),
                FakeRetriever("lexical", (), events, chunks.chunk_set_id),
            )

        def http_factory(**kwargs):
            events.append(("transport_open", attempt))
            assert kwargs == {"verify": True, "retries": 0}

            def handle(request):
                operation = request.url.path.split(":")[-1]
                events.append((operation, attempt))
                if operation == "countTokens":
                    return httpx.Response(200, json={"totalTokens": 123})
                if attempt == 2:
                    return httpx.Response(503, content=PRIVATE.encode())
                payload = {
                    "status": "answered",
                    "answer_blocks": [{"text": ANSWER, "citation_ids": ["1"]}],
                    "limitation": None,
                    "clarification_question": None,
                }
                return httpx.Response(200, json=_envelope(json.dumps(payload)))

            transport = TrackingTransport(handle)
            transports.append(transport)
            return transport

        # AppTest creates its own loopback socket pair before the submission.
        # Block socket connections during the actual application runtime.
        with (
            patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden")),
        ):
            return actual_run(
                config=config, question=question, api_key=api_key,
                chroma_client_factory=open_client, retrieval_initializer=initialize,
                http_transport_factory=http_factory,
            )

    with (
        patch("src.application.local_config.load_local_app_config", return_value=config) as load_config,
        patch("src.application.local_runtime.load_gemini_api_key", return_value=FAKE_KEY) as load_key,
        patch("src.application.local_runtime.run_local_question", side_effect=forward) as runtime,
        patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")),
        patch("src.application.local_runtime.httpx.HTTPTransport", side_effect=AssertionError("real HTTP forbidden")),
    ):
        app = AppTest.from_file(MAIN, default_timeout=10).run()
        assert not app.exception
        assert runtime.call_count == load_config.call_count == load_key.call_count == 0
        assert events == []

        app.text_area[0].set_value("First synthetic MDR question?")
        app.form_submit_button[0].click().run()
        assert not app.exception
        assert [x.value for x in app.subheader] == ["Validated answer"]
        assert ANSWER + " [1]" in [x.value for x in app.text]
        assert f"Source identity: {chunk.source.source_identity}" in [x.value for x in app.text]
        assert f"Document: {chunk.source.document_id}" in [x.value for x in app.text]
        assert (f"[1] {chunk.source.short_title}, {chunk.source.version} — Article 15; Annex I"
                in [x.value for x in app.text])
        assert not any(str(x.value).startswith("Citations:") for x in app.caption)
        assert runtime.call_count == load_config.call_count == load_key.call_count == 1
        assert [c.close_count for c in clients] == [1]
        assert [t.close_count for t in transports] == [1]
        first_events = list(events)
        assert [event[0] for event in first_events] == [
            "runtime", "client_open", "initialize", "transport_open", "dense", "lexical",
            "countTokens", "generateContent",
        ]
        assert first_events[4:6] == [
            ("dense", "First synthetic MDR question?", 3),
            ("lexical", "First synthetic MDR question?", 2),
        ]

        app.run()
        assert not app.exception
        assert ANSWER + " [1]" in [x.value for x in app.text]
        assert events == first_events
        assert runtime.call_count == load_config.call_count == load_key.call_count == 1
        assert [c.close_count for c in clients] == [1]
        assert [t.close_count for t in transports] == [1]

        app.text_area[0].set_value("Failing synthetic MDR question?")
        app.form_submit_button[0].click().run()
        assert not app.exception
        assert [x.value for x in app.error] == [
            "The provider request or validated response failed; no partial answer is shown."
        ]
        displayed = "\n".join(str(x.value) for collection in
                              (app.text, app.caption, app.subheader, app.error) for x in collection)
        assert ANSWER not in displayed and PRIVATE not in displayed
        assert "Validated answer" not in displayed
        assert runtime.call_count == load_config.call_count == load_key.call_count == 2
        assert [c.close_count for c in clients] == [1, 1]
        assert [t.close_count for t in transports] == [1, 1]

        app.text_area[0].set_value("Recovery synthetic MDR question?")
        app.form_submit_button[0].click().run()
        assert not app.exception and not app.error
        assert ANSWER + " [1]" in [x.value for x in app.text]
        assert f"Source identity: {chunk.source.source_identity}" in [x.value for x in app.text]
        assert runtime.call_count == load_config.call_count == load_key.call_count == 3
        assert [c.close_count for c in clients] == [1, 1, 1]
        assert [t.close_count for t in transports] == [1, 1, 1]
        for stage in ("runtime", "client_open", "initialize", "transport_open", "dense", "lexical", "countTokens", "generateContent"):
            assert sum(event[0] == stage for event in events) == 3
        assert [event[1] for event in events if event[0] == "generateContent"] == [1, 2, 3]
