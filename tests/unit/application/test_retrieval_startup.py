"""Offline tests for explicit corpus and retrieval initialization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from inspect import Parameter, Signature, signature
from pathlib import Path
from typing import get_type_hints
from unittest.mock import Mock, call

import pytest

from src.application import retrieval_startup
from src.ingestion.chunk_models import ChunkSet, ChunkingConfig
from src.ingestion.models import (
    ManifestSchemaError,
    RegulatoryRegime,
    SourceSchemaError,
)
from src.retrieval.dense_index import (
    DenseIndexError,
    collection_metadata,
    dense_collection_name,
)
from src.retrieval.dense_query import DenseQueryError, DenseRetriever
from src.retrieval.embeddings import (
    EmbeddingModelConfig,
    TransformerEmbedder,
)
from src.retrieval.lexical_query import BM25Config, BM25Retriever
from tests.unit.ingestion.test_chunk_models import _config, _guidance_source
from tests.unit.retrieval.test_dense_index import (
    _FakeOpeningClient,
    _FakeStoredCollection,
    _embedding_config,
)
from tests.unit.retrieval.test_lexical_query import (
    _ivdr_source,
    _synthetic_chunk_set,
)


class _TrackingCollection(_FakeStoredCollection):
    """Expose forbidden write methods around the existing opening fake."""

    def __init__(self, metadata: object, record_count: int) -> None:
        super().__init__(metadata, record_count=record_count)
        self.write_calls: list[str] = []

    def _write(self, operation: str) -> None:
        self.write_calls.append(operation)
        raise AssertionError("Dense collection writes are forbidden.")

    def add(self, **values: object) -> None:
        del values
        self._write("add")

    def update(self, **values: object) -> None:
        del values
        self._write("update")

    def upsert(self, **values: object) -> None:
        del values
        self._write("upsert")

    def delete(self, **values: object) -> None:
        del values
        self._write("delete")


class _TrackingClient(_FakeOpeningClient):
    """Record ownership operations without importing or opening Chroma."""

    def __init__(
        self,
        name: str,
        collection: _TrackingCollection | None,
    ) -> None:
        super().__init__(name, collection)
        self.close_calls = 0
        self.write_calls: list[str] = []

    def close(self) -> None:
        self.close_calls += 1

    def delete_collection(self, name: str) -> None:
        self.write_calls.append(f"delete:{name}")
        raise AssertionError("Dense client writes are forbidden.")


def _dual_source():
    source = _guidance_source()
    return replace(
        source,
        document_id="test-dual-guidance",
        official_title="Test dual-applicable guidance",
        short_title="Dual guidance",
        local_path="data/rag_ready/test_dual_guidance.json",
        sha256="d" * 64,
        regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
        permitted_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
    )


def _mixed_chunk_set() -> ChunkSet:
    return _synthetic_chunk_set(
        (_guidance_source(), ("mdrterm evidence",)),
        (_dual_source(), ("dualterm evidence",)),
        (_ivdr_source(), ("ivdrterm evidence",)),
    )


def _tracking_client(
    chunk_set: ChunkSet,
    embedding_config: EmbeddingModelConfig,
) -> tuple[_TrackingClient, _TrackingCollection]:
    collection = _TrackingCollection(
        collection_metadata(chunk_set, embedding_config),
        record_count=len(chunk_set.chunks),
    )
    client = _TrackingClient(
        dense_collection_name(chunk_set, embedding_config),
        collection,
    )
    return client, collection


def _mock_loading_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    chunk_set: ChunkSet,
) -> tuple[Mock, Mock, Mock, object]:
    loaded_sources = object()
    manifest_loader = Mock(return_value=chunk_set.manifest)
    corpus_loader = Mock(return_value=loaded_sources)
    chunk_builder = Mock(return_value=chunk_set)
    monkeypatch.setattr(retrieval_startup, "load_manifest", manifest_loader)
    monkeypatch.setattr(retrieval_startup, "load_corpus", corpus_loader)
    monkeypatch.setattr(retrieval_startup, "build_chunk_set", chunk_builder)
    return manifest_loader, corpus_loader, chunk_builder, loaded_sources


def _forbid_model_loading(monkeypatch: pytest.MonkeyPatch) -> Mock:
    loader = Mock(side_effect=AssertionError("Model loading is forbidden."))
    monkeypatch.setattr(
        "src.retrieval.embeddings._load_transformers_backend",
        loader,
    )
    return loader


def test_contract_is_explicit_keyword_only_and_typed() -> None:
    function_signature = signature(retrieval_startup.initialize_retrieval)
    assert tuple(function_signature.parameters) == (
        "project_root",
        "manifest_path",
        "chunking_config",
        "embedding_config",
        "bm25_config",
        "dense_client",
    )
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY
        and parameter.default is Signature.empty
        for parameter in function_signature.parameters.values()
    )
    assert get_type_hints(retrieval_startup.initialize_retrieval) == {
        "project_root": str | Path,
        "manifest_path": str | Path,
        "chunking_config": ChunkingConfig,
        "embedding_config": EmbeddingModelConfig,
        "bm25_config": BM25Config,
        "dense_client": object,
        "return": tuple[ChunkSet, DenseRetriever, BM25Retriever],
    }


def test_stage_order_and_unchanged_argument_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    project_root = Path("synthetic-project-root")
    manifest_path = Path("synthetic-manifest.json")
    chunking_config = _config()
    embedding_config = _embedding_config()
    bm25_config = BM25Config(k1=1.2, b=0.75)
    dense_client = object()
    manifest = object()
    loaded_sources = object()
    chunk_set = object()
    lexical_retriever = object()
    embedder = object()
    dense_retriever = object()

    def stage(name: str, result: object):
        def run(*args: object, **kwargs: object) -> object:
            del args, kwargs
            events.append(name)
            return result

        return Mock(side_effect=run)

    manifest_loader = stage("manifest", manifest)
    corpus_loader = stage("corpus", loaded_sources)
    chunk_builder = stage("chunks", chunk_set)
    lexical_constructor = stage("lexical", lexical_retriever)
    embedder_constructor = stage("embedder", embedder)
    dense_constructor = stage("dense", dense_retriever)
    monkeypatch.setattr(retrieval_startup, "load_manifest", manifest_loader)
    monkeypatch.setattr(retrieval_startup, "load_corpus", corpus_loader)
    monkeypatch.setattr(retrieval_startup, "build_chunk_set", chunk_builder)
    monkeypatch.setattr(
        retrieval_startup,
        "BM25Retriever",
        lexical_constructor,
    )
    monkeypatch.setattr(
        retrieval_startup,
        "TransformerEmbedder",
        embedder_constructor,
    )
    monkeypatch.setattr(
        retrieval_startup,
        "DenseRetriever",
        dense_constructor,
    )

    result = retrieval_startup.initialize_retrieval(
        project_root=project_root,
        manifest_path=manifest_path,
        chunking_config=chunking_config,
        embedding_config=embedding_config,
        bm25_config=bm25_config,
        dense_client=dense_client,
    )

    assert events == [
        "manifest",
        "corpus",
        "chunks",
        "lexical",
        "embedder",
        "dense",
    ]
    assert result == (chunk_set, dense_retriever, lexical_retriever)
    assert manifest_loader.call_args == call(
        manifest_path,
        project_root=project_root,
        verify_files=True,
    )
    assert corpus_loader.call_args == call(
        manifest,
        project_root=project_root,
    )
    assert chunk_builder.call_args == call(
        manifest,
        loaded_sources,
        chunking_config,
    )
    assert lexical_constructor.call_args == call(
        chunk_set=chunk_set,
        config=bm25_config,
    )
    assert embedder_constructor.call_args == call(embedding_config)
    assert dense_constructor.call_args == call(
        client=dense_client,
        chunk_set=chunk_set,
        embedder=embedder,
    )


def test_real_constructors_open_once_without_loading_or_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_set = _mixed_chunk_set()
    chunk_snapshot = deepcopy(chunk_set)
    embedding_config = _embedding_config()
    bm25_config = BM25Config(k1=1.2, b=0.75)
    client, collection = _tracking_client(chunk_set, embedding_config)
    manifest_loader, corpus_loader, chunk_builder, loaded_sources = (
        _mock_loading_boundaries(monkeypatch, chunk_set)
    )
    model_loader = _forbid_model_loading(monkeypatch)
    project_root = Path("synthetic-project-root")
    manifest_path = Path("synthetic-manifest.json")

    returned_chunk_set, dense_retriever, lexical_retriever = (
        retrieval_startup.initialize_retrieval(
            project_root=project_root,
            manifest_path=manifest_path,
            chunking_config=chunk_set.chunking_config,
            embedding_config=embedding_config,
            bm25_config=bm25_config,
            dense_client=client,
        )
    )

    assert returned_chunk_set is chunk_set
    assert isinstance(dense_retriever, DenseRetriever)
    assert isinstance(lexical_retriever, BM25Retriever)
    assert dense_retriever._chunk_set_id == chunk_set.chunk_set_id
    assert lexical_retriever.chunk_set_id == chunk_set.chunk_set_id
    assert dense_retriever._embedder.config is embedding_config
    assert client.opened == [{
        "name": dense_collection_name(chunk_set, embedding_config),
        "embedding_function": None,
    }]
    assert collection.checks == ["metadata", "count"]
    assert client.created is False
    assert client.write_calls == []
    assert collection.write_calls == []
    assert client.close_calls == 0
    model_loader.assert_not_called()

    candidates = lexical_retriever.retrieve(
        "mdrterm dualterm ivdrterm",
        candidate_count=3,
    )
    assert {candidate.chunk_id for candidate in candidates} == {
        chunk_set.chunks[0].chunk_id,
        chunk_set.chunks[1].chunk_id,
    }
    assert all(candidate.metadata.permits_mdr for candidate in candidates)
    assert chunk_set.chunks[2].chunk_id not in {
        candidate.chunk_id for candidate in candidates
    }
    assert chunk_set == chunk_snapshot
    assert manifest_loader.call_args.kwargs["project_root"] is project_root
    assert corpus_loader.call_args.kwargs["project_root"] is project_root
    assert chunk_builder.call_args.args == (
        chunk_set.manifest,
        loaded_sources,
        chunk_set.chunking_config,
    )


@pytest.mark.parametrize(
    "failure_mode",
    ("missing", "metadata", "count"),
)
def test_dense_open_failures_preserve_existing_cause_behavior(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    chunk_set = _mixed_chunk_set()
    embedding_config = _embedding_config()
    client, collection = _tracking_client(chunk_set, embedding_config)
    if failure_mode == "missing":
        client.collection = None
    elif failure_mode == "metadata":
        collection.stored_metadata = {
            **collection.stored_metadata,
            "embedding_revision": "unexpected-revision",
        }
    else:
        collection.record_count += 1
    _mock_loading_boundaries(monkeypatch, chunk_set)
    model_loader = _forbid_model_loading(monkeypatch)

    with pytest.raises(DenseQueryError) as caught:
        retrieval_startup.initialize_retrieval(
            project_root=Path("synthetic-project-root"),
            manifest_path=Path("synthetic-manifest.json"),
            chunking_config=chunk_set.chunking_config,
            embedding_config=embedding_config,
            bm25_config=BM25Config(k1=1.2, b=0.75),
            dense_client=client,
        )

    assert isinstance(caught.value.__cause__, DenseIndexError)
    if failure_mode == "missing":
        assert isinstance(caught.value.__cause__.__cause__, LookupError)
    assert len(client.opened) == 1
    assert client.created is False
    assert client.write_calls == []
    assert collection.write_calls == []
    assert client.close_calls == 0
    model_loader.assert_not_called()


@pytest.mark.parametrize("failure_stage", ("manifest", "source"))
def test_loading_failure_stops_before_dense_access(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    manifest = object()
    manifest_failure = ManifestSchemaError("synthetic manifest failure")
    source_failure = SourceSchemaError("synthetic source failure")
    if failure_stage == "manifest":
        expected_failure = manifest_failure
        manifest_loader = Mock(side_effect=expected_failure)
        corpus_loader = Mock()
    else:
        expected_failure = source_failure
        manifest_loader = Mock(return_value=manifest)
        corpus_loader = Mock(side_effect=expected_failure)
    later_stages = {
        name: Mock()
        for name in (
            "build_chunk_set",
            "BM25Retriever",
            "TransformerEmbedder",
            "DenseRetriever",
        )
    }
    monkeypatch.setattr(retrieval_startup, "load_manifest", manifest_loader)
    monkeypatch.setattr(retrieval_startup, "load_corpus", corpus_loader)
    for name, mocked_stage in later_stages.items():
        monkeypatch.setattr(retrieval_startup, name, mocked_stage)
    client = _TrackingClient("unused", None)

    with pytest.raises(type(expected_failure)) as caught:
        retrieval_startup.initialize_retrieval(
            project_root=Path("synthetic-project-root"),
            manifest_path=Path("synthetic-manifest.json"),
            chunking_config=_config(),
            embedding_config=_embedding_config(),
            bm25_config=BM25Config(k1=1.2, b=0.75),
            dense_client=client,
        )

    assert caught.value is expected_failure
    if failure_stage == "manifest":
        corpus_loader.assert_not_called()
    else:
        corpus_loader.assert_called_once_with(
            manifest,
            project_root=Path("synthetic-project-root"),
        )
    for mocked_stage in later_stages.values():
        mocked_stage.assert_not_called()
    assert client.opened == []
    assert client.created is False
    assert client.write_calls == []
    assert client.close_calls == 0
