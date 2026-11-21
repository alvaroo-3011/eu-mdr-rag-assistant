"""Synthetic tests for identity-bound Chroma dense-index construction."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from src.ingestion.chunk_models import (
    ChunkSet,
    compute_chunk_set_id,
    compute_chunking_config_id,
    compute_corpus_identity,
)
from src.ingestion.models import (
    CorpusManifest,
    HierarchyEntry,
    LegalLocatorPart,
    PageIdentity,
)
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
from src.retrieval.dense_index import (
    DenseIndexError,
    build_dense_index,
    chunk_metadata,
    collection_metadata,
    dense_collection_name,
    open_dense_collection,
)
from src.retrieval.embeddings import EmbeddingModelConfig, GTE_MODERNBERT_BASE
from tests.unit.ingestion.test_chunk_models import (
    _chunk,
    _config,
    _guidance_source,
    _manifest,
)


def _chunk_set() -> ChunkSet:
    """Return one complete synthetic ChunkSet with citation metadata."""

    source = _guidance_source()
    manifest: CorpusManifest = _manifest(source)
    config = _config()
    corpus_identity = compute_corpus_identity(manifest)
    config_id = compute_chunking_config_id(config)
    chunk_set_id = compute_chunk_set_id(corpus_identity, config_id)
    hierarchy = (
        HierarchyEntry(
            level=1,
            kind="section",
            block_id=f"{source.source_identity}::section-1",
            title="Scope",
        ),
    )
    locator = (
        LegalLocatorPart(kind="section", value="1"),
    )
    chunk = _chunk(
        chunk_set_id,
        source,
        text="Synthetic regulatory evidence.",
        hierarchies=(hierarchy,),
        pages=(PageIdentity(source_index=2, pdf_page_number=3),),
        legal_locators=(locator,),
    )

    return ChunkSet(
        schema_version="1.0",
        manifest=manifest,
        corpus_identity=corpus_identity,
        chunking_config=config,
        chunking_config_id=config_id,
        chunk_set_id=chunk_set_id,
        chunks=(chunk,),
    )


def _embedding_config() -> EmbeddingModelConfig:
    """Return a tiny synthetic embedding identity."""

    return EmbeddingModelConfig(
        model_id="test/model",
        revision="abc123",
        dimension=3,
        max_sequence_length=32,
        device="cpu",
        normalize_embeddings=True,
    )


class _FakeCollection:
    def __init__(
        self,
        metadata: dict[str, object],
        *,
        fail_add: bool = False,
        count_offset: int = 0,
    ) -> None:
        self.metadata = metadata
        self.fail_add = fail_add
        self.count_offset = count_offset
        self.records: list[dict[str, object]] = []

    def add(self, **values: object) -> None:
        if self.fail_add:
            raise RuntimeError("synthetic add failure")

        ids = values["ids"]
        embeddings = values["embeddings"]
        documents = values["documents"]
        metadatas = values["metadatas"]
        for index, chunk_id in enumerate(ids):
            self.records.append(
                {
                    "id": chunk_id,
                    "embedding": embeddings[index].copy(),
                    "document": documents[index],
                    "metadata": metadatas[index],
                }
            )

    def count(self) -> int:
        return len(self.records) + self.count_offset


class _FakeClient:
    def __init__(
        self,
        *,
        fail_create: bool = False,
        fail_add: bool = False,
        count_offset: int = 0,
        max_batch_size: int = 100,
    ) -> None:
        self.fail_create = fail_create
        self.fail_add = fail_add
        self.count_offset = count_offset
        self.max_batch_size = max_batch_size
        self.create_values: dict[str, object] | None = None
        self.collection: _FakeCollection | None = None
        self.deleted: list[str] = []

    def create_collection(self, **values: object) -> _FakeCollection:
        if self.fail_create:
            raise ValueError("collection already exists")
        self.create_values = values
        self.collection = _FakeCollection(
            values["metadata"],
            fail_add=self.fail_add,
            count_offset=self.count_offset,
        )
        return self.collection

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def get_max_batch_size(self) -> int:
        return self.max_batch_size


class DenseIndexTests(unittest.TestCase):
    """Validate metadata, complete writes, and fail-closed rollback."""

    def test_chunk_metadata_preserves_citation_and_filter_fields(self) -> None:
        chunk = _chunk_set().chunks[0]

        metadata = chunk_metadata(chunk)

        self.assertEqual(metadata["chunk_id"], chunk.chunk_id)
        self.assertEqual(metadata["document_id"], "test-guidance")
        self.assertEqual(metadata["version"], "1.0")
        self.assertTrue(metadata["permits_mdr"])
        self.assertFalse(metadata["permits_ivdr"])
        self.assertEqual(
            json.loads(metadata["pages_json"]),
            [{"pdf_page_number": 3, "source_index": 2}],
        )
        self.assertEqual(
            json.loads(metadata["legal_locators_json"]),
            [[{"kind": "section", "value": "1"}]],
        )
        self.assertEqual(
            json.loads(metadata["hierarchies_json"])[0][0]["title"],
            "Scope",
        )

    def test_build_creates_fresh_cosine_collection(self) -> None:
        chunk_set = _chunk_set()
        config = _embedding_config()
        client = _FakeClient()
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        descriptor = build_dense_index(
            client=client,
            collection_name="test-dense-index",
            chunk_set=chunk_set,
            embedding_config=config,
            embeddings=vectors,
        )

        self.assertEqual(descriptor.record_count, 1)
        self.assertEqual(descriptor.chunk_set_id, chunk_set.chunk_set_id)
        self.assertEqual(descriptor.embedding_dimension, 3)
        self.assertEqual(
            client.create_values["configuration"],
            {"hnsw": {"space": "cosine"}},
        )
        self.assertIsNone(client.create_values["embedding_function"])
        self.assertFalse(client.create_values["get_or_create"])
        self.assertEqual(
            client.create_values["metadata"]["chunk_set_id"],
            chunk_set.chunk_set_id,
        )
        self.assertEqual(
            client.create_values["metadata"]["embedding_revision"],
            "abc123",
        )
        self.assertEqual(
            client.collection.records[0]["document"],
            "Synthetic regulatory evidence.",
        )
        self.assertEqual(client.deleted, [])

    def test_embedding_shape_and_values_fail_before_collection_creation(
        self,
    ) -> None:
        chunk_set = _chunk_set()
        config = _embedding_config()

        invalid_vectors = (
            np.zeros((1, 2), dtype=np.float32),
            np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
            "not a matrix",
        )
        for vectors in invalid_vectors:
            with self.subTest(vectors=repr(vectors)):
                client = _FakeClient()
                with self.assertRaises(DenseIndexError):
                    build_dense_index(
                        client=client,
                        collection_name="test-dense-index",
                        chunk_set=chunk_set,
                        embedding_config=config,
                        embeddings=vectors,
                    )
                self.assertIsNone(client.create_values)

    def test_existing_collection_collision_fails_without_deletion(self) -> None:
        client = _FakeClient(fail_create=True)

        with self.assertRaises(DenseIndexError):
            build_dense_index(
                client=client,
                collection_name="existing-index",
                chunk_set=_chunk_set(),
                embedding_config=_embedding_config(),
                embeddings=np.array([[1.0, 0.0, 0.0]]),
            )

        self.assertEqual(client.deleted, [])

    def test_partial_add_failure_rolls_back_new_collection(self) -> None:
        client = _FakeClient(fail_add=True)

        with self.assertRaises(DenseIndexError):
            build_dense_index(
                client=client,
                collection_name="failed-index",
                chunk_set=_chunk_set(),
                embedding_config=_embedding_config(),
                embeddings=np.array([[1.0, 0.0, 0.0]]),
            )

        self.assertEqual(client.deleted, ["failed-index"])

    def test_incomplete_count_rolls_back_new_collection(self) -> None:
        client = _FakeClient(count_offset=-1)

        with self.assertRaisesRegex(DenseIndexError, "incomplete"):
            build_dense_index(
                client=client,
                collection_name="incomplete-index",
                chunk_set=_chunk_set(),
                embedding_config=_embedding_config(),
                embeddings=np.array([[1.0, 0.0, 0.0]]),
            )

        self.assertEqual(client.deleted, ["incomplete-index"])

    def test_invalid_arguments_and_batch_size_are_rejected(self) -> None:
        chunk_set = _chunk_set()
        config = _embedding_config()
        vectors = np.array([[1.0, 0.0, 0.0]])

        with self.assertRaises(DenseIndexError):
            build_dense_index(
                client=_FakeClient(),
                collection_name=" ",
                chunk_set=chunk_set,
                embedding_config=config,
                embeddings=vectors,
            )

        client = _FakeClient(max_batch_size=0)
        with self.assertRaises(DenseIndexError):
            build_dense_index(
                client=client,
                collection_name="invalid-batch-index",
                chunk_set=chunk_set,
                embedding_config=config,
                embeddings=vectors,
            )
        self.assertEqual(client.deleted, ["invalid-batch-index"])


class _FakeStoredCollection:
    """Existing records with observable validation and no storage engine."""

    def __init__(self, metadata: object, record_count: int = 1) -> None:
        self.stored_metadata = metadata
        self.record_count = record_count
        self.checks: list[str] = []
        self.fail_count = False

    @property
    def metadata(self) -> object:
        self.checks.append("metadata")
        return self.stored_metadata

    def count(self) -> int:
        self.checks.append("count")
        if self.fail_count:
            raise RuntimeError("synthetic count failure")
        return self.record_count


class _FakeOpeningClient:
    def __init__(
        self,
        name: str,
        collection: _FakeStoredCollection | None,
    ) -> None:
        self.name = name
        self.collection = collection
        self.opened: list[dict[str, object]] = []
        self.created = False

    def get_collection(
        self,
        *,
        name: str,
        embedding_function: None,
    ) -> _FakeStoredCollection:
        self.opened.append(
            {"name": name, "embedding_function": embedding_function}
        )
        if name != self.name or self.collection is None:
            raise LookupError("synthetic missing collection")
        return self.collection

    def create_collection(self, **values: object) -> None:
        self.created = True
        raise AssertionError("Opening must never create a collection.")

    def get_or_create_collection(self, **values: object) -> None:
        self.created = True
        raise AssertionError("Opening must never create a collection.")


class DenseCollectionOpeningTests(unittest.TestCase):
    """Validate shared naming and fail-closed opening using only fakes."""

    def setUp(self) -> None:
        self.chunk_set = _chunk_set()
        self.config = _embedding_config()
        self.name = dense_collection_name(self.chunk_set, self.config)
        self.metadata = collection_metadata(self.chunk_set, self.config)
        self.collection = _FakeStoredCollection(self.metadata.copy())
        self.client = _FakeOpeningClient(self.name, self.collection)

    def _open(self) -> _FakeStoredCollection:
        return open_dense_collection(
            client=self.client,
            chunk_set=self.chunk_set,
            embedding_config=self.config,
        )

    def test_frozen_baseline_preserves_expected_collection_name(self) -> None:
        # Snapshot of the approved corpus identity; no corpus files, model,
        # or database are needed to reproduce the collection-name contract.
        corpus_identity = (
            "corpus:sha256:"
            "7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab"
        )
        config_id = compute_chunking_config_id(BASELINE_CHUNKING_CONFIG)
        self.assertEqual(
            config_id,
            "chunking-config:sha256:"
            "84eab6ff2f1bbd9cfb55f20cd63ac21d0cf29125d7e521fa7c9d7fa94e3ff634",
        )
        identity = Mock(spec=ChunkSet)
        identity.chunk_set_id = compute_chunk_set_id(corpus_identity, config_id)

        self.assertEqual(
            dense_collection_name(identity, GTE_MODERNBERT_BASE),
            "mdr-dense-623682d019d735ec03ccc2b4",
        )

    def test_collection_name_uses_every_index_defining_identity(self) -> None:
        for changes in (
            {"model_id": "another/model"},
            {"revision": "another-revision"},
            {"dimension": 4},
        ):
            with self.subTest(changes=changes):
                self.assertNotEqual(
                    dense_collection_name(
                        self.chunk_set, replace(self.config, **changes)
                    ),
                    self.name,
                )
        other_identity = Mock(spec=ChunkSet, chunk_set_id="another-chunk-set")
        self.assertNotEqual(
            dense_collection_name(other_identity, self.config), self.name
        )
        self.assertEqual(
            dense_collection_name(self.chunk_set, self.config), self.name
        )

    def test_opens_exact_expected_collection_without_creating(self) -> None:
        opened = self._open()

        self.assertIs(opened, self.collection)
        self.assertEqual(
            self.client.opened,
            [{"name": self.name, "embedding_function": None}],
        )
        self.assertFalse(self.client.created)
        self.assertEqual(self.collection.checks, ["metadata", "count"])

    def test_missing_collection_raises_without_creating(self) -> None:
        self.client.collection = None

        with self.assertRaisesRegex(DenseIndexError, "missing or unavailable"):
            self._open()
        self.assertFalse(self.client.created)

    def test_missing_or_malformed_metadata_is_rejected_before_count(self) -> None:
        for metadata in (None, [], "not metadata", {}):
            with self.subTest(metadata=metadata):
                self.collection.stored_metadata = metadata
                self.collection.checks.clear()
                with self.assertRaisesRegex(DenseIndexError, "metadata"):
                    self._open()
                self.assertEqual(self.collection.checks, ["metadata"])
                self.assertFalse(self.client.created)

    def test_missing_or_extra_metadata_fields_are_rejected(self) -> None:
        missing_field = self.metadata.copy()
        del missing_field["embedding_revision"]
        for metadata in (missing_field, {**self.metadata, "unexpected": "value"}):
            with self.subTest(metadata=metadata):
                self.collection.stored_metadata = metadata
                with self.assertRaisesRegex(DenseIndexError, "metadata keys"):
                    self._open()

    def _assert_metadata_mismatch(self, field: str, value: object) -> None:
        self.collection.stored_metadata = {**self.metadata, field: value}
        with self.assertRaisesRegex(DenseIndexError, f"field={field}"):
            self._open()
        self.assertEqual(self.collection.checks, ["metadata"])
        self.assertFalse(self.client.created)

    def test_corpus_identity_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("corpus_identity", "stale-corpus")

    def test_chunk_set_identity_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("chunk_set_id", "stale-chunk-set")

    def test_chunking_configuration_identity_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("chunking_config_id", "stale-config")

    def test_embedding_model_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("embedding_model_id", "another/model")

    def test_embedding_revision_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("embedding_revision", "another-revision")

    def test_dimension_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("embedding_dimension", 4)

    def test_distance_metric_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("distance_metric", "l2")

    def test_schema_version_mismatch_is_rejected(self) -> None:
        self._assert_metadata_mismatch("schema_version", "2.0")

    def test_metadata_value_types_must_match_exactly(self) -> None:
        for value in (3.0, "3", True, None, [3]):
            with self.subTest(value=value):
                self.collection.checks.clear()
                self._assert_metadata_mismatch("embedding_dimension", value)

    def test_record_count_mismatch_prevents_return(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count):
                self.collection.record_count = count
                self.collection.checks.clear()
                with self.assertRaisesRegex(DenseIndexError, "count mismatch"):
                    self._open()
                self.assertEqual(self.collection.checks, ["metadata", "count"])
                self.assertFalse(self.client.created)

    def test_malformed_record_count_is_rejected(self) -> None:
        for count in (True, 1.0, "1", None):
            with self.subTest(count=count):
                self.collection.record_count = count
                with self.assertRaisesRegex(DenseIndexError, "invalid record count"):
                    self._open()

    def test_count_failure_is_wrapped_and_collection_is_not_returned(self) -> None:
        self.collection.fail_count = True
        with self.assertRaisesRegex(DenseIndexError, "Unable to read.*count"):
            self._open()
        self.assertFalse(self.client.created)

    def test_invalid_inputs_are_rejected_before_opening(self) -> None:
        for changes in (
            {"chunk_set": None},
            {"embedding_config": None},
        ):
            arguments = {
                "client": self.client,
                "chunk_set": self.chunk_set,
                "embedding_config": self.config,
                **changes,
            }
            with self.subTest(changes=changes):
                with self.assertRaises(DenseIndexError):
                    open_dense_collection(**arguments)
                self.assertEqual(self.client.opened, [])
                self.assertFalse(self.client.created)


if __name__ == "__main__":
    unittest.main()
