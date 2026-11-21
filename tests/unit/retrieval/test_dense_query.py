"""Synthetic query tests: no production model, corpus, or database execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np

from src.retrieval.dense_index import (
    DenseIndexError, chunk_metadata, collection_metadata, dense_collection_name,
    open_dense_collection,
)
from src.retrieval.dense_query import DenseQueryError, DenseRetriever
from src.retrieval.dense_results import DenseResultError, deserialize_chunk_metadata
from tests.unit.ingestion.test_chunk_models import _chunk, _chunk_set, _manifest
from tests.unit.retrieval.test_dense_index import (
    _FakeOpeningClient, _FakeStoredCollection, _embedding_config,
)
from tests.unit.retrieval.test_dense_results import _representative_chunk


class _FakeEmbedder:
    def __init__(self, config) -> None:
        self.config = config
        self.vector: object = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.calls: list[str] = []
        self.failure: BaseException | None = None

    def embed_query(self, query: str) -> object:
        self.calls.append(query)
        if self.failure is not None:
            raise self.failure
        return self.vector


class _FakeQueryCollection(_FakeStoredCollection):
    def __init__(self, metadata, response) -> None:
        super().__init__(metadata, record_count=2)
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.failure: BaseException | None = None

    def query(self, **values: object) -> object:
        self.calls.append(values)
        if self.failure is not None:
            raise self.failure
        return self.response


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        # Any accidental heavyweight import or fallback fails without loading it.
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {"chromadb": None, "torch": None,
                                   "transformers": None}).start()
        self.loader = patch(
            "src.retrieval.embeddings._load_transformers_backend",
            side_effect=AssertionError("Production model loading is forbidden."),
        ).start()
        first = replace(_representative_chunk(), order=0,
                        previous_chunk_id=None, next_chunk_id=None)
        second = _chunk(first.chunk_set_id, first.source, order=1,
                        text="Second synthetic candidate.")
        self.chunks = (
            replace(first, next_chunk_id=second.chunk_id),
            replace(second, previous_chunk_id=first.chunk_id),
        )
        self.chunk_set = _chunk_set(_manifest(first.source), self.chunks)
        self.config = _embedding_config()
        self.embedder = _FakeEmbedder(self.config)
        self.response = {
            "ids": [[c.chunk_id for c in self.chunks]],
            "documents": [[c.text for c in self.chunks]],
            "metadatas": [[chunk_metadata(c) for c in self.chunks]],
            "distances": [[0.8, 0.1]],
            "embeddings": None,
            "uris": None,
            "included": ["documents", "metadatas", "distances"],
        }
        self.collection = _FakeQueryCollection(
            collection_metadata(self.chunk_set, self.config), self.response,
        )
        self.client = _FakeOpeningClient(
            dense_collection_name(self.chunk_set, self.config), self.collection,
        )
        self.retriever = self._construct()

    def _construct(self, **changes) -> DenseRetriever:
        return DenseRetriever(**{
            "client": self.client, "chunk_set": self.chunk_set,
            "embedder": self.embedder, **changes,
        })

    def _retrieve(self, count: int = 2):
        return self.retriever.retrieve("  Original query  ", candidate_count=count)

    def test_construction_uses_shared_opener_and_exact_config(self) -> None:
        self.client.opened.clear()
        self.collection.checks.clear()
        with patch("src.retrieval.dense_query.open_dense_collection",
                   wraps=open_dense_collection) as opener:
            self._construct()
        self.assertIs(opener.call_args.kwargs["embedding_config"], self.config)
        self.assertIs(opener.call_args.kwargs["chunk_set"], self.chunk_set)
        self.assertIs(opener.call_args.kwargs["client"], self.client)
        self.assertEqual(self.client.opened, [{
            "name": dense_collection_name(self.chunk_set, self.config),
            "embedding_function": None,
        }])
        self.assertEqual(self.collection.checks, ["metadata", "count"])
        self.assertFalse(self.client.created)
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.collection.calls, [])

    def test_invalid_chunk_set_and_config_fail_before_opening(self) -> None:
        for changes in ({"chunk_set": None},
                        {"embedder": _FakeEmbedder(None)}):
            with self.subTest(changes=changes):
                self.client.opened.clear()
                with self.assertRaises(DenseQueryError) as caught:
                    self._construct(**changes)
                self.assertIsInstance(caught.exception.__cause__, DenseIndexError)
                self.assertEqual(self.client.opened, [])

    def test_config_access_failure_retains_cause(self) -> None:
        failure = RuntimeError("config unavailable")

        class BrokenEmbedder:
            @property
            def config(self):
                raise failure

        with self.assertRaises(DenseQueryError) as caught:
            self._construct(embedder=BrokenEmbedder())
        self.assertIs(caught.exception.__cause__, failure)

    def test_every_collection_identity_mismatch_fails_closed(self) -> None:
        original = self.collection.stored_metadata.copy()
        for field in original:
            with self.subTest(field=field):
                self.collection.stored_metadata = {**original, field: None}
                with self.assertRaises(DenseQueryError) as caught:
                    self._construct()
                self.assertIsInstance(caught.exception.__cause__, DenseIndexError)
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.collection.calls, [])

    def test_embedder_identity_selects_exact_collection(self) -> None:
        for changes in ({"model_id": "other/model"}, {"revision": "other"},
                        {"dimension": 4}):
            with self.subTest(changes=changes):
                with self.assertRaises(DenseQueryError):
                    self._construct(embedder=_FakeEmbedder(replace(self.config, **changes)))
                self.assertEqual(self.client.opened[-1]["name"],
                                 dense_collection_name(self.chunk_set,
                                                       replace(self.config, **changes)))

    def test_missing_collection_and_count_failures_are_wrapped(self) -> None:
        self.client.collection = None
        with self.assertRaises(DenseQueryError) as caught:
            self._construct()
        self.assertIsInstance(caught.exception.__cause__.__cause__, LookupError)
        self.client.collection = self.collection
        for count in (0, 3, True):
            with self.subTest(count=count):
                self.collection.record_count = count
                with self.assertRaises(DenseQueryError):
                    self._construct()
        self.collection.record_count = 2
        self.collection.fail_count = True
        with self.assertRaises(DenseQueryError) as caught:
            self._construct()
        self.assertIsInstance(caught.exception.__cause__.__cause__, RuntimeError)
        self.assertFalse(self.client.created)

    def test_invalid_queries_and_counts_have_no_inference_or_query(self) -> None:
        for query in (None, "", " \t\n", b"query", 1, []):
            with self.subTest(query=query), self.assertRaises(DenseQueryError):
                self.retriever.retrieve(query, candidate_count=2)
        for count in (None, True, False, 0, -1, 1.5, "2"):
            with self.subTest(count=count), self.assertRaises(DenseQueryError):
                self._retrieve(count)
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.collection.calls, [])

    def test_candidate_count_is_required(self) -> None:
        with self.assertRaises(TypeError):
            self.retriever.retrieve("query")
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.collection.calls, [])

    def test_success_has_exact_arguments_and_preserves_text_and_order(self) -> None:
        result = self._retrieve()
        self.assertEqual(self.embedder.calls, ["  Original query  "])
        self.assertEqual(self.collection.calls, [{
            "query_embeddings": [[1.0, 0.0, 0.0]], "n_results": 2,
            "where": {"permits_mdr": True},
            "include": ["documents", "metadatas", "distances"],
        }])
        self.assertEqual(tuple(c.chunk_id for c in result), tuple(c.chunk_id for c in self.chunks))
        self.assertEqual(tuple(c.rank for c in result), (1, 2))
        self.assertEqual(tuple(c.cosine_distance for c in result), (0.8, 0.1))
        self.loader.assert_not_called()

    def test_output_is_immutable_and_preserves_all_trusted_metadata(self) -> None:
        result = self._retrieve()
        self.assertIs(type(result), tuple)
        for candidate, chunk in zip(result, self.chunks):
            self.assertEqual(candidate.document, chunk.text)
            self.assertEqual(candidate.metadata,
                             deserialize_chunk_metadata(chunk_metadata(chunk)))
        with self.assertRaises(TypeError):
            result[0] = result[1]
        with self.assertRaises(FrozenInstanceError):
            result[0].rank = 2
        with self.assertRaises(FrozenInstanceError):
            result[0].metadata.pages[0].source_index = 9
        self.response["metadatas"][0][0]["official_title"] = "changed"
        self.assertEqual(result[0].metadata.official_title, self.chunks[0].source.official_title)

    def test_mdr_only_and_dual_permission_are_both_accepted(self) -> None:
        raw = self.response["metadatas"][0][0]
        raw.update(permitted_regimes_json='["MDR"]', permits_ivdr=False)
        result = self._retrieve()
        self.assertFalse(result[0].metadata.permits_ivdr)
        self.assertTrue(result[1].metadata.permits_ivdr)
        self.assertTrue(all(c.metadata.permits_mdr for c in result))

    def test_ivdr_only_result_is_rejected_even_after_valid_first_result(self) -> None:
        self.response["metadatas"][0][1].update(
            permitted_regimes_json='["IVDR"]', permits_mdr=False,
        )
        with self.assertRaisesRegex(DenseQueryError, "permit MDR"):
            self._retrieve()

    def test_fewer_results_than_requested_are_valid(self) -> None:
        self.assertEqual(len(self._retrieve(10)), 2)

    def test_invalid_vector_shapes_and_values_prevent_query(self) -> None:
        for vector in (None, 1.0, [], [1, 0], [1, 0, 0, 0], [[1, 0, 0]],
                       [[1], [0], [0]], [[1], [0, 0]], ["1", "0", "0"],
                       [True, False, False], [1j, 0, 0], [object(), 0, 0],
                       [np.nan, 0, 0], [np.inf, 0, 0], [-np.inf, 0, 0]):
            with self.subTest(vector=repr(vector)):
                self.embedder.vector = vector
                before = len(self.embedder.calls)
                with self.assertRaises(DenseQueryError):
                    self._retrieve()
                self.assertEqual(len(self.embedder.calls), before + 1)
                self.assertEqual(self.collection.calls, [])

    def test_numeric_vector_is_not_normalized_or_mutated(self) -> None:
        vector = np.array([2.0, -0.5, 0.125], dtype=np.float64)
        self.embedder.vector = vector
        self._retrieve()
        self.assertEqual(self.collection.calls[0]["query_embeddings"], [vector.tolist()])
        np.testing.assert_array_equal(vector, [2.0, -0.5, 0.125])

    def test_embedding_and_query_failures_retain_causes(self) -> None:
        failure = RuntimeError("synthetic failure")
        self.embedder.failure = failure
        with self.assertRaises(DenseQueryError) as caught:
            self._retrieve()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(self.collection.calls, [])
        self.embedder.failure = None
        self.collection.failure = failure
        with self.assertRaises(DenseQueryError) as caught:
            self._retrieve()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertEqual(len(self.collection.calls), 1)

    def test_base_exceptions_are_not_caught(self) -> None:
        for target in (self.embedder, self.collection):
            with self.subTest(target=type(target).__name__):
                target.failure = KeyboardInterrupt()
                with self.assertRaises(KeyboardInterrupt):
                    self._retrieve()
                target.failure = None

    def test_non_mapping_and_missing_columns_are_rejected(self) -> None:
        for response in (None, [], "response"):
            with self.subTest(response=response), self.assertRaises(DenseQueryError):
                self.collection.response = response
                self._retrieve()
        for field in ("ids", "documents", "metadatas", "distances"):
            changed = deepcopy(self.response)
            del changed[field]
            self.collection.response = changed
            with self.subTest(field=field), self.assertRaises(DenseQueryError):
                self._retrieve()

    def test_required_columns_have_exact_single_row_nesting(self) -> None:
        for field in ("ids", "documents", "metadatas", "distances"):
            row = self.response[field][0]
            for value in (None, [], {}, "row", row, [None], ["row"],
                          [tuple(row)], (row,), [row, row]):
                self.collection.response = {**self.response, field: value}
                with self.subTest(field=field, value=repr(value)):
                    with self.assertRaises(DenseQueryError):
                        self._retrieve()

    def test_misaligned_empty_and_excessive_results_are_rejected(self) -> None:
        for field in ("ids", "documents", "metadatas", "distances"):
            self.collection.response = {**self.response, field: [self.response[field][0][:1]]}
            with self.subTest(field=field), self.assertRaises(DenseQueryError):
                self._retrieve()
        self.collection.response = {field: [[]] for field in
                                    ("ids", "documents", "metadatas", "distances")}
        with self.assertRaises(DenseQueryError):
            self._retrieve()
        self.collection.response = self.response
        with self.assertRaises(DenseQueryError):
            self._retrieve(1)

    def test_duplicate_or_invalid_ids_and_documents_are_rejected(self) -> None:
        for field in ("ids", "documents"):
            for value in ("", " \t", None, 1, True):
                changed = deepcopy(self.response)
                changed[field][0][1] = value
                self.collection.response = changed
                with self.subTest(field=field, value=value):
                    with self.assertRaises(DenseQueryError):
                        self._retrieve()
        changed = deepcopy(self.response)
        changed["ids"][0][1] = changed["ids"][0][0]
        self.collection.response = changed
        with self.assertRaisesRegex(DenseQueryError, "duplicate"):
            self._retrieve()

    def test_invalid_metadata_retains_decoder_cause(self) -> None:
        original = self.response["metadatas"][0][1]
        for raw in (None, {}, {**original, "pages_json": "["},
                    {**original, "permits_mdr": 1}):
            changed = deepcopy(self.response)
            changed["metadatas"][0][1] = raw
            self.collection.response = changed
            with self.subTest(raw=raw), self.assertRaises(DenseQueryError) as caught:
                self._retrieve()
            self.assertIsInstance(caught.exception.__cause__, DenseResultError)

    def test_chunk_set_and_chunk_id_mismatches_are_rejected(self) -> None:
        for field in ("chunk_set_id", "chunk_id"):
            changed = deepcopy(self.response)
            changed["metadatas"][0][1][field] = "wrong-identity"
            self.collection.response = changed
            with self.subTest(field=field), self.assertRaises(DenseQueryError):
                self._retrieve()

    def test_invalid_distances_reject_entire_response(self) -> None:
        for value in (True, False, None, "0.1", 1j, np.nan, np.inf, -np.inf,
                      10 ** 1000):
            changed = deepcopy(self.response)
            changed["distances"][0][1] = value
            self.collection.response = changed
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(DenseQueryError) as caught:
                    self._retrieve()
                self.assertIsInstance(caught.exception.__cause__, DenseResultError)

    def test_finite_distances_have_no_threshold_or_range_repair(self) -> None:
        self.response["distances"] = [[2.5, -0.01]]
        self.assertEqual(tuple(c.cosine_distance for c in self._retrieve()), (2.5, -0.01))


if __name__ == "__main__":
    unittest.main()
