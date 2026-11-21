"""Synthetic hybrid orchestration tests; all parameters are diagnostic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import inspect
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.unit.retrieval.test_lexical_query import (
    _forbid_production_imports, _ivdr_source, _synthetic_chunk_set,
)

with _forbid_production_imports():
    from src.retrieval.hybrid_query import HybridError, retrieve_hybrid

from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_query import DenseQueryError
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.fusion import FusionError, fuse_candidates
from src.retrieval.lexical_query import LexicalCandidate, LexicalError
from tests.unit.ingestion.test_chunk_models import (
    _chunk_set, _guidance_source, _manifest, _mdr_source,
)
from tests.unit.retrieval.test_dense_results import _representative_chunk


_MODULE = "src.retrieval.hybrid_query"


class _FakeRetriever:
    def __init__(self, name, result, events):
        self.name = name
        self.result = result
        self.events = events
        self.calls = []
        self.failure = None

    def retrieve(self, query, *, candidate_count):
        self.events.append(self.name)
        self.calls.append((query, candidate_count))
        if self.failure is not None:
            raise self.failure
        return self.result


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.guards = [
            self.enterContext(patch(
                target, side_effect=AssertionError("Production execution forbidden."),
            ))
            for target in (
                "src.retrieval.dense_query.DenseRetriever.__init__",
                "src.retrieval.dense_query.DenseRetriever.retrieve",
                "src.retrieval.lexical_query.BM25Retriever.__init__",
                "src.retrieval.lexical_query.BM25Retriever.retrieve",
                "src.retrieval.embeddings.TransformerEmbedder.__init__",
                "src.retrieval.embeddings._load_transformers_backend",
                "src.retrieval.dense_index.open_dense_collection",
                "src.retrieval.dense_index.build_dense_index",
            )
        ]
        self.chunk_set = _synthetic_chunk_set(
            (_guidance_source(), ("Synthetic A.", "Synthetic B.")),
            (_mdr_source(), ("Synthetic C.",)),
        )
        self.chunks = self.chunk_set.chunks
        self.metadata = tuple(
            deserialize_chunk_metadata(chunk_metadata(c)) for c in self.chunks
        )
        self.events = []
        self.dense = _FakeRetriever("dense", self._dense((0, 1)), self.events)
        self.lexical = _FakeRetriever("lexical", self._lexical((2, 0)), self.events)
        self.lexical.chunk_set_id = self.chunk_set.chunk_set_id

        def real_fusion(**kwargs):
            self.events.append("fusion")
            return fuse_candidates(**kwargs)

        self.fusion = self.enterContext(patch(
            f"{_MODULE}.fuse_candidates", side_effect=real_fusion,
        ))

    def tearDown(self):
        for guard in self.guards:
            guard.assert_not_called()

    def _dense(self, indexes):
        return tuple(
            DenseCandidate(rank, self.chunks[i].chunk_id, self.chunks[i].text,
                           self.metadata[i], 0.9 - rank / 10)
            for rank, i in enumerate(indexes, 1)
        )

    def _lexical(self, indexes):
        return tuple(
            LexicalCandidate(rank, self.chunks[i].chunk_id, self.chunks[i].text,
                             self.metadata[i], 10.0 / rank)
            for rank, i in enumerate(indexes, 1)
        )

    def _arguments(self, **changes):
        return {
            "chunk_set": self.chunk_set, "dense_retriever": self.dense,
            "lexical_retriever": self.lexical, "dense_candidate_count": 2,
            "lexical_candidate_count": 2, "rank_constant": 2.0, **changes,
        }

    def _retrieve(self, query="  PRRC Ω? \n", **changes):
        return retrieve_hybrid(query, **self._arguments(**changes))

    def _assert_calls(self, dense, lexical, fusion):
        self.assertEqual(len(self.dense.calls), dense)
        self.assertEqual(len(self.lexical.calls), lexical)
        self.assertEqual(self.fusion.call_count, fusion)

    def _assert_preflight(self, query="PRRC", **changes):
        with self.assertRaisesRegex(HybridError, "preflight") as caught:
            self._retrieve(query, **changes)
        self.assertIsNotNone(caught.exception.__cause__)
        self._assert_calls(0, 0, 0)
        self.assertEqual(self.events, [])
        return caught.exception

    def _reset_calls(self):
        self.dense.calls.clear()
        self.lexical.calls.clear()
        self.events.clear()
        self.fusion.reset_mock()

    def test_public_signature_explicit_parameters_and_exports(self):
        parameters = inspect.signature(retrieve_hybrid).parameters
        self.assertEqual(tuple(parameters), (
            "query", "chunk_set", "dense_retriever", "lexical_retriever",
            "dense_candidate_count", "lexical_candidate_count", "rank_constant",
        ))
        for name, parameter in parameters.items():
            self.assertIs(parameter.default, inspect.Parameter.empty)
            if name != "query":
                self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        for name in parameters:
            kwargs = {"query": "PRRC", **self._arguments()}
            del kwargs[name]
            with self.subTest(missing=name), self.assertRaises(TypeError):
                retrieve_hybrid(**kwargs)
        with self.assertRaises(TypeError):
            retrieve_hybrid("PRRC", self.chunk_set)
        from src.retrieval.hybrid_query import __all__
        self.assertEqual(
            __all__,
            ["HybridError", "HybridRetrievalConfig", "retrieve_hybrid"],
        )
        self.assertTrue(issubclass(HybridError, ValueError))
        self._assert_calls(0, 0, 0)

    def test_query_counts_order_and_original_fusion_arguments(self):
        query = "  PRRC Ω? \n"
        dense, lexical = self.dense.result, self.lexical.result
        self._retrieve(query, dense_candidate_count=3, lexical_candidate_count=7)
        self._assert_calls(1, 1, 1)
        self.assertEqual(self.events, ["dense", "lexical", "fusion"])
        self.assertEqual(self.dense.calls, [(query, 3)])
        self.assertEqual(self.lexical.calls, [(query, 7)])
        self.assertIs(self.dense.calls[0][0], query)
        self.assertIs(self.lexical.calls[0][0], query)
        kwargs = self.fusion.call_args.kwargs
        self.assertIs(kwargs["chunk_set"], self.chunk_set)
        self.assertIs(kwargs["dense_candidates"], dense)
        self.assertIs(kwargs["lexical_candidates"], lexical)
        self.assertEqual(kwargs["rank_constant"], 2.0)

    def test_methods_and_identity_resolved_once_before_first_call(self):
        events = self.events

        class AccessOnce:
            def __init__(inner, fake):
                inner.fake = fake
                inner.lookups = 0
                inner.identities = 0

            @property
            def retrieve(inner):
                inner.lookups += 1
                if inner.lookups != 1:
                    raise AssertionError("Method resolved more than once")
                events.append(f"resolve-{inner.fake.name}")
                return inner.fake.retrieve

            @property
            def chunk_set_id(inner):
                inner.identities += 1
                events.append("lexical-identity")
                return inner.fake.chunk_set_id

        dense, lexical = AccessOnce(self.dense), AccessOnce(self.lexical)
        self._retrieve(dense_retriever=dense, lexical_retriever=lexical)
        self.assertEqual(self.events, [
            "resolve-dense", "resolve-lexical", "lexical-identity",
            "dense", "lexical", "fusion",
        ])
        self.assertEqual((dense.lookups, lexical.lookups), (1, 1))
        self.assertEqual((dense.identities, lexical.identities), (0, 1))
        self._assert_calls(1, 1, 1)

    def test_retained_lexical_method_survives_attribute_change_during_dense(self):
        lexical = self.lexical
        original = self.dense.retrieve

        def change_method(query, *, candidate_count):
            lexical.retrieve = None
            return original(query, candidate_count=candidate_count)

        self.dense.retrieve = change_method
        self._retrieve()
        self._assert_calls(1, 1, 1)

    def test_invalid_query_precedes_all_backend_calls(self):
        for query in (None, True, 1, [], {}, "", " \n\t", "\u2003"):
            with self.subTest(query=query):
                self._assert_preflight(query)

    def test_invalid_context_precedes_all_backend_calls(self):
        for context in (None, {}, (), self.chunks, "chunk-set"):
            with self.subTest(context=context):
                self._assert_preflight(chunk_set=context)

    def test_invalid_either_count_precedes_all_backend_calls(self):
        for name in ("dense_candidate_count", "lexical_candidate_count"):
            for value in (None, 0, -1, True, False, 1.0, "1", math.nan, math.inf):
                with self.subTest(name=name, value=value):
                    self._assert_preflight(**{name: value})

    def test_invalid_rank_constant_including_negative_underflow(self):
        for value in (
            None, True, False, "2", 1j, -1, math.nan, math.inf, -math.inf,
            Fraction(-1, 10 ** 400), 10 ** 400,
        ):
            with self.subTest(value=value):
                self._assert_preflight(rank_constant=value)

    def test_rank_constant_conversion_and_comparison_causes(self):
        failure = RuntimeError("numeric contract failed")

        class BrokenFloat(float):
            def __float__(self):
                raise failure

        class BrokenComparison(float):
            def __lt__(self, other):
                raise failure

        for value in (BrokenFloat(2), BrokenComparison(2)):
            with self.subTest(value=value):
                error = self._assert_preflight(rank_constant=value)
                self.assertIs(error.__cause__, failure)

    def test_zero_and_fractional_constants_normalized_before_fusion(self):
        for value in (0, 0.0, -0.0, Fraction(1, 2), Fraction(2)):
            with self.subTest(value=value):
                self._retrieve(rank_constant=value)
                self._assert_calls(1, 1, 1)
                forwarded = self.fusion.call_args.kwargs["rank_constant"]
                self.assertIs(type(forwarded), float)
                self.assertEqual(forwarded, float(value))
                self._reset_calls()

    def test_missing_or_noncallable_retrieval_methods(self):
        for name in ("dense_retriever", "lexical_retriever"):
            for value in (None, object(), SimpleNamespace(retrieve=None),
                          SimpleNamespace(retrieve=1)):
                with self.subTest(name=name, value=value):
                    self._assert_preflight(**{name: value})

    def test_retrieval_method_property_failures_retain_causes(self):
        failure = RuntimeError("method access failed")

        class BrokenMethod:
            @property
            def retrieve(self):
                raise failure

        for name in ("dense_retriever", "lexical_retriever"):
            with self.subTest(name=name):
                error = self._assert_preflight(**{name: BrokenMethod()})
                self.assertIs(error.__cause__, failure)

    def test_lexical_identity_missing_invalid_or_mismatched_even_if_empty(self):
        self.lexical.result = ()
        del self.lexical.chunk_set_id
        self._assert_preflight()
        for identity in (None, True, 1, "", " ", "wrong-set"):
            self.lexical.chunk_set_id = identity
            with self.subTest(identity=identity):
                self._assert_preflight()

    def test_lexical_identity_property_failure_retains_cause(self):
        failure = RuntimeError("identity access failed")

        class BrokenIdentity:
            retrieve = self.lexical.retrieve

            @property
            def chunk_set_id(self):
                raise failure

        error = self._assert_preflight(lexical_retriever=BrokenIdentity())
        self.assertIs(error.__cause__, failure)

    def test_dense_failure_retains_chain_and_skips_lexical_and_fusion(self):
        original = RuntimeError("embedding or query failed")
        try:
            raise DenseQueryError("dense failed") from original
        except DenseQueryError as failure:
            self.dense.failure = failure
        with self.assertRaisesRegex(HybridError, "dense") as caught:
            self._retrieve()
        self.assertIs(caught.exception.__cause__, self.dense.failure)
        self.assertIs(caught.exception.__cause__.__cause__, original)
        self._assert_calls(1, 0, 0)
        self.assertEqual(self.events, ["dense"])

    def test_lexical_failure_retains_cause_without_partial_output(self):
        for failure in (LexicalError("lexical failed"), RuntimeError("unexpected")):
            self.lexical.failure = failure
            with self.subTest(failure=failure):
                with self.assertRaisesRegex(HybridError, "lexical") as caught:
                    self._retrieve()
                self.assertIs(caught.exception.__cause__, failure)
                self._assert_calls(1, 1, 0)
                self.assertEqual(self.events, ["dense", "lexical"])
                self._reset_calls()

    def test_dense_empty_malformed_and_oversized_envelopes_stop_immediately(self):
        class TupleSubclass(tuple):
            pass

        for result in (
            (), None, [], {}, "results", iter(self.dense.result),
            TupleSubclass(self.dense.result), self.dense.result * 2,
        ):
            self.dense.result = result
            with self.subTest(result=result):
                with self.assertRaisesRegex(HybridError, "dense") as caught:
                    self._retrieve()
                self.assertIsInstance(caught.exception.__cause__, ValueError)
                self._assert_calls(1, 0, 0)
                self.assertEqual(self.events, ["dense"])
                self._reset_calls()

    def test_lexical_malformed_and_oversized_envelopes_stop_before_fusion(self):
        class TupleSubclass(tuple):
            pass

        for result in (
            None, [], {}, "results", iter(self.lexical.result),
            TupleSubclass(self.lexical.result), self.lexical.result * 2,
        ):
            self.lexical.result = result
            with self.subTest(result=result):
                with self.assertRaisesRegex(HybridError, "lexical"):
                    self._retrieve()
                self._assert_calls(1, 1, 0)
                self.assertEqual(self.events, ["dense", "lexical"])
                self._reset_calls()

    def test_empty_lexical_and_fewer_than_requested_are_successful(self):
        self.dense.result = self._dense((1,))
        self.lexical.result = ()
        result = self._retrieve(dense_candidate_count=5, lexical_candidate_count=7)
        self._assert_calls(1, 1, 1)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0].dense_candidate, self.dense.result[0])
        self.assertIsNone(result[0].lexical_candidate)
        self.assertEqual(result[0].rrf_score, 1 / 3)

    def test_returned_identity_mismatch_is_delegated_to_real_fusion(self):
        for branch in (self.dense, self.lexical):
            original = branch.result
            candidate = original[0]
            branch.result = (replace(
                candidate, metadata=replace(candidate.metadata, chunk_set_id="wrong"),
            ),)
            with self.subTest(branch=branch.name):
                with self.assertRaisesRegex(HybridError, "fusion") as caught:
                    self._retrieve()
                self.assertIsInstance(caught.exception.__cause__, FusionError)
                self._assert_calls(1, 1, 1)
                self.assertEqual(self.events, ["dense", "lexical", "fusion"])
            branch.result = original
            self._reset_calls()

    def test_candidate_level_validation_occurs_only_in_real_fusion(self):
        candidate = self.dense.result[0]
        unknown = replace(candidate, chunk_id="unknown",
                          metadata=replace(candidate.metadata, chunk_id="unknown"))
        for result in (
            (None,), (self.lexical.result[0],), (unknown,),
            (replace(candidate, rank=2),),
            (candidate, replace(candidate, rank=2)),
            (replace(candidate, document="fabricated"),),
            (replace(candidate, metadata=replace(candidate.metadata, version="fake")),),
        ):
            self.dense.result = result
            with self.subTest(result=result):
                with self.assertRaisesRegex(HybridError, "fusion") as caught:
                    self._retrieve()
                self.assertIsInstance(caught.exception.__cause__, FusionError)
                self._assert_calls(1, 1, 1)
                self._reset_calls()

    def test_original_mdr_permission_rechecked_by_fusion(self):
        chunk_set = _synthetic_chunk_set((_ivdr_source(), ("IVDR evidence.",)))
        chunk = chunk_set.chunks[0]
        self.dense.result = (DenseCandidate(
            1, chunk.chunk_id, chunk.text,
            deserialize_chunk_metadata(chunk_metadata(chunk)), 0.5,
        ),)
        self.lexical.result = ()
        self.lexical.chunk_set_id = chunk_set.chunk_set_id
        with self.assertRaisesRegex(HybridError, "fusion") as caught:
            self._retrieve(chunk_set=chunk_set)
        self.assertIsInstance(caught.exception.__cause__, FusionError)
        self._assert_calls(1, 1, 1)

    def test_fusion_failure_retains_cause_and_never_retries(self):
        failure = RuntimeError("fusion calculation failed")
        self.fusion.side_effect = failure
        with self.assertRaisesRegex(HybridError, "fusion") as caught:
            self._retrieve()
        self.assertIs(caught.exception.__cause__, failure)
        self._assert_calls(1, 1, 1)

    def test_base_exceptions_escape_each_execution_stage(self):
        for stage in ("dense", "lexical", "fusion"):
            for failure in (KeyboardInterrupt(), SystemExit()):
                if stage == "fusion":
                    self.fusion.side_effect = failure
                else:
                    getattr(self, stage).failure = failure
                with self.subTest(stage=stage, failure=type(failure)):
                    with self.assertRaises(type(failure)) as caught:
                        self._retrieve()
                    self.assertIs(caught.exception, failure)
                    self._assert_calls(1, int(stage != "dense"), int(stage == "fusion"))
                self.dense.failure = self.lexical.failure = None
                self._reset_calls()

    def test_base_exception_during_preflight_escapes_without_calls(self):
        failure = KeyboardInterrupt()

        class Interrupted:
            @property
            def retrieve(self):
                raise failure

        with self.assertRaises(KeyboardInterrupt) as caught:
            self._retrieve(lexical_retriever=Interrupted())
        self.assertIs(caught.exception, failure)
        self._assert_calls(0, 0, 0)

    def test_real_fusion_full_union_independent_arithmetic_and_immutability(self):
        dense, lexical = self.dense.result, self.lexical.result
        before = deepcopy((self.chunk_set, dense, lexical))
        result = self._retrieve()
        self.assertEqual([c.chunk_id for c in result], [
            self.chunks[i].chunk_id for i in (0, 2, 1)
        ])
        self.assertEqual(len(result), 3)  # Larger than either branch's count.
        for candidate, oracle in zip(result, (Fraction(7, 12), Fraction(1, 3), Fraction(1, 4))):
            self.assertAlmostEqual(candidate.rrf_score, float(oracle), places=15)
        self.assertEqual([c.rank for c in result], [1, 2, 3])
        self.assertIs(type(result), tuple)
        self.assertIs(result[0].dense_candidate, dense[0])
        self.assertIs(result[0].lexical_candidate, lexical[1])
        self.assertIs(result[1].lexical_candidate, lexical[0])
        self.assertIs(result[2].dense_candidate, dense[1])
        self.assertIsNone(result[1].dense_candidate)
        self.assertIsNone(result[2].lexical_candidate)
        self.assertEqual(result[0].lexical_candidate.rank, 2)
        self.assertEqual(result[0].dense_candidate.cosine_distance, dense[0].cosine_distance)
        self.assertEqual(result[0].lexical_candidate.bm25_score, lexical[1].bm25_score)
        for candidate in result:
            index = next(i for i, c in enumerate(self.chunks) if c.chunk_id == candidate.chunk_id)
            self.assertEqual(candidate.document, self.chunks[index].text)
            self.assertEqual(candidate.metadata, self.metadata[index])
        with self.assertRaises(TypeError):
            result[0] = result[1]
        with self.assertRaises(FrozenInstanceError):
            result[0].rank = 9
        with self.assertRaises(FrozenInstanceError):
            result[0].metadata.version = "changed"
        self.assertEqual((self.chunk_set, dense, lexical), before)
        self._assert_calls(1, 1, 1)

    def test_real_fusion_cross_source_ties_follow_global_position(self):
        self.assertEqual([c.order for c in self.chunks], [0, 1, 0])
        self.dense.result = self._dense((2, 1, 0))
        self.lexical.result = self._lexical((1, 2, 0))
        result = self._retrieve(dense_candidate_count=3, lexical_candidate_count=3)
        self.assertEqual([c.chunk_id for c in result], [
            self.chunks[i].chunk_id for i in (1, 2, 0)
        ])
        self.assertEqual(result[0].rrf_score, result[1].rrf_score)
        self.assertAlmostEqual(result[0].rrf_score, float(Fraction(7, 12)), places=15)
        self.assertAlmostEqual(result[2].rrf_score, float(Fraction(2, 5)), places=15)
        self._assert_calls(1, 1, 1)

    def test_rich_metadata_and_dual_permission_preserved(self):
        chunk = replace(_representative_chunk(), order=0,
                        previous_chunk_id=None, next_chunk_id=None)
        chunk_set = _chunk_set(_manifest(chunk.source), (chunk,))
        metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
        self.dense.result = (DenseCandidate(1, chunk.chunk_id, chunk.text, metadata, -0.1),)
        self.lexical.result = (LexicalCandidate(1, chunk.chunk_id, chunk.text, metadata, 0.0),)
        self.lexical.chunk_set_id = chunk_set.chunk_set_id
        result = self._retrieve(chunk_set=chunk_set)
        self.assertEqual(result[0].metadata, metadata)
        self.assertTrue(metadata.permits_mdr and metadata.permits_ivdr)
        self.assertTrue(metadata.pages and metadata.annotations and metadata.has_table)
        self.assertTrue(metadata.legal_locators and metadata.hierarchies)
        self.assertIs(result[0].dense_candidate, self.dense.result[0])
        self.assertIs(result[0].lexical_candidate, self.lexical.result[0])
        self.assertEqual(result[0].dense_candidate.cosine_distance, -0.1)
        self.assertEqual(result[0].lexical_candidate.bm25_score, 0.0)
        self._assert_calls(1, 1, 1)

    def test_returns_fusion_tuple_directly_without_reconstruction(self):
        original = fuse_candidates(
            chunk_set=self.chunk_set, dense_candidates=self.dense.result,
            lexical_candidates=self.lexical.result, rank_constant=2.0,
        )
        self.fusion.side_effect = None
        self.fusion.return_value = original
        self.assertIs(self._retrieve(), original)
        self._assert_calls(1, 1, 1)


if __name__ == "__main__":
    unittest.main()
