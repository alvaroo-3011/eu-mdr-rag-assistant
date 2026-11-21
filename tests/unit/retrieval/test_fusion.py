"""Synthetic fusion tests; k=2 is an arithmetic fixture, not a default."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import math
import sys
import unittest
from unittest.mock import patch

from tests.unit.retrieval.test_lexical_query import (
    _forbid_production_imports, _ivdr_source, _synthetic_chunk_set,
)

with _forbid_production_imports():
    from src.retrieval.fusion import FusionError, FusedCandidate, fuse_candidates

from src.ingestion.models import PageIdentity, RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import (
    DenseCandidate, DenseResultError, deserialize_chunk_metadata,
)
from src.retrieval.lexical_query import LexicalCandidate
from tests.unit.ingestion.test_chunk_models import (
    _chunk, _chunk_set, _guidance_source, _manifest, _mdr_source,
)
from tests.unit.retrieval.test_dense_results import _representative_chunk


_MODULE = "src.retrieval.fusion"


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.guards = [
            self.enterContext(patch(
                target, side_effect=AssertionError("Real execution is forbidden."),
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
        first = replace(
            _representative_chunk(), order=0,
            previous_chunk_id=None, next_chunk_id=None,
        )
        chunks = [first] + [
            _chunk(first.chunk_set_id, first.source, order=i, text=f"Synthetic evidence {i}.")
            for i in (1, 2)
        ]
        self.chunks = tuple(
            replace(
                chunk, previous_chunk_id=chunks[i - 1].chunk_id if i else None,
                next_chunk_id=chunks[i + 1].chunk_id if i + 1 < len(chunks) else None,
            )
            for i, chunk in enumerate(chunks)
        )
        self.chunk_set = _chunk_set(_manifest(first.source), self.chunks)
        self.metadata = tuple(
            deserialize_chunk_metadata(chunk_metadata(chunk)) for chunk in self.chunks
        )

    def _dense(self, indexes, scores=None):
        return tuple(
            DenseCandidate(
                rank, self.chunks[i].chunk_id, self.chunks[i].text, self.metadata[i],
                scores[rank - 1] if scores is not None else 0.5,
            )
            for rank, i in enumerate(indexes, 1)
        )

    def _lexical(self, indexes, scores=None):
        return tuple(
            LexicalCandidate(
                rank, self.chunks[i].chunk_id, self.chunks[i].text, self.metadata[i],
                scores[rank - 1] if scores is not None else 1.0,
            )
            for rank, i in enumerate(indexes, 1)
        )

    def _fuse(self, dense=(), lexical=(), **changes):
        return fuse_candidates(**{
            "chunk_set": self.chunk_set, "dense_candidates": dense,
            "lexical_candidates": lexical, "rank_constant": 2,
            **changes,
        })

    def test_independent_arithmetic_oracle_and_full_union(self):
        dense, lexical = self._dense((0, 1)), self._lexical((2, 0))
        result = self._fuse(dense, lexical)
        self.assertEqual([c.chunk_id for c in result], [self.chunks[i].chunk_id for i in (0, 2, 1)])
        self.assertAlmostEqual(result[0].rrf_score, 7 / 12, places=15)
        self.assertAlmostEqual(result[1].rrf_score, 1 / 3, places=15)
        self.assertAlmostEqual(result[2].rrf_score, 1 / 4, places=15)
        self.assertEqual([c.rank for c in result], [1, 2, 3])
        self.assertIs(result[0].dense_candidate, dense[0])
        self.assertIs(result[0].lexical_candidate, lexical[1])
        self.assertEqual(result[0].lexical_candidate.rank, 2)

    def test_single_branch_and_both_empty(self):
        for dense, lexical in ((self._dense((2, 0)), ()), ((), self._lexical((2, 0)))):
            with self.subTest(dense=bool(dense)):
                result = self._fuse(dense, lexical)
                self.assertEqual([c.chunk_id for c in result], [self.chunks[i].chunk_id for i in (2, 0)])
                self.assertEqual([c.rrf_score for c in result], [1 / 3, 1 / 4])
                self.assertIs(result[0].dense_candidate, dense[0] if dense else None)
                self.assertIs(result[0].lexical_candidate, lexical[0] if lexical else None)
        self.assertEqual(self._fuse(), ())

    def test_disjoint_candidates_and_cross_branch_merge(self):
        result = self._fuse(self._dense((0,)), self._lexical((1,)))
        self.assertEqual(len(result), 2)
        merged = self._fuse(self._dense((0,)), self._lexical((0,)))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].rrf_score, 2 / 3)

    def test_duplicate_ids_within_each_branch_fail(self):
        for dense, lexical in ((self._dense((0, 0)), ()), ((), self._lexical((0, 0)))):
            with self.subTest(dense=bool(dense)), self.assertRaisesRegex(FusionError, "Duplicate"):
                self._fuse(dense, lexical)

    def test_branch_rank_gaps_repetitions_and_reordering_fail(self):
        for make, field in ((self._dense, "dense_candidates"), (self._lexical, "lexical_candidates")):
            candidates = make((0, 1))
            for malformed in (
                (replace(candidates[0], rank=2), candidates[1]),
                (candidates[0], replace(candidates[1], rank=1)),
                (candidates[0], replace(candidates[1], rank=3)),
                candidates[::-1],
            ):
                with self.subTest(field=field, malformed=malformed), self.assertRaises(FusionError):
                    self._fuse(**{field: malformed})

    def test_invalid_containers_and_branch_types(self):
        for field in ("dense_candidates", "lexical_candidates"):
            for value in (None, [], {}, "", iter(()), (None,)):
                with self.subTest(field=field, value=value), self.assertRaises(FusionError):
                    self._fuse(**{field: value})
        for values in (
            {"dense_candidates": self._lexical((0,))},
            {"lexical_candidates": self._dense((0,))},
        ):
            with self.subTest(values=values), self.assertRaises(FusionError):
                self._fuse(**values)

    def test_empty_inputs_still_validate_context_and_constant(self):
        for context in (None, (), self.chunks, {}):
            with self.subTest(context=context), self.assertRaises(FusionError):
                self._fuse(chunk_set=context)
        for value in (-1, True, False, None, "2", 1j, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(FusionError):
                self._fuse(rank_constant=value)

    def test_constant_is_explicit_keyword_only_and_normalized(self):
        values = {
            "chunk_set": self.chunk_set, "dense_candidates": (), "lexical_candidates": (),
        }
        with self.assertRaises(TypeError):
            fuse_candidates(**values)
        with self.assertRaises(TypeError):
            fuse_candidates(self.chunk_set, (), (), 2)
        for zero in (0, 0.0, -0.0):
            with self.subTest(zero=zero):
                self.assertEqual(self._fuse(self._dense((0,)), rank_constant=zero)[0].rrf_score, 1)
        self.assertEqual(
            self._fuse(self._dense((0,)), rank_constant=Fraction(2)),
            self._fuse(self._dense((0,)), rank_constant=2.0),
        )
        with patch(f"{_MODULE}._contribution", return_value=0.5) as contribution:
            self._fuse(self._dense((0,)), rank_constant=2)
        self.assertIs(type(contribution.call_args.args[0]), float)

    def test_negative_fraction_underflow_rejected_with_empty_inputs(self):
        with self.assertRaisesRegex(FusionError, "nonnegative"):
            self._fuse(rank_constant=Fraction(-1, 10 ** 400))

    def test_negative_fraction_underflow_rejected_with_populated_inputs(self):
        with self.assertRaisesRegex(FusionError, "nonnegative"):
            self._fuse(
                self._dense((0,)), self._lexical((0,)),
                rank_constant=Fraction(-1, 10 ** 400),
            )

    def test_positive_fractional_constant_independent_arithmetic(self):
        result = self._fuse(
            self._dense((0,)), self._lexical((1, 0)), rank_constant=Fraction(1, 2),
        )
        self.assertEqual([c.chunk_id for c in result], [self.chunks[i].chunk_id for i in (0, 1)])
        # k=1/2: ranks (1, 2) contribute 2/3 + 2/5 = 16/15.
        self.assertAlmostEqual(result[0].rrf_score, 16 / 15, places=15)
        self.assertAlmostEqual(result[1].rrf_score, 2 / 3, places=15)

    def test_largest_finite_constant_positive_contributions_and_scores(self):
        constant = sys.float_info.max
        dense = self._dense((0, 1))
        single = self._fuse(dense, rank_constant=constant)
        combined = self._fuse(dense, self._lexical((1,)), rank_constant=constant)
        for candidate in (*single, *combined):
            self.assertTrue(math.isfinite(candidate.rrf_score))
            self.assertGreater(candidate.rrf_score, 0)
        # Adding these small ranks rounds to the same largest finite float.
        # Single-branch scores expose each contribution without private helpers.
        self.assertEqual([c.rrf_score for c in single], [1 / constant, 1 / constant])
        self.assertEqual([c.chunk_id for c in combined], [self.chunks[i].chunk_id for i in (1, 0)])
        self.assertEqual([c.rrf_score for c in combined], [2 / constant, 1 / constant])

    def test_unknown_id_and_wrong_chunk_set_identity(self):
        candidate = self._dense((0,))[0]
        unknown = replace(
            candidate, chunk_id="unknown-id", metadata=replace(candidate.metadata, chunk_id="unknown-id"),
        )
        wrong_set = replace(candidate, metadata=replace(candidate.metadata, chunk_set_id="wrong-set"))
        for value in (unknown, wrong_set):
            with self.subTest(value=value), self.assertRaises(FusionError):
                self._fuse((value,))

    def test_conflicting_and_jointly_fabricated_text_fail(self):
        dense, lexical = self._dense((0,)), self._lexical((0,))
        fake_dense = replace(dense[0], document="Fabricated evidence.")
        fake_lexical = replace(lexical[0], document="Fabricated evidence.")
        for left, right in (((fake_dense,), lexical), (dense, (fake_lexical,)), ((fake_dense,), (fake_lexical,))):
            with self.subTest(left=left, right=right), self.assertRaises(FusionError):
                self._fuse(left, right)

    def test_conflicting_and_jointly_fabricated_metadata_fail(self):
        dense, lexical = self._dense((0,)), self._lexical((0,))
        for metadata in (
            replace(self.metadata[0], version="fabricated version"),
            replace(self.metadata[0], pages=(PageIdentity(2, 3), PageIdentity(5, 6))),
            replace(self.metadata[0], annotations=(replace(self.metadata[0].annotations[0], text="fabricated"),)),
        ):
            fake_dense = replace(dense[0], metadata=metadata)
            fake_lexical = replace(lexical[0], metadata=metadata)
            for left, right in ((dense, (fake_lexical,)), ((fake_dense,), (fake_lexical,))):
                with self.subTest(metadata=metadata), self.assertRaises(FusionError):
                    self._fuse(left, right)

    def test_ivdr_rejected_and_original_permission_rechecked(self):
        chunk_set = _synthetic_chunk_set((_ivdr_source(), ("Synthetic IVDR evidence.",)))
        original = chunk_set.chunks[0]
        metadata = deserialize_chunk_metadata(chunk_metadata(original))
        fabricated_permission = replace(
            metadata, regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
            permitted_regimes=(RegulatoryRegime.MDR,), permits_mdr=True, permits_ivdr=False,
        )
        for model in (DenseCandidate, LexicalCandidate):
            field = "dense_candidates" if model is DenseCandidate else "lexical_candidates"
            for projection in (metadata, fabricated_permission):
                candidate = model(1, original.chunk_id, original.text, projection, 1.0)
                with self.subTest(model=model, projection=projection), self.assertRaises(FusionError):
                    self._fuse(chunk_set=chunk_set, **{field: (candidate,)})

    def test_dual_permission_rich_metadata_text_and_provenance_preserved(self):
        dense, lexical = self._dense((0,)), self._lexical((0,))
        result = self._fuse(dense, lexical)[0]
        self.assertTrue(result.metadata.permits_mdr and result.metadata.permits_ivdr)
        self.assertEqual(result.document, self.chunks[0].text)
        self.assertEqual(result.metadata, self.metadata[0])
        self.assertTrue(result.metadata.annotations and result.metadata.has_table)
        self.assertTrue(result.metadata.pages and result.metadata.legal_locators)
        self.assertIs(result.dense_candidate, dense[0])
        self.assertIs(result.lexical_candidate, lexical[0])

    def test_distinct_ids_with_identical_text_remain_distinct(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("Same text.", "Same text.")))
        dense = tuple(
            DenseCandidate(rank, c.chunk_id, c.text, deserialize_chunk_metadata(chunk_metadata(c)), 0.5)
            for rank, c in enumerate(chunk_set.chunks, 1)
        )
        result = self._fuse(dense, chunk_set=chunk_set)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0].chunk_id, result[1].chunk_id)

    def test_exact_ties_use_global_position_not_source_local_order(self):
        chunk_set = _synthetic_chunk_set(
            (_guidance_source(), ("A", "B")), (_mdr_source(), ("C",)),
        )
        self.assertEqual([c.order for c in chunk_set.chunks], [0, 1, 0])
        metadata = tuple(deserialize_chunk_metadata(chunk_metadata(c)) for c in chunk_set.chunks)
        dense = tuple(
            DenseCandidate(rank, chunk_set.chunks[i].chunk_id, chunk_set.chunks[i].text, metadata[i], 0.5)
            for rank, i in enumerate((2, 1, 0), 1)
        )
        lexical = tuple(
            LexicalCandidate(rank, chunk_set.chunks[i].chunk_id, chunk_set.chunks[i].text, metadata[i], 1.0)
            for rank, i in enumerate((1, 2, 0), 1)
        )
        result = self._fuse(dense, lexical, chunk_set=chunk_set)
        self.assertEqual([c.chunk_id for c in result], [chunk_set.chunks[i].chunk_id for i in (1, 2, 0)])
        self.assertEqual(result[0].rrf_score, result[1].rrf_score)
        self.assertAlmostEqual(result[0].rrf_score, 7 / 12, places=15)
        self.assertEqual(result, self._fuse(dense, lexical, chunk_set=chunk_set))

    def test_raw_scores_do_not_affect_ranking_or_thresholds(self):
        first = self._fuse(self._dense((0, 1), (0.8, 0.1)), self._lexical((2, 0), (2, 1)))
        second = self._fuse(self._dense((0, 1), (-100, 999)), self._lexical((2, 0), (-1, 0)))
        self.assertEqual([(c.chunk_id, c.rrf_score) for c in first], [(c.chunk_id, c.rrf_score) for c in second])
        self.assertEqual(second[0].dense_candidate.cosine_distance, -100)
        self.assertEqual(second[0].lexical_candidate.bm25_score, 0)

    def test_revalidate_invalid_candidate_fields_and_nonfinite_raw_scores(self):
        for make, branch, score_field in (
            (self._dense, "dense_candidates", "cosine_distance"),
            (self._lexical, "lexical_candidates", "bm25_score"),
        ):
            for field, values in {
                "rank": (True, 0, 1.0), "chunk_id": ("", "wrong-id"),
                "document": ("", " ", None), "metadata": (None, {}),
                score_field: (True, "1", None, 1j, math.nan, math.inf, -math.inf),
            }.items():
                for value in values:
                    # Simulate corrupt inputs that bypassed a frozen constructor.
                    corrupt = replace(make((0,))[0])
                    object.__setattr__(corrupt, field, value)
                    with self.subTest(branch=branch, field=field, value=value), self.assertRaises(FusionError):
                        self._fuse(**{branch: (corrupt,)})

    def test_metadata_exact_types_not_just_equality_are_validated(self):
        metadata = replace(self.metadata[0])
        object.__setattr__(metadata, "permits_mdr", 1)
        self.assertEqual(metadata, self.metadata[0])  # Python bool/int equality trap.
        candidate = replace(self._dense((0,))[0], metadata=metadata)
        with self.assertRaises(FusionError) as caught:
            self._fuse((candidate,))
        self.assertIsInstance(caught.exception.__cause__, DenseResultError)

    def test_output_frozen_slotted_tuple_and_inputs_unchanged(self):
        dense, lexical = self._dense((0, 1)), self._lexical((2, 0))
        before = deepcopy((self.chunk_set, dense, lexical))
        result = self._fuse(dense, lexical)
        self.assertIs(type(result), tuple)
        self.assertFalse(hasattr(result[0], "__dict__"))
        self.assertEqual(tuple(f.name for f in fields(result[0])), (
            "rank", "chunk_id", "document", "metadata", "rrf_score",
            "dense_candidate", "lexical_candidate",
        ))
        with self.assertRaises(TypeError):
            result[0] = result[1]
        with self.assertRaises(FrozenInstanceError):
            result[0].rank = 9
        with self.assertRaises(FrozenInstanceError):
            result[0].metadata.version = "changed"
        self.assertEqual((self.chunk_set, dense, lexical), before)

    def test_direct_fused_candidate_validation(self):
        candidate = self._fuse(self._dense((0,)))[0]
        invalid = {
            "rank": (True, 0, 1.0), "chunk_id": ("", "wrong"),
            "document": ("", "changed"), "metadata": (None, {}),
            "rrf_score": (True, 0, -1, "1", math.nan, math.inf),
            "dense_candidate": (None, self._lexical((0,))[0], self._dense((1,))[0]),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(FusionError):
                    replace(candidate, **{field: value})
        self.assertIs(type(replace(candidate, rrf_score=1).rrf_score), float)

    def test_invalid_computed_contributions_and_scores(self):
        for target in ("_contribution", "math.fsum"):
            for value in (0, -1, True, math.nan, math.inf, -math.inf):
                with self.subTest(target=target, value=value), patch(f"{_MODULE}.{target}", return_value=value):
                    with self.assertRaises(FusionError):
                        self._fuse(self._dense((0,)), self._lexical((0,)))

    def test_late_validation_failure_precedes_all_scoring(self):
        dense = self._dense((0, 1))
        bad_second = replace(dense[1], document="fabricated")
        before = deepcopy(self.chunk_set)
        with patch(f"{_MODULE}.math.fsum", wraps=math.fsum) as scorer:
            with self.assertRaises(FusionError):
                self._fuse((dense[0], bad_second))
        scorer.assert_not_called()
        self.assertEqual(self.chunk_set, before)

    def test_late_calculation_and_output_failures_retain_causes(self):
        failure = RuntimeError("late failure")
        with patch(f"{_MODULE}.math.fsum", side_effect=[1.0, failure]):
            with self.assertRaises(FusionError) as caught:
                self._fuse(self._dense((0, 1)))
        self.assertIs(caught.exception.__cause__, failure)

        def build(*values):
            if values[0] == 2:
                raise failure
            return FusedCandidate(*values)

        with patch(f"{_MODULE}.FusedCandidate", side_effect=build):
            with self.assertRaises(FusionError) as caught:
                self._fuse(self._dense((0, 1)))
        self.assertIs(caught.exception.__cause__, failure)

    def test_metadata_and_numeric_failures_retain_causes(self):
        failure = RuntimeError("boundary failed")
        for target in ("chunk_metadata", "deserialize_chunk_metadata"):
            with self.subTest(target=target), patch(f"{_MODULE}.{target}", side_effect=failure):
                with self.assertRaises(FusionError) as caught:
                    self._fuse(self._dense((0,)))
                self.assertIs(caught.exception.__cause__, failure)

        class BrokenFloat(float):
            def __float__(self):
                raise failure

        with self.assertRaises(FusionError) as caught:
            self._fuse(rank_constant=BrokenFloat(2))
        self.assertIs(caught.exception.__cause__, failure)

        class BrokenComparison(float):
            def __lt__(self, other):
                raise failure

        with self.assertRaises(FusionError) as caught:
            self._fuse(rank_constant=BrokenComparison(2))
        self.assertIs(caught.exception.__cause__, failure)
        with self.assertRaises(FusionError) as caught:
            self._fuse(rank_constant=10 ** 400)
        self.assertIsInstance(caught.exception.__cause__, OverflowError)

    def test_base_exceptions_propagate(self):
        dense = self._dense((0,))
        for target, failure in (("chunk_metadata", KeyboardInterrupt()), ("math.fsum", SystemExit())):
            with self.subTest(target=target), patch(f"{_MODULE}.{target}", side_effect=failure):
                with self.assertRaises(type(failure)) as caught:
                    self._fuse(dense)
                self.assertIs(caught.exception, failure)

    def test_no_retrieval_corpus_model_or_database_operations(self):
        self._fuse(self._dense((0, 1)), self._lexical((2, 0)))
        for guard in self.guards:
            guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
