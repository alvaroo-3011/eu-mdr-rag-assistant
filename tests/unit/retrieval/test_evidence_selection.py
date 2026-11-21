"""Synthetic selection tests: no corpus, model, retrieval, or database access."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import math
import unittest
from unittest.mock import patch

from tests.unit.retrieval.test_lexical_query import (
    _forbid_production_imports, _ivdr_source, _synthetic_chunk_set,
)

with _forbid_production_imports():
    from src.retrieval.evidence_selection import (
        EvidenceSelection, EvidenceSelectionError, SelectedEvidence, select_evidence,
    )

from src.ingestion.models import PageIdentity, RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.fusion import FusedCandidate, fuse_candidates
from src.retrieval.lexical_query import LexicalCandidate
from tests.unit.ingestion.test_chunk_models import _chunk, _chunk_set, _manifest, _selection
from tests.unit.retrieval.test_dense_results import _representative_chunk


_MODULE = "src.retrieval.evidence_selection"


def _change_record(candidate, **changes):
    """Keep branch/fused records internally consistent for original-match tests."""

    record = {
        field: changes.get(field, getattr(candidate, field))
        for field in ("chunk_id", "document", "metadata")
    }
    branches = {
        field: replace(getattr(candidate, field), **record)
        if getattr(candidate, field) is not None else None
        for field in ("dense_candidate", "lexical_candidate")
    }
    return replace(candidate, **record, **branches)


class EvidenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.guards = [
            self.enterContext(patch(target, side_effect=AssertionError("Real execution is forbidden.")))
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
            _chunk(
                first.chunk_set_id, first.source, order=i,
                text=f"Synthetic distinct evidence {i}.",
                block_selections=(_selection(first.source, f"body-{i}", block_order=i + 1),),
            )
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
        # Synthetic rank constant 2 is only a fixture. These branches produce
        # a fused order different from original ChunkSet order: 2, 0, 1.
        self.dense = self._dense((2, 1, 0))
        self.lexical = self._lexical((2, 0))
        self.fused = self._fuse(self.dense, self.lexical)

    def _dense(self, indexes):
        return tuple(
            DenseCandidate(rank, self.chunks[i].chunk_id, self.chunks[i].text, self.metadata[i], 0.5)
            for rank, i in enumerate(indexes, 1)
        )

    def _lexical(self, indexes):
        return tuple(
            LexicalCandidate(rank, self.chunks[i].chunk_id, self.chunks[i].text, self.metadata[i], 1.0)
            for rank, i in enumerate(indexes, 1)
        )

    def _fuse(self, dense=(), lexical=()):
        return fuse_candidates(
            chunk_set=self.chunk_set, dense_candidates=dense,
            lexical_candidates=lexical, rank_constant=2,
        )

    def _select(self, **changes):
        return select_evidence(**{
            "chunk_set": self.chunk_set, "fused_candidates": self.fused,
            "max_chunks": 2, **changes,
        })

    def test_prefix_uses_fused_order_and_reports_exact_remaining_ids(self):
        self.assertEqual([c.chunk_id for c in self.fused], [self.chunks[i].chunk_id for i in (2, 0, 1)])
        for count in (1, 2):
            with self.subTest(count=count):
                result = self._select(max_chunks=count)
                self.assertEqual(tuple(p.candidate for p in result.selected), self.fused[:count])
                self.assertEqual(result.omitted_chunk_ids, tuple(c.chunk_id for c in self.fused[count:]))
                self.assertTrue(result.has_omitted_candidates)
                self.assertEqual(result.max_chunks, count)
                self.assertEqual(result.chunk_set_id, self.chunk_set.chunk_set_id)

    def test_equal_or_larger_limit_retains_entire_union_without_padding(self):
        for count in (3, 4, 10 ** 100):
            with self.subTest(count=count):
                result = self._select(max_chunks=count)
                self.assertEqual(tuple(p.candidate for p in result.selected), self.fused)
                self.assertEqual(result.omitted_chunk_ids, ())
                self.assertFalse(result.has_omitted_candidates)

    def test_original_objects_and_raw_branch_provenance_are_retained(self):
        before = deepcopy((self.chunk_set, self.fused))
        result = self._select(max_chunks=3)
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        for candidate, pair in zip(self.fused, result.selected):
            self.assertIs(pair.candidate, candidate)
            self.assertIs(pair.chunk, by_id[candidate.chunk_id])
            self.assertIs(pair.candidate.dense_candidate, candidate.dense_candidate)
            self.assertIs(pair.candidate.lexical_candidate, candidate.lexical_candidate)
        self.assertIs(result.selected[0].candidate.dense_candidate, self.dense[0])
        self.assertIs(result.selected[1].candidate.dense_candidate, self.dense[2])
        self.assertEqual(result.selected[1].candidate.dense_candidate.rank, 3)
        self.assertEqual((self.chunk_set, self.fused), before)

    def test_complete_structured_context_and_dual_applicability_survive(self):
        result = self._select()
        pair = next(p for p in result.selected if p.chunk is self.chunks[0])
        chunk = pair.chunk
        for field in (
            "source", "block_selections", "hierarchies", "pages", "legal_locators",
            "table_segments", "annotations",
        ):
            self.assertIs(getattr(chunk, field), getattr(self.chunks[0], field))
        self.assertEqual(chunk.table_segments[0].table.rows, (("A", "I"),))
        self.assertEqual(chunk.annotations[0].raw_html, "<b>M1</b>")
        self.assertEqual(chunk.pages, (PageIdentity(2, 3), PageIdentity(4, 5)))
        self.assertEqual(chunk.next_chunk_id, self.chunks[1].chunk_id)
        self.assertTrue(pair.candidate.metadata.permits_mdr)
        self.assertTrue(pair.candidate.metadata.permits_ivdr)
        # The rich table record survives even though it has no candidate field.
        self.assertFalse(hasattr(pair.candidate.metadata, "table_segments"))

    def test_outputs_are_frozen_and_slotted(self):
        result = self._select()
        for instance, field in ((result, "max_chunks"), (result.selected[0], "chunk")):
            with self.assertRaises(FrozenInstanceError):
                setattr(instance, field, None)
            self.assertFalse(hasattr(instance, "__dict__"))

    def test_empty_input_is_valid_but_still_checks_context_and_limit(self):
        result = self._select(fused_candidates=())
        self.assertEqual(result.selected, ())
        self.assertEqual(result.omitted_chunk_ids, ())
        self.assertFalse(result.has_omitted_candidates)
        for changes in ({"max_chunks": 0}, {"max_chunks": True}, {"chunk_set": None}):
            with self.subTest(changes=changes), self.assertRaises(EvidenceSelectionError):
                self._select(fused_candidates=(), **changes)

    def test_arguments_are_required_and_keyword_only(self):
        values = dict(chunk_set=self.chunk_set, fused_candidates=self.fused, max_chunks=2)
        for field in values:
            with self.subTest(missing=field), self.assertRaises(TypeError):
                select_evidence(**{k: v for k, v in values.items() if k != field})
        with self.assertRaises(TypeError):
            select_evidence(self.chunk_set, self.fused, 2)

    def test_invalid_limits_are_rejected(self):
        for count in (0, -1, True, False, None, "2", 2.0, math.nan, math.inf, 1j):
            with self.subTest(count=count), self.assertRaises(EvidenceSelectionError):
                self._select(max_chunks=count)

    def test_invalid_contexts_and_containers_are_rejected(self):
        for context in (None, self.chunks, {}, object()):
            with self.subTest(context=context), self.assertRaises(EvidenceSelectionError):
                self._select(chunk_set=context)
        class TupleSubclass(tuple):
            pass
        for values in (None, [], {}, "", list(self.fused), iter(self.fused), TupleSubclass(self.fused)):
            with self.subTest(values=type(values)), self.assertRaises(EvidenceSelectionError):
                self._select(fused_candidates=values)

    def test_wrong_candidate_types_fail_even_beyond_limit(self):
        for value in (None, object(), self.dense[0], self.lexical[0]):
            with self.subTest(value=type(value)), self.assertRaises(EvidenceSelectionError):
                self._select(fused_candidates=(*self.fused[:2], value), max_chunks=1)

    def test_duplicate_tail_record_is_not_hidden_by_truncation(self):
        duplicate = replace(self.fused[0], rank=3)
        with self.assertRaisesRegex(EvidenceSelectionError, "Duplicate"):
            self._select(fused_candidates=(*self.fused[:2], duplicate), max_chunks=1)

    def test_fused_rank_gaps_repetitions_and_reordering_fail(self):
        for values in (
            (replace(self.fused[0], rank=2), *self.fused[1:]),
            (*self.fused[:2], replace(self.fused[2], rank=2)),
            (*self.fused[:2], replace(self.fused[2], rank=4)),
            self.fused[::-1],
        ):
            with self.subTest(ranks=[v.rank for v in values]), self.assertRaises(EvidenceSelectionError):
                self._select(fused_candidates=values, max_chunks=1)

    def test_unknown_tail_id_is_not_hidden_by_truncation(self):
        chunk = _chunk(
            self.chunk_set.chunk_set_id, self.chunks[0].source, order=3,
            text="Not a member of the supplied ChunkSet.",
        )
        metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
        branch = DenseCandidate(3, chunk.chunk_id, chunk.text, metadata, 0.5)
        unknown = FusedCandidate(3, chunk.chunk_id, chunk.text, metadata, 0.1, branch, None)
        with self.assertRaisesRegex(EvidenceSelectionError, "absent"):
            self._select(fused_candidates=(*self.fused[:2], unknown), max_chunks=1)

    def test_foreign_chunk_set_and_wrong_metadata_identity_fail(self):
        foreign = _synthetic_chunk_set((self.chunks[0].source, ("Foreign evidence.",)))
        with self.assertRaises(EvidenceSelectionError):
            self._select(chunk_set=foreign)
        tail = self.fused[-1]
        altered = _change_record(tail, metadata=replace(tail.metadata, chunk_set_id="another-set"))
        with self.assertRaisesRegex(EvidenceSelectionError, "ChunkSet identity"):
            self._select(fused_candidates=(*self.fused[:2], altered), max_chunks=1)

    def test_original_text_mismatch_in_tail_fails(self):
        altered = _change_record(self.fused[-1], document="Altered evidence.")
        with self.assertRaisesRegex(EvidenceSelectionError, "text differs"):
            self._select(fused_candidates=(*self.fused[:2], altered), max_chunks=1)

    def test_original_metadata_mismatches_in_tail_fail(self):
        tail = self.fused[-1]
        for changes in (
            {"official_title": "Another title"}, {"version": "Another version"},
            {"pages": (PageIdentity(0, 1),)}, {"hierarchies": ()},
            {"legal_locators": ()}, {"has_table": True},
            {"previous_chunk_id": None}, {"chunk_order": 99},
            {"official_url": "https://example.eu/another"},
        ):
            # The tail is a plain chunk; use a genuinely changed empty field.
            if changes in ({"hierarchies": ()}, {"legal_locators": ()}):
                original = self.fused[1]
                altered = _change_record(original, metadata=replace(original.metadata, **changes))
                values = (self.fused[0], altered, self.fused[2])
            else:
                altered = _change_record(tail, metadata=replace(tail.metadata, **changes))
                values = (*self.fused[:2], altered)
            with self.subTest(changes=changes), self.assertRaisesRegex(EvidenceSelectionError, "metadata differs"):
                self._select(fused_candidates=values, max_chunks=1)

    def test_invalid_fused_fields_are_revalidated_after_unsafe_mutation(self):
        for field, values in (
            ("rank", (True, 0)), ("rrf_score", (0, -1, True, math.nan, math.inf)),
            ("chunk_id", (None, [])), ("metadata", (None,)),
        ):
            for value in values:
                damaged = deepcopy(self.fused[-1])
                object.__setattr__(damaged, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(EvidenceSelectionError):
                    self._select(fused_candidates=(*self.fused[:2], damaged), max_chunks=1)

    def test_invalid_nested_branch_and_metadata_are_revalidated(self):
        for field in ("branch", "metadata"):
            damaged = deepcopy(self.fused[-1])
            if field == "branch":
                object.__setattr__(damaged.dense_candidate, "document", "Conflicting text")
            else:
                object.__setattr__(damaged.metadata, "chunk_order", True)
            with self.subTest(field=field), self.assertRaises(EvidenceSelectionError):
                self._select(fused_candidates=(*self.fused[:2], damaged), max_chunks=1)

    def test_score_order_is_checked_without_reordering(self):
        altered = replace(self.fused[-1], rrf_score=10.0)
        values = (*self.fused[:2], altered)
        before = deepcopy(values)
        with self.assertRaisesRegex(EvidenceSelectionError, "ordering"):
            self._select(fused_candidates=values, max_chunks=1)
        self.assertEqual(values, before)

    def test_exact_ties_require_global_chunk_set_order(self):
        tied = self._fuse(self._dense((2,)), self._lexical((0,)))
        self.assertEqual(tied[0].chunk_id, self.chunks[0].chunk_id)
        self.assertEqual(len(self._select(fused_candidates=tied).selected), 2)
        wrong = tuple(replace(c, rank=i) for i, c in enumerate(tied[::-1], 1))
        with self.assertRaisesRegex(EvidenceSelectionError, "ordering"):
            self._select(fused_candidates=wrong, max_chunks=1)

    def test_duplicate_or_gapped_branch_ranks_in_omitted_tail_fail(self):
        tail = self.fused[-1]
        for rank in (1, 99):
            altered = replace(tail, dense_candidate=replace(tail.dense_candidate, rank=rank))
            with self.subTest(rank=rank), self.assertRaisesRegex(EvidenceSelectionError, "branch ranks"):
                self._select(fused_candidates=(*self.fused[:2], altered), max_chunks=1)

    def test_single_branch_unions_are_valid(self):
        for values in (self._fuse(self._dense((2, 0))), self._fuse((), self._lexical((1, 2)))):
            with self.subTest(dense=values[0].dense_candidate is not None):
                result = self._select(fused_candidates=values, max_chunks=1)
                self.assertIs(result.selected[0].candidate, values[0])
                self.assertEqual(result.omitted_chunk_ids, (values[1].chunk_id,))

    def test_original_mdr_permission_is_required_even_with_forged_metadata(self):
        for source in (
            _ivdr_source(),
            replace(_ivdr_source(), regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR)),
        ):
            chunk_set = _synthetic_chunk_set((source, ("IVDR-permitted evidence only.",)))
            chunk = chunk_set.chunks[0]
            metadata = replace(
                deserialize_chunk_metadata(chunk_metadata(chunk)),
                regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
                permitted_regimes=(RegulatoryRegime.MDR,), permits_mdr=True, permits_ivdr=False,
            )
            branch = DenseCandidate(1, chunk.chunk_id, chunk.text, metadata, 0.5)
            forged = FusedCandidate(1, chunk.chunk_id, chunk.text, metadata, 1.0, branch, None)
            with self.subTest(regimes=source.regulatory_regimes), self.assertRaisesRegex(EvidenceSelectionError, "permit MDR"):
                self._select(chunk_set=chunk_set, fused_candidates=(forged,))

    def test_changed_original_structured_content_fails_canonical_chunk_checks(self):
        damaged_set = deepcopy(self.chunk_set)
        # Projection has_table remains true, but the structured cell changes.
        table = damaged_set.chunks[0].table_segments[0].table
        object.__setattr__(table, "rows", (("Altered cell", "I"),))
        with self.assertRaises(EvidenceSelectionError) as caught:
            self._select(chunk_set=damaged_set, max_chunks=1)
        self.assertIsNotNone(caught.exception.__cause__)

    def test_ordinary_failures_preserve_the_cause_without_partial_output(self):
        failure = RuntimeError("synthetic serialization failure")
        with patch(f"{_MODULE}.chunk_metadata", side_effect=failure):
            with self.assertRaises(EvidenceSelectionError) as caught:
                self._select()
        self.assertIs(caught.exception.__cause__, failure)

    def test_base_exception_propagates(self):
        failure = KeyboardInterrupt("synthetic interrupt")
        with patch(f"{_MODULE}.chunk_metadata", side_effect=failure):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self._select()
        self.assertIs(caught.exception, failure)

    def test_selection_does_not_read_files_retrieve_or_recompute_fusion(self):
        with patch("builtins.open", side_effect=AssertionError("No file access")), patch(
            "src.retrieval.fusion.fuse_candidates", side_effect=AssertionError("No fusion")
        ):
            self.assertEqual(len(self._select().selected), 2)
        for guard in self.guards:
            guard.assert_not_called()

    def test_raw_scores_are_not_thresholded_and_rrf_is_not_recomputed(self):
        dense = tuple(replace(c, cosine_distance=-50.0) for c in self.dense)
        lexical = tuple(replace(c, bm25_score=-100.0) for c in self.lexical)
        values = self._fuse(dense, lexical)
        # Positive, ordered scores are preserved, not treated as probabilities.
        values = (replace(values[0], rrf_score=100.0), *values[1:])
        result = self._select(fused_candidates=values)
        self.assertIs(result.selected[0].candidate, values[0])
        self.assertEqual(result.selected[0].candidate.rrf_score, 100.0)
        self.assertEqual(result.selected[0].candidate.dense_candidate.cosine_distance, -50.0)
        self.assertEqual(result.selected[0].candidate.lexical_candidate.bm25_score, -100.0)

    def test_direct_pair_construction_checks_consistency(self):
        with self.assertRaises(EvidenceSelectionError):
            SelectedEvidence(candidate=self.fused[0], chunk=self.chunks[0])
        with self.assertRaises(EvidenceSelectionError):
            SelectedEvidence(candidate=None, chunk=self.chunks[0])
        with self.assertRaises(EvidenceSelectionError):
            SelectedEvidence(candidate=self.fused[0], chunk=None)

    def test_direct_result_construction_rejects_inconsistent_envelopes(self):
        result = self._select()
        values = {field.name: getattr(result, field.name) for field in fields(result)}
        for changes in (
            {"chunk_set_id": ""}, {"chunk_set_id": "another-set"},
            {"max_chunks": True}, {"max_chunks": 1},
            {"selected": list(result.selected)}, {"selected": (None,)},
            {"selected": result.selected[::-1]},
            {"omitted_chunk_ids": []}, {"omitted_chunk_ids": (None,)},
            {"omitted_chunk_ids": (result.selected[0].chunk.chunk_id,)},
        ):
            with self.subTest(changes=changes), self.assertRaises(EvidenceSelectionError):
                EvidenceSelection(**{**values, **changes})


if __name__ == "__main__":
    unittest.main()
