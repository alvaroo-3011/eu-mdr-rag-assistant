"""Synthetic BM25 acceptance tests: no real corpus, model, or database.

The explicit 1.2/0.75 pair is experimental, not a production default. Score
oracles use independently simplified expressions, not production helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import hashlib
import json
import math
import sys
import unittest
from unittest.mock import patch
import unicodedata

# Guard the new module's initial import as well as every test's execution.
_FORBIDDEN_MODULES = {
    name: None for name in (
        "chromadb", "torch", "transformers", "src.ingestion.corpus_loader",
        "src.ingestion.chunk_set_builder",
    )
}


@contextmanager
def _forbid_production_imports():
    # Restore only guarded entries. Restoring all sys.modules would unload
    # freshly imported NumPy extensions, which cannot be imported twice.
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in _FORBIDDEN_MODULES}
    sys.modules.update(_FORBIDDEN_MODULES)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


with _forbid_production_imports():
    from src.retrieval.lexical_query import (
        BM25Config, BM25Retriever, LexicalCandidate, LexicalError,
        _index_identity, _tokenize,
    )

from src.ingestion.models import RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseResultError, deserialize_chunk_metadata
from tests.unit.ingestion.test_chunk_models import (
    _chunk, _chunk_set, _chunk_set_id, _guidance_source, _manifest, _mdr_source,
    _selection,
)
from tests.unit.retrieval.test_dense_results import _representative_chunk


_MODULE = "src.retrieval.lexical_query"


def _synthetic_chunk_set(*groups):
    """Build valid source-ordered chunks with distinct selections and links."""

    manifest = _manifest(*(source for source, texts in groups))
    chunk_set_id = _chunk_set_id(manifest)
    chunks = []
    for source, texts in groups:
        source_chunks = [
            _chunk(
                chunk_set_id, source, order=order, text=text,
                block_selections=(_selection(source, f"block-{order}", block_order=order),),
            )
            for order, text in enumerate(texts)
        ]
        chunks.extend(
            replace(
                chunk,
                previous_chunk_id=source_chunks[index - 1].chunk_id if index else None,
                next_chunk_id=(
                    source_chunks[index + 1].chunk_id
                    if index + 1 < len(source_chunks) else None
                ),
            )
            for index, chunk in enumerate(source_chunks)
        )
    return _chunk_set(manifest, tuple(chunks))


def _ivdr_source():
    return replace(
        _guidance_source(), document_id="test-ivdr", sha256="c" * 64,
        local_path="data/rag_ready/test_ivdr.json",
        regulatory_regimes=(RegulatoryRegime.IVDR,),
        permitted_regimes=(RegulatoryRegime.IVDR,),
    )


class _SyntheticTestCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.guards = [
            self.enterContext(patch(
                target, side_effect=AssertionError("Production execution is forbidden."),
            ))
            for target in (
                "src.retrieval.embeddings._load_transformers_backend",
                "src.retrieval.dense_index.open_dense_collection",
                "src.retrieval.dense_index.build_dense_index",
            )
        ]

    def _retriever(self, *texts, config=None):
        return BM25Retriever(
            chunk_set=_synthetic_chunk_set((_guidance_source(), texts)),
            config=config if config is not None else BM25Config(k1=1.2, b=0.75),
        )


class ConfigurationAndCandidateTests(_SyntheticTestCase):
    def setUp(self):
        super().setUp()
        self.chunk = _representative_chunk()
        self.metadata = deserialize_chunk_metadata(chunk_metadata(self.chunk))
        self.candidate = LexicalCandidate(
            1, self.chunk.chunk_id, self.chunk.text, self.metadata, 1,
        )

    def test_required_arguments_and_keyword_only_configuration(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        config = BM25Config(k1=1.2, b=0.75)
        for call in (
            lambda: BM25Config(), lambda: BM25Config(k1=1.2),
            lambda: BM25Config(b=0.75), lambda: BM25Config(1.2, 0.75),
            lambda: BM25Config(k1=1.2, b=0.75, tokenizer_version="arbitrary"),
            lambda: BM25Retriever(chunk_set=chunk_set),
            lambda: BM25Retriever(config=config),
            lambda: BM25Retriever(chunk_set, config),
            lambda: BM25Retriever(chunk_set=chunk_set, config=config).retrieve("prrc"),
            lambda: BM25Retriever(chunk_set=chunk_set, config=config).retrieve("prrc", 1),
        ):
            with self.subTest(call=call), self.assertRaises(TypeError):
                call()

    def test_configuration_normalizes_real_numbers_and_signed_zero(self):
        config = BM25Config(k1=Fraction(6, 5), b=-0.0)
        self.assertEqual(config.k1, 1.2)
        self.assertIs(type(config.k1), float)
        self.assertIs(type(config.b), float)
        self.assertEqual(math.copysign(1, config.b), 1)
        self.assertEqual(BM25Config(k1=1, b=1), BM25Config(k1=1.0, b=1.0))

    def test_invalid_configuration_values(self):
        for field, values in (
            ("k1", (0, -1, True, False, None, "1.2", 1j, math.nan, math.inf, -math.inf)),
            ("b", (-0.01, 1.01, True, False, None, "0.75", 1j, math.nan, math.inf)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(LexicalError):
                    BM25Config(**{"k1": 1.2, "b": 0.75, field: value})

    def test_float_conversion_failures_preserve_causes(self):
        failure = RuntimeError("broken numeric conversion")

        class BrokenFloat(float):
            def __float__(self):
                raise failure

        for construct in (
            lambda: BM25Config(k1=BrokenFloat(1), b=0.75),
            lambda: replace(self.candidate, bm25_score=BrokenFloat(1)),
        ):
            with self.assertRaises(LexicalError) as caught:
                construct()
            self.assertIs(caught.exception.__cause__, failure)
        with self.assertRaises(LexicalError) as caught:
            BM25Config(k1=10 ** 400, b=0.75)
        self.assertIsInstance(caught.exception.__cause__, OverflowError)

    def test_configuration_and_candidates_are_frozen_and_slotted(self):
        config = BM25Config(k1=1.2, b=0.75)
        for instance, field in ((config, "k1"), (self.candidate, "rank")):
            with self.assertRaises(FrozenInstanceError):
                setattr(instance, field, 2)
            self.assertFalse(hasattr(instance, "__dict__"))
        self.assertEqual(
            tuple(field.name for field in fields(self.candidate)),
            ("rank", "chunk_id", "document", "metadata", "bm25_score"),
        )

    def test_invalid_candidate_fields(self):
        invalid = {
            "rank": (0, -1, True, 1.0, "1"),
            "chunk_id": (None, "", " ", "wrong-id"),
            "document": (None, "", "\n", 1),
            "metadata": (None, {}, self.chunk),
            "bm25_score": (True, None, "1", 1j, math.nan, math.inf, -math.inf),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(LexicalError):
                    replace(self.candidate, **{field: value})

    def test_candidate_score_is_finite_real_without_sign_threshold(self):
        for score in (0, -1, Fraction(1, 2)):
            candidate = replace(self.candidate, bm25_score=score)
            self.assertEqual(candidate.bm25_score, float(score))
            self.assertIs(type(candidate.bm25_score), float)


class TokenizerTests(_SyntheticTestCase):
    def test_regulatory_tokenization_examples(self):
        examples = {
            "Article 15": ("article", "15"),
            "Annex VIII": ("annex", "viii"),
            "Class IIa / IIb / III": ("class", "iia", "iib", "iii"),
            "PRRC": ("prrc",),
            "MDCG 2019-7 Rev.1": ("mdcg", "2019", "7", "rev", "1"),
            "Regulation (EU) 2017/745": ("regulation", "eu", "2017", "745"),
            "Article 015; 1.5": ("article", "015", "1", "5"),
        }
        for text, tokens in examples.items():
            with self.subTest(text=text):
                self.assertEqual(_tokenize(text), tokens)

    def test_unicode_case_punctuation_and_single_character_tokens(self):
        self.assertEqual(
            _tokenize("ＰＲＲＣ Annex Ⅷ risk-based/risk–based_risk—based; ÉTUDE A 7"),
            ("prrc", "annex", "viii", "risk", "based", "risk", "based",
             "risk", "based", "étude", "a", "7"),
        )
        self.assertEqual(_tokenize("the devices PRRC's"), ("the", "devices", "prrc", "s"))

    def test_numeric_and_identifier_distinctions_are_not_expanded(self):
        self.assertNotEqual(_tokenize("015"), _tokenize("15"))
        self.assertNotEqual(_tokenize("2019-07"), _tokenize("2019-7"))
        self.assertNotEqual(_tokenize("II a"), _tokenize("IIa"))
        self.assertEqual(_tokenize("!!! ___ / —"), ())
        self.assertEqual(self._retriever("PRRC").retrieve("regulatory compliance", candidate_count=1), ())

    def test_document_and_query_use_same_tokens_without_changing_text(self):
        text = "  ＰＲＲＣ / Regulation (EU) 2017/745; ÉTUDE\n"
        retriever = self._retriever(text)
        candidates = retriever.retrieve("prrc étude", candidate_count=1)
        self.assertEqual(candidates[0].document, text)
        self.assertEqual(candidates[0].chunk_id, retriever._records[0].chunk.chunk_id)
        self.assertEqual(retriever.retrieve("etude", candidate_count=1), ())


class ScoringAndOrderingTests(_SyntheticTestCase):
    def test_independent_tiny_corpus_score_oracle(self):
        retriever = self._retriever("prrc prrc", "prrc device", "device")
        candidates = retriever.retrieve("prrc", candidate_count=5)
        self.assertEqual([c.document for c in candidates], ["prrc prrc", "prrc device"])
        self.assertAlmostEqual(candidates[0].bm25_score, (220 / 169) * math.log(8 / 5), places=12)
        self.assertAlmostEqual(candidates[1].bm25_score, (110 / 119) * math.log(8 / 5), places=12)

    def test_document_frequency_counts_once_despite_repetitions(self):
        candidates = self._retriever(
            "prrc prrc prrc prrc", "device", config=BM25Config(k1=1.2, b=0),
        ).retrieve("prrc", candidate_count=5)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].bm25_score, (22 / 13) * math.log(2), places=12)

    def test_frequency_saturation(self):
        candidates = self._retriever(
            "prrc", "prrc prrc", "prrc prrc prrc prrc", config=BM25Config(k1=1.2, b=0),
        ).retrieve("prrc", candidate_count=3)
        scores = {c.document: c.bm25_score for c in candidates}
        one, two, four = (scores[" ".join(["prrc"] * n)] for n in (1, 2, 4))
        self.assertAlmostEqual(two / one, 11 / 8, places=12)
        self.assertAlmostEqual(four / one, 22 / 13, places=12)
        self.assertLess((four - two) / 2, two - one)

    def test_length_effect_and_b_endpoints(self):
        texts = ("prrc", "prrc device device device")
        ordinary = self._retriever(*texts).retrieve("prrc", candidate_count=2)
        self.assertGreater(ordinary[0].bm25_score, ordinary[1].bm25_score)
        no_length = self._retriever(
            *texts, config=BM25Config(k1=1.2, b=0),
        ).retrieve("prrc", candidate_count=2)
        self.assertEqual(no_length[0].bm25_score, no_length[1].bm25_score)
        full_length = self._retriever(
            *texts, config=BM25Config(k1=1.2, b=1),
        ).retrieve("prrc", candidate_count=2)
        self.assertAlmostEqual(full_length[0].bm25_score, (55 / 37) * math.log(6 / 5), places=12)
        self.assertAlmostEqual(full_length[1].bm25_score, (55 / 73) * math.log(6 / 5), places=12)

    def test_multiple_terms_and_query_frequency_invariance(self):
        retriever = self._retriever("prrc device", "prrc", "device", config=BM25Config(k1=1.2, b=0))
        candidates = retriever.retrieve("prrc device", candidate_count=3)
        self.assertAlmostEqual(candidates[0].bm25_score, 2 * math.log(8 / 5), places=12)
        self.assertEqual([c.document for c in candidates], ["prrc device", "prrc", "device"])
        self.assertEqual(candidates, retriever.retrieve("device PRRC prrc device", candidate_count=3))
        self.assertEqual(candidates, retriever.retrieve("prrc unseen device", candidate_count=3))

    def test_sorted_unique_query_tokens_and_overlap_only_scoring(self):
        retriever = self._retriever("prrc device", "nothing")
        with patch.object(BM25Retriever, "_score", autospec=True, return_value=1) as scorer:
            candidates = retriever.retrieve("PRRC device prrc", candidate_count=2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(scorer.call_args.args[2], ("device", "prrc"))

    def test_common_terms_and_nonpositive_scores_are_not_overlap_filters(self):
        retriever = self._retriever("prrc", "prrc", "device")
        candidates = retriever.retrieve("prrc", candidate_count=3)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(c.bm25_score > 0 for c in candidates))
        for score in (0, -1):
            with self.subTest(score=score), patch.object(BM25Retriever, "_score", return_value=score):
                candidates = retriever.retrieve("prrc", candidate_count=3)
                self.assertEqual(len(candidates), 2)
                self.assertTrue(all(c.bm25_score == score for c in candidates))
        ubiquitous = self._retriever("prrc", "prrc").retrieve("prrc", candidate_count=2)
        self.assertAlmostEqual(ubiquitous[0].bm25_score, math.log(6 / 5), places=12)

    def test_cross_source_ties_follow_global_not_source_local_order(self):
        chunk_set = _synthetic_chunk_set(
            (_guidance_source(), ("prrc", "prrc")), (_mdr_source(), ("prrc",)),
        )
        self.assertEqual([c.order for c in chunk_set.chunks], [0, 1, 0])
        retriever = BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        candidates = retriever.retrieve("prrc", candidate_count=3)
        self.assertEqual([c.chunk_id for c in candidates], [c.chunk_id for c in chunk_set.chunks])
        self.assertEqual(candidates, retriever.retrieve("prrc", candidate_count=3))

    def test_counts_ranks_tuple_and_no_padding(self):
        retriever = self._retriever("prrc", "prrc", "device")
        for count, expected in ((1, 1), (2, 2), (3, 2), (100, 2)):
            with self.subTest(count=count):
                candidates = retriever.retrieve("prrc", candidate_count=count)
                self.assertIs(type(candidates), tuple)
                self.assertEqual(len(candidates), expected)
                self.assertEqual([c.rank for c in candidates], list(range(1, expected + 1)))
        self.assertEqual(retriever.retrieve("unknown", candidate_count=5), ())


class PermissionAndMetadataTests(_SyntheticTestCase):
    def test_ivdr_is_excluded_before_statistics_and_ranking(self):
        source = _guidance_source()
        texts = ("prrc prrc", "prrc device", "device")
        baseline = _synthetic_chunk_set((source, texts))
        expanded = _synthetic_chunk_set((_ivdr_source(), ("prrc " * 100,)), (source, texts))
        results = [
            BM25Retriever(chunk_set=cs, config=BM25Config(k1=1.2, b=0.75)).retrieve("prrc", candidate_count=5)
            for cs in (baseline, expanded)
        ]
        # Adding a manifest source changes canonical IDs, but not eligible scores/order.
        self.assertEqual(
            [(c.document, c.bm25_score, c.metadata.document_id) for c in results[0]],
            [(c.document, c.bm25_score, c.metadata.document_id) for c in results[1]],
        )
        self.assertAlmostEqual(results[1][0].bm25_score, (220 / 169) * math.log(8 / 5), places=12)

    def test_dual_permission_retained_but_declared_mdr_without_permission_excluded(self):
        dual = replace(
            _guidance_source(), regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
            permitted_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
        )
        disallowed = replace(_ivdr_source(), regulatory_regimes=dual.regulatory_regimes)
        chunk_set = _synthetic_chunk_set((disallowed, ("prrc",)), (dual, ("prrc",)))
        candidates = BM25Retriever(
            chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75),
        ).retrieve("prrc", candidate_count=5)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].metadata.permits_mdr)
        self.assertTrue(candidates[0].metadata.permits_ivdr)

    def test_filter_precedes_tokenization_and_metadata_boundaries(self):
        chunk_set = _synthetic_chunk_set((_ivdr_source(), ("!!!",)), (_guidance_source(), ("prrc",)))
        with (
            patch(f"{_MODULE}._tokenize", wraps=_tokenize) as tokenizer,
            patch(f"{_MODULE}.chunk_metadata", wraps=chunk_metadata) as serializer,
            patch(f"{_MODULE}.deserialize_chunk_metadata", wraps=deserialize_chunk_metadata) as decoder,
        ):
            BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        tokenizer.assert_called_once_with("prrc")
        serializer.assert_called_once_with(chunk_set.chunks[1])
        self.assertEqual(decoder.call_count, 1)

    def test_tokenless_eligible_document_reports_id(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("!!!",)))
        with self.assertRaises(LexicalError) as caught:
            BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        self.assertIn(chunk_set.chunks[0].chunk_id, str(caught.exception))

    def test_all_ivdr_is_valid_empty_index(self):
        chunk_set = _synthetic_chunk_set((_ivdr_source(), ("!!!", "prrc")))
        retriever = BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        self.assertEqual(retriever.retrieve("prrc", candidate_count=5), ())
        self.assertEqual(retriever.chunk_set_id, chunk_set.chunk_set_id)
        with self.assertRaises(LexicalError):
            retriever.retrieve("!!!", candidate_count=1)

    def test_complete_metadata_text_ids_and_chunk_set_are_preserved(self):
        first = replace(_representative_chunk(), order=0, previous_chunk_id=None, next_chunk_id=None)
        second = _chunk(first.chunk_set_id, first.source, order=1, text="Synthetic second chunk.")
        chunks = (
            replace(first, next_chunk_id=second.chunk_id),
            replace(second, previous_chunk_id=first.chunk_id),
        )
        chunk_set = _chunk_set(_manifest(first.source), chunks)
        before = deepcopy(chunk_set)
        retriever = BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        candidates = retriever.retrieve("synthetic", candidate_count=5)
        original = {c.chunk_id: c for c in chunks}
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            chunk = original[candidate.chunk_id]
            self.assertEqual(candidate.document, chunk.text)
            self.assertEqual(candidate.metadata, deserialize_chunk_metadata(chunk_metadata(chunk)))
            self.assertEqual(candidate.metadata.chunk_set_id, chunk_set.chunk_set_id)
            with self.assertRaises(FrozenInstanceError):
                candidate.metadata.chunk_order = 10
        self.assertEqual(chunk_set, before)
        self.assertIs(chunk_set.chunks, chunks)

    def test_metadata_fields_are_not_appended_to_scoring_text(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        retriever = BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        self.assertEqual(retriever.retrieve("guidance", candidate_count=1), ())

    def test_metadata_identity_id_permission_and_type_are_rechecked(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        metadata = deserialize_chunk_metadata(chunk_metadata(chunk_set.chunks[0]))
        invalid = (
            replace(metadata, chunk_set_id="unexpected"),
            replace(metadata, chunk_id="unexpected"),
            replace(
                metadata, regulatory_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
                permitted_regimes=(RegulatoryRegime.IVDR,), permits_mdr=False, permits_ivdr=True,
            ),
            None,
        )
        for value in invalid:
            with self.subTest(value=value), patch(f"{_MODULE}.deserialize_chunk_metadata", return_value=value):
                with self.assertRaises(LexicalError):
                    BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))

    def test_malformed_metadata_preserves_decoder_error(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        raw = chunk_metadata(chunk_set.chunks[0])
        raw["pages_json"] = "not JSON"
        with patch(f"{_MODULE}.chunk_metadata", return_value=raw):
            with self.assertRaises(LexicalError) as caught:
                BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        self.assertIsInstance(caught.exception.__cause__, DenseResultError)


class IdentityAndFailureTests(_SyntheticTestCase):
    def test_read_only_properties(self):
        retriever = self._retriever("prrc")
        self.assertEqual(retriever.config, BM25Config(k1=1.2, b=0.75))
        for name in ("config", "chunk_set_id", "index_id"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(retriever, name, "replacement")

    def test_identity_stability_parameter_and_eligible_chunk_sensitivity(self):
        first = self._retriever("prrc")
        self.assertEqual(first.index_id, self._retriever("prrc").index_id)
        self.assertNotEqual(first.index_id, self._retriever("prrc", config=BM25Config(k1=2, b=0.75)).index_id)
        self.assertNotEqual(first.index_id, self._retriever("prrc", config=BM25Config(k1=1.2, b=0)).index_id)
        changed = self._retriever("device")
        self.assertEqual(first.chunk_set_id, changed.chunk_set_id)
        self.assertNotEqual(first.index_id, changed.index_id)
        self.assertEqual(
            self._retriever("prrc", config=BM25Config(k1=1, b=-0.0)).index_id,
            self._retriever("prrc", config=BM25Config(k1=1.0, b=0.0)).index_id,
        )

    def test_identity_binds_order_versions_unicode_and_fixed_policies(self):
        retriever = self._retriever("prrc", "device")
        ids = tuple(record.chunk.chunk_id for record in retriever._records)
        self.assertNotEqual(retriever.index_id, _index_identity(retriever.chunk_set_id, ids[::-1], retriever.config))
        for name in ("_TOKENIZER_VERSION", "_SCORER_VERSION"):
            with self.subTest(name=name), patch(f"{_MODULE}.{name}", "future-version"):
                self.assertNotEqual(retriever.index_id, _index_identity(retriever.chunk_set_id, ids, retriever.config))
        with patch(f"{_MODULE}.unicodedata.unidata_version", "future-unicode"):
            self.assertNotEqual(retriever.index_id, _index_identity(retriever.chunk_set_id, ids, retriever.config))
        with patch(f"{_MODULE}._POLICIES", (("filter", "different-policy"),)):
            self.assertNotEqual(retriever.index_id, _index_identity(retriever.chunk_set_id, ids, retriever.config))

    def test_identity_canonical_payload_and_no_query_count_or_dense_identity(self):
        dumps = json.dumps
        with patch(f"{_MODULE}.json.dumps", wraps=dumps) as serializer:
            retriever = self._retriever("prrc")
        payload = serializer.call_args.args[0]
        self.assertEqual(payload, {
            "schema_version": "1.0", "chunk_set_id": retriever.chunk_set_id,
            "eligible_chunk_ids": tuple(r.chunk.chunk_id for r in retriever._records),
            "k1": 1.2, "b": 0.75,
            "tokenizer_version": "unicode-nfkc-casefold-alnum-v1",
            "scorer_version": "bm25-positive-idf-k1-plus-one-v1",
            "unicode_database_version": unicodedata.unidata_version,
            "policies": {
                "filter": "permitted-regimes-mdr-v1", "text": "original-chunk-text-v1",
                "query_tokens": "sorted-unique-v1", "tie_order": "original-chunkset-position-v1",
                "tokenless_query": "error-v1", "tokenless_document": "error-v1",
            },
        })
        canonical = dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_id = "lexical-index:sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(retriever.index_id, expected_id)
        retriever.retrieve("prrc", candidate_count=1)
        retriever.retrieve("unknown", candidate_count=5)
        self.assertEqual(retriever.index_id, expected_id)

    def test_invalid_constructor_types(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        config = BM25Config(k1=1.2, b=0.75)
        for values in (
            {"chunk_set": None, "config": config},
            {"chunk_set": chunk_set.chunks, "config": config},
            {"chunk_set": chunk_set, "config": None},
            {"chunk_set": chunk_set, "config": {"k1": 1.2, "b": 0.75}},
        ):
            with self.subTest(values=values), self.assertRaises(LexicalError):
                BM25Retriever(**values)

    def test_invalid_queries_and_counts(self):
        retriever = self._retriever("prrc")
        for query in (None, 1, True, [], "", " \n", "!!!", "___"):
            with self.subTest(query=query), self.assertRaises(LexicalError):
                retriever.retrieve(query, candidate_count=1)
        for count in (None, 0, -1, True, False, 1.0, "1", math.nan, math.inf):
            with self.subTest(count=count), self.assertRaises(LexicalError):
                retriever.retrieve("prrc", candidate_count=count)

    def test_constructor_internal_failures_preserve_causes(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        for target in ("_tokenize", "chunk_metadata", "deserialize_chunk_metadata", "_index_identity"):
            failure = RuntimeError(target)
            with self.subTest(target=target), patch(f"{_MODULE}.{target}", side_effect=failure):
                with self.assertRaises(LexicalError) as caught:
                    BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
                self.assertIs(caught.exception.__cause__, failure)

    def test_query_tokenizer_failure_preserves_cause(self):
        retriever = self._retriever("prrc")
        failure = RuntimeError("tokenizer unavailable")
        with patch(f"{_MODULE}._tokenize", side_effect=failure):
            with self.assertRaises(LexicalError) as caught:
                retriever.retrieve("prrc", candidate_count=1)
        self.assertIs(caught.exception.__cause__, failure)

    def test_nonfinite_and_nonreal_scores_fail_closed(self):
        retriever = self._retriever("prrc")
        for score in (math.nan, math.inf, -math.inf, True, "1", None):
            with self.subTest(score=score), patch.object(BM25Retriever, "_score", return_value=score):
                with self.assertRaises(LexicalError):
                    retriever.retrieve("prrc", candidate_count=1)
        overflowing = self._retriever("prrc prrc", config=BM25Config(k1=1e308, b=0))
        with self.assertRaises(LexicalError):
            overflowing.retrieve("prrc", candidate_count=1)

    def test_scoring_failure_in_later_match_prevents_partial_results(self):
        retriever = self._retriever("prrc", "prrc")
        failure = RuntimeError("second score failed")
        with patch.object(BM25Retriever, "_score", side_effect=[1.0, failure]):
            with self.assertRaises(LexicalError) as caught:
                retriever.retrieve("prrc", candidate_count=1)
        self.assertIs(caught.exception.__cause__, failure)

    def test_candidate_failure_prevents_partial_results(self):
        retriever = self._retriever("prrc", "prrc")
        failure = RuntimeError("second candidate failed")

        def build(*values):
            if values[0] == 2:
                raise failure
            return LexicalCandidate(*values)

        with patch(f"{_MODULE}.LexicalCandidate", side_effect=build):
            with self.assertRaises(LexicalError) as caught:
                retriever.retrieve("prrc", candidate_count=2)
        self.assertIs(caught.exception.__cause__, failure)

    def test_base_exceptions_are_not_swallowed(self):
        chunk_set = _synthetic_chunk_set((_guidance_source(), ("prrc",)))
        with patch(f"{_MODULE}._tokenize", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
        retriever = self._retriever("prrc")
        with patch.object(BM25Retriever, "_score", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                retriever.retrieve("prrc", candidate_count=1)

    def test_no_production_model_database_or_corpus_execution(self):
        self._retriever("prrc").retrieve("prrc", candidate_count=1)
        for guard in self.guards:
            guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
