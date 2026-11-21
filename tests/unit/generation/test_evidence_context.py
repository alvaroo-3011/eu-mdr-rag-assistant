"""Synthetic context tests; no frozen corpus, SDK, model, or API execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.unit.retrieval.test_lexical_query import (
    _forbid_production_imports, _ivdr_source, _synthetic_chunk_set,
)

with _forbid_production_imports():
    from src.generation.evidence_context import (
        EVIDENCE_CONTEXT_FORMAT, EvidenceCitation, EvidenceContext,
        EvidenceContextError, build_evidence_context,
    )

from src.ingestion.chunk_models import Chunk, ChunkTableSegment, compute_chunk_id
from src.ingestion.models import LegalLocatorPart, TableData
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.evidence_selection import EvidenceSelection, SelectedEvidence, select_evidence
from src.retrieval.fusion import FusedCandidate, fuse_candidates
from src.retrieval.lexical_query import LexicalCandidate
from tests.unit.ingestion.test_chunk_models import (
    _chunk, _chunk_set, _chunk_set_id, _manifest, _mdr_source, _selection,
)
from tests.unit.retrieval.test_dense_results import _representative_chunk


_MODULE = "src.generation.evidence_context"
_IDENTITY_FIELDS = (
    "chunk_set_id", "source", "text", "block_selections", "hierarchies", "pages",
    "legal_locators", "table_segments", "annotations",
)


def _rebuild(chunk, **changes):
    values = {field.name: getattr(chunk, field.name) for field in fields(chunk) if field.name != "chunk_id"}
    values.update(changes)
    return Chunk(chunk_id=compute_chunk_id(**{name: values[name] for name in _IDENTITY_FIELDS}), **values)


def _single_selection(chunk):
    metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
    dense = DenseCandidate(1, chunk.chunk_id, chunk.text, metadata, 0.5)
    fused = FusedCandidate(1, chunk.chunk_id, chunk.text, metadata, 0.5, dense, None)
    return EvidenceSelection(
        chunk_set_id=chunk.chunk_set_id, max_chunks=1,
        selected=(SelectedEvidence(candidate=fused, chunk=chunk),), omitted_chunk_ids=(),
    )


class EvidenceContextTests(unittest.TestCase):
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
        base = _representative_chunk()
        self.manifest = _manifest(base.source, _mdr_source())
        set_id = _chunk_set_id(self.manifest)
        selections = (
            replace(base.block_selections[0], start_char=2, end_char=8),
            replace(base.block_selections[1], table_row_start=2, table_row_end=4),
        )
        table = replace(
            base.table_segments[0].table, headers=(None, "Class"),
            rows=((None, ""), ('“α”\\path', "I\nII")),
        )
        rich = _rebuild(
            base, chunk_set_id=set_id, order=0, previous_chunk_id=None, next_chunk_id=None,
            text='Évidence α & "quotes" \\ paths\nSecond line\tend.',
            block_selections=selections,
            table_segments=(ChunkTableSegment(selections[1].block_id, table, 2, 4, False),),
        )
        plain = _chunk(
            set_id, base.source, order=1, text="Omitted audit-only fixture.",
            block_selections=(_selection(base.source, "later", block_order=3),),
        )
        rich = replace(rich, next_chunk_id=plain.chunk_id)
        plain = replace(plain, previous_chunk_id=rich.chunk_id)
        source = self.manifest.sources[1]
        table_selection = _selection(source, "html-table")
        html_table = TableData(
            headers=(), rows=(("HTML cell", None),), extraction_method="html",
            source_page=None, table_number=None, pdfplumber_bbox=None, header_detected=None,
        )
        regulation = _chunk(
            set_id, source, text="Synthetic regulation with an HTML table.",
            block_selections=(table_selection,),
            legal_locators=((LegalLocatorPart("article", "15"), LegalLocatorPart("paragraph", "3")),),
            table_segments=(ChunkTableSegment(table_selection.block_id, html_table, 0, 1, True),),
        )
        self.chunks = (rich, plain, regulation)
        self.chunk_set = _chunk_set(self.manifest, self.chunks)
        metadata = tuple(deserialize_chunk_metadata(chunk_metadata(chunk)) for chunk in self.chunks)
        self.dense = tuple(
            DenseCandidate(rank, self.chunks[i].chunk_id, self.chunks[i].text, metadata[i], 0.2 * rank)
            for rank, i in enumerate((2, 0, 1), 1)
        )
        self.lexical = tuple(
            LexicalCandidate(rank, self.chunks[i].chunk_id, self.chunks[i].text, metadata[i], 5.0 / rank)
            for rank, i in enumerate((0, 2), 1)
        )
        self.fused = fuse_candidates(
            chunk_set=self.chunk_set, dense_candidates=self.dense,
            lexical_candidates=self.lexical, rank_constant=2,
        )
        self.selection = select_evidence(chunk_set=self.chunk_set, fused_candidates=self.fused, max_chunks=2)

    def _build(self, **changes):
        return build_evidence_context(**{"selection": self.selection, "chunk_set": self.chunk_set, **changes})

    def test_versioned_compact_json_is_deterministic_and_unicode_faithful(self):
        first, second = self._build(), self._build()
        text = first.model_context_json
        self.assertEqual(text.encode("utf-8"), second.model_context_json.encode("utf-8"))
        self.assertEqual(text, first.model_context_json)
        payload = json.loads(text)
        self.assertEqual(payload["format"], "mdr-evidence-context/v2")
        self.assertEqual(EVIDENCE_CONTEXT_FORMAT, payload["format"])
        self.assertEqual(payload["chunk_set_id"], self.chunk_set.chunk_set_id)
        self.assertEqual(payload["budget_status"], "unbudgeted")
        self.assertEqual(payload["evidence"][0]["text"], self.chunks[0].text)
        self.assertIn("Évidence α", text)
        self.assertNotIn("\\u00c9", text)
        self.assertNotIn("\n", text)  # Source newlines are escaped inside strings.
        self.assertEqual(text, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def test_full_guidance_structure_matches_an_independent_field_oracle(self):
        evidence = json.loads(self._build().model_context_json)["evidence"][0]
        chunk = self.chunks[0]
        prefix = chunk.source.source_identity
        self.assertEqual(evidence["citation_id"], "1")
        self.assertEqual(evidence["chunk_id"], chunk.chunk_id)
        self.assertEqual(evidence["source"], {
            "source_identity": prefix, "document_id": "test-guidance",
            "official_title": "Test guidance", "short_title": "Guidance", "version": "1.0",
            "source_type": "mdcg_guidance", "authority": "Medical Device Coordination Group",
            "authority_tier": "mdcg_guidance", "official_url": "https://example.eu/guidance",
            "sha256": "a" * 64, "regulatory_regimes": ["IVDR", "MDR"],
            "permitted_regimes": ["MDR", "IVDR"],
        })
        self.assertEqual(evidence["pages"], [
            {"source_index": 2, "pdf_page_number": 3},
            {"source_index": 4, "pdf_page_number": 5},
        ])
        self.assertEqual(evidence["hierarchies"], [[
            {"level": 1, "kind": "chapter", "block_id": prefix + "::heading-1", "title": "Scope"},
            {"level": 2, "kind": "section", "block_id": prefix + "::heading-2", "title": ""},
        ], [{"level": 1, "kind": "chapter", "block_id": prefix + "::heading-1", "title": "Scope"}]])
        self.assertEqual(evidence["legal_locators"], [
            [{"kind": "article", "value": "15"}, {"kind": "paragraph", "value": "3"}],
            [{"kind": "annex", "value": "I"}, {"kind": "section", "value": "23.2"}],
        ])
        self.assertNotIn("block_selections", evidence)
        self.assertEqual(self._build().resolve_citation("1").block_selections, chunk.block_selections)
        self.assertEqual(evidence["table_segments"], [{
            "block_id": prefix + "::table", "source_row_start": 2, "source_row_end": 4,
            "is_complete_table": False,
            "table": {"headers": [None, "Class"], "rows": [[None, ""], ['“α”\\path', "I\nII"]],
                      "extraction_method": "pdfplumber", "source_page": 1, "table_number": 1,
                      "pdfplumber_bbox": [10.0, 20.0, 300.0, 400.0], "header_detected": True},
        }])
        self.assertEqual(evidence["annotations"], [
            {"kind": "amendment", "identifier": "M1", "target_id": None, "raw_html": "<b>M1</b>", "text": ""},
            {"kind": "footnote_reference", "identifier": "1", "target_id": prefix + "::body",
             "raw_html": "<sup>1</sup>", "text": "See note 1."},
        ])

    def test_regulation_has_no_invented_pages_and_html_table_nulls_survive(self):
        entry = json.loads(self._build().model_context_json)["evidence"][1]
        self.assertEqual(entry["source"]["source_type"], "regulation")
        self.assertEqual(entry["pages"], [])
        self.assertEqual(entry["hierarchies"], [])
        self.assertEqual(entry["annotations"], [])
        self.assertEqual(entry["legal_locators"], [[{"kind": "article", "value": "15"}, {"kind": "paragraph", "value": "3"}]])
        self.assertEqual(entry["table_segments"][0]["table"], {
            "headers": [], "rows": [["HTML cell", None]], "extraction_method": "html",
            "source_page": None, "table_number": None, "pdfplumber_bbox": None, "header_detected": None,
        })

    def test_citation_ids_resolve_exactly_to_authoritative_original_chunks(self):
        context = self._build()
        self.assertEqual(tuple(c.citation_id for c in context.citations), ("1", "2"))
        self.assertIs(context.resolve_citation("1"), self.chunks[0])
        self.assertIs(context.resolve_citation("2"), self.chunks[2])
        for value in (1, True, None, "", "0", "3", "01", " 1", "1 ", "[1]", "١", "-1"):
            with self.subTest(value=value), self.assertRaises(EvidenceContextError):
                context.resolve_citation(value)

    def test_citation_numbering_is_package_local_not_a_global_chunk_identity(self):
        original = self._build()
        other = self._build(selection=_single_selection(self.chunks[2]))
        self.assertIs(original.resolve_citation("1"), self.chunks[0])
        self.assertIs(other.resolve_citation("1"), self.chunks[2])
        self.assertNotEqual(original.resolve_citation("1").chunk_id, other.resolve_citation("1").chunk_id)

    def test_audit_envelope_retains_exact_selection_objects_and_provenance(self):
        before = deepcopy((self.selection, self.chunk_set))
        context = self._build()
        self.assertIs(context.selection, self.selection)
        self.assertIs(context.selection.selected[0].candidate, self.fused[0])
        self.assertIs(context.selection.selected[0].chunk, self.chunks[0])
        self.assertIs(context.selection.selected[0].candidate.dense_candidate, self.dense[1])
        self.assertIs(context.selection.selected[0].candidate.lexical_candidate, self.lexical[0])
        self.assertEqual(context.selection.omitted_chunk_ids, (self.chunks[1].chunk_id,))
        self.assertEqual(context.selection.max_chunks, 2)
        self.assertEqual(context.selection.selected[0].chunk.source.local_path, "data/rag_ready/test_guidance.json")
        self.assertEqual((self.selection, self.chunk_set), before)

    def test_model_projection_excludes_operational_and_retrieval_audit_fields(self):
        payload = json.loads(self._build().model_context_json)
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()
        self.assertFalse(keys(payload) & {
            "local_path", "content_format", "validation_status", "index_status",
            "rank", "rrf_score", "bm25_score", "cosine_distance", "dense_candidate",
            "lexical_candidate", "max_chunks", "omitted_chunk_ids", "previous_chunk_id", "next_chunk_id",
        })
        self.assertEqual([v["chunk_id"] for v in payload["evidence"]], [self.chunks[0].chunk_id, self.chunks[2].chunk_id])

    def test_empty_selection_produces_empty_unbudgeted_context(self):
        selection = select_evidence(chunk_set=self.chunk_set, fused_candidates=(), max_chunks=3)
        context = self._build(selection=selection)
        self.assertEqual(context.citations, ())
        self.assertEqual(context.budget_status, "unbudgeted")
        self.assertEqual(json.loads(context.model_context_json), {
            "format": "mdr-evidence-context/v2", "budget_status": "unbudgeted",
            "chunk_set_id": self.chunk_set.chunk_set_id, "evidence": [],
        })
        with self.assertRaises(EvidenceContextError):
            context.resolve_citation("1")

    def test_large_context_is_not_trimmed_or_implicitly_token_budgeted(self):
        source = self.chunks[0].source
        text = "Large synthetic evidence α\n" * 3000
        chunk_set = _synthetic_chunk_set((source, (text,)))
        chunk = chunk_set.chunks[0]
        context = self._build(selection=_single_selection(chunk), chunk_set=chunk_set)
        self.assertGreater(len(context.model_context_json), 64_000)
        self.assertEqual(json.loads(context.model_context_json)["evidence"][0]["text"], text)
        self.assertEqual(context.budget_status, "unbudgeted")

    def test_arguments_are_required_and_keyword_only(self):
        with self.assertRaises(TypeError):
            build_evidence_context(selection=self.selection)
        with self.assertRaises(TypeError):
            build_evidence_context(chunk_set=self.chunk_set)
        with self.assertRaises(TypeError):
            build_evidence_context(self.selection, self.chunk_set)

    def test_invalid_selection_and_chunk_set_types_fail(self):
        for value in (None, (), {}, object(), self.selection.selected):
            with self.subTest(value=type(value)), self.assertRaises(EvidenceContextError):
                self._build(selection=value)
        for value in (None, self.chunks, {}, object()):
            with self.subTest(context=type(value)), self.assertRaises(EvidenceContextError):
                self._build(chunk_set=value)

    def test_valid_direct_constructed_or_copied_selection_is_revalidated_and_accepted(self):
        copied = deepcopy(self.selection)
        context = self._build(selection=copied)
        self.assertIs(context.selection, copied)
        self.assertIsNot(copied.selected[0].chunk, self.chunks[0])
        self.assertIs(context.resolve_citation("1"), self.chunks[0])
        self.assertEqual(context.model_context_json, self._build().model_context_json)

    def test_wrong_chunk_set_identity_fails_even_for_empty_selection(self):
        empty = EvidenceSelection(chunk_set_id="another-set", max_chunks=1, selected=(), omitted_chunk_ids=())
        with self.assertRaisesRegex(EvidenceContextError, "ChunkSet identity"):
            self._build(selection=empty)
        foreign = _synthetic_chunk_set((self.chunks[0].source, ("Foreign original.",)))
        with self.assertRaises(EvidenceContextError):
            self._build(chunk_set=foreign)

    def test_selected_record_absent_from_authoritative_set_fails(self):
        chunk = _chunk(self.chunk_set.chunk_set_id, self.chunks[0].source, text="Unknown record.")
        selection = _single_selection(chunk)
        with self.assertRaisesRegex(EvidenceContextError, "absent"):
            self._build(selection=selection)

    def test_nonprojected_source_difference_is_rejected_by_full_record_comparison(self):
        original = self.selection.selected[0]
        changed_chunk = replace(original.chunk, source=replace(original.chunk.source, local_path="data/rag_ready/another.json"))
        self.assertEqual(changed_chunk.chunk_id, original.chunk.chunk_id)
        self.assertEqual(chunk_metadata(changed_chunk), chunk_metadata(original.chunk))
        changed_pair = SelectedEvidence(candidate=original.candidate, chunk=changed_chunk)
        selection = replace(self.selection, selected=(changed_pair, *self.selection.selected[1:]))
        with self.assertRaisesRegex(EvidenceContextError, "full record differs"):
            self._build(selection=selection)

    def test_unknown_omitted_id_is_rejected_without_adding_omissions_to_context(self):
        selection = replace(self.selection, omitted_chunk_ids=("unknown-omitted",))
        with self.assertRaisesRegex(EvidenceContextError, "Omitted chunk is absent"):
            self._build(selection=selection)

    def test_ivdr_only_originals_cannot_be_cited_or_hidden_as_omitted_candidates(self):
        chunk_set = _synthetic_chunk_set(
            (self.chunks[0].source, ("MDR-permitted fixture.",)),
            (_ivdr_source(), ("IVDR-only fixture.",)),
        )
        ivdr = chunk_set.chunks[1]
        with self.assertRaisesRegex(EvidenceContextError, "permit MDR"):
            EvidenceCitation(citation_id="1", chunk=ivdr)
        selection = replace(_single_selection(chunk_set.chunks[0]), omitted_chunk_ids=(ivdr.chunk_id,))
        with self.assertRaisesRegex(EvidenceContextError, "permit MDR"):
            self._build(selection=selection, chunk_set=chunk_set)

    def test_tampered_selection_fields_are_revalidated(self):
        for field, value in (("max_chunks", True), ("selected", []), ("omitted_chunk_ids", [])):
            damaged = deepcopy(self.selection)
            object.__setattr__(damaged, field, value)
            with self.subTest(field=field), self.assertRaises(EvidenceContextError):
                self._build(selection=damaged)

    def test_tampered_selected_pair_and_provenance_are_revalidated(self):
        for mode in ("table", "branch", "metadata", "rank"):
            damaged = deepcopy(self.selection)
            item = damaged.selected[0]
            if mode == "table":
                object.__setattr__(item.chunk.table_segments[0].table, "rows", (("Altered", "cell"),))
            elif mode == "branch":
                object.__setattr__(item.candidate.dense_candidate, "document", "Conflicting text")
            elif mode == "metadata":
                object.__setattr__(item.candidate.metadata, "chunk_order", True)
            else:
                object.__setattr__(item.candidate, "rank", True)
            with self.subTest(mode=mode), self.assertRaises(EvidenceContextError):
                self._build(selection=damaged)

    def test_increasing_selected_scores_are_rejected_without_reranking(self):
        pair = self.selection.selected[1]
        pair = replace(pair, candidate=replace(pair.candidate, rrf_score=100.0))
        selection = replace(self.selection, selected=(self.selection.selected[0], pair))
        with self.assertRaisesRegex(EvidenceContextError, "ordering"):
            self._build(selection=selection)

    def test_equal_scores_require_global_original_order(self):
        pairs = tuple(
            replace(pair, candidate=replace(pair.candidate, rank=index, rrf_score=1.0))
            for index, pair in enumerate(self.selection.selected[::-1], 1)
        )
        selection = replace(self.selection, selected=pairs)
        with self.assertRaisesRegex(EvidenceContextError, "ordering"):
            self._build(selection=selection)

    def test_direct_context_citations_must_align_with_selected_records(self):
        good = self._build()
        for citations in (
            [], (), (None, None), good.citations[::-1],
            (good.citations[0], good.citations[0]),
            (replace(good.citations[0], citation_id="9"), good.citations[1]),
        ):
            with self.subTest(citations=citations), self.assertRaises(EvidenceContextError):
                EvidenceContext(selection=self.selection, citations=citations)
        wrong = EvidenceCitation(citation_id="1", chunk=self.chunks[1])
        with self.assertRaisesRegex(EvidenceContextError, "differs"):
            EvidenceContext(selection=self.selection, citations=(wrong, good.citations[1]))

    def test_invalid_citation_records_fail(self):
        for value in (None, 1, True, "", "0", "-1", "01", "1 ", "١"):
            with self.subTest(value=value), self.assertRaises(EvidenceContextError):
                EvidenceCitation(citation_id=value, chunk=self.chunks[0])
        with self.assertRaises(EvidenceContextError):
            EvidenceCitation(citation_id="1", chunk=None)

    def test_context_and_citation_records_are_frozen_and_slotted(self):
        context = self._build()
        for instance, name in ((context, "selection"), (context.citations[0], "citation_id")):
            with self.assertRaises(FrozenInstanceError):
                setattr(instance, name, None)
            self.assertFalse(hasattr(instance, "__dict__"))

    def test_nonfinite_table_coordinates_cannot_enter_json(self):
        for value in (math.nan, math.inf, -math.inf):
            original = self.chunks[0]
            table = replace(original.table_segments[0].table, pdfplumber_bbox=(value, 0, 1, 1))
            chunk = _rebuild(
                original, table_segments=(replace(original.table_segments[0], table=table),),
                next_chunk_id=None,
            )
            # Existing identity models allow these numeric coordinates, but
            # model context must still reject nonstandard JSON constants.
            # Restore valid source-neighbor links without changing chunk IDs.
            chunk = replace(chunk, next_chunk_id=self.chunks[1].chunk_id)
            plain = replace(self.chunks[1], previous_chunk_id=chunk.chunk_id)
            chunk_set = _chunk_set(self.manifest, (chunk, plain, self.chunks[2]))
            with self.subTest(value=value), self.assertRaises(EvidenceContextError):
                self._build(selection=_single_selection(chunk), chunk_set=chunk_set)

    def test_serializer_errors_preserve_causes_and_base_exception_propagates(self):
        failure = RuntimeError("Synthetic serialization failure")
        # Patch this module's JSON reference, not the shared json.dumps used by
        # canonical identity validation in the ingestion and selection layers.
        with patch(f"{_MODULE}.json", SimpleNamespace(dumps=Mock(side_effect=failure))):
            with self.assertRaises(EvidenceContextError) as caught:
                self._build()
        self.assertIs(caught.exception.__cause__, failure)
        interrupt = KeyboardInterrupt("Synthetic interrupt")
        with patch(f"{_MODULE}.json", SimpleNamespace(dumps=Mock(side_effect=interrupt))):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self._build()
        self.assertIs(caught.exception, interrupt)

    def test_invalid_unicode_output_fails_encoding_instead_of_replacing_text(self):
        with patch(f"{_MODULE}.json", SimpleNamespace(dumps=Mock(return_value="\ud800"))):
            with self.assertRaises(EvidenceContextError) as caught:
                self._build()
        self.assertIsInstance(caught.exception.__cause__, UnicodeEncodeError)

    def test_construction_and_lookup_do_not_read_files_retrieve_or_select_again(self):
        with patch("builtins.open", side_effect=AssertionError("No file access")), patch(
            "src.retrieval.evidence_selection.select_evidence", side_effect=AssertionError("No reselection")
        ), patch("src.retrieval.fusion.fuse_candidates", side_effect=AssertionError("No fusion")):
            context = self._build()
            self.assertIs(context.resolve_citation("1"), self.chunks[0])
            self.assertIn('"budget_status":"unbudgeted"', context.model_context_json)
        for guard in self.guards:
            guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
