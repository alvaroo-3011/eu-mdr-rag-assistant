"""Synthetic tests for the trusted dense-result metadata boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from types import MappingProxyType
import unittest

from src.ingestion.chunk_models import Chunk, ChunkTableSegment, compute_chunk_id
from src.ingestion.models import (
    AnnotationKind,
    AuthorityTier,
    HierarchyEntry,
    LegalLocatorPart,
    PageIdentity,
    RegulatoryRegime,
    SourceAnnotation,
    SourceType,
)
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import (
    CHUNK_METADATA_KEYS,
    DenseAnnotation,
    DenseCandidate,
    DenseResultError,
    TrustedChunkMetadata,
    deserialize_chunk_metadata,
)
from tests.unit.ingestion.test_chunk_models import (
    _chunk, _chunk_set_id, _guidance_source, _manifest, _mdr_source,
    _selection, _table,
)


def _representative_chunk() -> Chunk:
    source = replace(
        _guidance_source(),
        regulatory_regimes=(RegulatoryRegime.IVDR, RegulatoryRegime.MDR),
        permitted_regimes=(RegulatoryRegime.MDR, RegulatoryRegime.IVDR),
    )
    selections = (
        _selection(source, "body", block_order=0),
        _selection(source, "table", block_order=1),
    )
    hierarchy = (
        HierarchyEntry(1, "chapter", f"{source.source_identity}::heading-1", "Scope"),
        HierarchyEntry(2, "section", f"{source.source_identity}::heading-2", ""),
    )
    annotations = (
        SourceAnnotation(AnnotationKind.AMENDMENT, "M1", None, "<b>M1</b>", ""),
        SourceAnnotation(
            AnnotationKind.FOOTNOTE_REFERENCE, "1", selections[0].block_id,
            "<sup>1</sup>", "See note 1.",
        ),
    )
    values = {
        "chunk_set_id": _chunk_set_id(_manifest(source)),
        "source": source,
        "text": "Synthetic guidance with a table.",
        "block_selections": selections,
        "hierarchies": (hierarchy, (hierarchy[0],)),
        "pages": (PageIdentity(2, 3), PageIdentity(4, 5)),
        "legal_locators": (
            (LegalLocatorPart("article", "15"), LegalLocatorPart("paragraph", "3")),
            (LegalLocatorPart("annex", "I"), LegalLocatorPart("section", "23.2")),
        ),
        "table_segments": (
            ChunkTableSegment(selections[1].block_id, _table((("A", "I"),)), 0, 1, True),
        ),
        "annotations": annotations,
    }
    return Chunk(
        chunk_id=compute_chunk_id(**values),
        order=1,
        previous_chunk_id="previous-chunk",
        next_chunk_id="next-chunk",
        **values,
    )


class TrustedChunkMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = _representative_chunk()
        self.raw = chunk_metadata(self.chunk)

    def _bad(self, field: str, value: object) -> None:
        with self.assertRaisesRegex(DenseResultError, field):
            deserialize_chunk_metadata({**self.raw, field: value})

    def test_serializer_keys_and_all_source_fields_are_preserved(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        self.assertIsInstance(metadata, TrustedChunkMetadata)
        self.assertEqual(self.raw.keys(), CHUNK_METADATA_KEYS)
        self.assertEqual(metadata.schema_version, "1.0")
        self.assertEqual(metadata.chunk_id, self.chunk.chunk_id)
        self.assertEqual(metadata.chunk_set_id, self.chunk.chunk_set_id)
        self.assertEqual(metadata.chunk_order, self.chunk.order)
        for field in (
            "document_id", "official_title", "short_title", "version", "source_type",
            "authority", "authority_tier", "official_url", "regulatory_regimes",
            "permitted_regimes", "source_identity",
        ):
            self.assertEqual(getattr(metadata, field), getattr(self.chunk.source, field))
        self.assertEqual(metadata.source_sha256, self.chunk.source.sha256)
        self.assertIs(metadata.source_type, SourceType.MDCG_GUIDANCE)
        self.assertIs(metadata.authority_tier, AuthorityTier.MDCG_GUIDANCE)

    def test_page_identities_and_order_are_preserved(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        self.assertEqual(metadata.pages, self.chunk.pages)
        self.assertEqual(metadata.pages[0], PageIdentity(2, 3))

    def test_multi_level_hierarchies_and_empty_titles_are_preserved(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        self.assertEqual(metadata.hierarchies, self.chunk.hierarchies)
        self.assertEqual(metadata.hierarchies[0][1].title, "")

    def test_legal_locator_paths_and_order_are_preserved(self) -> None:
        self.assertEqual(
            deserialize_chunk_metadata(self.raw).legal_locators,
            self.chunk.legal_locators,
        )

    def test_table_presence_and_source_block_order_are_preserved(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        self.assertTrue(metadata.has_table)
        self.assertEqual(
            metadata.block_ids,
            tuple(selection.block_id for selection in self.chunk.block_selections),
        )

    def test_annotations_preserve_order_nulls_and_empty_text(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        self.assertEqual(
            metadata.annotations,
            tuple(
                DenseAnnotation(a.kind, a.identifier, a.target_id, a.text)
                for a in self.chunk.annotations
            ),
        )
        self.assertIsNone(metadata.annotations[0].target_id)
        self.assertEqual(metadata.annotations[0].text, "")
        self.assertFalse(hasattr(metadata.annotations[0], "raw_html"))

    def test_neighbor_ids_and_absent_neighbor_markers_are_preserved(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        self.assertEqual(metadata.previous_chunk_id, "previous-chunk")
        self.assertEqual(metadata.next_chunk_id, "next-chunk")
        absent = chunk_metadata(replace(self.chunk, previous_chunk_id=None, next_chunk_id=None))
        decoded = deserialize_chunk_metadata(absent)
        self.assertEqual(absent["previous_chunk_id"], "")
        self.assertIsNone(decoded.previous_chunk_id)
        self.assertIsNone(decoded.next_chunk_id)

    def test_regime_order_and_applicability_flags_are_preserved(self) -> None:
        both = deserialize_chunk_metadata(self.raw)
        self.assertEqual(both.regulatory_regimes, (RegulatoryRegime.IVDR, RegulatoryRegime.MDR))
        self.assertTrue(both.permits_mdr)
        self.assertTrue(both.permits_ivdr)
        for regime in (RegulatoryRegime.MDR, RegulatoryRegime.IVDR):
            source = replace(_guidance_source(), regulatory_regimes=(regime,), permitted_regimes=(regime,))
            chunk = _chunk(_chunk_set_id(_manifest(source)), source)
            metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
            self.assertEqual(metadata.permits_mdr, regime is RegulatoryRegime.MDR)
            self.assertEqual(metadata.permits_ivdr, regime is RegulatoryRegime.IVDR)

    def test_empty_citation_collections_and_no_table_are_preserved(self) -> None:
        source = _mdr_source()
        raw = chunk_metadata(_chunk(_chunk_set_id(_manifest(source)), source))
        metadata = deserialize_chunk_metadata(raw)
        for field in ("pages", "hierarchies", "legal_locators", "annotations"):
            self.assertEqual(getattr(metadata, field), ())
        self.assertFalse(metadata.has_table)

    def test_empty_paths_and_repeated_annotation_occurrences_are_preserved(self) -> None:
        source = _guidance_source()
        chunk = _chunk(
            _chunk_set_id(_manifest(source)), source,
            hierarchies=((),), legal_locators=((), ()),
        )
        raw = chunk_metadata(chunk)
        annotation = dict(kind="footnote_definition", identifier=None, target_id=None, text="")
        raw["annotations_json"] = json.dumps([annotation, annotation])
        metadata = deserialize_chunk_metadata(raw)
        self.assertEqual(metadata.hierarchies, ((),))
        self.assertEqual(metadata.legal_locators, ((), ()))
        self.assertEqual(len(metadata.annotations), 2)
        self.assertEqual(metadata.annotations[0], metadata.annotations[1])
        self.assertIsNone(metadata.annotations[0].identifier)

    def test_mapping_input_is_supported_and_not_mutated(self) -> None:
        before = self.raw.copy()
        metadata = deserialize_chunk_metadata(MappingProxyType(self.raw))
        self.assertEqual(self.raw, before)
        self.raw["official_title"] = "changed"
        self.assertEqual(metadata.official_title, before["official_title"])

    def test_non_mapping_missing_and_unexpected_keys_are_rejected(self) -> None:
        for raw in (None, [], "metadata"):
            with self.subTest(raw=raw), self.assertRaises(DenseResultError):
                deserialize_chunk_metadata(raw)
        for field in self.raw:
            missing = self.raw.copy()
            del missing[field]
            with self.subTest(field=field), self.assertRaisesRegex(DenseResultError, field):
                deserialize_chunk_metadata(missing)
        with self.assertRaisesRegex(DenseResultError, "unexpected"):
            deserialize_chunk_metadata({**self.raw, "unexpected": 1})

    def test_exact_scalar_types_including_bool_and_int_are_required(self) -> None:
        for field, original in self.raw.items():
            bad = True if type(original) is int else 1 if type(original) is bool else None
            with self.subTest(field=field):
                self._bad(field, bad)
        for field, value in (("chunk_order", 1.0), ("has_table", 1.0), ("document_id", RegulatoryRegime.MDR)):
            with self.subTest(field=field):
                self._bad(field, value)

    def test_invalid_scalar_values_are_rejected(self) -> None:
        for field, value in (
            ("schema_version", "2.0"), ("chunk_order", -1), ("chunk_id", " "),
            ("document_id", "Invalid ID"), ("source_sha256", "a" * 63),
            ("official_url", "http://example.eu"), ("source_type", "unknown"),
            ("official_url", "https://[invalid"),
            ("authority_tier", "unknown"), ("permits_mdr", False),
            ("previous_chunk_id", " "), ("next_chunk_id", self.chunk.chunk_id),
        ):
            with self.subTest(field=field, value=value):
                self._bad(field, value)

    def test_json_fields_reject_malformed_and_nonstandard_json(self) -> None:
        for field in (key for key in self.raw if key.endswith("_json")):
            for value in ("[", "[NaN]", "[Infinity]", "[-Infinity]"):
                with self.subTest(field=field, value=value):
                    self._bad(field, value)
        self._bad("pages_json", '[{"source_index":2,"source_index":2,"pdf_page_number":3}]')

    def test_json_fields_reject_wrong_container_shapes(self) -> None:
        for field in (key for key in self.raw if key.endswith("_json")):
            for value in ("{}", "null", '"text"', "1"):
                with self.subTest(field=field, value=value):
                    self._bad(field, value)
        self._bad("hierarchies_json", '[{"level":1}]')
        self._bad("legal_locators_json", '[{"kind":"article","value":"15"}]')

    def test_nested_fields_must_be_exact(self) -> None:
        for field in ("pages_json", "hierarchies_json", "legal_locators_json", "annotations_json"):
            original = json.loads(self.raw[field])
            nested = original[0][0] if field in ("hierarchies_json", "legal_locators_json") else original[0]
            for missing in tuple(nested):
                changed = json.loads(self.raw[field])
                target = changed[0][0] if field in ("hierarchies_json", "legal_locators_json") else changed[0]
                del target[missing]
                with self.subTest(field=field, missing=missing):
                    self._bad(field, json.dumps(changed))
            nested["unexpected"] = "extra"
            with self.subTest(field=field, extra=True):
                self._bad(field, json.dumps(original))

    def test_every_nested_scalar_requires_its_exact_type(self) -> None:
        for field in ("pages_json", "hierarchies_json", "legal_locators_json", "annotations_json"):
            decoded = json.loads(self.raw[field])
            original = decoded[0][0] if field in ("hierarchies_json", "legal_locators_json") else decoded[0]
            for key, value in original.items():
                changed = json.loads(self.raw[field])
                target = changed[0][0] if field in ("hierarchies_json", "legal_locators_json") else changed[0]
                target[key] = True if type(value) is int else 123
                with self.subTest(field=field, key=key):
                    self._bad(field, json.dumps(changed))

    def test_invalid_nested_types_enums_and_model_values_are_rejected(self) -> None:
        cases = (
            ("pages_json", [[]]),
            ("hierarchies_json", [[None]]),
            ("legal_locators_json", [[False]]),
            ("annotations_json", [[]]),
            ("pages_json", [{"source_index": True, "pdf_page_number": 2}]),
            ("pages_json", [{"source_index": 2.0, "pdf_page_number": 3}]),
            ("pages_json", [{"source_index": 2, "pdf_page_number": 4}]),
            ("hierarchies_json", [[{"level": 0, "kind": "section", "block_id": "heading", "title": ""}]]),
            ("hierarchies_json", [[{"level": 1, "kind": "section", "block_id": "heading", "title": None}]]),
            ("legal_locators_json", [[{"kind": "article", "value": 15}]]),
            ("legal_locators_json", [[{"kind": "article", "value": ""}]]),
            ("annotations_json", [{"kind": "unknown", "identifier": None, "target_id": None, "text": ""}]),
            ("annotations_json", [{"kind": "amendment", "identifier": "", "target_id": None, "text": ""}]),
            ("annotations_json", [{"kind": "amendment", "identifier": None, "target_id": False, "text": ""}]),
            ("regulatory_regimes_json", ["unknown"]),
            ("permitted_regimes_json", [True]),
            ("block_ids_json", [1]),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self._bad(field, json.dumps(value))

    def test_cross_field_and_collection_invariants_are_rejected(self) -> None:
        for field, value in (
            ("pages_json", [dict(source_index=4, pdf_page_number=5), dict(source_index=2, pdf_page_number=3)]),
            ("pages_json", [dict(source_index=2, pdf_page_number=3)] * 2),
            ("regulatory_regimes_json", []),
            ("permitted_regimes_json", ["MDR", "MDR"]),
            ("block_ids_json", []),
            ("block_ids_json", ["wrong-source::body"]),
        ):
            with self.subTest(field=field):
                # Cross-field errors name decoded fields rather than _json keys.
                with self.assertRaisesRegex(DenseResultError, field.removesuffix("_json")):
                    deserialize_chunk_metadata({**self.raw, field: json.dumps(value)})
        for changes in (
            {"regulatory_regimes_json": '["MDR"]'},
            {"source_type": "regulation"},
            {"previous_chunk_id": "same", "next_chunk_id": "same"},
        ):
            with self.subTest(changes=changes), self.assertRaises(DenseResultError):
                deserialize_chunk_metadata({**self.raw, **changes})

    def test_direct_models_are_validated_frozen_and_deeply_immutable(self) -> None:
        metadata = deserialize_chunk_metadata(self.raw)
        for model, field, value in (
            (metadata, "chunk_order", 9),
            (metadata.annotations[0], "text", "changed"),
            (metadata.pages[0], "source_index", 9),
        ):
            with self.subTest(model=type(model).__name__):
                self.assertFalse(hasattr(model, "__dict__"))
                with self.assertRaises(FrozenInstanceError):
                    setattr(model, field, value)
        with self.assertRaises(TypeError):
            metadata.hierarchies[0][0] = metadata.hierarchies[0][1]
        for changes in ({"pages": []}, {"permits_mdr": 1}, {"annotations": ({},)}):
            with self.subTest(changes=changes), self.assertRaises(DenseResultError):
                replace(metadata, **changes)


class DenseCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        chunk = _representative_chunk()
        self.arguments = {
            "rank": 1,
            "chunk_id": chunk.chunk_id,
            "document": chunk.text,
            "metadata": deserialize_chunk_metadata(chunk_metadata(chunk)),
            "cosine_distance": 0.25,
        }

    def test_finite_real_distances_become_floats_without_thresholds(self) -> None:
        for distance in (0, 1, -0.01, 2.5):
            with self.subTest(distance=distance):
                candidate = DenseCandidate(**{**self.arguments, "cosine_distance": distance})
                self.assertIs(type(candidate.cosine_distance), float)
                self.assertEqual(candidate.cosine_distance, float(distance))
        self.assertEqual(DenseCandidate(**{**self.arguments, "rank": 1000}).rank, 1000)

    def test_invalid_ranks_ids_documents_and_metadata_are_rejected(self) -> None:
        for field, values in (
            ("rank", (0, -1, True, 1.0)),
            ("chunk_id", ("", " ", None, 1, "wrong-chunk")),
            ("document", ("", " ", None, 1)),
            ("metadata", (None, {}, "metadata")),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(DenseResultError, field):
                        DenseCandidate(**{**self.arguments, field: value})

    def test_nan_infinity_and_non_real_distances_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), True, "0.2", None, 1j, 10 ** 1000):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(DenseResultError, "cosine_distance"):
                    DenseCandidate(**{**self.arguments, "cosine_distance": value})

    def test_candidate_is_frozen_and_slotted(self) -> None:
        candidate = DenseCandidate(**self.arguments)
        self.assertFalse(hasattr(candidate, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            candidate.rank = 2


if __name__ == "__main__":
    unittest.main()
