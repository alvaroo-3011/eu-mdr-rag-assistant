"""Synthetic unit tests for the immutable chunk-schema boundary."""

from __future__ import annotations

import re
import unittest
from dataclasses import FrozenInstanceError, replace

from src.ingestion.chunk_models import (
    BlockSelection,
    Chunk,
    ChunkSchemaError,
    ChunkSet,
    ChunkTableSegment,
    ChunkingConfig,
    OverlapPolicy,
    OversizedTextPolicy,
    SizeUnit,
    TableSplitPolicy,
    compute_chunk_id,
    compute_chunk_set_id,
    compute_chunking_config_id,
    compute_corpus_identity,
)
from src.ingestion.models import (
    AuthorityTier,
    ContentFormat,
    CorpusManifest,
    HierarchyEntry,
    IndexStatus,
    LegalLocatorPart,
    PageIdentity,
    RegulatoryRegime,
    SourceManifestEntry,
    SourceType,
    TableData,
    ValidationStatus,
)


_IDENTITY_PATTERN = re.compile(r"^.+:sha256:[0-9a-f]{64}$")


def _guidance_source() -> SourceManifestEntry:
    """Return one complete synthetic Marker source entry."""

    return SourceManifestEntry(
        document_id="test-guidance",
        official_title="Test guidance",
        short_title="Guidance",
        version="1.0",
        regulatory_regimes=(RegulatoryRegime.MDR,),
        permitted_regimes=(RegulatoryRegime.MDR,),
        source_type=SourceType.MDCG_GUIDANCE,
        authority="Medical Device Coordination Group",
        authority_tier=AuthorityTier.MDCG_GUIDANCE,
        official_url="https://example.eu/guidance",
        local_path="data/rag_ready/test_guidance.json",
        content_format=ContentFormat.MARKER_JSON,
        sha256="a" * 64,
        validation_status=ValidationStatus.VALIDATED,
        index_status=IndexStatus.NOT_BUILT,
    )


def _mdr_source() -> SourceManifestEntry:
    """Return one complete synthetic EUR-Lex source entry."""

    return SourceManifestEntry(
        document_id="test-mdr",
        official_title="Test Medical Device Regulation",
        short_title="Test MDR",
        version="2026-07-19",
        regulatory_regimes=(RegulatoryRegime.MDR,),
        permitted_regimes=(RegulatoryRegime.MDR,),
        source_type=SourceType.REGULATION,
        authority="European Union",
        authority_tier=AuthorityTier.PRIMARY_LEGISLATION,
        official_url="https://example.eu/regulation",
        local_path="data/rag_ready/test_mdr.html",
        content_format=ContentFormat.EURLEX_HTML,
        sha256="b" * 64,
        validation_status=ValidationStatus.VALIDATED,
        index_status=IndexStatus.NOT_BUILT,
    )


def _manifest(
    *sources: SourceManifestEntry,
) -> CorpusManifest:
    """Return a typed synthetic manifest in the supplied source order."""

    return CorpusManifest(
        schema_version="1.0",
        corpus_id="test-corpus",
        expected_source_count=len(sources),
        sources=tuple(sources),
    )


def _config() -> ChunkingConfig:
    """Return the explicitly supplied approved provisional config."""

    table_split_policy = (
        TableSplitPolicy.RETAIN_WHOLE_OR_ROW_GROUPS_WITH_REPEATED_HEADERS
    )
    return ChunkingConfig(
        schema_version="1.0",
        algorithm_version="B/1/2",
        target_size=4_000,
        hard_max_size=6_000,
        size_unit=SizeUnit.UNICODE_CHARACTERS,
        overlap_policy=OverlapPolicy.NONE,
        table_split_policy=table_split_policy,
        oversized_text_policy=OversizedTextPolicy.REJECT,
        include_hierarchy_in_text=True,
    )


def _chunk_set_id(manifest: CorpusManifest) -> str:
    """Return the correct chunk-set identity for the fixture config."""

    return compute_chunk_set_id(
        compute_corpus_identity(manifest),
        compute_chunking_config_id(_config()),
    )


def _selection(
    source: SourceManifestEntry,
    structural_id: str = "document",
    *,
    block_order: int = 0,
    start_char: int | None = None,
    end_char: int | None = None,
    table_row_start: int | None = None,
    table_row_end: int | None = None,
) -> BlockSelection:
    """Build one canonical synthetic block selection."""

    return BlockSelection(
        block_id=f"{source.source_identity}::{structural_id}",
        source_block_id=structural_id,
        block_order=block_order,
        start_char=start_char,
        end_char=end_char,
        table_row_start=table_row_start,
        table_row_end=table_row_end,
    )


def _chunk(
    chunk_set_id: str,
    source: SourceManifestEntry,
    *,
    order: int = 0,
    text: str = "Synthetic evidence.",
    block_selections: tuple[BlockSelection, ...] | None = None,
    hierarchies: tuple[tuple[HierarchyEntry, ...], ...] = (),
    pages: tuple[PageIdentity, ...] = (),
    legal_locators: tuple[tuple[LegalLocatorPart, ...], ...] = (),
    table_segments: tuple[ChunkTableSegment, ...] = (),
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> Chunk:
    """Build a Chunk with its correctly computed evidence identity."""

    selections = (
        block_selections
        if block_selections is not None
        else (_selection(source),)
    )
    annotations = ()
    chunk_id = compute_chunk_id(
        chunk_set_id=chunk_set_id,
        source=source,
        text=text,
        block_selections=selections,
        hierarchies=hierarchies,
        pages=pages,
        legal_locators=legal_locators,
        table_segments=table_segments,
        annotations=annotations,
    )
    return Chunk(
        chunk_id=chunk_id,
        chunk_set_id=chunk_set_id,
        source=source,
        order=order,
        text=text,
        block_selections=selections,
        hierarchies=hierarchies,
        pages=pages,
        legal_locators=legal_locators,
        table_segments=table_segments,
        annotations=annotations,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
    )


def _chunk_set(
    manifest: CorpusManifest,
    chunks: tuple[Chunk, ...],
) -> ChunkSet:
    """Build a ChunkSet with all correctly computed stored identities."""

    config = _config()
    corpus_identity = compute_corpus_identity(manifest)
    config_id = compute_chunking_config_id(config)
    return ChunkSet(
        schema_version="1.0",
        manifest=manifest,
        corpus_identity=corpus_identity,
        chunking_config=config,
        chunking_config_id=config_id,
        chunk_set_id=compute_chunk_set_id(
            corpus_identity,
            config_id,
        ),
        chunks=chunks,
    )


def _table(rows: tuple[tuple[str | None, ...], ...]) -> TableData:
    """Return structured table data for a selected row group."""

    return TableData(
        headers=("Device", "Class"),
        rows=rows,
        extraction_method="pdfplumber",
        source_page=1,
        table_number=1,
        pdfplumber_bbox=(10.0, 20.0, 300.0, 400.0),
        header_detected=True,
    )


class ChunkingConfigTests(unittest.TestCase):
    """Validate explicit policies, size limits, and config identity."""

    def test_approved_b_1_2_configuration_is_valid(self) -> None:
        config = _config()

        self.assertEqual(config.target_size, 4_000)
        self.assertEqual(config.hard_max_size, 6_000)
        self.assertIs(config.overlap_policy, OverlapPolicy.NONE)
        self.assertRegex(config.chunking_config_id, _IDENTITY_PATTERN)

    def test_models_are_immutable_and_slotted(self) -> None:
        config = _config()

        with self.assertRaises(FrozenInstanceError):
            config.target_size = 2_000  # type: ignore[misc]

        self.assertFalse(hasattr(config, "__dict__"))

    def test_unsupported_schema_version_is_rejected(self) -> None:
        with self.assertRaises(ChunkSchemaError):
            replace(_config(), schema_version="2.0")

    def test_invalid_sizes_are_rejected(self) -> None:
        invalid_changes = (
            {"target_size": 0},
            {"target_size": True},
            {"hard_max_size": False},
            {"target_size": 6_001, "hard_max_size": 6_000},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ChunkSchemaError):
                    replace(_config(), **changes)

    def test_plain_strings_are_rejected_for_enum_fields(self) -> None:
        invalid_changes = (
            {"size_unit": "unicode_characters"},
            {"overlap_policy": "none"},
            {
                "table_split_policy": (
                    "retain_whole_or_row_groups_with_repeated_headers"
                )
            },
            {"oversized_text_policy": "reject"},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ChunkSchemaError):
                    replace(_config(), **changes)


class SelectionAndTableSegmentTests(unittest.TestCase):
    """Validate half-open source selections and table row segments."""

    def test_selection_ranges_are_paired_and_mutually_exclusive(self) -> None:
        source = _guidance_source()

        with self.assertRaises(ChunkSchemaError):
            _selection(source, start_char=0)
        with self.assertRaises(ChunkSchemaError):
            _selection(source, table_row_end=2)
        with self.assertRaises(ChunkSchemaError):
            _selection(
                source,
                start_char=0,
                end_char=5,
                table_row_start=0,
                table_row_end=1,
            )

    def test_invalid_and_boolean_range_values_are_rejected(self) -> None:
        source = _guidance_source()
        invalid_ranges = (
            {"start_char": True, "end_char": 2},
            {"start_char": -1, "end_char": 2},
            {"start_char": 2, "end_char": 2},
            {"table_row_start": False, "table_row_end": 1},
            {"table_row_start": 2, "table_row_end": 1},
        )

        for values in invalid_ranges:
            with self.subTest(values=values):
                with self.assertRaises(ChunkSchemaError):
                    _selection(source, **values)

    def test_table_segment_requires_matching_row_count(self) -> None:
        source = _guidance_source()

        valid = ChunkTableSegment(
            block_id=f"{source.source_identity}::table",
            table=_table((("A", "I"), ("B", "II"))),
            source_row_start=3,
            source_row_end=5,
            is_complete_table=False,
        )

        self.assertEqual(len(valid.table.rows), 2)

        with self.assertRaises(ChunkSchemaError):
            ChunkTableSegment(
                block_id=f"{source.source_identity}::table",
                table=_table((("A", "I"),)),
                source_row_start=3,
                source_row_end=5,
                is_complete_table=False,
            )


class ChunkTests(unittest.TestCase):
    """Validate standalone chunk structure and evidence identity."""

    def test_valid_guidance_and_eurlex_chunks(self) -> None:
        guidance = _guidance_source()
        mdr = _mdr_source()
        manifest = _manifest(guidance, mdr)
        chunk_set_id = _chunk_set_id(manifest)

        guidance_chunk = _chunk(
            chunk_set_id,
            guidance,
            pages=(PageIdentity(0, 1),),
        )
        mdr_chunk = _chunk(
            chunk_set_id,
            mdr,
            legal_locators=((LegalLocatorPart("article", "1"),),),
        )

        self.assertEqual(guidance_chunk.pages[0].pdf_page_number, 1)
        self.assertEqual(
            mdr_chunk.legal_locators[0][0].value,
            "1",
        )

    def test_hierarchy_levels_must_strictly_increase(self) -> None:
        source = _guidance_source()
        chunk_set_id = _chunk_set_id(_manifest(source))

        for levels in ((2, 1), (1, 1)):
            hierarchy = tuple(
                HierarchyEntry(
                    level=level,
                    kind="section",
                    block_id=(
                        f"{source.source_identity}::heading-{index}"
                    ),
                    title=f"Heading {index}",
                )
                for index, level in enumerate(levels)
            )

            with self.subTest(levels=levels):
                with self.assertRaisesRegex(
                    ChunkSchemaError,
                    "strictly increasing",
                ):
                    _chunk(
                        chunk_set_id,
                        source,
                        hierarchies=(hierarchy,),
                    )

    def test_empty_text_and_empty_selections_are_rejected(self) -> None:
        source = _guidance_source()
        chunk_set_id = _chunk_set_id(_manifest(source))

        with self.assertRaises(ChunkSchemaError):
            _chunk(chunk_set_id, source, text="")
        with self.assertRaises(ChunkSchemaError):
            _chunk(
                chunk_set_id,
                source,
                block_selections=(),
            )

    def test_wrong_block_id_source_prefix_is_rejected(self) -> None:
        source = _guidance_source()
        chunk_set_id = _chunk_set_id(_manifest(source))
        selection = replace(
            _selection(source),
            block_id="wrong-source::document",
        )

        with self.assertRaises(ChunkSchemaError):
            _chunk(
                chunk_set_id,
                source,
                block_selections=(selection,),
            )

    def test_eurlex_chunk_rejects_page_identity(self) -> None:
        source = _mdr_source()
        chunk_set_id = _chunk_set_id(_manifest(source))

        with self.assertRaises(ChunkSchemaError):
            _chunk(
                chunk_set_id,
                source,
                pages=(PageIdentity(0, 1),),
            )

    def test_table_selection_must_match_its_segment(self) -> None:
        source = _guidance_source()
        chunk_set_id = _chunk_set_id(_manifest(source))
        selection = _selection(
            source,
            "table",
            table_row_start=2,
            table_row_end=4,
        )
        segment = ChunkTableSegment(
            block_id=selection.block_id,
            table=_table((("A", "I"), ("B", "II"))),
            source_row_start=3,
            source_row_end=5,
            is_complete_table=False,
        )

        with self.assertRaises(ChunkSchemaError):
            _chunk(
                chunk_set_id,
                source,
                block_selections=(selection,),
                table_segments=(segment,),
            )

    def test_deterministic_identity_reproduction(self) -> None:
        source = _guidance_source()
        manifest = _manifest(source)
        chunk_set_id = _chunk_set_id(manifest)
        first = _chunk(chunk_set_id, source)
        second = _chunk(chunk_set_id, source)

        self.assertEqual(
            compute_corpus_identity(manifest),
            compute_corpus_identity(manifest),
        )
        self.assertEqual(first.chunk_id, second.chunk_id)
        self.assertRegex(first.chunk_id, _IDENTITY_PATTERN)

    def test_manifest_and_config_metadata_change_identities(self) -> None:
        source = _guidance_source()
        manifest = _manifest(source)
        changed_manifest = _manifest(
            replace(source, official_title="Changed official title")
        )
        changed_config = replace(
            _config(),
            include_hierarchy_in_text=False,
        )

        self.assertNotEqual(
            compute_corpus_identity(manifest),
            compute_corpus_identity(changed_manifest),
        )
        self.assertNotEqual(
            compute_chunking_config_id(_config()),
            compute_chunking_config_id(changed_config),
        )

    def test_text_selection_and_locator_change_chunk_identity(self) -> None:
        source = _guidance_source()
        chunk_set_id = _chunk_set_id(_manifest(source))
        original = _chunk(chunk_set_id, source)
        changed_text = _chunk(
            chunk_set_id,
            source,
            text="Different evidence.",
        )
        changed_selection = _chunk(
            chunk_set_id,
            source,
            block_selections=(
                _selection(source, start_char=0, end_char=5),
            ),
        )
        changed_locator = _chunk(
            chunk_set_id,
            source,
            legal_locators=((LegalLocatorPart("section", "A"),),),
        )

        changed_ids = {
            changed_text.chunk_id,
            changed_selection.chunk_id,
            changed_locator.chunk_id,
        }
        self.assertEqual(len(changed_ids), 3)
        self.assertNotIn(original.chunk_id, changed_ids)

    def test_tampered_chunk_identity_is_rejected(self) -> None:
        source = _guidance_source()
        chunk = _chunk(_chunk_set_id(_manifest(source)), source)

        with self.assertRaises(ChunkSchemaError):
            replace(
                chunk,
                chunk_id=f"chunk:sha256:{'0' * 64}",
            )


class ChunkSetTests(unittest.TestCase):
    """Validate corpus-wide coverage, order, and neighbor invariants."""

    def test_tampered_chunk_set_identity_fields_are_rejected(self) -> None:
        source = _guidance_source()
        manifest = _manifest(source)
        chunk_set_id = _chunk_set_id(manifest)
        valid = _chunk_set(
            manifest,
            (_chunk(chunk_set_id, source),),
        )
        replacements = (
            {"corpus_identity": f"corpus:sha256:{'0' * 64}"},
            {
                "chunking_config_id": (
                    f"chunking-config:sha256:{'0' * 64}"
                )
            },
            {"chunk_set_id": f"chunk-set:sha256:{'0' * 64}"},
        )

        for changes in replacements:
            with self.subTest(changes=changes):
                with self.assertRaises(ChunkSchemaError):
                    replace(valid, **changes)

    def test_unsupported_chunk_set_schema_version_is_rejected(self) -> None:
        source = _guidance_source()
        manifest = _manifest(source)
        chunk_set_id = _chunk_set_id(manifest)
        valid = _chunk_set(
            manifest,
            (_chunk(chunk_set_id, source),),
        )

        with self.assertRaises(ChunkSchemaError):
            replace(valid, schema_version="2.0")

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        source = _guidance_source()
        manifest = _manifest(source)
        chunk_set_id = _chunk_set_id(manifest)
        first = _chunk(chunk_set_id, source, order=0)
        duplicate = _chunk(chunk_set_id, source, order=1)

        self.assertEqual(first.chunk_id, duplicate.chunk_id)
        with self.assertRaises(ChunkSchemaError):
            _chunk_set(manifest, (first, duplicate))

    def test_missing_and_unlisted_sources_are_rejected(self) -> None:
        guidance = _guidance_source()
        mdr = _mdr_source()
        complete_manifest = _manifest(guidance, mdr)
        complete_id = _chunk_set_id(complete_manifest)

        with self.assertRaises(ChunkSchemaError):
            _chunk_set(
                complete_manifest,
                (_chunk(complete_id, guidance),),
            )

        guidance_manifest = _manifest(guidance)
        guidance_id = _chunk_set_id(guidance_manifest)
        with self.assertRaises(ChunkSchemaError):
            _chunk_set(
                guidance_manifest,
                (_chunk(guidance_id, mdr),),
            )

    def test_incorrect_global_and_source_local_order_is_rejected(self) -> None:
        guidance = _guidance_source()
        mdr = _mdr_source()
        manifest = _manifest(guidance, mdr)
        chunk_set_id = _chunk_set_id(manifest)
        guidance_chunk = _chunk(chunk_set_id, guidance)
        mdr_chunk = _chunk(chunk_set_id, mdr)

        with self.assertRaises(ChunkSchemaError):
            _chunk_set(manifest, (mdr_chunk, guidance_chunk))

        single_manifest = _manifest(guidance)
        single_id = _chunk_set_id(single_manifest)
        with self.assertRaises(ChunkSchemaError):
            _chunk_set(
                single_manifest,
                (_chunk(single_id, guidance, order=1),),
            )

    def test_broken_asymmetric_and_cross_source_links_are_rejected(
        self,
    ) -> None:
        guidance = _guidance_source()
        guidance_manifest = _manifest(guidance)
        guidance_id = _chunk_set_id(guidance_manifest)
        first = _chunk(guidance_id, guidance, order=0, text="First")
        second = _chunk(guidance_id, guidance, order=1, text="Second")

        with self.assertRaises(ChunkSchemaError):
            _chunk_set(guidance_manifest, (first, second))

        asymmetric_first = replace(
            first,
            next_chunk_id=second.chunk_id,
        )
        with self.assertRaises(ChunkSchemaError):
            _chunk_set(
                guidance_manifest,
                (asymmetric_first, second),
            )

        mdr = _mdr_source()
        combined_manifest = _manifest(guidance, mdr)
        combined_id = _chunk_set_id(combined_manifest)
        guidance_chunk = _chunk(
            combined_id,
            guidance,
            text="Guidance",
        )
        mdr_chunk = _chunk(combined_id, mdr, text="Regulation")
        guidance_chunk = replace(
            guidance_chunk,
            next_chunk_id=mdr_chunk.chunk_id,
        )
        mdr_chunk = replace(
            mdr_chunk,
            previous_chunk_id=guidance_chunk.chunk_id,
        )

        with self.assertRaises(ChunkSchemaError):
            _chunk_set(
                combined_manifest,
                (guidance_chunk, mdr_chunk),
            )

    def test_fully_valid_multi_source_chunk_set(self) -> None:
        guidance = _guidance_source()
        mdr = _mdr_source()
        manifest = _manifest(guidance, mdr)
        chunk_set_id = _chunk_set_id(manifest)

        first = _chunk(
            chunk_set_id,
            guidance,
            order=0,
            text="First guidance chunk.",
            pages=(PageIdentity(0, 1),),
        )
        second = _chunk(
            chunk_set_id,
            guidance,
            order=1,
            text="Second guidance chunk.",
            pages=(PageIdentity(1, 2),),
        )
        first = replace(first, next_chunk_id=second.chunk_id)
        second = replace(second, previous_chunk_id=first.chunk_id)
        regulation = _chunk(
            chunk_set_id,
            mdr,
            order=0,
            text="Regulation chunk.",
            legal_locators=((LegalLocatorPart("article", "1"),),),
        )

        chunk_set = _chunk_set(
            manifest,
            (first, second, regulation),
        )

        self.assertEqual(len(chunk_set.chunks), 3)
        self.assertEqual(chunk_set.chunks[0].next_chunk_id, second.chunk_id)
        self.assertIsNone(chunk_set.chunks[2].previous_chunk_id)


if __name__ == "__main__":
    unittest.main()
