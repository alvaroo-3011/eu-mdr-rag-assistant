"""Synthetic tests for immutable source-level chunk construction."""

from __future__ import annotations

from functools import lru_cache
import unittest

from src.ingestion.chunk_builder import (
    ChunkBuilderError,
    build_source_chunks,
)
from src.ingestion.chunk_models import (
    ChunkingConfig,
    OverlapPolicy,
    OversizedTextPolicy,
    SizeUnit,
    TableSplitPolicy,
)
from src.ingestion.models import (
    AnnotationKind,
    AuthorityTier,
    ContentFormat,
    HierarchyEntry,
    IndexStatus,
    LegalLocatorPart,
    LoadedSource,
    ManifestSchemaError,
    PageIdentity,
    RegulatoryRegime,
    SourceAnnotation,
    SourceBlock,
    SourceManifestEntry,
    SourceType,
    TableData,
    ValidationStatus,
)


@lru_cache(maxsize=2)
def _source(*, with_pages: bool = False) -> SourceManifestEntry:
    """Build valid synthetic source metadata without loading the manifest."""

    formats = tuple(
        content_format
        for content_format in ContentFormat
        if not with_pages or content_format.value != "eurlex_html"
    )

    # The source model owns the approved enum relationships. Trying the small
    # finite enum product keeps this test independent of enum member names
    # while still constructing a fully validated synthetic source.
    for content_format in formats:
        suffix = ".html" if "html" in content_format.value else ".json"
        for source_type in SourceType:
            for authority_tier in AuthorityTier:
                for validation_status in ValidationStatus:
                    for index_status in IndexStatus:
                        for regime in RegulatoryRegime:
                            try:
                                return SourceManifestEntry(
                                    document_id=(
                                        "synthetic-paged-source"
                                        if with_pages
                                        else "synthetic-source"
                                    ),
                                    official_title="Synthetic source",
                                    short_title="Synthetic",
                                    version="1.0",
                                    regulatory_regimes=(regime,),
                                    permitted_regimes=(regime,),
                                    source_type=source_type,
                                    authority="Synthetic authority",
                                    authority_tier=authority_tier,
                                    official_url="https://example.com/source",
                                    local_path=f"data/rag_ready/source{suffix}",
                                    content_format=content_format,
                                    sha256="0" * 64,
                                    validation_status=validation_status,
                                    index_status=index_status,
                                )
                            except ManifestSchemaError:
                                continue

    raise AssertionError("No valid synthetic manifest combination found.")


def _block(
    source: SourceManifestEntry,
    block_type: str,
    *,
    order: int,
    text: str = "",
    hierarchy: tuple[HierarchyEntry, ...] = (),
    page: PageIdentity | None = None,
    locator: tuple[LegalLocatorPart, ...] = (),
    table: TableData | None = None,
    annotations: tuple[SourceAnnotation, ...] = (),
) -> SourceBlock:
    """Build one normalized block using the source's canonical prefix."""

    suffix = f"block-{order}-{block_type.lower()}"
    return SourceBlock(
        block_id=f"{source.source_identity}::{suffix}",
        source_block_id=f"source-{suffix}",
        block_type=block_type,
        order=order,
        parent_id=None,
        child_ids=(),
        hierarchy=hierarchy,
        page=page,
        legal_locator=locator,
        raw_html="",
        searchable_text=text,
        table=table,
        annotations=annotations,
        images=None,
    )


def _config(
    *,
    target_size: int = 4_000,
    hard_max_size: int = 6_000,
) -> ChunkingConfig:
    """Return the approved no-overlap synthetic configuration."""

    return ChunkingConfig(
        schema_version="1.0",
        algorithm_version="B/1/2",
        target_size=target_size,
        hard_max_size=hard_max_size,
        size_unit=SizeUnit.UNICODE_CHARACTERS,
        overlap_policy=OverlapPolicy.NONE,
        table_split_policy=(
            TableSplitPolicy
            .RETAIN_WHOLE_OR_ROW_GROUPS_WITH_REPEATED_HEADERS
        ),
        oversized_text_policy=OversizedTextPolicy.REJECT,
        include_hierarchy_in_text=True,
    )


class SourceChunkBuilderTests(unittest.TestCase):
    """Validate deterministic Chunk construction without loading the corpus."""

    def test_single_prose_chunk_has_complete_selection(self) -> None:
        source = _source()
        body = _block(
            source,
            "Paragraph",
            order=0,
            text="Original body text.",
        )

        chunks = build_source_chunks(
            LoadedSource(source=source, blocks=(body,)),
            chunk_set_id="chunk-set:test",
            config=_config(),
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.text, body.searchable_text)
        self.assertEqual(chunk.order, 0)
        self.assertIsNone(chunk.previous_chunk_id)
        self.assertIsNone(chunk.next_chunk_id)
        self.assertEqual(len(chunk.block_selections), 1)
        selection = chunk.block_selections[0]
        self.assertEqual(selection.block_id, body.block_id)
        self.assertEqual(selection.source_block_id, body.source_block_id)
        self.assertEqual(selection.block_order, 0)
        self.assertIsNone(selection.start_char)
        self.assertIsNone(selection.table_row_start)

    def test_context_pages_and_repeated_locators_are_preserved(self) -> None:
        source = _source(with_pages=True)
        heading = _block(source, "Section", order=0, text="Scope")
        hierarchy = (
            HierarchyEntry(
                level=1,
                kind="Section",
                block_id=heading.block_id,
                title="Scope",
            ),
        )
        locator = (LegalLocatorPart("paragraph", "1"),)
        first = _block(
            source,
            "Paragraph",
            order=1,
            text="First.",
            hierarchy=hierarchy,
            page=PageIdentity(0, 1),
            locator=locator,
        )
        second = _block(
            source,
            "Paragraph",
            order=2,
            text="Second.",
            hierarchy=hierarchy,
            page=PageIdentity(1, 2),
            locator=locator,
        )

        chunk = build_source_chunks(
            LoadedSource(source=source, blocks=(heading, first, second)),
            chunk_set_id="chunk-set:test",
            config=_config(),
        )[0]

        self.assertEqual(chunk.hierarchies, (hierarchy,))
        self.assertEqual(
            chunk.pages,
            (PageIdentity(0, 1), PageIdentity(1, 2)),
        )
        self.assertEqual(chunk.legal_locators, (locator, locator))
        self.assertEqual(
            tuple(item.block_order for item in chunk.block_selections),
            (1, 2),
        )

    def test_complete_table_segment_is_preserved(self) -> None:
        source = _source()
        table = TableData(
            headers=("Class", "Requirement"),
            rows=(("IIa", "Synthetic requirement"),),
            extraction_method="synthetic",
            source_page=None,
            table_number=1,
            pdfplumber_bbox=None,
            header_detected=True,
        )
        block = _block(
            source,
            "Table",
            order=0,
            text="Class | Requirement",
            table=table,
        )

        chunk = build_source_chunks(
            LoadedSource(source=source, blocks=(block,)),
            chunk_set_id="chunk-set:test",
            config=_config(),
        )[0]

        self.assertEqual(len(chunk.table_segments), 1)
        segment = chunk.table_segments[0]
        self.assertEqual(segment.block_id, block.block_id)
        self.assertIs(segment.table, table)
        self.assertEqual(segment.source_row_start, 0)
        self.assertEqual(segment.source_row_end, 1)
        self.assertTrue(segment.is_complete_table)

    def test_amendment_annotation_is_attached_without_marker_text(self) -> None:
        source = _source()
        annotation = SourceAnnotation(
            kind=AnnotationKind.AMENDMENT,
            identifier="M1",
            target_id=None,
            raw_html="<a>marker</a>",
            text="M1: amended",
        )
        marker = _block(
            source,
            "AmendmentMarker",
            order=0,
            text="MARKER",
            annotations=(annotation,),
        )
        body = _block(
            source,
            "Paragraph",
            order=1,
            text="Amended body.",
        )

        chunk = build_source_chunks(
            LoadedSource(source=source, blocks=(marker, body)),
            chunk_set_id="chunk-set:test",
            config=_config(),
        )[0]

        self.assertEqual(chunk.annotations, (annotation,))
        self.assertEqual(chunk.text, "Amended body.")
        self.assertNotIn("MARKER", chunk.text)

    def test_ids_orders_and_neighbor_links_are_deterministic(self) -> None:
        source = _source()
        blocks = tuple(
            _block(
                source,
                "Paragraph",
                order=index,
                text=f"Body {index}.",
            )
            for index in range(3)
        )
        loaded = LoadedSource(source=source, blocks=blocks)
        config = _config(target_size=8, hard_max_size=20)

        first_build = build_source_chunks(
            loaded,
            chunk_set_id="chunk-set:test",
            config=config,
        )
        second_build = build_source_chunks(
            loaded,
            chunk_set_id="chunk-set:test",
            config=config,
        )

        self.assertEqual(first_build, second_build)
        self.assertEqual(
            tuple(chunk.order for chunk in first_build),
            (0, 1, 2),
        )
        self.assertEqual(
            first_build[0].next_chunk_id,
            first_build[1].chunk_id,
        )
        self.assertEqual(
            first_build[1].previous_chunk_id,
            first_build[0].chunk_id,
        )
        self.assertEqual(
            first_build[1].next_chunk_id,
            first_build[2].chunk_id,
        )
        self.assertEqual(
            first_build[2].previous_chunk_id,
            first_build[1].chunk_id,
        )
        self.assertIsNone(first_build[0].previous_chunk_id)
        self.assertIsNone(first_build[2].next_chunk_id)

    def test_invalid_arguments_are_rejected(self) -> None:
        source = _source()
        body = _block(source, "Paragraph", order=0, text="Body.")
        loaded = LoadedSource(source=source, blocks=(body,))

        invalid_cases = (
            (object(), "chunk-set:test", _config(), "loaded_source"),
            (loaded, "", _config(), "chunk_set_id"),
            (loaded, "chunk-set:test", object(), "config"),
        )
        for loaded_value, set_id, config, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ChunkBuilderError, message):
                    build_source_chunks(  # type: ignore[arg-type]
                        loaded_value,
                        chunk_set_id=set_id,
                        config=config,
                    )

    def test_source_without_retrievable_content_is_rejected(self) -> None:
        source = _source()
        excluded = _block(
            source,
            "PageHeader",
            order=0,
            text="Furniture only.",
        )

        with self.assertRaisesRegex(
            ChunkBuilderError,
            "no retrievable chunk candidates",
        ):
            build_source_chunks(
                LoadedSource(source=source, blocks=(excluded,)),
                chunk_set_id="chunk-set:test",
                config=_config(),
            )

    def test_table_route_without_table_data_is_rejected(self) -> None:
        source = _source()
        invalid_table = _block(
            source,
            "Table",
            order=0,
            text="Rendered table text.",
            table=None,
        )

        with self.assertRaisesRegex(
            ChunkBuilderError,
            "has no TableData",
        ):
            build_source_chunks(
                LoadedSource(source=source, blocks=(invalid_table,)),
                chunk_set_id="chunk-set:test",
                config=_config(),
            )


if __name__ == "__main__":
    unittest.main()
