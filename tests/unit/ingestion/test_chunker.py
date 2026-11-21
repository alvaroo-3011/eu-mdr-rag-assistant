"""
Synthetic tests for deterministic source-block routing and rendering.

These tests do not load or modify the frozen corpus.
"""

from __future__ import annotations

import unittest

from src.ingestion.chunker import (
    BLOCK_TYPE_ROUTES,
    BlockRoute,
    ChunkerError,
    classify_block_type,
    render_context,
    route_block,
    route_blocks,
    CandidateGroupingResult,
    ChunkCandidate,
    PreparedChunkCandidate,
    group_routed_blocks,
    prepare_chunk_candidates,
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
    HierarchyEntry,
    LegalLocatorPart,
    SourceAnnotation,
    SourceBlock,
    TableData,
    PageIdentity,
)


def _source_block(
    block_type: str,
    *,
    order: int = 0,
    searchable_text: str = "",
    hierarchy: tuple[HierarchyEntry, ...] = (),
    legal_locator: tuple[LegalLocatorPart, ...] = (),
    table: TableData | None = None,
    annotations: tuple[SourceAnnotation, ...] = (),
    page: PageIdentity | None = None,
) -> SourceBlock:
    """Build one normalized synthetic source block."""

    structural_name = block_type.lower()

    return SourceBlock(
        block_id=f"test-source::block-{order}-{structural_name}",
        source_block_id=f"source-{order}-{structural_name}",
        block_type=block_type,
        order=order,
        parent_id=None,
        child_ids=(),
        hierarchy=hierarchy,
        page=page,
        legal_locator=legal_locator,
        raw_html="",
        searchable_text=searchable_text,
        table=table,
        annotations=annotations,
        images=None,
    )


class BlockClassificationTests(unittest.TestCase):
    """Validate explicit and fail-closed routing."""

    def test_every_known_block_type_has_expected_route(self) -> None:
        expected = {
            "AmendmentMarker": BlockRoute.ANNOTATION_ONLY,
            "Annex": BlockRoute.STRUCTURAL_CONTEXT,
            "AnnexHeading": BlockRoute.STRUCTURAL_CONTEXT,
            "AnnexPoint": BlockRoute.BODY,
            "Annexes": BlockRoute.EXCLUDED,
            "Article": BlockRoute.STRUCTURAL_CONTEXT,
            "Caption": BlockRoute.CAPTION,
            "Chapter": BlockRoute.STRUCTURAL_CONTEXT,
            "ConcludingFormula": BlockRoute.BODY,
            "Diagram": BlockRoute.EXCLUDED,
            "Document": BlockRoute.EXCLUDED,
            "DocumentTitle": BlockRoute.STRUCTURAL_CONTEXT,
            "EnactingTerms": BlockRoute.EXCLUDED,
            "Footnote": BlockRoute.BODY,
            "FootnoteDefinition": BlockRoute.BODY,
            "Footnotes": BlockRoute.EXCLUDED,
            "Indent": BlockRoute.BODY,
            "ListGroup": BlockRoute.EXCLUDED,
            "ListItem": BlockRoute.BODY,
            "Page": BlockRoute.EXCLUDED,
            "PageFooter": BlockRoute.EXCLUDED,
            "PageHeader": BlockRoute.EXCLUDED,
            "Paragraph": BlockRoute.BODY,
            "Picture": BlockRoute.EXCLUDED,
            "Point": BlockRoute.BODY,
            "Section": BlockRoute.STRUCTURAL_CONTEXT,
            "SectionHeader": BlockRoute.STRUCTURAL_CONTEXT,
            "Subparagraph": BlockRoute.BODY,
            "Table": BlockRoute.TABLE,
            "TableGroup": BlockRoute.EXCLUDED,
            "TableOfContents": BlockRoute.EXCLUDED,
            "Text": BlockRoute.BODY,
        }

        self.assertEqual(
            dict(BLOCK_TYPE_ROUTES),
            expected,
        )

        for block_type, expected_route in expected.items():
            with self.subTest(block_type=block_type):
                self.assertEqual(
                    classify_block_type(block_type),
                    expected_route,
                )

    def test_unknown_block_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ChunkerError,
            "Unknown block type",
        ):
            classify_block_type("UnexpectedBlock")

    def test_non_block_object_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ChunkerError,
            "unsupported object at position 0",
        ):
            route_blocks((object(),))  # type: ignore[arg-type]


class BlockRenderingTests(unittest.TestCase):
    """Validate deterministic context and body preservation."""

    def test_hierarchy_and_locator_are_rendered_in_source_order(
        self,
    ) -> None:
        hierarchy = (
            HierarchyEntry(
                level=1,
                kind="Chapter",
                block_id="test-source::chapter-1",
                title="General provisions",
            ),
            HierarchyEntry(
                level=2,
                kind="Article",
                block_id="test-source::article-1",
                title="Scope",
            ),
        )
        locator = (
            LegalLocatorPart(
                kind="article",
                value="1",
            ),
            LegalLocatorPart(
                kind="paragraph",
                value="2",
            ),
        )
        original_body = (
            "  This body retains its original whitespace.\n"
            "Second line.  "
        )
        block = _source_block(
            "Paragraph",
            searchable_text=original_body,
            hierarchy=hierarchy,
            legal_locator=locator,
        )

        routed = route_block(block)

        expected_context = (
            "Hierarchy: Chapter: General provisions > Article: Scope\n"
            "Legal locator: article: 1 > paragraph: 2"
        )

        self.assertEqual(
            routed.context_text,
            expected_context,
        )
        self.assertEqual(
            routed.body_text,
            original_body,
        )
        self.assertEqual(
            routed.rendered_text,
            f"{expected_context}\n\n{original_body}",
        )
        self.assertEqual(
            block.searchable_text,
            original_body,
        )

    def test_empty_hierarchy_title_is_not_invented(self) -> None:
        hierarchy = (
            HierarchyEntry(
                level=1,
                kind="SectionHeader",
                block_id="test-source::empty-heading",
                title="",
            ),
        )
        block = _source_block(
            "Paragraph",
            searchable_text="Body text.",
            hierarchy=hierarchy,
        )

        self.assertEqual(
            render_context(block),
            "Hierarchy: SectionHeader",
        )

    def test_empty_body_does_not_create_context_only_evidence(
        self,
    ) -> None:
        hierarchy = (
            HierarchyEntry(
                level=1,
                kind="Chapter",
                block_id="test-source::chapter-1",
                title="General provisions",
            ),
        )
        original_body = " \n\t "
        block = _source_block(
            "Paragraph",
            searchable_text=original_body,
            hierarchy=hierarchy,
        )

        routed = route_block(block)

        self.assertEqual(
            routed.body_text,
            original_body,
        )
        self.assertEqual(
            routed.context_text,
            "Hierarchy: Chapter: General provisions",
        )
        self.assertIsNone(routed.rendered_text)

    def test_table_and_caption_keep_distinct_routes(self) -> None:
        table = TableData(
            headers=("Device class", "Requirement"),
            rows=(("IIa", "Synthetic requirement"),),
            extraction_method="synthetic",
            source_page=None,
            table_number=None,
            pdfplumber_bbox=None,
            header_detected=None,
        )
        table_block = _source_block(
            "Table",
            order=0,
            searchable_text="Device class | Requirement",
            table=table,
        )
        caption_block = _source_block(
            "Caption",
            order=1,
            searchable_text="Table 1. Device requirements.",
        )

        routed = route_blocks(
            (table_block, caption_block)
        )

        self.assertEqual(
            routed[0].route,
            BlockRoute.TABLE,
        )
        self.assertIs(
            routed[0].block.table,
            table,
        )
        self.assertEqual(
            routed[0].rendered_text,
            table_block.searchable_text,
        )

        self.assertEqual(
            routed[1].route,
            BlockRoute.CAPTION,
        )
        self.assertEqual(
            routed[1].rendered_text,
            caption_block.searchable_text,
        )

    def test_structural_and_excluded_blocks_are_not_evidence(
        self,
    ) -> None:
        cases = (
            ("Article", BlockRoute.STRUCTURAL_CONTEXT),
            ("TableOfContents", BlockRoute.EXCLUDED),
            ("PageHeader", BlockRoute.EXCLUDED),
            ("Document", BlockRoute.EXCLUDED),
        )

        for order, (block_type, expected_route) in enumerate(cases):
            with self.subTest(block_type=block_type):
                block = _source_block(
                    block_type,
                    order=order,
                    searchable_text="Must not become body evidence.",
                )
                routed = route_block(block)

                self.assertEqual(
                    routed.route,
                    expected_route,
                )
                self.assertIsNone(routed.rendered_text)


class AnnotationAndOrderingTests(unittest.TestCase):
    """Validate annotation preservation and stable source order."""

    def test_amendment_marker_is_preserved_without_rendering_symbol(
        self,
    ) -> None:
        annotation = SourceAnnotation(
            kind=AnnotationKind.AMENDMENT,
            identifier="M1",
            target_id=None,
            raw_html='<a title="M1">▼M1</a>',
            text="▼M1 — M1: amended",
        )
        marker = _source_block(
            "AmendmentMarker",
            searchable_text="▼M1",
            annotations=(annotation,),
        )

        routed = route_block(marker)

        self.assertEqual(
            routed.route,
            BlockRoute.ANNOTATION_ONLY,
        )
        self.assertIs(
            routed.block,
            marker,
        )
        self.assertEqual(
            routed.annotations,
            (annotation,),
        )
        self.assertEqual(
            routed.body_text,
            "▼M1",
        )
        self.assertIsNone(routed.rendered_text)

    def test_routing_preserves_supplied_source_order(self) -> None:
        blocks = (
            _source_block(
                "Text",
                order=0,
                searchable_text="First.",
            ),
            _source_block(
                "Table",
                order=1,
                searchable_text="Second.",
            ),
            _source_block(
                "Caption",
                order=2,
                searchable_text="Third.",
            ),
        )

        routed = route_blocks(blocks)

        self.assertEqual(
            tuple(item.block.block_id for item in routed),
            tuple(block.block_id for block in blocks),
        )
        self.assertEqual(
            tuple(item.block.order for item in routed),
            (0, 1, 2),
        )

def _config(
    *,
    target_size: int = 4_000,
    hard_max_size: int = 6_000,
) -> ChunkingConfig:
    """Return an explicit synthetic chunking configuration."""

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


class CandidateGroupingTests(unittest.TestCase):
    """Validate deterministic grouping without constructing Chunks."""

    def test_locator_and_page_changes_do_not_force_boundary(
        self,
    ) -> None:
        hierarchy = (
            HierarchyEntry(
                level=1,
                kind="Article",
                block_id="test-source::article-1",
                title="Scope",
            ),
        )
        first = _source_block(
            "Paragraph",
            order=0,
            searchable_text="First paragraph.",
            hierarchy=hierarchy,
            legal_locator=(
                LegalLocatorPart("paragraph", "1"),
            ),
            page=PageIdentity(0, 1),
        )
        second = _source_block(
            "Paragraph",
            order=1,
            searchable_text="Second paragraph.",
            hierarchy=hierarchy,
            legal_locator=(
                LegalLocatorPart("paragraph", "2"),
            ),
            page=PageIdentity(1, 2),
        )

        result = group_routed_blocks(
            route_blocks((first, second)),
            _config(),
        )

        self.assertIsInstance(
            result,
            CandidateGroupingResult,
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].blocks[0].block,
            first,
        )
        self.assertEqual(
            result.candidates[0].blocks[1].block,
            second,
        )

    def test_hierarchy_change_forces_boundary(self) -> None:
        first_hierarchy = (
            HierarchyEntry(
                level=1,
                kind="Section",
                block_id="test-source::section-1",
                title="First section",
            ),
        )
        second_hierarchy = (
            HierarchyEntry(
                level=1,
                kind="Section",
                block_id="test-source::section-2",
                title="Second section",
            ),
        )
        blocks = (
            _source_block(
                "Text",
                order=0,
                searchable_text="First.",
                hierarchy=first_hierarchy,
            ),
            _source_block(
                "Text",
                order=1,
                searchable_text="Second.",
                hierarchy=second_hierarchy,
            ),
        )

        result = group_routed_blocks(
            route_blocks(blocks),
            _config(),
        )

        self.assertEqual(len(result.candidates), 2)

    def test_target_size_starts_new_candidate(self) -> None:
        blocks = (
            _source_block(
                "Text",
                order=0,
                searchable_text="123456",
            ),
            _source_block(
                "Text",
                order=1,
                searchable_text="abcdef",
            ),
        )

        result = group_routed_blocks(
            route_blocks(blocks),
            _config(
                target_size=10,
                hard_max_size=20,
            ),
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(
            tuple(
                candidate.character_count
                for candidate in result.candidates
            ),
            (6, 6),
        )

    def test_single_block_may_exceed_target_but_not_hard_max(
        self,
    ) -> None:
        block = _source_block(
            "Text",
            searchable_text="123456789012345",
        )

        result = group_routed_blocks(
            route_blocks((block,)),
            _config(
                target_size=10,
                hard_max_size=20,
            ),
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].character_count,
            15,
        )

    def test_block_above_hard_max_requires_future_splitting(
        self,
    ) -> None:
        block = _source_block(
            "Text",
            searchable_text="x" * 21,
        )

        with self.assertRaisesRegex(
            ChunkerError,
            "requires splitting",
        ):
            group_routed_blocks(
                route_blocks((block,)),
                _config(
                    target_size=10,
                    hard_max_size=20,
                ),
            )

    def test_table_remains_an_atomic_selected_block(self) -> None:
        table = TableData(
            headers=("Class", "Requirement"),
            rows=(("IIa", "Synthetic requirement"),),
            extraction_method="synthetic",
            source_page=None,
            table_number=None,
            pdfplumber_bbox=None,
            header_detected=None,
        )
        block = _source_block(
            "Table",
            searchable_text="Class | Requirement",
            table=table,
        )

        result = group_routed_blocks(
            route_blocks((block,)),
            _config(),
        )
        candidate = result.candidates[0]

        self.assertIsInstance(candidate, ChunkCandidate)
        self.assertEqual(len(candidate.blocks), 1)
        self.assertIs(
            candidate.blocks[0].block.table,
            table,
        )

    def test_annotation_records_are_deferred_without_loss(
        self,
    ) -> None:
        annotation = SourceAnnotation(
            kind=AnnotationKind.AMENDMENT,
            identifier="M1",
            target_id=None,
            raw_html="<a>▼M1</a>",
            text="▼M1 — amended",
        )
        marker = _source_block(
            "AmendmentMarker",
            order=0,
            searchable_text="▼M1",
            annotations=(annotation,),
        )
        body = _source_block(
            "Paragraph",
            order=1,
            searchable_text="Amended body.",
        )

        routed = route_blocks((marker, body))
        result = group_routed_blocks(
            routed,
            _config(),
        )

        self.assertEqual(
            result.annotation_records,
            (routed[0],),
        )
        self.assertEqual(
            result.annotation_records[0].annotations,
            (annotation,),
        )
        self.assertEqual(len(result.candidates), 1)

    def test_amendment_marker_forces_candidate_boundary(self) -> None:
        for marker_symbol in ("▼M1", "►M1"):
            with self.subTest(marker_symbol=marker_symbol):
                annotation = SourceAnnotation(
                    kind=AnnotationKind.AMENDMENT,
                    identifier="M1",
                    target_id=None,
                    raw_html=f"<a>{marker_symbol}</a>",
                    text=f"{marker_symbol} — amended",
                )
                blocks = (
                    _source_block(
                        "Paragraph",
                        order=0,
                        searchable_text="Before amendment.",
                    ),
                    _source_block(
                        "AmendmentMarker",
                        order=1,
                        searchable_text=marker_symbol,
                        annotations=(annotation,),
                    ),
                    _source_block(
                        "Paragraph",
                        order=2,
                        searchable_text="After amendment.",
                    ),
                )

                routed = route_blocks(blocks)
                result = group_routed_blocks(routed, _config())

                self.assertEqual(len(result.candidates), 2)
                self.assertEqual(
                    tuple(candidate.text for candidate in result.candidates),
                    ("Before amendment.", "After amendment."),
                )
                self.assertEqual(result.annotation_records, (routed[1],))
                self.assertTrue(
                    all(
                        marker_symbol not in candidate.text
                        for candidate in result.candidates
                    )
                )
                self.assertEqual(
                    tuple(
                        block.block.order
                        for candidate in result.candidates
                        for block in candidate.blocks
                    ),
                    (0, 2),
                )

    def test_excluded_blocks_do_not_change_body_order(self) -> None:
        blocks = (
            _source_block(
                "Text",
                order=0,
                searchable_text="First.",
            ),
            _source_block(
                "PageHeader",
                order=1,
                searchable_text="Page header",
            ),
            _source_block(
                "Text",
                order=2,
                searchable_text="Second.",
            ),
        )

        result = group_routed_blocks(
            route_blocks(blocks),
            _config(),
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            tuple(
                routed.block.order
                for routed in result.candidates[0].blocks
            ),
            (0, 2),
        )

class CandidatePreparationTests(unittest.TestCase):
    """Validate annotation attachment using synthetic blocks only."""

    @staticmethod
    def annotation(
        identifier: str = "M1",
        target_id: str | None = None,
    ) -> SourceAnnotation:
        return SourceAnnotation(
            kind=AnnotationKind.AMENDMENT,
            identifier=identifier,
            target_id=target_id,
            raw_html=f"<a>{identifier}</a>",
            text=f"{identifier}: amended",
        )

    def test_explicit_target_attaches_to_canonical_member_block(self) -> None:
        first = _source_block("Paragraph", order=0, searchable_text="First.")
        target = _source_block("Table", order=1, searchable_text="Table body.")
        annotation = self.annotation(target_id=target.block_id)
        marker = _source_block(
            "AmendmentMarker", order=2, searchable_text="MARKER",
            annotations=(annotation,),
        )
        last = _source_block("Caption", order=3, searchable_text="Last.")
        result = group_routed_blocks(
            route_blocks((first, target, marker, last)), _config(),
        )

        prepared = prepare_chunk_candidates(result)

        self.assertEqual(len(prepared), 2)
        self.assertIsInstance(prepared[0], PreparedChunkCandidate)
        for item, candidate in zip(prepared, result.candidates):
            self.assertIs(item.candidate, candidate)
        self.assertEqual(prepared[0].annotations, (annotation,))
        self.assertEqual(prepared[1].annotations, ())
        self.assertEqual(prepared[0].candidate.text, "First.\n\nTable body.")
        self.assertEqual(prepared[1].candidate.text, "Last.")

    def test_targetless_markers_attach_to_first_following_candidate(self) -> None:
        first_annotation = self.annotation("M1")
        second_annotation = self.annotation("M2")
        body_annotation = self.annotation("M3")
        blocks = (
            _source_block("Paragraph", order=0, searchable_text="Before."),
            _source_block(
                "AmendmentMarker", order=1, searchable_text="MARKER1",
                annotations=(first_annotation,),
            ),
            _source_block(
                "AmendmentMarker", order=2, searchable_text="MARKER2",
                annotations=(second_annotation,),
            ),
            _source_block(
                "Paragraph", order=3, searchable_text="After.",
                annotations=(body_annotation,),
            ),
            _source_block("Paragraph", order=4, searchable_text="Later."),
        )
        result = group_routed_blocks(
            route_blocks(blocks), _config(target_size=10, hard_max_size=20),
        )

        prepared = prepare_chunk_candidates(result)

        self.assertEqual(
            tuple(item.annotations for item in prepared),
            ((), (first_annotation, second_annotation, body_annotation), ()),
        )
        self.assertEqual(
            tuple(item.candidate.text for item in prepared),
            ("Before.", "After.", "Later."),
        )

    def test_body_table_and_caption_annotations_merge_in_source_order(
        self,
    ) -> None:
        existing = tuple(self.annotation(f"M{index}") for index in range(3))
        deferred = self.annotation("M3")
        blocks = (
            _source_block(
                "AmendmentMarker", order=0, annotations=(deferred,),
            ),
            *(
                _source_block(
                    block_type, order=index + 1,
                    searchable_text=f"Body {index}.",
                    annotations=(existing[index],),
                )
                for index, block_type in enumerate(("Paragraph", "Table", "Caption"))
            ),
        )
        result = group_routed_blocks(route_blocks(blocks), _config())

        prepared = prepare_chunk_candidates(result)

        self.assertEqual(prepared[0].annotations, (deferred,) + existing)
        self.assertEqual(result.candidates[0].annotations, existing)
        self.assertEqual(
            prepared[0].candidate.text, "Body 0.\n\nBody 1.\n\nBody 2.",
        )

    def test_explicit_record_annotations_merge_between_candidate_blocks(self) -> None:
        first_annotation = self.annotation("M1")
        body_annotation = self.annotation("M4")
        target = _source_block(
            "Paragraph", order=2, searchable_text="Target.",
            annotations=(body_annotation,),
        )
        record_annotations = (
            self.annotation("M3", target.block_id),
            self.annotation("M2", target.block_id),
        )
        blocks = (
            _source_block(
                "Paragraph", order=0, searchable_text="First.",
                annotations=(first_annotation,),
            ),
            _source_block(
                "PageHeader", order=1, annotations=record_annotations,
            ),
            target,
        )
        result = group_routed_blocks(route_blocks(blocks), _config())

        prepared = prepare_chunk_candidates(result)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(
            prepared[0].annotations,
            (first_annotation,) + record_annotations + (body_annotation,),
        )
        self.assertEqual(prepared[0].candidate.text, "First.\n\nTarget.")

    def test_equal_annotations_from_separate_occurrences_are_all_retained(
        self,
    ) -> None:
        annotation = self.annotation()
        equal_annotation = self.annotation()
        self.assertIsNot(annotation, equal_annotation)
        blocks = (
            _source_block(
                "AmendmentMarker", order=0,
                annotations=(equal_annotation, equal_annotation),
            ),
            _source_block(
                "Paragraph", order=1, searchable_text="Body.",
                annotations=(annotation, equal_annotation),
            ),
        )
        result = group_routed_blocks(route_blocks(blocks), _config())

        prepared = prepare_chunk_candidates(result)

        expected = (equal_annotation, equal_annotation, annotation, equal_annotation)
        self.assertEqual(prepared[0].annotations, expected)
        for actual, occurrence in zip(prepared[0].annotations, expected):
            self.assertIs(actual, occurrence)

    def test_unresolved_explicit_target_is_rejected(self) -> None:
        body = _source_block("Paragraph", order=1, searchable_text="Body.")
        excluded = _source_block("PageHeader", order=2, searchable_text="Header.")
        empty = _source_block("Paragraph", order=3)
        for target_id in ("test-source::missing", body.source_block_id,
                          excluded.block_id, empty.block_id):
            with self.subTest(target_id=target_id):
                marker = _source_block(
                    "AmendmentMarker", order=0,
                    annotations=(self.annotation(target_id=target_id),),
                )
                result = group_routed_blocks(
                    route_blocks((marker, body, excluded, empty)), _config(),
                )

                with self.assertRaisesRegex(ChunkerError, "target is not retrievable"):
                    prepare_chunk_candidates(result)

    def test_trailing_targetless_annotation_is_rejected(self) -> None:
        marker = _source_block(
            "AmendmentMarker", order=1, annotations=(self.annotation(),),
        )
        body = _source_block("Paragraph", order=0, searchable_text="Body.")
        for blocks in ((body, marker), (marker,)):
            with self.subTest(block_count=len(blocks)):
                result = group_routed_blocks(route_blocks(blocks), _config())

                with self.assertRaisesRegex(ChunkerError, "no following candidate"):
                    prepare_chunk_candidates(result)

    def test_annotation_free_markers_remain_valid_boundaries(self) -> None:
        first = _source_block("Paragraph", order=0, searchable_text="First.")
        marker = _source_block(
            "AmendmentMarker", order=1, searchable_text="MARKER",
        )
        last = _source_block("Paragraph", order=2, searchable_text="Last.")
        trailing = _source_block("AmendmentMarker", order=3)
        result = group_routed_blocks(
            route_blocks((first, marker, last, trailing)), _config(),
        )

        prepared = prepare_chunk_candidates(result)

        self.assertEqual(len(result.annotation_records), 2)
        self.assertEqual(tuple(item.annotations for item in prepared), ((), ()))
        self.assertEqual(
            tuple(item.candidate.text for item in prepared), ("First.", "Last."),
        )
        empty_result = group_routed_blocks(route_blocks((trailing,)), _config())
        self.assertEqual(prepare_chunk_candidates(empty_result), ())


if __name__ == "__main__":
    unittest.main()
