"""
Focused tests for normalized source-record models.

These tests use synthetic records only. They do not read, modify, or
rewrite any frozen corpus source.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.ingestion.models import (
    AnnotationKind,
    AuthorityTier,
    ContentFormat,
    HierarchyEntry,
    IndexStatus,
    LegalLocatorPart,
    LoadedSource,
    PageIdentity,
    RegulatoryRegime,
    SourceAnnotation,
    SourceBlock,
    SourceManifestEntry,
    SourceSchemaError,
    SourceType,
    TableData,
    ValidationStatus,
)


def _guidance_source() -> SourceManifestEntry:
    """Return trusted manifest metadata for synthetic guidance."""

    return SourceManifestEntry(
        document_id="test-guidance",
        official_title="Test MDCG Guidance",
        short_title="Test guidance",
        version="1.0",
        regulatory_regimes=(RegulatoryRegime.MDR,),
        permitted_regimes=(RegulatoryRegime.MDR,),
        source_type=SourceType.MDCG_GUIDANCE,
        authority="Medical Device Coordination Group",
        authority_tier=AuthorityTier.MDCG_GUIDANCE,
        official_url="https://example.eu/test-guidance",
        local_path="data/rag_ready/test_guidance.json",
        content_format=ContentFormat.MARKER_JSON,
        sha256="a" * 64,
        validation_status=ValidationStatus.VALIDATED,
        index_status=IndexStatus.NOT_BUILT,
    )


def _mdr_source() -> SourceManifestEntry:
    """Return trusted manifest metadata for synthetic MDR HTML."""

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
        official_url="https://example.eu/test-mdr",
        local_path="data/rag_ready/test_mdr.html",
        content_format=ContentFormat.EURLEX_HTML,
        sha256="b" * 64,
        validation_status=ValidationStatus.VALIDATED,
        index_status=IndexStatus.NOT_BUILT,
    )


def _block_id(
    source: SourceManifestEntry,
    structural_id: str,
) -> str:
    """Build a canonical block ID for one synthetic source."""

    return f"{source.source_identity}::{structural_id}"


def _source_block(
    source: SourceManifestEntry,
    structural_id: str,
    *,
    order: int,
    source_block_id: str | None = None,
    parent: str | None = None,
    children: tuple[str, ...] = (),
    hierarchy: tuple[HierarchyEntry, ...] = (),
    page: PageIdentity | None = None,
    legal_locator: tuple[LegalLocatorPart, ...] = (),
    raw_html: str = "",
    searchable_text: str = "",
    table: TableData | None = None,
    annotations: tuple[SourceAnnotation, ...] = (),
    images: dict[str, object] | None = None,
) -> SourceBlock:
    """Build one normalized synthetic block."""

    parent_id = (
        _block_id(source, parent)
        if parent is not None
        else None
    )

    child_ids = tuple(
        _block_id(source, child)
        for child in children
    )

    return SourceBlock(
        block_id=_block_id(source, structural_id),
        source_block_id=source_block_id,
        block_type="SyntheticBlock",
        order=order,
        parent_id=parent_id,
        child_ids=child_ids,
        hierarchy=hierarchy,
        page=page,
        legal_locator=legal_locator,
        raw_html=raw_html,
        searchable_text=searchable_text,
        table=table,
        annotations=annotations,
        images=images,
    )


class SourceValueModelTests(unittest.TestCase):
    """Validate individual page, table, and block values."""

    def test_empty_hierarchy_title_is_preserved(self) -> None:
        source = _guidance_source()

        entry = HierarchyEntry(
            level=1,
            kind="section",
            block_id=_block_id(source, "empty-heading"),
            title="",
        )

        self.assertEqual(entry.title, "")

    def test_page_identity_preserves_both_numbering_systems(
        self,
    ) -> None:
        page = PageIdentity(
            source_index=0,
            pdf_page_number=1,
        )

        self.assertEqual(page.source_index, 0)
        self.assertEqual(page.pdf_page_number, 1)

    def test_inconsistent_page_numbers_are_rejected(self) -> None:
        with self.assertRaises(SourceSchemaError):
            PageIdentity(
                source_index=0,
                pdf_page_number=2,
            )

    def test_structured_table_preserves_pdfplumber_data(
        self,
    ) -> None:
        table = TableData(
            headers=("Device class", "Requirement"),
            rows=(("IIa", "Example requirement"),),
            extraction_method="pdfplumber",
            source_page=4,
            table_number=2,
            pdfplumber_bbox=(10.0, 20.0, 300.0, 400.0),
            header_detected=True,
        )

        self.assertEqual(
            table.headers,
            ("Device class", "Requirement"),
        )
        self.assertEqual(
            table.rows,
            (("IIa", "Example requirement"),),
        )
        self.assertEqual(
            table.pdfplumber_bbox,
            (10.0, 20.0, 300.0, 400.0),
        )

    def test_invalid_table_bbox_is_rejected(self) -> None:
        with self.assertRaises(SourceSchemaError):
            TableData(
                headers=(),
                rows=(),
                extraction_method="pdfplumber",
                source_page=0,
                table_number=1,
                pdfplumber_bbox=(10.0, 20.0, 30.0),
                header_detected=False,
            )

    def test_hierarchy_must_be_outermost_to_innermost(
        self,
    ) -> None:
        source = _guidance_source()

        reversed_hierarchy = (
            HierarchyEntry(
                level=2,
                kind="subsection",
                block_id=_block_id(source, "inner"),
                title="Inner heading",
            ),
            HierarchyEntry(
                level=1,
                kind="section",
                block_id=_block_id(source, "outer"),
                title="Outer heading",
            ),
        )

        with self.assertRaises(SourceSchemaError):
            _source_block(
                source,
                "content",
                order=0,
                hierarchy=reversed_hierarchy,
            )


class ValidLoadedSourceTests(unittest.TestCase):
    """Confirm valid guidance and MDR structures are accepted."""

    def test_valid_guidance_structure_is_preserved(self) -> None:
        source = _guidance_source()

        root = "/document"
        page = "/page/0"
        heading = "/page/0/SectionHeader/0"
        table = "/page/0/Table/1"
        footnote = "/page/0/Footnote/2"

        page_identity = PageIdentity(
            source_index=0,
            pdf_page_number=1,
        )

        heading_entry = HierarchyEntry(
            level=1,
            kind="section",
            block_id=_block_id(source, heading),
            title="Scope",
        )

        table_data = TableData(
            headers=("Device", "Class"),
            rows=(("Example device", "IIa"),),
            extraction_method="pdfplumber",
            source_page=1,
            table_number=1,
            pdfplumber_bbox=(10.0, 20.0, 300.0, 400.0),
            header_detected=True,
        )

        footnote_reference = SourceAnnotation(
            kind=AnnotationKind.FOOTNOTE_REFERENCE,
            identifier="1",
            target_id=_block_id(source, footnote),
            raw_html='<sup><a href="#fn1">1</a></sup>',
            text="1",
        )

        blocks = (
            _source_block(
                source,
                root,
                order=0,
                source_block_id=root,
                children=(page,),
            ),
            _source_block(
                source,
                page,
                order=1,
                source_block_id=page,
                parent=root,
                children=(heading, table, footnote),
                page=page_identity,
                images={
                    "diagram": {
                        "path": "diagram.png",
                    }
                },
            ),
            _source_block(
                source,
                heading,
                order=2,
                source_block_id=heading,
                parent=page,
                page=page_identity,
                raw_html="<h1>Scope</h1>",
                searchable_text="Scope",
            ),
            _source_block(
                source,
                table,
                order=3,
                source_block_id=table,
                parent=page,
                hierarchy=(heading_entry,),
                page=page_identity,
                raw_html="<table><tr><td>IIa</td></tr></table>",
                searchable_text="Device Class Example device IIa",
                table=table_data,
                annotations=(footnote_reference,),
            ),
            _source_block(
                source,
                footnote,
                order=4,
                source_block_id=footnote,
                parent=page,
                hierarchy=(heading_entry,),
                page=page_identity,
                raw_html='<p id="fn1">1 Footnote text</p>',
                searchable_text="1 Footnote text",
            ),
        )

        loaded_source = LoadedSource(
            source=source,
            blocks=blocks,
        )

        self.assertIs(loaded_source.source, source)
        self.assertEqual(len(loaded_source.blocks), 5)
        self.assertEqual(
            loaded_source.blocks[3].table,
            table_data,
        )
        self.assertEqual(
            loaded_source.blocks[3].annotations[0].target_id,
            _block_id(source, footnote),
        )
        self.assertEqual(
            loaded_source.blocks[1].page.pdf_page_number,
            1,
        )

    def test_valid_mdr_legal_structure_has_no_pages(self) -> None:
        source = _mdr_source()

        root = "document"
        chapter = "chapter-I"
        article = "chapter-I/article-1"
        paragraph = "chapter-I/article-1/paragraph-1"

        chapter_heading = HierarchyEntry(
            level=1,
            kind="chapter",
            block_id=_block_id(source, chapter),
            title="CHAPTER I",
        )

        article_heading = HierarchyEntry(
            level=2,
            kind="article",
            block_id=_block_id(source, article),
            title="Article 1 — Subject matter and scope",
        )

        blocks = (
            _source_block(
                source,
                root,
                order=0,
                children=(chapter,),
            ),
            _source_block(
                source,
                chapter,
                order=1,
                source_block_id="cpt_I",
                parent=root,
                children=(article,),
                legal_locator=(
                    LegalLocatorPart(
                        kind="chapter",
                        value="I",
                    ),
                ),
                raw_html='<div id="cpt_I">CHAPTER I</div>',
                searchable_text="CHAPTER I",
            ),
            _source_block(
                source,
                article,
                order=2,
                source_block_id="art_1",
                parent=chapter,
                children=(paragraph,),
                hierarchy=(chapter_heading,),
                legal_locator=(
                    LegalLocatorPart(
                        kind="chapter",
                        value="I",
                    ),
                    LegalLocatorPart(
                        kind="article",
                        value="1",
                    ),
                ),
                raw_html='<div id="art_1">Article 1</div>',
                searchable_text="Article 1 Subject matter and scope",
            ),
            _source_block(
                source,
                paragraph,
                order=3,
                parent=article,
                hierarchy=(
                    chapter_heading,
                    article_heading,
                ),
                legal_locator=(
                    LegalLocatorPart(
                        kind="chapter",
                        value="I",
                    ),
                    LegalLocatorPart(
                        kind="article",
                        value="1",
                    ),
                    LegalLocatorPart(
                        kind="paragraph",
                        value="1",
                    ),
                ),
                raw_html="<div>1. This Regulation...</div>",
                searchable_text="1. This Regulation...",
            ),
        )

        loaded_source = LoadedSource(
            source=source,
            blocks=blocks,
        )

        self.assertTrue(
            all(
                block.page is None
                for block in loaded_source.blocks
            )
        )
        self.assertEqual(
            loaded_source.blocks[3].legal_locator[-1].value,
            "1",
        )


class FailClosedLoadedSourceTests(unittest.TestCase):
    """Reject unsafe or internally inconsistent loaded sources."""

    def test_wrong_source_identity_prefix_is_rejected(self) -> None:
        source = _guidance_source()

        block = replace(
            _source_block(
                source,
                "document",
                order=0,
            ),
            block_id="another-document::document",
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(block,),
            )

    def test_duplicate_block_ids_are_rejected(self) -> None:
        source = _guidance_source()

        first = _source_block(
            source,
            "duplicate",
            order=0,
        )
        second = replace(
            first,
            order=1,
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(first, second),
            )

    def test_non_contiguous_block_order_is_rejected(self) -> None:
        source = _guidance_source()

        block = _source_block(
            source,
            "document",
            order=1,
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(block,),
            )

    def test_asymmetric_parent_child_relationship_is_rejected(
        self,
    ) -> None:
        source = _guidance_source()

        root = _source_block(
            source,
            "document",
            order=0,
            children=("child",),
        )
        child_without_parent = _source_block(
            source,
            "child",
            order=1,
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(root, child_without_parent),
            )

    def test_unknown_hierarchy_reference_is_rejected(self) -> None:
        source = _guidance_source()

        unknown_heading = HierarchyEntry(
            level=1,
            kind="section",
            block_id=_block_id(source, "missing-heading"),
            title="Missing heading",
        )

        block = _source_block(
            source,
            "document",
            order=0,
            hierarchy=(unknown_heading,),
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(block,),
            )

    def test_unknown_footnote_target_is_rejected(self) -> None:
        source = _guidance_source()

        missing_footnote = SourceAnnotation(
            kind=AnnotationKind.FOOTNOTE_REFERENCE,
            identifier="1",
            target_id=_block_id(source, "missing-footnote"),
            raw_html="<sup>1</sup>",
            text="1",
        )

        block = _source_block(
            source,
            "document",
            order=0,
            annotations=(missing_footnote,),
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(block,),
            )

    def test_eurlex_html_page_identity_is_rejected(self) -> None:
        source = _mdr_source()

        block = _source_block(
            source,
            "document",
            order=0,
            page=PageIdentity(
                source_index=0,
                pdf_page_number=1,
            ),
        )

        with self.assertRaises(SourceSchemaError):
            LoadedSource(
                source=source,
                blocks=(block,),
            )


if __name__ == "__main__":
    unittest.main()
