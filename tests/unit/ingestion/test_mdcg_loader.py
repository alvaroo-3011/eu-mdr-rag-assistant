"""Focused tests for loading validated Marker JSON guidance sources.

The real-corpus test reads frozen data without modifying it. Every malformed
source used by fail-closed tests is created inside a temporary directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from src.ingestion.manifest import load_manifest
from src.ingestion.mdcg_loader import MdcgLoaderError, load_mdcg_source
from src.ingestion.models import (
    AuthorityTier,
    ContentFormat,
    CorpusManifest,
    IndexStatus,
    RegulatoryRegime,
    SourceManifestEntry,
    SourceType,
    ValidationStatus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"
REPRESENTATIVE_FILENAME = "mdcg_2021_24.json"


def _node(
    source_id: str,
    block_type: str,
    *,
    html: str = "",
    children: list[dict[str, Any]] | None = None,
    hierarchy: dict[str, str] | None = None,
    images: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one ordinary synthetic Marker node."""

    return {
        "id": source_id,
        "block_type": block_type,
        "html": html,
        "polygon": None,
        "bbox": None,
        "children": children,
        "section_hierarchy": hierarchy or {},
        "images": images,
    }


def _table_node(
    *,
    source_page: int = 1,
    extraction_method: str = "pdfplumber",
    hierarchy_target: str = "/page/0/SectionHeader/0",
) -> dict[str, Any]:
    """Return one enhanced pdfplumber table node."""

    node = _node(
        "/page/0/Table/1",
        "Table",
        html=(
            "<table><tr><th>Device</th><th>Class</th></tr>"
            "<tr><td>Example</td><td>IIa</td></tr></table>"
        ),
        hierarchy={"1": hierarchy_target},
        images={"table-image": {"path": "table.png"}},
    )
    node.update(
        {
            "source_page": source_page,
            "table_number": 1,
            "extraction_method": extraction_method,
            "table_data": {
                "headers": ["Device", "Class"],
                "rows": [["Example", "IIa"]],
                "pdfplumber_bbox": [10.0, 20.0, 300.0, 400.0],
                "header_detected": True,
            },
        }
    )
    return node


def _valid_marker_document() -> dict[str, Any]:
    """Return a two-page source with a cross-page hierarchy reference."""

    heading_id = "/page/0/SectionHeader/0"
    heading = _node(
        heading_id,
        "SectionHeader",
        html="<h1> Scope &amp; purpose </h1>",
    )
    table = _table_node(hierarchy_target=heading_id)
    footnote = _node(
        "/page/0/Footnote/2",
        "Footnote",
        html="<p>1 Footnote text</p>",
        hierarchy={"1": heading_id},
    )
    first_page = _node(
        "/page/0/Page/0",
        "Page",
        children=[heading, table, footnote],
    )
    cross_page_text = _node(
        "/page/1/Text/0",
        "Text",
        html="<p>  Continued   guidance. </p>",
        hierarchy={"1": heading_id},
    )
    second_page = _node(
        "/page/1/Page/0",
        "Page",
        children=[cross_page_text],
    )

    return {
        "children": [first_page, second_page],
        "block_type": "Document",
        "metadata": {"synthetic": True},
    }


def _write_source_and_manifest(
    project_root: Path,
    *,
    document: dict[str, Any] | None = None,
    raw_content: bytes | None = None,
) -> tuple[CorpusManifest, SourceManifestEntry, Path]:
    """Write one disposable source and return its trusted typed manifest."""

    if (document is None) == (raw_content is None):
        raise ValueError("Provide exactly one source representation.")

    content = (
        json.dumps(document, ensure_ascii=False).encode("utf-8")
        if document is not None
        else raw_content
    )
    assert content is not None

    source_path = (
        project_root
        / "data"
        / "rag_ready"
        / "test_guidance.json"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(content)

    source = SourceManifestEntry(
        document_id="test-guidance",
        official_title="Test guidance",
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
        sha256=hashlib.sha256(content).hexdigest(),
        validation_status=ValidationStatus.VALIDATED,
        index_status=IndexStatus.NOT_BUILT,
    )
    manifest = CorpusManifest(
        schema_version="1.0",
        corpus_id="test-corpus",
        expected_source_count=1,
        sources=(source,),
    )
    return manifest, source, source_path


def _load_disposable_document(document: dict[str, Any]) -> None:
    """Load one temporary document, allowing a test to assert rejection."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        project_root = Path(temporary_directory)
        manifest, source, _ = _write_source_and_manifest(
            project_root,
            document=document,
        )
        load_mdcg_source(
            manifest,
            source.document_id,
            project_root=project_root,
        )


class RealCorpusMdcgLoaderTests(unittest.TestCase):
    """Protect the observed signature of the representative guidance."""

    def test_representative_guidance_preserves_structure(self) -> None:
        manifest = load_manifest(
            REAL_MANIFEST_PATH,
            project_root=PROJECT_ROOT,
        )
        source = next(
            item
            for item in manifest.sources
            if PurePosixPath(item.local_path).name
            == REPRESENTATIVE_FILENAME
        )

        loaded = load_mdcg_source(
            manifest,
            source.document_id,
            project_root=PROJECT_ROOT,
        )
        counts = Counter(block.block_type for block in loaded.blocks)
        tables = tuple(
            block.table
            for block in loaded.blocks
            if block.block_type == "Table"
        )
        empty_hierarchy_titles = {
            entry.block_id
            for block in loaded.blocks
            for entry in block.hierarchy
            if entry.title == ""
        }

        self.assertEqual(len(loaded.blocks), 767)
        self.assertEqual(counts["Page"], 59)
        self.assertEqual(counts["Table"], 37)
        self.assertEqual(counts["Footnote"], 30)
        self.assertEqual(len(empty_hierarchy_titles), 75)
        self.assertTrue(all(table is not None for table in tables))
        self.assertTrue(
            all(
                table.extraction_method == "pdfplumber"
                for table in tables
                if table is not None
            )
        )


class ValidSyntheticMdcgLoaderTests(unittest.TestCase):
    """Confirm normalized records preserve trusted source information."""

    def test_valid_source_preserves_traceability_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, source, _ = _write_source_and_manifest(
                project_root,
                document=_valid_marker_document(),
            )

            loaded = load_mdcg_source(
                manifest,
                source.document_id,
                project_root=project_root,
            )
            blocks = {
                block.source_block_id: block
                for block in loaded.blocks
                if block.source_block_id is not None
            }
            root = loaded.blocks[0]
            heading = blocks["/page/0/SectionHeader/0"]
            table = blocks["/page/0/Table/1"]
            footnote = blocks["/page/0/Footnote/2"]
            cross_page_text = blocks["/page/1/Text/0"]

            self.assertEqual(len(loaded.blocks), 7)
            self.assertIs(loaded.source, source)
            self.assertEqual(root.block_type, "Document")
            self.assertEqual(len(root.child_ids), 2)
            self.assertEqual(heading.searchable_text, "Scope & purpose")
            self.assertEqual(table.page.pdf_page_number, 1)
            self.assertEqual(table.table.headers, ("Device", "Class"))
            self.assertEqual(table.table.rows, (("Example", "IIa"),))
            self.assertEqual(table.images["table-image"]["path"], "table.png")
            self.assertEqual(footnote.searchable_text, "1 Footnote text")
            self.assertEqual(cross_page_text.page.pdf_page_number, 2)
            self.assertEqual(cross_page_text.searchable_text, "Continued guidance.")
            self.assertEqual(len(cross_page_text.hierarchy), 1)
            self.assertEqual(
                cross_page_text.hierarchy[0].block_id,
                heading.block_id,
            )
            self.assertEqual(
                cross_page_text.hierarchy[0].title,
                "Scope & purpose",
            )


class FailClosedMdcgLoaderTests(unittest.TestCase):
    """Reject changed, malformed, or structurally unsafe guidance sources."""

    def test_unlisted_document_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, _, _ = _write_source_and_manifest(
                project_root,
                document=_valid_marker_document(),
            )

            with self.assertRaisesRegex(
                MdcgLoaderError,
                "not listed",
            ):
                load_mdcg_source(
                    manifest,
                    "unlisted-guidance",
                    project_root=project_root,
                )

    def test_changed_source_after_manifest_creation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, source, source_path = _write_source_and_manifest(
                project_root,
                document=_valid_marker_document(),
            )
            source_path.write_bytes(b'{"changed": true}')

            with self.assertRaisesRegex(
                MdcgLoaderError,
                "Checksum mismatch",
            ):
                load_mdcg_source(
                    manifest,
                    source.document_id,
                    project_root=project_root,
                )

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw_content = (
            b'{"children": [], "block_type": "Document", '
            b'"block_type": "Document", "metadata": {}}'
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, source, _ = _write_source_and_manifest(
                project_root,
                raw_content=raw_content,
            )

            with self.assertRaisesRegex(
                MdcgLoaderError,
                "Duplicate JSON key",
            ):
                load_mdcg_source(
                    manifest,
                    source.document_id,
                    project_root=project_root,
                )

    def test_unexpected_node_field_is_rejected(self) -> None:
        document = _valid_marker_document()
        document["children"][0]["unexpected"] = "unsafe"

        with self.assertRaisesRegex(MdcgLoaderError, "unexpected keys"):
            _load_disposable_document(document)

    def test_noncontiguous_top_level_pages_are_rejected(self) -> None:
        document = _valid_marker_document()
        second_page = document["children"][1]
        second_page["id"] = "/page/2/Page/0"
        second_page["children"][0]["id"] = "/page/2/Text/0"

        with self.assertRaisesRegex(MdcgLoaderError, "contiguous"):
            _load_disposable_document(document)

    def test_unknown_hierarchy_target_is_rejected(self) -> None:
        document = _valid_marker_document()
        document["children"][1]["children"][0]["section_hierarchy"] = {
            "1": "/page/0/SectionHeader/999"
        }

        with self.assertRaisesRegex(
            MdcgLoaderError,
            "Unknown section_hierarchy target",
        ):
            _load_disposable_document(document)

    def test_duplicate_marker_block_id_is_rejected(self) -> None:
        document = _valid_marker_document()
        duplicate = copy.deepcopy(
            document["children"][0]["children"][2]
        )
        document["children"][0]["children"].append(duplicate)

        with self.assertRaisesRegex(MdcgLoaderError, "Duplicate Marker"):
            _load_disposable_document(document)

    def test_table_page_mismatch_is_rejected(self) -> None:
        document = _valid_marker_document()
        table = document["children"][0]["children"][1]
        table["source_page"] = 2

        with self.assertRaisesRegex(
            MdcgLoaderError,
            "does not match its Marker page",
        ):
            _load_disposable_document(document)

    def test_non_pdfplumber_table_is_rejected(self) -> None:
        document = _valid_marker_document()
        table = document["children"][0]["children"][1]
        table["extraction_method"] = "marker"

        with self.assertRaisesRegex(MdcgLoaderError, "must use pdfplumber"):
            _load_disposable_document(document)


if __name__ == "__main__":
    unittest.main()
