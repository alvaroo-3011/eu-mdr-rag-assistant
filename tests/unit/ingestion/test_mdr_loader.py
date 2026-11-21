"""Focused tests for loading the validated consolidated MDR HTML source.

The real-corpus tests read frozen data without modifying it. Malformed source
files used by fail-closed tests exist only inside temporary directories.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.ingestion.manifest import load_manifest
from src.ingestion.mdr_loader import MdrLoaderError, load_mdr_source
from src.ingestion.models import (
    AnnotationKind,
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


def _regulation_source(
    content: bytes,
    **changes: object,
) -> SourceManifestEntry:
    """Return trusted manifest metadata for one disposable MDR source."""

    source = SourceManifestEntry(
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
        sha256=hashlib.sha256(content).hexdigest(),
        validation_status=ValidationStatus.VALIDATED,
        index_status=IndexStatus.NOT_BUILT,
    )
    return replace(source, **changes)


def _manifest(source: SourceManifestEntry) -> CorpusManifest:
    """Return a one-source typed manifest."""

    return CorpusManifest(
        schema_version="1.0",
        corpus_id="test-corpus",
        expected_source_count=1,
        sources=(source,),
    )


def _write_disposable_source(
    project_root: Path,
    content: bytes,
    **source_changes: object,
) -> tuple[CorpusManifest, SourceManifestEntry, Path]:
    """Write a temporary source and return its typed manifest context."""

    source_path = project_root / "data" / "rag_ready" / "test_mdr.html"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(content)
    source = _regulation_source(content, **source_changes)
    return _manifest(source), source, source_path


def _real_loaded_source():
    """Load the sole manifest-approved EUR-Lex HTML source."""

    manifest = load_manifest(
        REAL_MANIFEST_PATH,
        project_root=PROJECT_ROOT,
    )
    sources = tuple(
        source
        for source in manifest.sources
        if source.content_format is ContentFormat.EURLEX_HTML
    )
    if len(sources) != 1:
        raise AssertionError(
            "Expected exactly one EUR-Lex HTML source in the corpus manifest."
        )
    source = sources[0]
    return load_mdr_source(
        manifest,
        source.document_id,
        project_root=PROJECT_ROOT,
    )


class RealCorpusMdrLoaderTests(unittest.TestCase):
    """Protect the observed structure and traceability of the regulation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = _real_loaded_source()

    def test_regulation_preserves_observed_structure(self) -> None:
        counts = Counter(block.block_type for block in self.loaded.blocks)
        annotations = Counter(
            annotation.kind
            for block in self.loaded.blocks
            for annotation in block.annotations
        )

        self.assertEqual(len(self.loaded.blocks), 2832)
        self.assertEqual(counts["Chapter"], 10)
        self.assertEqual(counts["Section"], 5)
        self.assertEqual(counts["Article"], 124)
        self.assertEqual(counts["Paragraph"], 539)
        self.assertEqual(counts["Point"], 837)
        self.assertEqual(counts["Indent"], 450)
        self.assertEqual(counts["Annex"], 17)
        self.assertEqual(counts["AnnexPoint"], 327)
        self.assertEqual(counts["Table"], 1)
        self.assertEqual(counts["FootnoteDefinition"], 13)
        self.assertEqual(annotations[AnnotationKind.AMENDMENT], 61)
        self.assertEqual(
            annotations[AnnotationKind.FOOTNOTE_REFERENCE],
            13,
        )
        self.assertTrue(
            all(block.page is None for block in self.loaded.blocks)
        )

    def test_legal_locators_and_table_are_preserved(self) -> None:
        article = next(
            block
            for block in self.loaded.blocks
            if block.source_block_id == "art_1"
        )
        point = next(
            block
            for block in self.loaded.blocks
            if tuple(
                (part.kind, part.value)
                for part in block.legal_locator
            )
            == (
                ("chapter", "I"),
                ("article", "1"),
                ("paragraph", "6"),
                ("point", "a"),
            )
        )
        table_block = next(
            block
            for block in self.loaded.blocks
            if block.block_type == "Table"
        )

        self.assertEqual(
            tuple(
                (part.kind, part.value)
                for part in article.legal_locator
            ),
            (("chapter", "I"), ("article", "1")),
        )
        self.assertEqual(point.block_type, "Point")
        self.assertIn("in vitro diagnostic", point.searchable_text)
        self.assertIsNotNone(table_block.table)
        assert table_block.table is not None
        self.assertEqual(
            table_block.table.headers,
            (
                "Council Directive 90/385/EEC",
                "Council Directive 93/42/EEC",
                "This Regulation",
            ),
        )
        self.assertEqual(len(table_block.table.rows), 68)


class FailClosedMdrLoaderTests(unittest.TestCase):
    """Reject untrusted or structurally invalid regulation inputs."""

    def test_unlisted_document_id_is_rejected(self) -> None:
        content = b"<html></html>"
        manifest = _manifest(_regulation_source(content))

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(MdrLoaderError, "not listed"):
                load_mdr_source(
                    manifest,
                    "unlisted-mdr",
                    project_root=temporary_directory,
                )

    def test_changed_source_after_manifest_creation_is_rejected(self) -> None:
        content = b"<html></html>"

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, source, source_path = _write_disposable_source(
                project_root,
                content,
            )
            source_path.write_bytes(b"<html><body>changed</body></html>")

            with self.assertRaisesRegex(MdrLoaderError, "Checksum mismatch"):
                load_mdr_source(
                    manifest,
                    source.document_id,
                    project_root=project_root,
                )

    def test_missing_approved_source_is_rejected(self) -> None:
        content = b"<html></html>"
        source = _regulation_source(content)

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                MdrLoaderError,
                "missing or is not a file",
            ):
                load_mdr_source(
                    _manifest(source),
                    source.document_id,
                    project_root=temporary_directory,
                )

    def test_duplicate_source_ids_are_rejected(self) -> None:
        content = (
            b'<html><body><div id="docHtml">'
            b'<div class="eli-container">'
            b'<span id="duplicate"></span>'
            b'<span id="duplicate"></span>'
            b'</div></div></body></html>'
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, source, _ = _write_disposable_source(
                project_root,
                content,
            )

            with self.assertRaisesRegex(MdrLoaderError, "duplicate IDs"):
                load_mdr_source(
                    manifest,
                    source.document_id,
                    project_root=project_root,
                )

    def test_missing_legal_container_is_rejected(self) -> None:
        content = b'<html><body><div id="docHtml"></div></body></html>'

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest, source, _ = _write_disposable_source(
                project_root,
                content,
            )

            with self.assertRaisesRegex(
                MdrLoaderError,
                "eli-container legal-content element",
            ):
                load_mdr_source(
                    manifest,
                    source.document_id,
                    project_root=project_root,
                )


if __name__ == "__main__":
    unittest.main()
