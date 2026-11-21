"""Tests for the corpus-wide ingestion boundary."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from src.ingestion.corpus_loader import load_corpus
from src.ingestion.manifest import load_manifest
from src.ingestion.models import (
    CorpusManifest,
    LoadedSource,
    SourceBlock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"


class CorpusLoaderTests(unittest.TestCase):
    """Focused unit tests for corpus dispatch and boundary validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(
            MANIFEST_PATH,
            project_root=PROJECT_ROOT,
            verify_files=True,
        )

    def _single_source_manifest(
        self,
        source,
    ) -> CorpusManifest:
        """Create a valid typed manifest containing one real source entry."""
        return CorpusManifest(
            schema_version=self.manifest.schema_version,
            corpus_id=self.manifest.corpus_id,
            expected_source_count=1,
            sources=(source,),
        )

    @staticmethod
    def _fake_loaded_source(source) -> LoadedSource:
        """Return the smallest valid loaded source for dispatch tests."""
        block = SourceBlock(
            block_id=f"{source.source_identity}::document",
            source_block_id=None,
            block_type="Document",
            order=0,
            parent_id=None,
            child_ids=(),
            hierarchy=(),
            page=None,
            legal_locator=(),
            raw_html="",
            searchable_text="",
            table=None,
            annotations=(),
            images=None,
        )

        return LoadedSource(
            source=source,
            blocks=(block,),
        )

    def test_dispatches_all_sources_in_manifest_order(self) -> None:
        dispatch_order: list[str] = []

        def load_marker(
            manifest,
            document_id,
            *,
            project_root,
        ):
            dispatch_order.append(document_id)
            return self._fake_loaded_source(
                manifest.get_source(document_id)
            )

        def load_html(
            manifest,
            document_id,
            *,
            project_root,
        ):
            dispatch_order.append(document_id)
            return self._fake_loaded_source(
                manifest.get_source(document_id)
            )

        with (
            patch(
                "src.ingestion.corpus_loader.load_mdcg_source",
                side_effect=load_marker,
            ) as marker_loader,
            patch(
                "src.ingestion.corpus_loader.load_mdr_source",
                side_effect=load_html,
            ) as html_loader,
        ):
            loaded = load_corpus(
                self.manifest,
                project_root=PROJECT_ROOT,
            )

        expected_document_ids = [
            source.document_id
            for source in self.manifest.sources
        ]

        self.assertIsInstance(loaded, tuple)
        self.assertEqual(len(loaded), 18)
        self.assertEqual(dispatch_order, expected_document_ids)
        self.assertEqual(
            tuple(item.source for item in loaded),
            self.manifest.sources,
        )
        self.assertEqual(marker_loader.call_count, 17)
        self.assertEqual(html_loader.call_count, 1)

    def test_rejects_expected_source_count_mismatch(self) -> None:
        invalid_manifest = Mock(spec=CorpusManifest)
        invalid_manifest.expected_source_count = 2
        invalid_manifest.sources = (self.manifest.sources[0],)

        with self.assertRaisesRegex(
            ValueError,
            "Manifest source count mismatch",
        ):
            load_corpus(
                invalid_manifest,
                project_root=PROJECT_ROOT,
            )

    def test_rejects_duplicate_manifest_source_identities(self) -> None:
        source = self.manifest.sources[0]
        invalid_manifest = Mock(spec=CorpusManifest)
        invalid_manifest.expected_source_count = 2
        invalid_manifest.sources = (source, source)

        with self.assertRaisesRegex(
            ValueError,
            "duplicate source identities",
        ):
            load_corpus(
                invalid_manifest,
                project_root=PROJECT_ROOT,
            )

    def test_rejects_unsupported_content_format(self) -> None:
        invalid_source = Mock()
        invalid_source.document_id = "unsupported-source"
        invalid_source.source_identity = (
            "unsupported-source@sha256:placeholder"
        )
        invalid_source.content_format = "unsupported_format"

        invalid_manifest = Mock(spec=CorpusManifest)
        invalid_manifest.expected_source_count = 1
        invalid_manifest.sources = (invalid_source,)

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported content format",
        ):
            load_corpus(
                invalid_manifest,
                project_root=PROJECT_ROOT,
            )

    def test_rejects_missing_loader_output(self) -> None:
        source = next(
            source
            for source in self.manifest.sources
            if source.content_format.value == "marker_json"
        )
        manifest = self._single_source_manifest(source)

        with patch(
            "src.ingestion.corpus_loader.load_mdcg_source",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                TypeError,
                "expected LoadedSource",
            ):
                load_corpus(
                    manifest,
                    project_root=PROJECT_ROOT,
                )

    def test_rejects_inconsistent_loaded_source(self) -> None:
        marker_sources = [
            source
            for source in self.manifest.sources
            if source.content_format.value == "marker_json"
        ]
        requested_source = marker_sources[0]
        wrong_source = marker_sources[1]
        manifest = self._single_source_manifest(requested_source)

        inconsistent_output = self._fake_loaded_source(
            wrong_source
        )

        with patch(
            "src.ingestion.corpus_loader.load_mdcg_source",
            return_value=inconsistent_output,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "does not match",
            ):
                load_corpus(
                    manifest,
                    project_root=PROJECT_ROOT,
                )


class RealCorpusLoaderIntegrationTests(unittest.TestCase):
    """Prove that every frozen manifest source normalizes successfully."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(
            MANIFEST_PATH,
            project_root=PROJECT_ROOT,
            verify_files=True,
        )
        cls.loaded_sources = load_corpus(
            cls.manifest,
            project_root=PROJECT_ROOT,
        )

    def test_loads_complete_real_corpus(self) -> None:
        self.assertIsInstance(self.loaded_sources, tuple)
        self.assertEqual(len(self.loaded_sources), 18)
        self.assertTrue(
            all(source.blocks for source in self.loaded_sources)
        )

    def test_real_corpus_matches_manifest_order(self) -> None:
        self.assertEqual(
            tuple(
                loaded.source.source_identity
                for loaded in self.loaded_sources
            ),
            tuple(
                source.source_identity
                for source in self.manifest.sources
            ),
        )

    def test_real_corpus_source_identities_are_unique(self) -> None:
        identities = [
            loaded.source.source_identity
            for loaded in self.loaded_sources
        ]
        self.assertEqual(len(identities), len(set(identities)))

    def test_real_corpus_preserves_table_header_detection_status(
        self,
    ) -> None:
        tables = [
            block.table
            for loaded_source in self.loaded_sources
            if loaded_source.source.content_format.value == "marker_json"
            for block in loaded_source.blocks
            if block.table is not None
        ]

        unavailable_count = sum(
            table.header_detected is None
            for table in tables
        )
        boolean_count = sum(
            isinstance(table.header_detected, bool)
            for table in tables
        )

        self.assertEqual(len(tables), 87)
        self.assertEqual(unavailable_count, 49)
        self.assertEqual(boolean_count, 38)

    def test_real_corpus_format_distribution(self) -> None:
        marker_count = sum(
            loaded.source.content_format.value == "marker_json"
            for loaded in self.loaded_sources
        )
        html_count = sum(
            loaded.source.content_format.value == "eurlex_html"
            for loaded in self.loaded_sources
        )

        self.assertEqual(marker_count, 17)
        self.assertEqual(html_count, 1)


if __name__ == "__main__":
    unittest.main()
