"""Focused tests for the authoritative corpus manifest.

The real corpus test reads the frozen sources but never modifies them.
All destructive or invalid scenarios are created inside temporary directories.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.ingestion.manifest import load_manifest
from src.ingestion.models import (
    ManifestSchemaError,
    ValidationStatus,
)


# test_manifest.py is located at:
# <project root>/tests/ingestion/test_manifest.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"


def _valid_source_entry(content: bytes) -> dict[str, Any]:
    """Return one complete, valid source entry for temporary tests."""

    return {
        "document_id": "test-guidance-v1",
        "official_title": "Test guidance",
        "short_title": "Test guidance",
        "version": "1.0",
        "regulatory_regimes": ["MDR"],
        "permitted_regimes": ["MDR"],
        "source_type": "mdcg_guidance",
        "authority": "Medical Device Coordination Group",
        "authority_tier": "mdcg_guidance",
        "official_url": "https://example.eu/test-guidance",
        "local_path": "data/rag_ready/test_guidance.json",
        "content_format": "marker_json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "validation_status": "validated",
        "index_status": "not_built",
    }


def _write_temporary_manifest(
    project_root: Path,
    sources: list[dict[str, Any]],
) -> Path:
    """Write a manifest inside a disposable test project."""

    manifest_path = project_root / "data" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "schema_version": "1.0",
        "corpus_id": "test-corpus",
        "expected_source_count": len(sources),
        "sources": sources,
    }

    manifest_path.write_text(
        json.dumps(manifest_data, indent=2),
        encoding="utf-8",
    )

    return manifest_path


class RealCorpusManifestTests(unittest.TestCase):
    """Integration checks against the approved 18-source corpus."""

    def test_real_manifest_and_all_frozen_sources_verify(self) -> None:
        manifest = load_manifest(
            REAL_MANIFEST_PATH,
            project_root=PROJECT_ROOT,
        )

        self.assertEqual(manifest.corpus_id, "eu-mdr-mvp")
        self.assertEqual(manifest.expected_source_count, 18)
        self.assertEqual(len(manifest.sources), 18)

        self.assertTrue(
            all(
                source.validation_status is ValidationStatus.VALIDATED
                for source in manifest.sources
            )
        )

        # CorpusManifest performs uniqueness validation, but these assertions
        # make the intended acceptance condition visible in the test output.
        self.assertEqual(
            len({source.document_id for source in manifest.sources}),
            18,
        )
        self.assertEqual(
            len({source.local_path for source in manifest.sources}),
            18,
        )
        self.assertEqual(
            len({source.sha256 for source in manifest.sources}),
            18,
        )


class FailClosedManifestTests(unittest.TestCase):
    """Verify that unsafe corpus states are rejected."""

    def test_missing_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = (
                Path(temporary_directory)
                / "data"
                / "corpus_manifest.json"
            )

            with self.assertRaises(ManifestSchemaError):
                load_manifest(missing_path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            manifest_path = (
                project_root / "data" / "corpus_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True)

            # A normal dict cannot contain duplicate keys, so this malformed
            # JSON must be written explicitly.
            manifest_path.write_text(
                """
                {
                    "schema_version": "1.0",
                    "schema_version": "1.0",
                    "corpus_id": "test-corpus",
                    "expected_source_count": 0,
                    "sources": []
                }
                """,
                encoding="utf-8",
            )

            with self.assertRaises(ManifestSchemaError):
                load_manifest(manifest_path, verify_files=False)

    def test_unvalidated_source_is_rejected(self) -> None:
        content = b'{"test": true}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_entry = _valid_source_entry(content)
            source_entry["validation_status"] = "not_validated"

            manifest_path = _write_temporary_manifest(
                project_root,
                [source_entry],
            )

            with self.assertRaises(ManifestSchemaError):
                load_manifest(
                    manifest_path,
                    project_root=project_root,
                )

    def test_missing_source_file_is_rejected(self) -> None:
        content = b'{"test": true}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_entry = _valid_source_entry(content)

            manifest_path = _write_temporary_manifest(
                project_root,
                [source_entry],
            )

            # The manifest exists, but its approved source was not created.
            with self.assertRaises(ManifestSchemaError):
                load_manifest(
                    manifest_path,
                    project_root=project_root,
                )

    def test_wrong_checksum_is_rejected(self) -> None:
        content = b'{"test": true}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_path = (
                project_root / "data" / "rag_ready" / "test_guidance.json"
            )
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(content)

            source_entry = _valid_source_entry(content)
            source_entry["sha256"] = "0" * 64

            manifest_path = _write_temporary_manifest(
                project_root,
                [source_entry],
            )

            with self.assertRaises(ManifestSchemaError):
                load_manifest(
                    manifest_path,
                    project_root=project_root,
                )

    def test_duplicate_source_path_is_rejected(self) -> None:
        content = b'{"test": true}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_path = (
                project_root / "data" / "rag_ready" / "test_guidance.json"
            )
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(content)

            first_source = _valid_source_entry(content)
            second_source = copy.deepcopy(first_source)

            # Keep a different document ID while deliberately duplicating
            # the physical source path.
            second_source["document_id"] = "second-test-guidance-v1"

            manifest_path = _write_temporary_manifest(
                project_root,
                [first_source, second_source],
            )

            with self.assertRaises(ManifestSchemaError):
                load_manifest(
                    manifest_path,
                    project_root=project_root,
                )

    def test_unexpected_manifest_field_is_rejected(self) -> None:
        content = b'{"test": true}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_entry = _valid_source_entry(content)
            source_entry["unapproved_field"] = "must not be ignored"

            manifest_path = _write_temporary_manifest(
                project_root,
                [source_entry],
            )

            with self.assertRaises(ManifestSchemaError):
                load_manifest(
                    manifest_path,
                    project_root=project_root,
                )


if __name__ == "__main__":
    unittest.main()