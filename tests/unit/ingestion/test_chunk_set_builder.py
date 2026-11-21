"""Synthetic tests for corpus-wide immutable ChunkSet construction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
import unittest

from src.ingestion.chunk_models import (
    ChunkingConfig,
    compute_chunk_set_id,
    compute_chunking_config_id,
    compute_corpus_identity,
)
from src.ingestion.chunk_set_builder import (
    ChunkSetBuilderError,
    build_chunk_set,
)
from src.ingestion.models import (
    CorpusManifest,
    LoadedSource,
)
from tests.unit.ingestion.test_chunk_builder import (
    _block,
    _config,
    _source,
)


def _synthetic_corpus(
    *,
    first_blocks=None,
    second_blocks=None,
) -> tuple[CorpusManifest, tuple[LoadedSource, LoadedSource]]:
    """Build two validated synthetic sources in authoritative order."""

    first_source = _source()
    suffix = PurePosixPath(first_source.local_path).suffix
    second_source = replace(
        first_source,
        document_id="synthetic-source-two",
        official_title="Synthetic source two",
        short_title="Synthetic two",
        official_url="https://example.com/source-two",
        local_path=f"data/rag_ready/source-two{suffix}",
        sha256="1" * 64,
    )

    if first_blocks is None:
        first_blocks = (
            _block(
                first_source,
                "Paragraph",
                order=0,
                text="First source body.",
            ),
        )
    if second_blocks is None:
        second_blocks = (
            _block(
                second_source,
                "Paragraph",
                order=0,
                text="Second source body.",
            ),
        )

    manifest = CorpusManifest(
        schema_version="1.0",
        corpus_id="synthetic-corpus",
        expected_source_count=2,
        sources=(first_source, second_source),
    )
    loaded = (
        LoadedSource(source=first_source, blocks=tuple(first_blocks)),
        LoadedSource(source=second_source, blocks=tuple(second_blocks)),
    )
    return manifest, loaded


class ChunkSetBuilderTests(unittest.TestCase):
    """Validate identities, coverage, ordering, and source boundaries."""

    def test_complete_chunk_set_has_canonical_identities(self) -> None:
        manifest, loaded = _synthetic_corpus()
        config = _config()

        result = build_chunk_set(manifest, loaded, config)

        corpus_identity = compute_corpus_identity(manifest)
        config_id = compute_chunking_config_id(config)
        self.assertEqual(result.corpus_identity, corpus_identity)
        self.assertEqual(result.chunking_config_id, config_id)
        self.assertEqual(
            result.chunk_set_id,
            compute_chunk_set_id(corpus_identity, config_id),
        )
        self.assertIs(result.manifest, manifest)
        self.assertIs(result.chunking_config, config)

    def test_chunks_follow_manifest_then_source_local_order(self) -> None:
        manifest, default_loaded = _synthetic_corpus()
        first_source, second_source = manifest.sources
        first_blocks = (
            _block(
                first_source,
                "Paragraph",
                order=0,
                text="First.",
            ),
            _block(
                first_source,
                "Paragraph",
                order=1,
                text="Second.",
            ),
        )
        second_blocks = default_loaded[1].blocks
        loaded = (
            LoadedSource(source=first_source, blocks=first_blocks),
            LoadedSource(source=second_source, blocks=second_blocks),
        )

        result = build_chunk_set(
            manifest,
            loaded,
            _config(target_size=8, hard_max_size=20),
        )

        self.assertEqual(
            tuple(chunk.source.document_id for chunk in result.chunks),
            (
                first_source.document_id,
                first_source.document_id,
                second_source.document_id,
            ),
        )
        self.assertEqual(
            tuple(chunk.order for chunk in result.chunks),
            (0, 1, 0),
        )

    def test_neighbor_links_never_cross_source_boundary(self) -> None:
        manifest, default_loaded = _synthetic_corpus()
        first_source, second_source = manifest.sources
        first_loaded = LoadedSource(
            source=first_source,
            blocks=(
                _block(
                    first_source,
                    "Paragraph",
                    order=0,
                    text="First.",
                ),
                _block(
                    first_source,
                    "Paragraph",
                    order=1,
                    text="Second.",
                ),
            ),
        )
        loaded = (first_loaded, default_loaded[1])

        result = build_chunk_set(
            manifest,
            loaded,
            _config(target_size=8, hard_max_size=20),
        )
        first, second, third = result.chunks

        self.assertIsNone(first.previous_chunk_id)
        self.assertEqual(first.next_chunk_id, second.chunk_id)
        self.assertEqual(second.previous_chunk_id, first.chunk_id)
        self.assertIsNone(second.next_chunk_id)
        self.assertIsNone(third.previous_chunk_id)
        self.assertIsNone(third.next_chunk_id)

    def test_result_is_deterministic_and_accepts_an_iterable(self) -> None:
        manifest, loaded = _synthetic_corpus()
        config = _config()

        first = build_chunk_set(manifest, iter(loaded), config)
        second = build_chunk_set(manifest, loaded, config)

        self.assertEqual(first, second)

    def test_missing_reordered_or_unapproved_sources_are_rejected(self) -> None:
        manifest, loaded = _synthetic_corpus()
        cases = (
            (loaded[:1], "exact manifest coverage"),
            ((loaded[1], loaded[0]), "manifest order"),
            ((loaded[0], loaded[0]), "exact approved source"),
        )

        for supplied, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    ChunkSetBuilderError,
                    message,
                ):
                    build_chunk_set(manifest, supplied, _config())

    def test_invalid_arguments_are_rejected(self) -> None:
        manifest, loaded = _synthetic_corpus()
        cases = (
            (object(), loaded, _config(), "manifest"),
            (manifest, None, _config(), "loaded_sources"),
            (manifest, loaded, object(), "config"),
        )

        for manifest_value, sources_value, config_value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    ChunkSetBuilderError,
                    message,
                ):
                    build_chunk_set(  # type: ignore[arg-type]
                        manifest_value,
                        sources_value,
                        config_value,
                    )

    def test_source_build_failure_identifies_document(self) -> None:
        manifest, loaded = _synthetic_corpus()
        first_source = manifest.sources[0]
        invalid_first = LoadedSource(
            source=first_source,
            blocks=(
                _block(
                    first_source,
                    "PageHeader",
                    order=0,
                    text="Furniture only.",
                ),
            ),
        )

        with self.assertRaisesRegex(
            ChunkSetBuilderError,
            first_source.document_id,
        ):
            build_chunk_set(
                manifest,
                (invalid_first, loaded[1]),
                _config(),
            )


if __name__ == "__main__":
    unittest.main()
