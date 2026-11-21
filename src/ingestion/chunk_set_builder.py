"""Construct one immutable chunk set from a validated loaded corpus.

This module coordinates source-level chunk construction in authoritative
manifest order. It performs no file loading, splitting, indexing, retrieval,
or model calls.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.ingestion.chunk_builder import (
    ChunkBuilderError,
    build_source_chunks,
)
from src.ingestion.chunk_models import (
    ChunkSchemaError,
    ChunkSet,
    ChunkingConfig,
    compute_chunk_set_id,
    compute_chunking_config_id,
    compute_corpus_identity,
)
from src.ingestion.models import (
    CorpusManifest,
    LoadedSource,
)


class ChunkSetBuilderError(ValueError):
    """Raised when a complete deterministic ChunkSet cannot be built."""


def _validated_loaded_sources(
    manifest: CorpusManifest,
    loaded_sources: Iterable[LoadedSource],
) -> tuple[LoadedSource, ...]:
    """Materialize and validate exact manifest coverage and order."""

    try:
        sources = tuple(loaded_sources)
    except TypeError as exc:
        raise ChunkSetBuilderError(
            "loaded_sources must be an iterable of LoadedSource objects."
        ) from exc

    if not all(isinstance(source, LoadedSource) for source in sources):
        invalid_position = next(
            index
            for index, source in enumerate(sources)
            if not isinstance(source, LoadedSource)
        )
        raise ChunkSetBuilderError(
            "loaded_sources contains an unsupported object at position "
            f"{invalid_position}."
        )

    if len(sources) != len(manifest.sources):
        raise ChunkSetBuilderError(
            "loaded_sources must provide exact manifest coverage: "
            f"expected={len(manifest.sources)}, actual={len(sources)}."
        )

    for position, (approved, loaded) in enumerate(
        zip(manifest.sources, sources)
    ):
        if loaded.source != approved:
            raise ChunkSetBuilderError(
                "loaded_sources must follow manifest order and contain the "
                "exact approved source entry: "
                f"position={position}, "
                f"expected={approved.document_id!r}, "
                f"actual={loaded.source.document_id!r}."
            )

    return sources


def build_chunk_set(
    manifest: CorpusManifest,
    loaded_sources: Iterable[LoadedSource],
    config: ChunkingConfig,
) -> ChunkSet:
    """Build the complete immutable ChunkSet in authoritative source order."""

    if not isinstance(manifest, CorpusManifest):
        raise ChunkSetBuilderError(
            "manifest must be a CorpusManifest."
        )
    if not isinstance(config, ChunkingConfig):
        raise ChunkSetBuilderError(
            "config must be a ChunkingConfig."
        )

    sources = _validated_loaded_sources(manifest, loaded_sources)
    corpus_identity = compute_corpus_identity(manifest)
    chunking_config_id = compute_chunking_config_id(config)
    chunk_set_id = compute_chunk_set_id(
        corpus_identity,
        chunking_config_id,
    )

    chunks = []
    for loaded_source in sources:
        try:
            chunks.extend(
                build_source_chunks(
                    loaded_source,
                    chunk_set_id=chunk_set_id,
                    config=config,
                )
            )
        except ChunkBuilderError as exc:
            raise ChunkSetBuilderError(
                "Source chunk construction failed: "
                f"document_id={loaded_source.source.document_id!r}."
            ) from exc

    try:
        return ChunkSet(
            schema_version="1.0",
            manifest=manifest,
            corpus_identity=corpus_identity,
            chunking_config=config,
            chunking_config_id=chunking_config_id,
            chunk_set_id=chunk_set_id,
            chunks=tuple(chunks),
        )
    except ChunkSchemaError as exc:
        raise ChunkSetBuilderError(
            "Constructed chunks violate the immutable ChunkSet contract."
        ) from exc


__all__ = [
    "ChunkSetBuilderError",
    "build_chunk_set",
]
