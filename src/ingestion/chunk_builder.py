"""Construct immutable chunks for one validated normalized source.

This module is the boundary between prepared chunk candidates and the
immutable chunk schema. It does not split content, construct a corpus-wide
ChunkSet, build indexes, perform retrieval, or call a generation model.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.chunk_models import (
    BlockSelection,
    Chunk,
    ChunkTableSegment,
    ChunkingConfig,
    compute_chunk_id,
)
from src.ingestion.chunker import (
    BlockRoute,
    ChunkerError,
    PreparedChunkCandidate,
    group_routed_blocks,
    prepare_chunk_candidates,
    route_blocks,
)
from src.ingestion.models import (
    HierarchyEntry,
    LegalLocatorPart,
    LoadedSource,
    PageIdentity,
    SourceAnnotation,
)


class ChunkBuilderError(ValueError):
    """Raised when immutable source chunks cannot be built safely."""


@dataclass(frozen=True, slots=True)
class _ChunkMaterial:
    """Identity-bearing values required to instantiate one Chunk."""

    text: str
    block_selections: tuple[BlockSelection, ...]
    hierarchies: tuple[tuple[HierarchyEntry, ...], ...]
    pages: tuple[PageIdentity, ...]
    legal_locators: tuple[tuple[LegalLocatorPart, ...], ...]
    table_segments: tuple[ChunkTableSegment, ...]
    annotations: tuple[SourceAnnotation, ...]


def _material_from_prepared(
    prepared: PreparedChunkCandidate,
) -> _ChunkMaterial:
    """Convert one prepared candidate into complete chunk schema values."""

    if not isinstance(prepared, PreparedChunkCandidate):
        raise ChunkBuilderError(
            "prepared must be a PreparedChunkCandidate."
        )

    candidate = prepared.candidate
    selections = tuple(
        BlockSelection(
            block_id=routed.block.block_id,
            source_block_id=routed.block.source_block_id,
            block_order=routed.block.order,
            start_char=None,
            end_char=None,
            table_row_start=None,
            table_row_end=None,
        )
        for routed in candidate.blocks
    )

    hierarchies = (
        (candidate.hierarchy,)
        if candidate.hierarchy
        else ()
    )

    pages: list[PageIdentity] = []
    legal_locators: list[tuple[LegalLocatorPart, ...]] = []
    table_segments: list[ChunkTableSegment] = []

    for routed in candidate.blocks:
        block = routed.block

        if block.page is not None and block.page not in pages:
            pages.append(block.page)

        if block.legal_locator:
            # Repeated paths are meaningful block-level occurrences and are
            # therefore preserved rather than silently deduplicated.
            legal_locators.append(block.legal_locator)

        if routed.route is BlockRoute.TABLE:
            if block.table is None:
                raise ChunkBuilderError(
                    "Table-routed block has no TableData: "
                    f"block_id={block.block_id!r}."
                )

            row_count = len(block.table.rows)
            if row_count == 0:
                raise ChunkBuilderError(
                    "Complete table segment requires at least one row: "
                    f"block_id={block.block_id!r}."
                )

            table_segments.append(
                ChunkTableSegment(
                    block_id=block.block_id,
                    table=block.table,
                    source_row_start=0,
                    source_row_end=row_count,
                    is_complete_table=True,
                )
            )
        elif block.table is not None:
            raise ChunkBuilderError(
                "Non-table route unexpectedly contains TableData: "
                f"block_id={block.block_id!r}, route={routed.route.value!r}."
            )

    return _ChunkMaterial(
        text=candidate.text,
        block_selections=selections,
        hierarchies=hierarchies,
        pages=tuple(pages),
        legal_locators=tuple(legal_locators),
        table_segments=tuple(table_segments),
        annotations=prepared.annotations,
    )


def build_source_chunks(
    loaded_source: LoadedSource,
    *,
    chunk_set_id: str,
    config: ChunkingConfig,
) -> tuple[Chunk, ...]:
    """Build deterministic, linked chunks for one LoadedSource."""

    if not isinstance(loaded_source, LoadedSource):
        raise ChunkBuilderError(
            "loaded_source must be a LoadedSource."
        )
    if not isinstance(chunk_set_id, str) or not chunk_set_id.strip():
        raise ChunkBuilderError(
            "chunk_set_id must be a non-empty string."
        )
    if not isinstance(config, ChunkingConfig):
        raise ChunkBuilderError(
            "config must be a ChunkingConfig."
        )

    try:
        grouping = group_routed_blocks(
            route_blocks(loaded_source.blocks),
            config,
        )
        prepared = prepare_chunk_candidates(grouping)
    except ChunkerError as exc:
        raise ChunkBuilderError(
            "Source chunk preparation failed: "
            f"document_id={loaded_source.source.document_id!r}."
        ) from exc

    if not prepared:
        raise ChunkBuilderError(
            "Loaded source produced no retrievable chunk candidates: "
            f"document_id={loaded_source.source.document_id!r}."
        )

    materials = tuple(
        _material_from_prepared(item)
        for item in prepared
    )
    chunk_ids = tuple(
        compute_chunk_id(
            chunk_set_id=chunk_set_id,
            source=loaded_source.source,
            text=material.text,
            block_selections=material.block_selections,
            hierarchies=material.hierarchies,
            pages=material.pages,
            legal_locators=material.legal_locators,
            table_segments=material.table_segments,
            annotations=material.annotations,
        )
        for material in materials
    )

    return tuple(
        Chunk(
            chunk_id=chunk_ids[index],
            chunk_set_id=chunk_set_id,
            source=loaded_source.source,
            order=index,
            text=material.text,
            block_selections=material.block_selections,
            hierarchies=material.hierarchies,
            pages=material.pages,
            legal_locators=material.legal_locators,
            table_segments=material.table_segments,
            annotations=material.annotations,
            previous_chunk_id=(
                chunk_ids[index - 1]
                if index > 0
                else None
            ),
            next_chunk_id=(
                chunk_ids[index + 1]
                if index + 1 < len(chunk_ids)
                else None
            ),
        )
        for index, material in enumerate(materials)
    )


__all__ = [
    "ChunkBuilderError",
    "build_source_chunks",
]
