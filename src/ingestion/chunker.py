"""
Deterministic routing, grouping, and annotation preparation for source blocks.

This module groups retrievable blocks, renders contextualized candidate text,
and attaches annotations for future chunk construction. It does not split
content, create chunks, or perform retrieval or indexing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from src.ingestion.models import (
    SourceAnnotation,
    SourceBlock,
)
from src.ingestion.chunk_models import ChunkingConfig


class ChunkerError(ValueError):
    """Raised when routing, grouping, or preparation cannot proceed safely."""


class BlockRoute(StrEnum):
    """Supported deterministic routes for normalized source blocks."""

    BODY = "body"
    TABLE = "table"
    CAPTION = "caption"
    STRUCTURAL_CONTEXT = "structural_context"
    ANNOTATION_ONLY = "annotation_only"
    EXCLUDED = "excluded"


# Every block type observed in the frozen corpus is listed explicitly.
# Unknown values must fail closed instead of silently becoming evidence.
BLOCK_TYPE_ROUTES: Mapping[str, BlockRoute] = MappingProxyType(
    {
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
)


_RENDERABLE_ROUTES = frozenset(
    {
        BlockRoute.BODY,
        BlockRoute.TABLE,
        BlockRoute.CAPTION,
    }
)


@dataclass(frozen=True, slots=True)
class RoutedBlock:
    """
    One source block with its deterministic route and rendered context.

    The complete original SourceBlock remains available. This preserves
    tables, annotations, source order, relationships, hierarchy, legal
    locators, pages, raw HTML, and the exact original searchable text.
    """

    block: SourceBlock
    route: BlockRoute
    context_text: str

    @property
    def body_text(self) -> str:
        """Return the exact source body without normalization."""

        return self.block.searchable_text

    @property
    def annotations(self) -> tuple[SourceAnnotation, ...]:
        """Expose preserved annotations for the future chunk builder."""

        return self.block.annotations

    @property
    def rendered_text(self) -> str | None:
        """
        Return contextualized retrievable text when appropriate.

        Context is added before the body, but the original body string is
        not stripped, rewritten, or otherwise modified. Empty body blocks
        remain routed records but do not produce context-only evidence.
        """

        if self.route not in _RENDERABLE_ROUTES:
            return None

        if not self.body_text.strip():
            return None

        if not self.context_text:
            return self.body_text

        return f"{self.context_text}\n\n{self.body_text}"


def classify_block_type(block_type: str) -> BlockRoute:
    """Return the explicit route for one known block type."""

    if not isinstance(block_type, str) or not block_type.strip():
        raise ChunkerError(
            "block_type must be a non-empty string."
        )

    try:
        return BLOCK_TYPE_ROUTES[block_type]
    except KeyError as exc:
        raise ChunkerError(
            f"Unknown block type: {block_type!r}."
        ) from exc


def render_context(block: SourceBlock) -> str:
    """
    Render hierarchy and legal-locator context deterministically.

    Hierarchy order and legal-locator order are preserved exactly as
    provided by the validated source model.
    """

    if not isinstance(block, SourceBlock):
        raise ChunkerError(
            "render_context requires a SourceBlock."
        )

    context_lines: list[str] = []

    if block.hierarchy:
        hierarchy_parts = tuple(
            (
                f"{entry.kind}: {entry.title}"
                if entry.title
                else entry.kind
            )
            for entry in block.hierarchy
        )
        context_lines.append(
            f"Hierarchy: {' > '.join(hierarchy_parts)}"
        )

    if block.legal_locator:
        locator_parts = tuple(
            f"{part.kind}: {part.value}"
            for part in block.legal_locator
        )
        context_lines.append(
            f"Legal locator: {' > '.join(locator_parts)}"
        )

    return "\n".join(context_lines)


def route_block(block: SourceBlock) -> RoutedBlock:
    """Classify and prepare one normalized source block."""

    if not isinstance(block, SourceBlock):
        raise ChunkerError(
            "route_block requires a SourceBlock."
        )

    return RoutedBlock(
        block=block,
        route=classify_block_type(block.block_type),
        context_text=render_context(block),
    )


def route_blocks(
    blocks: Iterable[SourceBlock],
) -> tuple[RoutedBlock, ...]:
    """
    Route blocks in supplied source order without sorting or grouping.

    LoadedSource already validates its tuple order. This function keeps
    that order unchanged and also rejects unsupported objects explicitly.
    """

    try:
        iterator = iter(blocks)
    except TypeError as exc:
        raise ChunkerError(
            "blocks must be an iterable of SourceBlock objects."
        ) from exc

    routed: list[RoutedBlock] = []

    for position, block in enumerate(iterator):
        if not isinstance(block, SourceBlock):
            raise ChunkerError(
                "blocks contains an unsupported object at position "
                f"{position}."
            )

        routed.append(route_block(block))

    return tuple(routed)

def _render_candidate_text(
    blocks: tuple[RoutedBlock, ...],
    *,
    include_hierarchy: bool,
) -> str:
    """
    Render one grouped candidate deterministically.

    Hierarchy context appears once. Each block retains its own legal
    locator so locator changes do not require a chunk boundary.
    """

    parts: list[str] = []

    if include_hierarchy and blocks[0].block.hierarchy:
        hierarchy_parts = tuple(
            (
                f"{entry.kind}: {entry.title}"
                if entry.title
                else entry.kind
            )
            for entry in blocks[0].block.hierarchy
        )
        parts.append(
            f"Hierarchy: {' > '.join(hierarchy_parts)}"
        )

    for routed in blocks:
        if routed.block.legal_locator:
            locator_parts = tuple(
                f"{part.kind}: {part.value}"
                for part in routed.block.legal_locator
            )
            parts.append(
                f"Legal locator: {' > '.join(locator_parts)}"
            )

        parts.append(routed.body_text)

    return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class ChunkCandidate:
    """
    Consecutive retrievable blocks prepared for future Chunk creation.

    Blocks remain complete and retain their original source metadata.
    """

    blocks: tuple[RoutedBlock, ...]
    include_hierarchy_in_text: bool

    def __post_init__(self) -> None:
        """Validate candidate membership and structural consistency."""

        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise ChunkerError(
                "ChunkCandidate.blocks must be a non-empty tuple."
            )

        if not isinstance(self.include_hierarchy_in_text, bool):
            raise ChunkerError(
                "include_hierarchy_in_text must be bool."
            )

        if not all(
            isinstance(routed, RoutedBlock)
            for routed in self.blocks
        ):
            raise ChunkerError(
                "ChunkCandidate.blocks contains an unsupported object."
            )

        if any(
            routed.route not in _RENDERABLE_ROUTES
            or not routed.body_text.strip()
            for routed in self.blocks
        ):
            raise ChunkerError(
                "ChunkCandidate may contain only non-empty retrievable "
                "blocks."
            )

        expected_hierarchy = self.blocks[0].block.hierarchy

        if any(
            routed.block.hierarchy != expected_hierarchy
            for routed in self.blocks[1:]
        ):
            raise ChunkerError(
                "Every block in a ChunkCandidate must share the same "
                "hierarchy path."
            )

    @property
    def hierarchy(self):
        """Return the shared hierarchy path."""

        return self.blocks[0].block.hierarchy

    @property
    def text(self) -> str:
        """Return deterministic candidate text."""

        return _render_candidate_text(
            self.blocks,
            include_hierarchy=self.include_hierarchy_in_text,
        )

    @property
    def character_count(self) -> int:
        """Return candidate size in Unicode characters."""

        return len(self.text)

    @property
    def annotations(self) -> tuple[SourceAnnotation, ...]:
        """Return annotations already attached to candidate blocks."""

        return tuple(
            annotation
            for routed in self.blocks
            for annotation in routed.annotations
        )


@dataclass(frozen=True, slots=True)
class CandidateGroupingResult:
    """
    Grouped candidates plus records awaiting annotation attachment.

    Annotation-only blocks are retained even when they contain no resolved
    annotation, because their document position may be needed later.
    """

    candidates: tuple[ChunkCandidate, ...]
    annotation_records: tuple[RoutedBlock, ...]

    def __post_init__(self) -> None:
        """Validate immutable grouping output."""

        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, ChunkCandidate)
            for candidate in self.candidates
        ):
            raise ChunkerError(
                "candidates must be a tuple of ChunkCandidate objects."
            )

        if not isinstance(self.annotation_records, tuple) or not all(
            isinstance(record, RoutedBlock)
            for record in self.annotation_records
        ):
            raise ChunkerError(
                "annotation_records must be a tuple of RoutedBlock "
                "objects."
            )


@dataclass(frozen=True, slots=True)
class PreparedChunkCandidate:
    """One unchanged candidate with its resolved, ordered annotations."""

    candidate: ChunkCandidate
    annotations: tuple[SourceAnnotation, ...]

    def __post_init__(self) -> None:
        """Validate immutable preparation output."""

        if not isinstance(self.candidate, ChunkCandidate):
            raise ChunkerError(
                "candidate must be a ChunkCandidate."
            )

        if not isinstance(self.annotations, tuple) or not all(
            isinstance(annotation, SourceAnnotation)
            for annotation in self.annotations
        ):
            raise ChunkerError(
                "annotations must be a tuple of SourceAnnotation objects."
            )


def prepare_chunk_candidates(
    result: CandidateGroupingResult,
) -> tuple[PreparedChunkCandidate, ...]:
    """
    Attach record annotations without changing candidate order or text.

    Merge candidate-block and attached record annotations in source-block
    order, preserving each block's annotation tuple order and every occurrence,
    including equal annotations.
    Explicit targets must be canonical IDs of retrievable candidate blocks;
    targetless records attach to the first candidate starting after them.
    """

    if not isinstance(result, CandidateGroupingResult):
        raise ChunkerError(
            "result must be a CandidateGroupingResult."
        )

    annotations: list[list[tuple[int, SourceAnnotation]]] = []
    candidate_by_block_id: dict[str, int] = {}

    for index, candidate in enumerate(result.candidates):
        annotations.append([
            (routed.block.order, annotation)
            for routed in candidate.blocks
            for annotation in routed.annotations
        ])
        for routed in candidate.blocks:
            candidate_by_block_id.setdefault(routed.block.block_id, index)

    for record in result.annotation_records:
        for annotation in record.annotations:
            if annotation.target_id is not None:
                index = candidate_by_block_id.get(annotation.target_id)
                if index is None:
                    raise ChunkerError(
                        "Annotation target is not retrievable: "
                        f"target_id={annotation.target_id!r}, "
                        f"record_id={record.block.block_id!r}."
                    )
            else:
                index = next(
                    (
                        position
                        for position, candidate in enumerate(result.candidates)
                        if candidate.blocks[0].block.order > record.block.order
                    ),
                    None,
                )
                if index is None:
                    raise ChunkerError(
                        "Targetless annotation has no following candidate: "
                        f"record_id={record.block.block_id!r}."
                    )

            annotations[index].append((record.block.order, annotation))

    return tuple(
        PreparedChunkCandidate(
            candidate,
            tuple(
                annotation
                for _, annotation in sorted(
                    annotations[index], key=lambda occurrence: occurrence[0],
                )
            ),
        )
        for index, candidate in enumerate(result.candidates)
    )


def group_routed_blocks(
    routed_blocks: Iterable[RoutedBlock],
    config: ChunkingConfig,
) -> CandidateGroupingResult:
    """
    Group consecutive retrievable blocks without splitting them.

    A hierarchy change forces a boundary. Page and legal-locator changes
    do not. Excluded and structural blocks do not become evidence.
    Annotation-only or otherwise non-retrievable annotated records are
    returned separately for future deterministic attachment.
    """

    if not isinstance(config, ChunkingConfig):
        raise ChunkerError(
            "config must be a ChunkingConfig."
        )

    try:
        iterator = iter(routed_blocks)
    except TypeError as exc:
        raise ChunkerError(
            "routed_blocks must be an iterable of RoutedBlock objects."
        ) from exc

    candidates: list[ChunkCandidate] = []
    annotation_records: list[RoutedBlock] = []
    current_blocks: list[RoutedBlock] = []

    def flush_current() -> None:
        """Store the current candidate and clear the working group."""

        if not current_blocks:
            return

        candidates.append(
            ChunkCandidate(
                blocks=tuple(current_blocks),
                include_hierarchy_in_text=(
                    config.include_hierarchy_in_text
                ),
            )
        )
        current_blocks.clear()

    for position, routed in enumerate(iterator):
        if not isinstance(routed, RoutedBlock):
            raise ChunkerError(
                "routed_blocks contains an unsupported object at "
                f"position {position}."
            )

        if routed.route is BlockRoute.ANNOTATION_ONLY:
            flush_current()
            annotation_records.append(routed)
            continue

        if routed.rendered_text is None:
            if routed.annotations:
                annotation_records.append(routed)
            continue

        single_candidate = ChunkCandidate(
            blocks=(routed,),
            include_hierarchy_in_text=(
                config.include_hierarchy_in_text
            ),
        )

        if single_candidate.character_count > config.hard_max_size:
            raise ChunkerError(
                "Rendered source block exceeds hard_max_size and "
                "requires splitting outside this checkpoint: "
                f"block_id={routed.block.block_id!r}, "
                f"size={single_candidate.character_count}, "
                f"hard_max_size={config.hard_max_size}."
            )

        if not current_blocks:
            current_blocks.append(routed)
            continue

        if (
            routed.block.hierarchy
            != current_blocks[0].block.hierarchy
        ):
            flush_current()
            current_blocks.append(routed)
            continue

        proposed = ChunkCandidate(
            blocks=tuple(current_blocks) + (routed,),
            include_hierarchy_in_text=(
                config.include_hierarchy_in_text
            ),
        )

        if proposed.character_count <= config.target_size:
            current_blocks.append(routed)
        else:
            flush_current()
            current_blocks.append(routed)

    flush_current()

    return CandidateGroupingResult(
        candidates=tuple(candidates),
        annotation_records=tuple(annotation_records),
    )

__all__ = [
    "BLOCK_TYPE_ROUTES",
    "BlockRoute",
    "ChunkerError",
    "RoutedBlock",
    "classify_block_type",
    "render_context",
    "route_block",
    "route_blocks",
    "CandidateGroupingResult",
    "ChunkCandidate",
    "PreparedChunkCandidate",
    "group_routed_blocks",
    "prepare_chunk_candidates",
]
