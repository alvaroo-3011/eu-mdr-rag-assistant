"""Immutable schema models and identities for future source chunking.

This module defines the validated storage boundary only. It does not choose
chunk boundaries, split source blocks, build embeddings, or perform retrieval.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.ingestion.models import (
    CorpusManifest,
    HierarchyEntry,
    LegalLocatorPart,
    PageIdentity,
    SourceAnnotation,
    SourceManifestEntry,
    TableData,
)


class ChunkSchemaError(ValueError):
    """Raised when chunk schema data violates the approved contract."""


class SizeUnit(StrEnum):
    """Supported units for measuring chunk size."""

    UNICODE_CHARACTERS = "unicode_characters"


class OverlapPolicy(StrEnum):
    """Supported policies for overlap between neighboring chunks."""

    NONE = "none"


class TableSplitPolicy(StrEnum):
    """Supported policies for tables that exceed the hard size limit."""

    RETAIN_WHOLE_OR_ROW_GROUPS_WITH_REPEATED_HEADERS = (
        "retain_whole_or_row_groups_with_repeated_headers"
    )


class OversizedTextPolicy(StrEnum):
    """Supported explicit outcomes for oversized non-table blocks."""

    REJECT = "reject"
    SPLIT_AT_CHARACTER_BOUNDARY = "split_at_character_boundary"


def _require_non_empty_text(field_name: str, value: object) -> None:
    """Require a non-empty string without coercing its value."""

    if not isinstance(value, str) or not value.strip():
        raise ChunkSchemaError(
            f"{field_name} must be a non-empty string."
        )


def _require_optional_text(field_name: str, value: object) -> None:
    """Require None or a non-empty string."""

    if value is not None:
        _require_non_empty_text(field_name, value)


def _require_non_negative_integer(
    field_name: str,
    value: object,
) -> None:
    """Require a non-negative integer while rejecting booleans."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ChunkSchemaError(
            f"{field_name} must be a non-negative integer."
        )


def _require_positive_integer(field_name: str, value: object) -> None:
    """Require a positive integer while rejecting booleans."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ChunkSchemaError(
            f"{field_name} must be a positive integer."
        )


def _canonical_identifier(prefix: str, payload: dict[str, Any]) -> str:
    """Return a recognizable identifier over canonical explicit JSON."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{prefix}:sha256:{digest}"


def _source_payload(source: SourceManifestEntry) -> dict[str, Any]:
    """Serialize every approved source-manifest field explicitly."""

    if not isinstance(source, SourceManifestEntry):
        raise ChunkSchemaError(
            "source must be a SourceManifestEntry."
        )

    return {
        "document_id": source.document_id,
        "official_title": source.official_title,
        "short_title": source.short_title,
        "version": source.version,
        "regulatory_regimes": [
            regime.value
            for regime in source.regulatory_regimes
        ],
        "permitted_regimes": [
            regime.value
            for regime in source.permitted_regimes
        ],
        "source_type": source.source_type.value,
        "authority": source.authority,
        "authority_tier": source.authority_tier.value,
        "official_url": source.official_url,
        "local_path": source.local_path,
        "content_format": source.content_format.value,
        "sha256": source.sha256,
        "validation_status": source.validation_status.value,
        "index_status": source.index_status.value,
    }


def compute_corpus_identity(manifest: CorpusManifest) -> str:
    """Compute identity from the complete ordered manifest contract."""

    if not isinstance(manifest, CorpusManifest):
        raise ChunkSchemaError(
            "manifest must be a CorpusManifest."
        )

    payload = {
        "schema_version": manifest.schema_version,
        "corpus_id": manifest.corpus_id,
        "expected_source_count": manifest.expected_source_count,
        "sources": [
            _source_payload(source)
            for source in manifest.sources
        ],
    }
    return _canonical_identifier("corpus", payload)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Explicit versioned configuration used to construct a chunk set."""

    schema_version: str
    algorithm_version: str
    target_size: int
    hard_max_size: int
    size_unit: SizeUnit
    overlap_policy: OverlapPolicy
    table_split_policy: TableSplitPolicy
    oversized_text_policy: OversizedTextPolicy
    include_hierarchy_in_text: bool

    def __post_init__(self) -> None:
        """Validate configuration without selecting implicit policies."""

        _require_non_empty_text(
            "ChunkingConfig.schema_version",
            self.schema_version,
        )
        _require_non_empty_text(
            "ChunkingConfig.algorithm_version",
            self.algorithm_version,
        )

        if self.schema_version != "1.0":
            raise ChunkSchemaError(
                "Unsupported ChunkingConfig schema_version: "
                f"{self.schema_version!r}."
            )

        _require_positive_integer(
            "ChunkingConfig.target_size",
            self.target_size,
        )
        _require_positive_integer(
            "ChunkingConfig.hard_max_size",
            self.hard_max_size,
        )

        if self.hard_max_size < self.target_size:
            raise ChunkSchemaError(
                "ChunkingConfig.hard_max_size must be at least "
                "target_size."
            )

        enum_fields = (
            ("size_unit", self.size_unit, SizeUnit),
            ("overlap_policy", self.overlap_policy, OverlapPolicy),
            (
                "table_split_policy",
                self.table_split_policy,
                TableSplitPolicy,
            ),
            (
                "oversized_text_policy",
                self.oversized_text_policy,
                OversizedTextPolicy,
            ),
        )

        for field_name, value, expected_type in enum_fields:
            if not isinstance(value, expected_type):
                raise ChunkSchemaError(
                    f"ChunkingConfig.{field_name} must be a valid "
                    f"{expected_type.__name__}."
                )

        if not isinstance(self.include_hierarchy_in_text, bool):
            raise ChunkSchemaError(
                "ChunkingConfig.include_hierarchy_in_text must be bool."
            )

    @property
    def chunking_config_id(self) -> str:
        """Return the deterministic identity of every config field."""

        return compute_chunking_config_id(self)


def compute_chunking_config_id(config: ChunkingConfig) -> str:
    """Compute identity from all explicit chunking configuration fields."""

    if not isinstance(config, ChunkingConfig):
        raise ChunkSchemaError(
            "config must be a ChunkingConfig."
        )

    payload = {
        "schema_version": config.schema_version,
        "algorithm_version": config.algorithm_version,
        "target_size": config.target_size,
        "hard_max_size": config.hard_max_size,
        "size_unit": config.size_unit.value,
        "overlap_policy": config.overlap_policy.value,
        "table_split_policy": config.table_split_policy.value,
        "oversized_text_policy": config.oversized_text_policy.value,
        "include_hierarchy_in_text": (
            config.include_hierarchy_in_text
        ),
    }
    return _canonical_identifier("chunking-config", payload)


def compute_chunk_set_id(
    corpus_identity: str,
    chunking_config_id: str,
) -> str:
    """Compute a chunk-set identity from corpus and config identities."""

    _require_non_empty_text("corpus_identity", corpus_identity)
    _require_non_empty_text(
        "chunking_config_id",
        chunking_config_id,
    )
    return _canonical_identifier(
        "chunk-set",
        {
            "corpus_identity": corpus_identity,
            "chunking_config_id": chunking_config_id,
        },
    )


@dataclass(frozen=True, slots=True)
class BlockSelection:
    """One complete or half-open partial selection from a source block."""

    block_id: str
    source_block_id: str | None
    block_order: int
    start_char: int | None
    end_char: int | None
    table_row_start: int | None
    table_row_end: int | None

    def __post_init__(self) -> None:
        """Validate selection mode and half-open range boundaries."""

        _require_non_empty_text(
            "BlockSelection.block_id",
            self.block_id,
        )
        _require_optional_text(
            "BlockSelection.source_block_id",
            self.source_block_id,
        )
        _require_non_negative_integer(
            "BlockSelection.block_order",
            self.block_order,
        )

        char_range = self._validate_range(
            "character",
            self.start_char,
            self.end_char,
        )
        table_range = self._validate_range(
            "table-row",
            self.table_row_start,
            self.table_row_end,
        )

        if char_range and table_range:
            raise ChunkSchemaError(
                "BlockSelection character and table-row ranges are "
                "mutually exclusive."
            )

    @staticmethod
    def _validate_range(
        range_name: str,
        start: int | None,
        end: int | None,
    ) -> bool:
        """Validate one optional half-open range and return its presence."""

        if (start is None) != (end is None):
            raise ChunkSchemaError(
                f"BlockSelection {range_name} range endpoints must be "
                "provided together."
            )

        if start is None:
            return False

        _require_non_negative_integer(
            f"BlockSelection.{range_name}_start",
            start,
        )
        _require_non_negative_integer(
            f"BlockSelection.{range_name}_end",
            end,
        )
        if end <= start:
            raise ChunkSchemaError(
                f"BlockSelection {range_name} range must use non-empty "
                "half-open [start, end) bounds."
            )
        return True


@dataclass(frozen=True, slots=True)
class ChunkTableSegment:
    """Structured rows retained from one selected source table block."""

    block_id: str
    table: TableData
    source_row_start: int
    source_row_end: int
    is_complete_table: bool

    def __post_init__(self) -> None:
        """Validate the represented half-open source-row range."""

        _require_non_empty_text(
            "ChunkTableSegment.block_id",
            self.block_id,
        )
        if not isinstance(self.table, TableData):
            raise ChunkSchemaError(
                "ChunkTableSegment.table must be a TableData."
            )

        _require_non_negative_integer(
            "ChunkTableSegment.source_row_start",
            self.source_row_start,
        )
        _require_non_negative_integer(
            "ChunkTableSegment.source_row_end",
            self.source_row_end,
        )
        if self.source_row_end <= self.source_row_start:
            raise ChunkSchemaError(
                "ChunkTableSegment row range must use non-empty "
                "half-open [start, end) bounds."
            )

        represented_count = self.source_row_end - self.source_row_start
        if len(self.table.rows) != represented_count:
            raise ChunkSchemaError(
                "ChunkTableSegment stored row count must equal its "
                "represented source-row range length."
            )

        if not isinstance(self.is_complete_table, bool):
            raise ChunkSchemaError(
                "ChunkTableSegment.is_complete_table must be bool."
            )


def _selection_payload(selection: BlockSelection) -> dict[str, Any]:
    """Serialize a block selection explicitly."""

    return {
        "block_id": selection.block_id,
        "source_block_id": selection.source_block_id,
        "block_order": selection.block_order,
        "start_char": selection.start_char,
        "end_char": selection.end_char,
        "table_row_start": selection.table_row_start,
        "table_row_end": selection.table_row_end,
    }


def _hierarchy_payload(entry: HierarchyEntry) -> dict[str, Any]:
    """Serialize one hierarchy entry explicitly."""

    return {
        "level": entry.level,
        "kind": entry.kind,
        "block_id": entry.block_id,
        "title": entry.title,
    }


def _page_payload(page: PageIdentity) -> dict[str, Any]:
    """Serialize one page identity explicitly."""

    return {
        "source_index": page.source_index,
        "pdf_page_number": page.pdf_page_number,
    }


def _locator_payload(part: LegalLocatorPart) -> dict[str, Any]:
    """Serialize one legal-locator component explicitly."""

    return {
        "kind": part.kind,
        "value": part.value,
    }


def _table_payload(table: TableData) -> dict[str, Any]:
    """Serialize all structured table values explicitly."""

    return {
        "headers": list(table.headers),
        "rows": [list(row) for row in table.rows],
        "extraction_method": table.extraction_method,
        "source_page": table.source_page,
        "table_number": table.table_number,
        "pdfplumber_bbox": (
            list(table.pdfplumber_bbox)
            if table.pdfplumber_bbox is not None
            else None
        ),
        "header_detected": table.header_detected,
    }


def _table_segment_payload(
    segment: ChunkTableSegment,
) -> dict[str, Any]:
    """Serialize one chunk table segment explicitly."""

    return {
        "block_id": segment.block_id,
        "table": _table_payload(segment.table),
        "source_row_start": segment.source_row_start,
        "source_row_end": segment.source_row_end,
        "is_complete_table": segment.is_complete_table,
    }


def _annotation_payload(annotation: SourceAnnotation) -> dict[str, Any]:
    """Serialize one source annotation explicitly."""

    return {
        "kind": annotation.kind.value,
        "identifier": annotation.identifier,
        "target_id": annotation.target_id,
        "raw_html": annotation.raw_html,
        "text": annotation.text,
    }


def compute_chunk_id(
    *,
    chunk_set_id: str,
    source: SourceManifestEntry,
    text: str,
    block_selections: tuple[BlockSelection, ...],
    hierarchies: tuple[tuple[HierarchyEntry, ...], ...],
    pages: tuple[PageIdentity, ...],
    legal_locators: tuple[tuple[LegalLocatorPart, ...], ...],
    table_segments: tuple[ChunkTableSegment, ...],
    annotations: tuple[SourceAnnotation, ...],
) -> str:
    """Compute evidence identity, excluding order and neighbor links."""

    _require_non_empty_text("chunk_set_id", chunk_set_id)
    if not isinstance(source, SourceManifestEntry):
        raise ChunkSchemaError(
            "source must be a SourceManifestEntry."
        )

    payload = {
        "chunk_set_id": chunk_set_id,
        "source_identity": source.source_identity,
        "text": text,
        "block_selections": [
            _selection_payload(selection)
            for selection in block_selections
        ],
        "hierarchies": [
            [_hierarchy_payload(entry) for entry in path]
            for path in hierarchies
        ],
        "pages": [_page_payload(page) for page in pages],
        "legal_locators": [
            [_locator_payload(part) for part in path]
            for path in legal_locators
        ],
        "table_segments": [
            _table_segment_payload(segment)
            for segment in table_segments
        ],
        "annotations": [
            _annotation_payload(annotation)
            for annotation in annotations
        ],
    }
    return _canonical_identifier("chunk", payload)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One immutable, source-traceable chunk of normalized evidence."""

    chunk_id: str
    chunk_set_id: str
    source: SourceManifestEntry
    order: int
    text: str
    block_selections: tuple[BlockSelection, ...]
    hierarchies: tuple[tuple[HierarchyEntry, ...], ...]
    pages: tuple[PageIdentity, ...]
    legal_locators: tuple[tuple[LegalLocatorPart, ...], ...]
    table_segments: tuple[ChunkTableSegment, ...]
    annotations: tuple[SourceAnnotation, ...]
    previous_chunk_id: str | None
    next_chunk_id: str | None

    def __post_init__(self) -> None:
        """Validate intrinsic chunk structure and stored identity."""

        _require_non_empty_text("Chunk.chunk_id", self.chunk_id)
        _require_non_empty_text(
            "Chunk.chunk_set_id",
            self.chunk_set_id,
        )
        if not isinstance(self.source, SourceManifestEntry):
            raise ChunkSchemaError(
                "Chunk.source must be a SourceManifestEntry."
            )
        _require_non_negative_integer("Chunk.order", self.order)
        _require_non_empty_text("Chunk.text", self.text)

        self._validate_block_selections()
        self._validate_hierarchies()
        self._validate_pages()
        self._validate_legal_locators()
        self._validate_table_segments()
        self._validate_annotations()
        self._validate_neighbors()

        expected_chunk_id = compute_chunk_id(
            chunk_set_id=self.chunk_set_id,
            source=self.source,
            text=self.text,
            block_selections=self.block_selections,
            hierarchies=self.hierarchies,
            pages=self.pages,
            legal_locators=self.legal_locators,
            table_segments=self.table_segments,
            annotations=self.annotations,
        )
        if self.chunk_id != expected_chunk_id:
            raise ChunkSchemaError(
                "Chunk.chunk_id does not match the canonical chunk "
                "identity."
            )

        # Character offsets, source-row ranges, hierarchy paths, pages, and
        # legal locators cannot be proven against SourceBlock content here.
        # Those cross-checks belong to the future chunk builder, which will
        # have the corresponding LoadedSource available.

    def _validate_block_selections(self) -> None:
        """Validate ordered, unique selections for this exact source."""

        if (
            not isinstance(self.block_selections, tuple)
            or not self.block_selections
            or not all(
                isinstance(selection, BlockSelection)
                for selection in self.block_selections
            )
        ):
            raise ChunkSchemaError(
                "Chunk.block_selections must be a non-empty immutable "
                "tuple of BlockSelection objects."
            )

        orders = tuple(
            selection.block_order
            for selection in self.block_selections
        )
        if any(
            first >= second
            for first, second in zip(orders, orders[1:])
        ):
            raise ChunkSchemaError(
                "Chunk selected block orders must be strictly increasing."
            )

        block_ids = tuple(
            selection.block_id
            for selection in self.block_selections
        )
        if len(set(block_ids)) != len(block_ids):
            raise ChunkSchemaError(
                "A source block must not appear more than once in a chunk."
            )

        required_prefix = f"{self.source.source_identity}::"
        if any(
            not block_id.startswith(required_prefix)
            for block_id in block_ids
        ):
            raise ChunkSchemaError(
                "Every selected block_id must use the chunk source's "
                "exact source-identity prefix."
            )

    def _validate_hierarchies(self) -> None:
        """Validate immutable, distinct hierarchy paths."""

        if not isinstance(self.hierarchies, tuple):
            raise ChunkSchemaError(
                "Chunk.hierarchies must be an immutable tuple."
            )
        for path in self.hierarchies:
            if not isinstance(path, tuple) or not all(
                isinstance(entry, HierarchyEntry)
                for entry in path
            ):
                raise ChunkSchemaError(
                    "Every Chunk hierarchy path must be an immutable "
                    "tuple of HierarchyEntry objects."
                )

            hierarchy_levels = tuple(entry.level for entry in path)
            if any(
                outer_level >= inner_level
                for outer_level, inner_level in zip(
                    hierarchy_levels,
                    hierarchy_levels[1:],
                )
            ):
                raise ChunkSchemaError(
                    "Chunk hierarchy levels must be strictly increasing "
                    "from outermost to innermost."
                )

        if len(set(self.hierarchies)) != len(self.hierarchies):
            raise ChunkSchemaError(
                "Chunk hierarchy paths must be distinct."
            )

    def _validate_pages(self) -> None:
        """Validate unique pages in document order."""

        if not isinstance(self.pages, tuple) or not all(
            isinstance(page, PageIdentity)
            for page in self.pages
        ):
            raise ChunkSchemaError(
                "Chunk.pages must be an immutable tuple of PageIdentity "
                "objects."
            )

        page_indexes = tuple(page.source_index for page in self.pages)
        if len(set(page_indexes)) != len(page_indexes):
            raise ChunkSchemaError(
                "Chunk page identities must be distinct."
            )
        if page_indexes != tuple(sorted(page_indexes)):
            raise ChunkSchemaError(
                "Chunk page identities must follow document order."
            )

        if self.source.content_format.value == "eurlex_html" and self.pages:
            raise ChunkSchemaError(
                "EUR-Lex HTML chunks must not contain page identities."
            )

    def _validate_legal_locators(self) -> None:
        """Validate immutable legal-locator paths."""

        if not isinstance(self.legal_locators, tuple):
            raise ChunkSchemaError(
                "Chunk.legal_locators must be an immutable tuple."
            )
        for path in self.legal_locators:
            if not isinstance(path, tuple) or not all(
                isinstance(part, LegalLocatorPart)
                for part in path
            ):
                raise ChunkSchemaError(
                    "Every Chunk legal-locator path must be an immutable "
                    "tuple of LegalLocatorPart objects."
                )

    def _validate_table_segments(self) -> None:
        """Validate table segments against this chunk's selections."""

        if not isinstance(self.table_segments, tuple) or not all(
            isinstance(segment, ChunkTableSegment)
            for segment in self.table_segments
        ):
            raise ChunkSchemaError(
                "Chunk.table_segments must be an immutable tuple of "
                "ChunkTableSegment objects."
            )

        selections_by_id = {
            selection.block_id: selection
            for selection in self.block_selections
        }
        segment_ids = tuple(
            segment.block_id
            for segment in self.table_segments
        )
        if len(set(segment_ids)) != len(segment_ids):
            raise ChunkSchemaError(
                "A selected block may have at most one table segment."
            )

        for segment in self.table_segments:
            selection = selections_by_id.get(segment.block_id)
            if selection is None:
                raise ChunkSchemaError(
                    "Every table segment must correspond to a selected "
                    "block."
                )
            if selection.start_char is not None:
                raise ChunkSchemaError(
                    "A table segment cannot correspond to a character "
                    "range selection."
                )

            if selection.table_row_start is None:
                if not segment.is_complete_table:
                    raise ChunkSchemaError(
                        "A complete-block table selection requires a "
                        "complete table segment."
                    )
            else:
                if segment.is_complete_table:
                    raise ChunkSchemaError(
                        "A partial table-row selection cannot use a "
                        "complete table segment."
                    )
                if (
                    selection.table_row_start
                    != segment.source_row_start
                    or selection.table_row_end
                    != segment.source_row_end
                ):
                    raise ChunkSchemaError(
                        "A partial table selection range must agree with "
                        "its table segment."
                    )

        for selection in self.block_selections:
            if (
                selection.table_row_start is not None
                and selection.block_id not in segment_ids
            ):
                raise ChunkSchemaError(
                    "Every partial table-row selection requires a matching "
                    "table segment."
                )

    def _validate_annotations(self) -> None:
        """Validate immutable trusted source annotations."""

        if not isinstance(self.annotations, tuple) or not all(
            isinstance(annotation, SourceAnnotation)
            for annotation in self.annotations
        ):
            raise ChunkSchemaError(
                "Chunk.annotations must be an immutable tuple of "
                "SourceAnnotation objects."
            )

    def _validate_neighbors(self) -> None:
        """Validate locally expressible neighbor-ID constraints."""

        _require_optional_text(
            "Chunk.previous_chunk_id",
            self.previous_chunk_id,
        )
        _require_optional_text(
            "Chunk.next_chunk_id",
            self.next_chunk_id,
        )
        if self.previous_chunk_id == self.chunk_id:
            raise ChunkSchemaError(
                "Chunk.previous_chunk_id cannot equal chunk_id."
            )
        if self.next_chunk_id == self.chunk_id:
            raise ChunkSchemaError(
                "Chunk.next_chunk_id cannot equal chunk_id."
            )
        if (
            self.previous_chunk_id is not None
            and self.previous_chunk_id == self.next_chunk_id
        ):
            raise ChunkSchemaError(
                "Chunk previous and next IDs cannot equal each other."
            )


@dataclass(frozen=True, slots=True)
class ChunkSet:
    """A complete validated chunk collection for one corpus/config pair."""

    schema_version: str
    manifest: CorpusManifest
    corpus_identity: str
    chunking_config: ChunkingConfig
    chunking_config_id: str
    chunk_set_id: str
    chunks: tuple[Chunk, ...]

    def __post_init__(self) -> None:
        """Validate identities, coverage, ordering, and neighbor links."""

        _require_non_empty_text(
            "ChunkSet.schema_version",
            self.schema_version,
        )
        if self.schema_version != "1.0":
            raise ChunkSchemaError(
                "Unsupported ChunkSet schema_version: "
                f"{self.schema_version!r}."
            )
        if not isinstance(self.manifest, CorpusManifest):
            raise ChunkSchemaError(
                "ChunkSet.manifest must be a CorpusManifest."
            )
        if not isinstance(self.chunking_config, ChunkingConfig):
            raise ChunkSchemaError(
                "ChunkSet.chunking_config must be a ChunkingConfig."
            )

        expected_corpus_identity = compute_corpus_identity(self.manifest)
        if self.corpus_identity != expected_corpus_identity:
            raise ChunkSchemaError(
                "ChunkSet.corpus_identity does not match the manifest."
            )

        expected_config_id = compute_chunking_config_id(
            self.chunking_config
        )
        if self.chunking_config_id != expected_config_id:
            raise ChunkSchemaError(
                "ChunkSet.chunking_config_id does not match its config."
            )

        expected_chunk_set_id = compute_chunk_set_id(
            self.corpus_identity,
            self.chunking_config_id,
        )
        if self.chunk_set_id != expected_chunk_set_id:
            raise ChunkSchemaError(
                "ChunkSet.chunk_set_id does not match its identities."
            )

        if (
            not isinstance(self.chunks, tuple)
            or not self.chunks
            or not all(isinstance(chunk, Chunk) for chunk in self.chunks)
        ):
            raise ChunkSchemaError(
                "ChunkSet.chunks must be a non-empty immutable tuple of "
                "Chunk objects."
            )

        chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ChunkSchemaError(
                "ChunkSet chunk IDs must be unique."
            )
        if any(
            chunk.chunk_set_id != self.chunk_set_id
            for chunk in self.chunks
        ):
            raise ChunkSchemaError(
                "Every chunk must contain the ChunkSet identity."
            )

        self._validate_source_coverage_and_order()
        self._validate_neighbor_links()

    def _validate_source_coverage_and_order(self) -> None:
        """Require exact manifest coverage and deterministic global order."""

        manifest_sources = {
            source.document_id: source
            for source in self.manifest.sources
        }
        manifest_positions = {
            source.document_id: index
            for index, source in enumerate(self.manifest.sources)
        }

        for chunk in self.chunks:
            approved_source = manifest_sources.get(
                chunk.source.document_id
            )
            if approved_source is None:
                raise ChunkSchemaError(
                    "ChunkSet contains a chunk from an unlisted source: "
                    f"{chunk.source.document_id!r}."
                )
            if chunk.source != approved_source:
                raise ChunkSchemaError(
                    "A chunk source does not equal its complete manifest "
                    f"entry: {chunk.source.document_id!r}."
                )

        represented_sources = {
            chunk.source.document_id
            for chunk in self.chunks
        }
        missing_sources = tuple(
            source.document_id
            for source in self.manifest.sources
            if source.document_id not in represented_sources
        )
        if missing_sources:
            raise ChunkSchemaError(
                "Every manifest source must produce at least one chunk; "
                f"missing: {missing_sources}."
            )

        global_order = tuple(
            (
                manifest_positions[chunk.source.document_id],
                chunk.order,
            )
            for chunk in self.chunks
        )
        if global_order != tuple(sorted(global_order)):
            raise ChunkSchemaError(
                "Global chunk order must follow manifest source order and "
                "then source-local chunk order."
            )

        for source in self.manifest.sources:
            source_orders = tuple(
                chunk.order
                for chunk in self.chunks
                if chunk.source.document_id == source.document_id
            )
            if source_orders != tuple(range(len(source_orders))):
                raise ChunkSchemaError(
                    "Source-local chunk order must be contiguous from zero "
                    f"for {source.document_id!r}."
                )

    def _validate_neighbor_links(self) -> None:
        """Require exact symmetric links within each source boundary."""

        for source in self.manifest.sources:
            source_chunks = tuple(
                chunk
                for chunk in self.chunks
                if chunk.source.document_id == source.document_id
            )
            for index, chunk in enumerate(source_chunks):
                expected_previous = (
                    source_chunks[index - 1].chunk_id
                    if index > 0
                    else None
                )
                expected_next = (
                    source_chunks[index + 1].chunk_id
                    if index + 1 < len(source_chunks)
                    else None
                )
                if chunk.previous_chunk_id != expected_previous:
                    raise ChunkSchemaError(
                        "Chunk previous link is missing, asymmetric, or "
                        "crosses a source boundary."
                    )
                if chunk.next_chunk_id != expected_next:
                    raise ChunkSchemaError(
                        "Chunk next link is missing, asymmetric, or crosses "
                        "a source boundary."
                    )


__all__ = [
    "BlockSelection",
    "Chunk",
    "ChunkSchemaError",
    "ChunkSet",
    "ChunkTableSegment",
    "ChunkingConfig",
    "OverlapPolicy",
    "OversizedTextPolicy",
    "SizeUnit",
    "TableSplitPolicy",
    "compute_chunk_id",
    "compute_chunk_set_id",
    "compute_chunking_config_id",
    "compute_corpus_identity",
]
