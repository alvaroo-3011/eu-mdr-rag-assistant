"""
Typed models for the authoritative manifest and normalized source records.

Manifest models describe trusted corpus identity and provenance. Source
models describe the normalized structural records produced by approved
loaders. This module does not read manifests or source files.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import urlparse


# Stable document IDs use lowercase words separated by hyphens.
_DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Source checksums are stored as lowercase SHA-256 hexadecimal strings.
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestSchemaError(ValueError):
    """Raised when manifest data violates the approved schema."""


class RegulatoryRegime(StrEnum):
    """Regulatory regimes that a source may cover."""

    MDR = "MDR"
    IVDR = "IVDR"


class SourceType(StrEnum):
    """Supported source categories in the frozen corpus."""

    REGULATION = "regulation"
    MDCG_GUIDANCE = "mdcg_guidance"
    MANUAL = "manual"
    Q_AND_A = "q_and_a"


class AuthorityTier(StrEnum):
    """
    Regulatory authority categories.

    These categories express source authority. They are not retrieval
    weights and do not choose any ranking configuration.
    """

    PRIMARY_LEGISLATION = "primary_legislation"
    MDCG_GUIDANCE = "mdcg_guidance"
    COMMISSION_SUPPORT = "commission_support"
    WORKING_GROUP_SUPPORT = "working_group_support"


class ContentFormat(StrEnum):
    """Supported frozen-source formats."""

    MARKER_JSON = "marker_json"
    EURLEX_HTML = "eurlex_html"


class ValidationStatus(StrEnum):
    """
    Allowed source validation states.

    Only validated sources are currently accepted. Additional states are
    intentionally not defined because the application must fail closed.
    """

    VALIDATED = "validated"


class IndexStatus(StrEnum):
    """
    Current index state.

    Index construction is outside this checkpoint, so only not_built is
    currently valid.
    """

    NOT_BUILT = "not_built"


def _require_non_empty(field_name: str, value: str) -> None:
    """Reject missing, non-string, or whitespace-only text values."""

    if not isinstance(value, str) or not value.strip():
        raise ManifestSchemaError(
            f"{field_name} must be a non-empty string."
        )


def _find_duplicates(values: tuple[str, ...]) -> list[str]:
    """Return sorted values that appear more than once."""

    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


@dataclass(frozen=True, slots=True)
class SourceManifestEntry:
    """
    Trusted identity and provenance metadata for one frozen source.

    Lists from JSON are converted to tuples by manifest.py so that loaded
    manifest objects are immutable.
    """

    document_id: str
    official_title: str
    short_title: str
    version: str
    regulatory_regimes: tuple[RegulatoryRegime, ...]
    permitted_regimes: tuple[RegulatoryRegime, ...]
    source_type: SourceType
    authority: str
    authority_tier: AuthorityTier
    official_url: str
    local_path: str
    content_format: ContentFormat
    sha256: str
    validation_status: ValidationStatus
    index_status: IndexStatus

    def __post_init__(self) -> None:
        """Validate intrinsic source metadata immediately."""

        for field_name in (
            "document_id",
            "official_title",
            "short_title",
            "version",
            "authority",
            "official_url",
            "local_path",
            "sha256",
        ):
            _require_non_empty(field_name, getattr(self, field_name))

        if not _DOCUMENT_ID_PATTERN.fullmatch(self.document_id):
            raise ManifestSchemaError(
                "document_id must contain lowercase letters or numbers "
                "separated by single hyphens: "
                f"{self.document_id!r}"
            )

        self._validate_regimes()
        self._validate_enum_fields()
        self._validate_url()
        self._validate_local_path()
        self._validate_checksum()
        self._validate_format_relationship()

    def _validate_regimes(self) -> None:
        """Ensure the permitted regimes are declared by the source."""

        if not isinstance(self.regulatory_regimes, tuple):
            raise ManifestSchemaError(
                "regulatory_regimes must be an immutable tuple."
            )

        if not isinstance(self.permitted_regimes, tuple):
            raise ManifestSchemaError(
                "permitted_regimes must be an immutable tuple."
            )

        if not self.regulatory_regimes:
            raise ManifestSchemaError(
                "regulatory_regimes must contain at least one regime."
            )

        if not self.permitted_regimes:
            raise ManifestSchemaError(
                "permitted_regimes must contain at least one regime."
            )

        if not all(
            isinstance(regime, RegulatoryRegime)
            for regime in self.regulatory_regimes
        ):
            raise ManifestSchemaError(
                "regulatory_regimes contains an unsupported value."
            )

        if not all(
            isinstance(regime, RegulatoryRegime)
            for regime in self.permitted_regimes
        ):
            raise ManifestSchemaError(
                "permitted_regimes contains an unsupported value."
            )

        if len(set(self.regulatory_regimes)) != len(
            self.regulatory_regimes
        ):
            raise ManifestSchemaError(
                "regulatory_regimes must not contain duplicates."
            )

        if len(set(self.permitted_regimes)) != len(
            self.permitted_regimes
        ):
            raise ManifestSchemaError(
                "permitted_regimes must not contain duplicates."
            )

        undeclared_regimes = set(self.permitted_regimes) - set(
            self.regulatory_regimes
        )

        if undeclared_regimes:
            values = sorted(regime.value for regime in undeclared_regimes)
            raise ManifestSchemaError(
                "permitted_regimes contains regimes not covered by the "
                f"source: {values}"
            )

    def _validate_enum_fields(self) -> None:
        """Reject strings or unsupported values where enums are required."""

        enum_fields = (
            ("source_type", self.source_type, SourceType),
            ("authority_tier", self.authority_tier, AuthorityTier),
            ("content_format", self.content_format, ContentFormat),
            (
                "validation_status",
                self.validation_status,
                ValidationStatus,
            ),
            ("index_status", self.index_status, IndexStatus),
        )

        for field_name, value, expected_type in enum_fields:
            if not isinstance(value, expected_type):
                raise ManifestSchemaError(
                    f"{field_name} must be a valid "
                    f"{expected_type.__name__} value."
                )

    def _validate_url(self) -> None:
        """Require a complete HTTPS official-source URL."""

        parsed_url = urlparse(self.official_url)

        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ManifestSchemaError(
                "official_url must be a complete HTTPS URL: "
                f"{self.official_url!r}"
            )

    def _validate_local_path(self) -> None:
        """
        Require a safe repository-relative path under data/rag_ready.

        Forward slashes are required in the manifest even on Windows. Python
        will convert the path appropriately when accessing the filesystem.
        """

        if "\\" in self.local_path:
            raise ManifestSchemaError(
                "local_path must use forward slashes, not backslashes."
            )

        path = PurePosixPath(self.local_path)

        if path.is_absolute() or ".." in path.parts:
            raise ManifestSchemaError(
                "local_path must be a safe repository-relative path."
            )

        if path.parts[:2] != ("data", "rag_ready"):
            raise ManifestSchemaError(
                "local_path must point inside data/rag_ready."
            )

    def _validate_checksum(self) -> None:
        """Require an exact lowercase SHA-256 value."""

        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ManifestSchemaError(
                "sha256 must contain exactly 64 lowercase hexadecimal "
                f"characters: {self.sha256!r}"
            )

    def _validate_format_relationship(self) -> None:
        """Ensure each content format has the expected file extension."""

        suffix = PurePosixPath(self.local_path).suffix.lower()

        if (
            self.content_format is ContentFormat.MARKER_JSON
            and suffix != ".json"
        ):
            raise ManifestSchemaError(
                "marker_json sources must use a .json local_path."
            )

        if (
            self.content_format is ContentFormat.EURLEX_HTML
            and suffix not in {".html", ".htm"}
        ):
            raise ManifestSchemaError(
                "eurlex_html sources must use an .html or .htm local_path."
            )

        if (
            self.source_type is SourceType.REGULATION
            and self.content_format is not ContentFormat.EURLEX_HTML
        ):
            raise ManifestSchemaError(
                "The regulation source must use eurlex_html."
            )

        if (
            self.source_type is not SourceType.REGULATION
            and self.content_format is not ContentFormat.MARKER_JSON
        ):
            raise ManifestSchemaError(
                "Non-regulation sources must use marker_json."
            )

    @property
    def source_identity(self) -> str:
        """
        Return the exact immutable source-version identity.

        The permanent document ID identifies the document lineage. The full
        checksum distinguishes this exact frozen version without relying on
        filenames or extraction-generated IDs.
        """

        return f"{self.document_id}@sha256:{self.sha256}"


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Immutable representation of the authoritative corpus manifest."""

    schema_version: str
    corpus_id: str
    expected_source_count: int
    sources: tuple[SourceManifestEntry, ...]

    def __post_init__(self) -> None:
        """Validate corpus-level structure and uniqueness."""

        _require_non_empty("schema_version", self.schema_version)
        _require_non_empty("corpus_id", self.corpus_id)

        if self.schema_version != "1.0":
            raise ManifestSchemaError(
                "Unsupported manifest schema_version: "
                f"{self.schema_version!r}"
            )

        if not _DOCUMENT_ID_PATTERN.fullmatch(self.corpus_id):
            raise ManifestSchemaError(
                "corpus_id must contain lowercase letters or numbers "
                "separated by single hyphens."
            )

        if (
            not isinstance(self.expected_source_count, int)
            or isinstance(self.expected_source_count, bool)
            or self.expected_source_count <= 0
        ):
            raise ManifestSchemaError(
                "expected_source_count must be a positive integer."
            )

        if not isinstance(self.sources, tuple):
            raise ManifestSchemaError(
                "sources must be an immutable tuple."
            )

        if len(self.sources) != self.expected_source_count:
            raise ManifestSchemaError(
                "Source count does not match expected_source_count: "
                f"expected {self.expected_source_count}, "
                f"found {len(self.sources)}."
            )

        if not all(
            isinstance(source, SourceManifestEntry)
            for source in self.sources
        ):
            raise ManifestSchemaError(
                "sources contains an invalid source entry."
            )

        self._reject_duplicate_values(
            "document IDs",
            tuple(source.document_id for source in self.sources),
        )
        self._reject_duplicate_values(
            "local paths",
            tuple(source.local_path for source in self.sources),
        )
        self._reject_duplicate_values(
            "checksums",
            tuple(source.sha256 for source in self.sources),
        )

    @staticmethod
    def _reject_duplicate_values(
        value_name: str,
        values: tuple[str, ...],
    ) -> None:
        """Reject duplicate source identifiers, paths, or checksums."""

        duplicates = _find_duplicates(values)

        if duplicates:
            raise ManifestSchemaError(
                f"Duplicate {value_name} found: {duplicates}"
            )

    def get_source(self, document_id: str) -> SourceManifestEntry:
        """Return one source by its stable document ID."""

        for source in self.sources:
            if source.document_id == document_id:
                return source

        raise KeyError(f"Unknown document_id: {document_id!r}")

class SourceSchemaError(ValueError):
    """Raised when normalized source records violate the approved schema."""


class AnnotationKind(StrEnum):
    """Supported preserved annotations within normalized source blocks."""

    AMENDMENT = "amendment"
    FOOTNOTE_REFERENCE = "footnote_reference"
    FOOTNOTE_DEFINITION = "footnote_definition"


def _require_source_text(field_name: str, value: object) -> None:
    """Require a non-empty string in a normalized source record."""

    if not isinstance(value, str) or not value.strip():
        raise SourceSchemaError(
            f"{field_name} must be a non-empty string."
        )


def _validate_optional_source_text(
    field_name: str,
    value: object,
) -> None:
    """Validate an optional string, rejecting empty strings."""

    if value is None:
        return

    _require_source_text(field_name, value)


def _validate_non_negative_integer(
    field_name: str,
    value: object,
) -> None:
    """Require an integer greater than or equal to zero."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise SourceSchemaError(
            f"{field_name} must be a non-negative integer."
        )


@dataclass(frozen=True, slots=True)
class PageIdentity:
    """
    Guidance-page identity.

    source_index follows the zero-based Marker page index. The
    human-facing PDF page number is always one greater.
    """

    source_index: int
    pdf_page_number: int

    def __post_init__(self) -> None:
        """Validate the relationship between source and displayed pages."""

        _validate_non_negative_integer(
            "source_index",
            self.source_index,
        )

        if (
            not isinstance(self.pdf_page_number, int)
            or isinstance(self.pdf_page_number, bool)
            or self.pdf_page_number < 1
        ):
            raise SourceSchemaError(
                "pdf_page_number must be a positive integer."
            )

        expected_page_number = self.source_index + 1

        if self.pdf_page_number != expected_page_number:
            raise SourceSchemaError(
                "pdf_page_number must equal source_index + 1: "
                f"expected {expected_page_number}, "
                f"received {self.pdf_page_number}."
            )


@dataclass(frozen=True, slots=True)
class HierarchyEntry:
    """
    One resolved heading in a block's semantic hierarchy.

    Hierarchy tuples are ordered from the outermost heading to the
    innermost heading.
    The title is empty when the source heading has no extractable text.
    """

    level: int
    kind: str
    block_id: str
    title: str

    def __post_init__(self) -> None:
        """Validate a resolved hierarchy entry."""

        if (
            not isinstance(self.level, int)
            or isinstance(self.level, bool)
            or self.level < 1
        ):
            raise SourceSchemaError(
                "HierarchyEntry.level must be a positive integer."
            )

        _require_source_text("HierarchyEntry.kind", self.kind)
        _require_source_text(
            "HierarchyEntry.block_id",
            self.block_id,
        )
        # Some valid Marker SectionHeader blocks have no extracted text.
        # Preserve that absence instead of inventing a heading title.
        if not isinstance(self.title, str):
            raise SourceSchemaError(
                "HierarchyEntry.title must be a string."
            )


@dataclass(frozen=True, slots=True)
class LegalLocatorPart:
    """
    One component of an ordered legal locator.

    Examples include an article, paragraph, point, annex, or annex
    section. The loader controls which locator kinds it emits.
    """

    kind: str
    value: str

    def __post_init__(self) -> None:
        """Reject incomplete legal-locator components."""

        _require_source_text("LegalLocatorPart.kind", self.kind)
        _require_source_text("LegalLocatorPart.value", self.value)


@dataclass(frozen=True, slots=True)
class TableData:
    """
    Structured representation of a genuine source table.

    Raw table HTML remains available through SourceBlock.raw_html.
    Fields specific to pdfplumber are optional so that the same model
    can represent genuine EUR-Lex HTML tables.
    """

    headers: tuple[str | None, ...]
    rows: tuple[tuple[str | None, ...], ...]
    extraction_method: str
    source_page: int | None
    table_number: int | None
    pdfplumber_bbox: tuple[int | float, ...] | None
    header_detected: bool | None

    def __post_init__(self) -> None:
        """Validate table structure without changing cell contents."""

        if not isinstance(self.headers, tuple):
            raise SourceSchemaError(
                "TableData.headers must be an immutable tuple."
            )

        if not all(
            cell is None or isinstance(cell, str)
            for cell in self.headers
        ):
            raise SourceSchemaError(
                "TableData.headers may contain only strings or None."
            )

        if not isinstance(self.rows, tuple):
            raise SourceSchemaError(
                "TableData.rows must be an immutable tuple."
            )

        for row in self.rows:
            if not isinstance(row, tuple):
                raise SourceSchemaError(
                    "Every TableData row must be an immutable tuple."
                )

            if not all(
                cell is None or isinstance(cell, str)
                for cell in row
            ):
                raise SourceSchemaError(
                    "TableData rows may contain only strings or None."
                )

        _require_source_text(
            "TableData.extraction_method",
            self.extraction_method,
        )

        if self.source_page is not None:
            _validate_non_negative_integer(
                "TableData.source_page",
                self.source_page,
            )

        if self.table_number is not None:
            if (
                not isinstance(self.table_number, int)
                or isinstance(self.table_number, bool)
                or self.table_number < 1
            ):
                raise SourceSchemaError(
                    "TableData.table_number must be a positive "
                    "integer or None."
                )

        if self.pdfplumber_bbox is not None:
            if (
                not isinstance(self.pdfplumber_bbox, tuple)
                or len(self.pdfplumber_bbox) != 4
            ):
                raise SourceSchemaError(
                    "TableData.pdfplumber_bbox must contain exactly "
                    "four coordinates."
                )

            if not all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                for coordinate in self.pdfplumber_bbox
            ):
                raise SourceSchemaError(
                    "TableData.pdfplumber_bbox coordinates must be "
                    "numeric."
                )

        if (
            self.header_detected is not None
            and not isinstance(self.header_detected, bool)
        ):
            raise SourceSchemaError(
                "TableData.header_detected must be bool or None."
            )


@dataclass(frozen=True, slots=True)
class SourceAnnotation:
    """
    Preserved amendment or footnote information.

    target_id points to another canonical SourceBlock when the source
    relationship can be resolved deterministically.
    """

    kind: AnnotationKind
    identifier: str | None
    target_id: str | None
    raw_html: str
    text: str

    def __post_init__(self) -> None:
        """Validate annotation identity and preserved content."""

        if not isinstance(self.kind, AnnotationKind):
            raise SourceSchemaError(
                "SourceAnnotation.kind must be an AnnotationKind."
            )

        _validate_optional_source_text(
            "SourceAnnotation.identifier",
            self.identifier,
        )
        _validate_optional_source_text(
            "SourceAnnotation.target_id",
            self.target_id,
        )

        if not isinstance(self.raw_html, str):
            raise SourceSchemaError(
                "SourceAnnotation.raw_html must be a string."
            )

        if not isinstance(self.text, str):
            raise SourceSchemaError(
                "SourceAnnotation.text must be a string."
            )


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """
    One normalized structural unit from an approved frozen source.

    Empty raw_html and searchable_text values are valid for structural
    containers, diagrams, pictures, and page furniture.
    """

    block_id: str
    source_block_id: str | None
    block_type: str
    order: int
    parent_id: str | None
    child_ids: tuple[str, ...]
    hierarchy: tuple[HierarchyEntry, ...]
    page: PageIdentity | None
    legal_locator: tuple[LegalLocatorPart, ...]
    raw_html: str
    searchable_text: str
    table: TableData | None
    annotations: tuple[SourceAnnotation, ...]
    images: Mapping[str, object] | None

    def __post_init__(self) -> None:
        """Validate the intrinsic structure of one normalized block."""

        _require_source_text("SourceBlock.block_id", self.block_id)
        _validate_optional_source_text(
            "SourceBlock.source_block_id",
            self.source_block_id,
        )
        _require_source_text(
            "SourceBlock.block_type",
            self.block_type,
        )
        _validate_non_negative_integer(
            "SourceBlock.order",
            self.order,
        )
        _validate_optional_source_text(
            "SourceBlock.parent_id",
            self.parent_id,
        )

        if self.parent_id == self.block_id:
            raise SourceSchemaError(
                "A SourceBlock cannot be its own parent."
            )

        if not isinstance(self.child_ids, tuple):
            raise SourceSchemaError(
                "SourceBlock.child_ids must be an immutable tuple."
            )

        if not all(
            isinstance(child_id, str) and child_id.strip()
            for child_id in self.child_ids
        ):
            raise SourceSchemaError(
                "SourceBlock.child_ids must contain non-empty strings."
            )

        duplicate_children = _find_duplicates(self.child_ids)

        if duplicate_children:
            raise SourceSchemaError(
                "SourceBlock.child_ids contains duplicates: "
                f"{duplicate_children}"
            )

        if self.block_id in self.child_ids:
            raise SourceSchemaError(
                "A SourceBlock cannot be its own child."
            )

        if not isinstance(self.hierarchy, tuple) or not all(
            isinstance(entry, HierarchyEntry)
            for entry in self.hierarchy
        ):
            raise SourceSchemaError(
                "SourceBlock.hierarchy must contain HierarchyEntry "
                "objects in an immutable tuple."
            )

        # A hierarchy path must run from the broadest heading to the
        # most specific heading, without repeated levels.
        hierarchy_levels = tuple(
            entry.level
            for entry in self.hierarchy
        )

        if any(
            outer_level >= inner_level
            for outer_level, inner_level in zip(
                hierarchy_levels,
                hierarchy_levels[1:],
            )
        ):
            raise SourceSchemaError(
                "SourceBlock.hierarchy levels must be strictly "
                "increasing from outermost to innermost."
            )

        if self.page is not None and not isinstance(
            self.page,
            PageIdentity,
        ):
            raise SourceSchemaError(
                "SourceBlock.page must be PageIdentity or None."
            )

        if not isinstance(self.legal_locator, tuple) or not all(
            isinstance(part, LegalLocatorPart)
            for part in self.legal_locator
        ):
            raise SourceSchemaError(
                "SourceBlock.legal_locator must contain "
                "LegalLocatorPart objects in an immutable tuple."
            )

        if not isinstance(self.raw_html, str):
            raise SourceSchemaError(
                "SourceBlock.raw_html must be a string."
            )

        if not isinstance(self.searchable_text, str):
            raise SourceSchemaError(
                "SourceBlock.searchable_text must be a string."
            )

        if self.table is not None and not isinstance(
            self.table,
            TableData,
        ):
            raise SourceSchemaError(
                "SourceBlock.table must be TableData or None."
            )

        if not isinstance(self.annotations, tuple) or not all(
            isinstance(annotation, SourceAnnotation)
            for annotation in self.annotations
        ):
            raise SourceSchemaError(
                "SourceBlock.annotations must contain "
                "SourceAnnotation objects in an immutable tuple."
            )

        if self.images is not None and not isinstance(
            self.images,
            Mapping,
        ):
            raise SourceSchemaError(
                "SourceBlock.images must be a mapping or None."
            )


@dataclass(frozen=True, slots=True)
class LoadedSource:
    """
    One manifest-approved source and its normalized structural blocks.

    Cross-block validation is performed here because an individual block
    cannot prove that its referenced parents, children, headings, and
    annotation targets actually exist.
    """

    source: SourceManifestEntry
    blocks: tuple[SourceBlock, ...]

    def __post_init__(self) -> None:
        """Validate document-wide block identity and relationships."""

        if not isinstance(self.source, SourceManifestEntry):
            raise SourceSchemaError(
                "LoadedSource.source must be a SourceManifestEntry."
            )

        if not isinstance(self.blocks, tuple):
            raise SourceSchemaError(
                "LoadedSource.blocks must be an immutable tuple."
            )

        if not self.blocks:
            raise SourceSchemaError(
                "LoadedSource.blocks must not be empty."
            )

        if not all(
            isinstance(block, SourceBlock)
            for block in self.blocks
        ):
            raise SourceSchemaError(
                "LoadedSource.blocks contains an unsupported object."
            )

        block_ids = tuple(block.block_id for block in self.blocks)
        duplicate_block_ids = _find_duplicates(block_ids)

        if duplicate_block_ids:
            raise SourceSchemaError(
                "LoadedSource contains duplicate block IDs: "
                f"{duplicate_block_ids}"
            )

        source_block_ids = tuple(
            block.source_block_id
            for block in self.blocks
            if block.source_block_id is not None
        )
        duplicate_source_ids = _find_duplicates(source_block_ids)

        if duplicate_source_ids:
            raise SourceSchemaError(
                "LoadedSource contains duplicate source block IDs: "
                f"{duplicate_source_ids}"
            )

        expected_orders = tuple(range(len(self.blocks)))
        actual_orders = tuple(block.order for block in self.blocks)

        if actual_orders != expected_orders:
            raise SourceSchemaError(
                "LoadedSource block orders must be contiguous and match "
                "tuple order, beginning at zero."
            )

        required_prefix = f"{self.source.source_identity}::"

        for block_id in block_ids:
            if not block_id.startswith(required_prefix):
                raise SourceSchemaError(
                    "Every block_id must begin with the exact source "
                    f"identity and '::': {required_prefix!r}"
                )

        blocks_by_id = {
            block.block_id: block
            for block in self.blocks
        }

        for block in self.blocks:
            self._validate_block_relationships(
                block,
                blocks_by_id,
            )
            self._validate_hierarchy_references(
                block,
                blocks_by_id,
            )
            self._validate_annotation_targets(
                block,
                blocks_by_id,
            )

        if self.source.content_format is ContentFormat.EURLEX_HTML:
            blocks_with_pages = tuple(
                block.block_id
                for block in self.blocks
                if block.page is not None
            )

            if blocks_with_pages:
                raise SourceSchemaError(
                    "EUR-Lex HTML blocks must not contain page "
                    f"identities: {blocks_with_pages}"
                )

    @staticmethod
    def _validate_block_relationships(
        block: SourceBlock,
        blocks_by_id: dict[str, SourceBlock],
    ) -> None:
        """Require complete, symmetric parent and child relationships."""

        if block.parent_id is not None:
            parent = blocks_by_id.get(block.parent_id)

            if parent is None:
                raise SourceSchemaError(
                    f"Unknown parent_id {block.parent_id!r} for "
                    f"{block.block_id!r}."
                )

            if block.block_id not in parent.child_ids:
                raise SourceSchemaError(
                    "Parent/child relationship is not symmetric for "
                    f"{block.block_id!r}."
                )

            if parent.order >= block.order:
                raise SourceSchemaError(
                    "A parent block must appear before its child: "
                    f"{block.block_id!r}."
                )

        child_orders: list[int] = []

        for child_id in block.child_ids:
            child = blocks_by_id.get(child_id)

            if child is None:
                raise SourceSchemaError(
                    f"Unknown child_id {child_id!r} for "
                    f"{block.block_id!r}."
                )

            if child.parent_id != block.block_id:
                raise SourceSchemaError(
                    "Child/parent relationship is not symmetric for "
                    f"{child_id!r}."
                )

            child_orders.append(child.order)

        if child_orders != sorted(child_orders):
            raise SourceSchemaError(
                "SourceBlock.child_ids must follow document order for "
                f"{block.block_id!r}."
            )

    @staticmethod
    def _validate_hierarchy_references(
        block: SourceBlock,
        blocks_by_id: dict[str, SourceBlock],
    ) -> None:
        """Require every resolved hierarchy heading to exist."""

        for entry in block.hierarchy:
            if entry.block_id not in blocks_by_id:
                raise SourceSchemaError(
                    "Unknown hierarchy block_id "
                    f"{entry.block_id!r} referenced by "
                    f"{block.block_id!r}."
                )

    @staticmethod
    def _validate_annotation_targets(
        block: SourceBlock,
        blocks_by_id: dict[str, SourceBlock],
    ) -> None:
        """Require resolved annotation targets to reference real blocks."""

        for annotation in block.annotations:
            if (
                annotation.target_id is not None
                and annotation.target_id not in blocks_by_id
            ):
                raise SourceSchemaError(
                    "Unknown annotation target_id "
                    f"{annotation.target_id!r} referenced by "
                    f"{block.block_id!r}."
                )


__all__ = [
    "AnnotationKind",
    "AuthorityTier",
    "ContentFormat",
    "CorpusManifest",
    "HierarchyEntry",
    "IndexStatus",
    "LegalLocatorPart",
    "LoadedSource",
    "ManifestSchemaError",
    "PageIdentity",
    "RegulatoryRegime",
    "SourceAnnotation",
    "SourceBlock",
    "SourceManifestEntry",
    "SourceSchemaError",
    "SourceType",
    "TableData",
    "ValidationStatus",
]
