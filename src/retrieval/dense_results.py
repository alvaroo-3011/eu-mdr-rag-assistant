"""Strict immutable dense-result metadata, without querying or model calls.

The stored metadata is a projection of Chunk, not enough to reconstruct one
or independently prove its canonical evidence hash. Annotation HTML, table
rows, and complete block selections are not present in this projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
import json
import math
from numbers import Real
import re
from types import UnionType
from typing import get_args, get_origin, get_type_hints
from urllib.parse import urlparse

from src.ingestion.models import (
    AnnotationKind,
    AuthorityTier,
    HierarchyEntry,
    LegalLocatorPart,
    PageIdentity,
    RegulatoryRegime,
    SourceType,
)


class DenseResultError(ValueError):
    """Raised when dense-result data violates the trusted metadata contract."""


def _nonblank(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise DenseResultError(f"{field}: must be a non-empty string.")


def _typed(value: object, kind: object, field: str, *, decode: bool = False):
    """Check typed values, or decode JSON shapes into existing frozen models."""

    origin = get_origin(kind)
    if origin is UnionType:
        # The metadata models use only optional strings, without defaults.
        if value is None:
            return None
        return _typed(value, get_args(kind)[0], field, decode=decode)
    if origin is tuple:
        container = list if decode else tuple
        if type(value) is not container:
            raise DenseResultError(f"{field}: must be a {container.__name__}.")
        item_type = get_args(kind)[0]
        return tuple(
            _typed(item, item_type, f"{field}[{index}]", decode=decode)
            for index, item in enumerate(value)
        )
    if decode and isinstance(kind, type) and issubclass(kind, StrEnum):
        if type(value) is not str:
            raise DenseResultError(f"{field}: enum value must be a string.")
        try:
            return kind(value)
        except ValueError as exc:
            raise DenseResultError(f"{field}: invalid {kind.__name__} value.") from exc
    if decode and is_dataclass(kind):
        expected = {item.name for item in fields(kind)}
        _exact_keys(value, expected, field)
        hints = get_type_hints(kind)
        arguments = {
            name: _typed(value[name], hints[name], f"{field}.{name}", decode=True)
            for name in hints
        }
        try:
            return kind(**arguments)
        except (ValueError, TypeError) as exc:
            raise DenseResultError(f"{field}: invalid {kind.__name__}: {exc}") from exc
    if type(value) is not kind:
        raise DenseResultError(f"{field}: must have exact type {kind.__name__}.")
    if is_dataclass(kind):
        for name, hint in get_type_hints(kind).items():
            _typed(getattr(value, name), hint, f"{field}.{name}")
    return value


def _exact_keys(value: object, expected: set[str] | frozenset[str], field: str) -> None:
    if not isinstance(value, Mapping):
        raise DenseResultError(f"{field}: must be a mapping.")
    if any(type(key) is not str for key in value):
        raise DenseResultError(f"{field}: keys must be strings.")
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing or unexpected:
        raise DenseResultError(
            f"{field}: metadata keys mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )


@dataclass(frozen=True, slots=True)
class DenseAnnotation:
    """Exactly the annotation fields stored by chunk_metadata, without HTML."""

    kind: AnnotationKind
    identifier: str | None
    target_id: str | None
    text: str

    def __post_init__(self) -> None:
        for name, hint in get_type_hints(type(self)).items():
            _typed(getattr(self, name), hint, f"annotations.{name}")
        for name in ("identifier", "target_id"):
            value = getattr(self, name)
            if value is not None:
                _nonblank(value, f"annotations.{name}")


@dataclass(frozen=True, slots=True)
class TrustedChunkMetadata:
    """Every stored chunk metadata field, with immutable decoded citations."""

    schema_version: str
    chunk_id: str
    chunk_set_id: str
    chunk_order: int
    document_id: str
    official_title: str
    short_title: str
    version: str
    source_type: SourceType
    authority: str
    authority_tier: AuthorityTier
    official_url: str
    source_sha256: str
    regulatory_regimes: tuple[RegulatoryRegime, ...]
    permitted_regimes: tuple[RegulatoryRegime, ...]
    permits_mdr: bool
    permits_ivdr: bool
    pages: tuple[PageIdentity, ...]
    hierarchies: tuple[tuple[HierarchyEntry, ...], ...]
    legal_locators: tuple[tuple[LegalLocatorPart, ...], ...]
    block_ids: tuple[str, ...]
    annotations: tuple[DenseAnnotation, ...]
    has_table: bool
    previous_chunk_id: str | None
    next_chunk_id: str | None

    def __post_init__(self) -> None:
        for name, hint in get_type_hints(type(self)).items():
            _typed(getattr(self, name), hint, name)
        for name in (
            "schema_version", "chunk_id", "chunk_set_id", "document_id",
            "official_title", "short_title", "version", "authority",
            "official_url", "source_sha256",
        ):
            _nonblank(getattr(self, name), name)
        if self.schema_version != "1.0":
            raise DenseResultError("schema_version: unsupported metadata schema.")
        if self.chunk_order < 0:
            raise DenseResultError("chunk_order: must be non-negative.")
        # These source fields are only a subset of SourceManifestEntry: do not
        # fabricate absent local_path/content_format/status fields to build it.
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.document_id) is None:
            raise DenseResultError("document_id: invalid stable document ID.")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise DenseResultError("source_sha256: must be lowercase SHA-256 hex.")
        try:
            url = urlparse(self.official_url)
        except ValueError as exc:
            raise DenseResultError("official_url: invalid HTTPS URL.") from exc
        if url.scheme != "https" or not url.netloc:
            raise DenseResultError("official_url: must be a complete HTTPS URL.")
        for name in ("regulatory_regimes", "permitted_regimes"):
            regimes = getattr(self, name)
            if not regimes or len(set(regimes)) != len(regimes):
                raise DenseResultError(f"{name}: must be non-empty and distinct.")
        if not set(self.permitted_regimes) <= set(self.regulatory_regimes):
            raise DenseResultError("permitted_regimes: includes an undeclared regime.")
        for flag, regime in (
            ("permits_mdr", RegulatoryRegime.MDR),
            ("permits_ivdr", RegulatoryRegime.IVDR),
        ):
            if getattr(self, flag) != (regime in self.permitted_regimes):
                raise DenseResultError(f"{flag}: disagrees with permitted_regimes.")
        indexes = tuple(page.source_index for page in self.pages)
        if indexes != tuple(sorted(set(indexes))):
            raise DenseResultError("pages: must be distinct and in document order.")
        if self.source_type is SourceType.REGULATION and self.pages:
            raise DenseResultError("pages: regulation HTML must not have page identities.")
        for index, path in enumerate(self.hierarchies):
            levels = tuple(entry.level for entry in path)
            if any(a >= b for a, b in zip(levels, levels[1:])):
                raise DenseResultError(f"hierarchies[{index}]: levels must increase.")
        if len(set(self.hierarchies)) != len(self.hierarchies):
            raise DenseResultError("hierarchies: paths must be distinct.")
        if not self.block_ids or len(set(self.block_ids)) != len(self.block_ids):
            raise DenseResultError("block_ids: must be non-empty and distinct.")
        for index, block_id in enumerate(self.block_ids):
            if not block_id.startswith(f"{self.source_identity}::"):
                raise DenseResultError(f"block_ids[{index}]: wrong source identity prefix.")
        for name in ("previous_chunk_id", "next_chunk_id"):
            neighbor = getattr(self, name)
            if neighbor is not None:
                _nonblank(neighbor, name)
                if neighbor == self.chunk_id:
                    raise DenseResultError(f"{name}: cannot refer to the same chunk.")
        if (
            self.previous_chunk_id is not None
            and self.previous_chunk_id == self.next_chunk_id
        ):
            raise DenseResultError("next_chunk_id: previous and next IDs must differ.")

    @property
    def source_identity(self) -> str:
        """Recover the exact source identity from its stored ID and checksum."""

        return f"{self.document_id}@sha256:{self.source_sha256}"


_JSON_FIELDS = frozenset({
    "regulatory_regimes", "permitted_regimes", "pages", "hierarchies",
    "legal_locators", "block_ids", "annotations",
})

# Derived from the typed projection; shared with the serializer to catch drift.
CHUNK_METADATA_KEYS = frozenset(
    f"{item.name}_json" if item.name in _JSON_FIELDS else item.name
    for item in fields(TrustedChunkMetadata)
)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant")


def deserialize_chunk_metadata(raw: object) -> TrustedChunkMetadata:
    """Strictly decode one complete stored metadata mapping, without defaults."""

    _exact_keys(raw, CHUNK_METADATA_KEYS, "metadata")
    values: dict[str, object] = {}
    for name, hint in get_type_hints(TrustedChunkMetadata).items():
        field = f"{name}_json" if name in _JSON_FIELDS else name
        value = raw[field]
        if name in _JSON_FIELDS:
            if type(value) is not str:
                raise DenseResultError(f"{field}: must be a JSON string.")
            try:
                value = json.loads(
                    value,
                    object_pairs_hook=_json_object,
                    parse_constant=_invalid_constant,
                )
            except (ValueError, TypeError, RecursionError) as exc:
                raise DenseResultError(f"{field}: invalid JSON: {exc}") from exc
        if name in ("previous_chunk_id", "next_chunk_id"):
            if type(value) is not str:
                raise DenseResultError(f"{field}: must be a string.")
            # Empty strings are the serializer's explicit absent-neighbor marker.
            value = None if value == "" else value
        values[name] = _typed(value, hint, field, decode=True)
    return TrustedChunkMetadata(**values)


@dataclass(frozen=True, slots=True)
class DenseCandidate:
    """One ranked dense candidate with validated, immutable citation metadata."""

    rank: int
    chunk_id: str
    document: str
    metadata: TrustedChunkMetadata
    cosine_distance: float

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise DenseResultError("rank: must be a positive non-boolean integer.")
        _nonblank(self.chunk_id, "chunk_id")
        _nonblank(self.document, "document")
        if not isinstance(self.metadata, TrustedChunkMetadata):
            raise DenseResultError("metadata: must be TrustedChunkMetadata.")
        if self.chunk_id != self.metadata.chunk_id:
            raise DenseResultError("chunk_id: must match trusted metadata.")
        if isinstance(self.cosine_distance, bool) or not isinstance(
            self.cosine_distance, Real
        ):
            raise DenseResultError("cosine_distance: must be a finite real number.")
        try:
            distance = float(self.cosine_distance)
        except (ValueError, TypeError, OverflowError) as exc:
            raise DenseResultError("cosine_distance: cannot be represented as float.") from exc
        if not math.isfinite(distance):
            raise DenseResultError("cosine_distance: must be finite.")
        object.__setattr__(self, "cosine_distance", distance)


__all__ = [
    "CHUNK_METADATA_KEYS",
    "DenseAnnotation",
    "DenseCandidate",
    "DenseResultError",
    "TrustedChunkMetadata",
    "deserialize_chunk_metadata",
]
