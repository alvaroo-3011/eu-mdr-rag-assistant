"""Strict loading and verification of the authoritative corpus manifest.

This module is deliberately limited to corpus identity and file integrity.
It does not parse source documents, create chunks, or build indexes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from .models import (
    AuthorityTier,
    ContentFormat,
    CorpusManifest,
    IndexStatus,
    ManifestSchemaError,
    RegulatoryRegime,
    SourceManifestEntry,
    SourceType,
    ValidationStatus,
)


DEFAULT_MANIFEST_PATH = Path("data/corpus_manifest.json")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "corpus_id",
        "expected_source_count",
        "sources",
    }
)

_SOURCE_KEYS = frozenset(
    {
        "document_id",
        "official_title",
        "short_title",
        "version",
        "regulatory_regimes",
        "permitted_regimes",
        "source_type",
        "authority",
        "authority_tier",
        "official_url",
        "local_path",
        "content_format",
        "sha256",
        "validation_status",
        "index_status",
    }
)

EnumType = TypeVar("EnumType", bound=StrEnum)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys.

    Python's normal JSON decoder silently keeps the final occurrence of a
    duplicate key. That behavior is unsafe for trusted corpus metadata.
    """

    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ManifestSchemaError(
                f"Duplicate JSON key found in manifest: {key!r}."
            )
        result[key] = value

    return result


def _require_exact_keys(
    value: dict[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    """Reject both missing fields and unapproved additional fields."""

    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected

    problems: list[str] = []

    if missing:
        problems.append(f"missing keys: {sorted(missing)}")

    if unexpected:
        problems.append(f"unexpected keys: {sorted(unexpected)}")

    if problems:
        raise ManifestSchemaError(
            f"{location} has an invalid structure ({'; '.join(problems)})."
        )


def _require_string(value: Any, location: str) -> str:
    """Require an actual JSON string."""

    if not isinstance(value, str):
        raise ManifestSchemaError(f"{location} must be a string.")

    return value


def _require_integer(value: Any, location: str) -> int:
    """Require an integer while excluding booleans.

    In Python, bool is a subclass of int, so it must be rejected explicitly.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestSchemaError(f"{location} must be an integer.")

    return value


def _parse_enum(
    enum_type: type[EnumType],
    value: Any,
    location: str,
) -> EnumType:
    """Convert a manifest string into an approved enum value."""

    raw_value = _require_string(value, location)

    try:
        return enum_type(raw_value)
    except ValueError as exc:
        allowed = [member.value for member in enum_type]
        raise ManifestSchemaError(
            f"{location} has unsupported value {raw_value!r}; "
            f"allowed values are {allowed}."
        ) from exc


def _parse_enum_list(
    enum_type: type[EnumType],
    value: Any,
    location: str,
) -> tuple[EnumType, ...]:
    """Convert a JSON array into an immutable tuple of enum values."""

    if not isinstance(value, list):
        raise ManifestSchemaError(f"{location} must be an array.")

    return tuple(
        _parse_enum(enum_type, item, f"{location}[{index}]")
        for index, item in enumerate(value)
    )


def _parse_source(
    value: Any,
    index: int,
) -> SourceManifestEntry:
    """Parse one source entry into the canonical typed model."""

    location = f"sources[{index}]"

    if not isinstance(value, dict):
        raise ManifestSchemaError(f"{location} must be an object.")

    _require_exact_keys(value, _SOURCE_KEYS, location)

    return SourceManifestEntry(
        document_id=_require_string(
            value["document_id"], f"{location}.document_id"
        ),
        official_title=_require_string(
            value["official_title"], f"{location}.official_title"
        ),
        short_title=_require_string(
            value["short_title"], f"{location}.short_title"
        ),
        version=_require_string(value["version"], f"{location}.version"),
        regulatory_regimes=_parse_enum_list(
            RegulatoryRegime,
            value["regulatory_regimes"],
            f"{location}.regulatory_regimes",
        ),
        permitted_regimes=_parse_enum_list(
            RegulatoryRegime,
            value["permitted_regimes"],
            f"{location}.permitted_regimes",
        ),
        source_type=_parse_enum(
            SourceType,
            value["source_type"],
            f"{location}.source_type",
        ),
        authority=_require_string(
            value["authority"], f"{location}.authority"
        ),
        authority_tier=_parse_enum(
            AuthorityTier,
            value["authority_tier"],
            f"{location}.authority_tier",
        ),
        official_url=_require_string(
            value["official_url"], f"{location}.official_url"
        ),
        local_path=_require_string(
            value["local_path"], f"{location}.local_path"
        ),
        content_format=_parse_enum(
            ContentFormat,
            value["content_format"],
            f"{location}.content_format",
        ),
        sha256=_require_string(value["sha256"], f"{location}.sha256"),
        validation_status=_parse_enum(
            ValidationStatus,
            value["validation_status"],
            f"{location}.validation_status",
        ),
        index_status=_parse_enum(
            IndexStatus,
            value["index_status"],
            f"{location}.index_status",
        ),
    )


def _read_manifest_json(manifest_path: Path) -> dict[str, Any]:
    """Read JSON without permitting duplicate object keys."""

    if not manifest_path.is_file():
        raise ManifestSchemaError(
            f"Corpus manifest is missing or is not a file: {manifest_path}"
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            data = json.load(
                manifest_file,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except UnicodeDecodeError as exc:
        raise ManifestSchemaError(
            f"Corpus manifest is not valid UTF-8: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestSchemaError(
            f"Corpus manifest contains invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ManifestSchemaError(
            f"Corpus manifest could not be read: {manifest_path}"
        ) from exc

    if not isinstance(data, dict):
        raise ManifestSchemaError(
            "The corpus manifest root must be a JSON object."
        )

    return data


def calculate_sha256(
    file_path: Path,
    *,
    block_size: int = 1024 * 1024,
) -> str:
    """Calculate a file's SHA-256 without loading the whole file into memory."""

    digest = hashlib.sha256()

    try:
        with file_path.open("rb") as source_file:
            for block in iter(
                lambda: source_file.read(block_size),
                b"",
            ):
                digest.update(block)
    except OSError as exc:
        raise ManifestSchemaError(
            f"Source file could not be read: {file_path}"
        ) from exc

    return digest.hexdigest()


def verify_source_files(
    manifest: CorpusManifest,
    project_root: Path,
    *,
    checksum_function: Callable[[Path], str] = calculate_sha256,
) -> None:
    """Verify location, uniqueness, existence, and checksum of every source."""

    resolved_project_root = project_root.resolve()
    approved_source_root = (
        resolved_project_root / "data" / "rag_ready"
    ).resolve()

    seen_resolved_paths: set[str] = set()

    for source in manifest.sources:
        # Manifest paths use forward slashes consistently, including on Windows.
        relative_path = Path(*PurePosixPath(source.local_path).parts)
        source_path = (resolved_project_root / relative_path).resolve()

        try:
            source_path.relative_to(approved_source_root)
        except ValueError as exc:
            raise ManifestSchemaError(
                f"Source {source.document_id!r} resolves outside "
                f"data/rag_ready: {source.local_path!r}."
            ) from exc

        # normcase makes this duplicate check case-insensitive on Windows.
        normalized_path = os.path.normcase(str(source_path))

        if normalized_path in seen_resolved_paths:
            raise ManifestSchemaError(
                f"Multiple manifest entries resolve to the same file: "
                f"{source_path}"
            )

        seen_resolved_paths.add(normalized_path)

        if not source_path.is_file():
            raise ManifestSchemaError(
                f"Approved source is missing or is not a regular file: "
                f"{source.local_path!r}."
            )

        actual_sha256 = checksum_function(source_path)

        if actual_sha256.lower() != source.sha256.lower():
            raise ManifestSchemaError(
                f"Checksum mismatch for {source.document_id!r}: "
                f"expected {source.sha256.lower()}, "
                f"found {actual_sha256.lower()}."
            )


def load_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    project_root: str | Path | None = None,
    verify_files: bool = True,
) -> CorpusManifest:
    """Load the authoritative manifest and optionally verify frozen sources.

    When ``project_root`` is omitted, it is derived from the normal manifest
    location ``<project root>/data/corpus_manifest.json``.
    """

    resolved_manifest_path = Path(manifest_path).resolve()
    data = _read_manifest_json(resolved_manifest_path)
    _require_exact_keys(data, _TOP_LEVEL_KEYS, "manifest")

    raw_sources = data["sources"]

    if not isinstance(raw_sources, list):
        raise ManifestSchemaError("manifest.sources must be an array.")

    manifest = CorpusManifest(
        schema_version=_require_string(
            data["schema_version"], "manifest.schema_version"
        ),
        corpus_id=_require_string(data["corpus_id"], "manifest.corpus_id"),
        expected_source_count=_require_integer(
            data["expected_source_count"],
            "manifest.expected_source_count",
        ),
        sources=tuple(
            _parse_source(source, index)
            for index, source in enumerate(raw_sources)
        ),
    )

    if verify_files:
        if project_root is None:
            # The approved manifest lives directly inside the data directory.
            resolved_project_root = resolved_manifest_path.parent.parent
        else:
            resolved_project_root = Path(project_root).resolve()

        verify_source_files(manifest, resolved_project_root)

    return manifest