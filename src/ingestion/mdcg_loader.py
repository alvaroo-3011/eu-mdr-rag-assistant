"""Load validated Marker JSON sources into normalized source records.

This module accepts only sources listed in an already parsed corpus manifest.
It rechecks the selected file's location and checksum before reading it. It
does not modify frozen sources, create chunks, or make retrieval decisions.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import calculate_sha256
from .models import (
    ContentFormat,
    CorpusManifest,
    HierarchyEntry,
    LoadedSource,
    PageIdentity,
    SourceBlock,
    SourceManifestEntry,
    SourceSchemaError,
    SourceType,
    TableData,
    ValidationStatus,
)


_ROOT_KEYS = frozenset({"children", "block_type", "metadata"})

_NODE_KEYS = frozenset(
    {
        "id",
        "block_type",
        "html",
        "polygon",
        "bbox",
        "children",
        "section_hierarchy",
        "images",
    }
)

_TABLE_EXTRA_KEYS = frozenset(
    {
        "source_page",
        "table_number",
        "extraction_method",
        "table_data",
    }
)

_MARKER_SOURCE_TYPES = frozenset(
    {
        SourceType.MDCG_GUIDANCE,
        SourceType.MANUAL,
        SourceType.Q_AND_A,
    }
)

_PAGE_ID_PATTERN = re.compile(r"^/page/(\d+)(?:/|$)")
_ROOT_STRUCTURAL_ID = "document"


class MdcgLoaderError(SourceSchemaError):
    """Raised when an approved Marker JSON source cannot be loaded safely."""


class _VisibleTextExtractor(HTMLParser):
    """Collect visible text from one preserved Marker HTML fragment."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Preserve text content in parser encounter order."""

        self.parts.append(data)


def _normalized_text(raw_html: str) -> str:
    """Return visible HTML text with deterministic whitespace only."""

    extractor = _VisibleTextExtractor()

    try:
        extractor.feed(raw_html)
        extractor.close()
    except Exception as exc:
        raise MdcgLoaderError(
            "A Marker HTML fragment could not be parsed."
        ) from exc

    return " ".join(" ".join(extractor.parts).split())


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""

    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise MdcgLoaderError(
                f"Duplicate JSON key found in Marker source: {key!r}."
            )
        result[key] = value

    return result


def _require_exact_keys(
    value: dict[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    """Reject missing fields and unapproved additional fields."""

    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected

    if missing or unexpected:
        problems: list[str] = []

        if missing:
            problems.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            problems.append(f"unexpected keys: {sorted(unexpected)}")

        raise MdcgLoaderError(
            f"{location} has an invalid structure "
            f"({'; '.join(problems)})."
        )


def _require_string(value: Any, location: str) -> str:
    """Require an actual string, while permitting empty HTML."""

    if not isinstance(value, str):
        raise MdcgLoaderError(f"{location} must be a string.")

    return value


def _require_non_empty_string(value: Any, location: str) -> str:
    """Require a non-empty string."""

    result = _require_string(value, location)

    if not result.strip():
        raise MdcgLoaderError(
            f"{location} must be a non-empty string."
        )

    return result


def _require_integer(value: Any, location: str) -> int:
    """Require an integer while explicitly rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise MdcgLoaderError(f"{location} must be an integer.")

    return value


def _canonical_block_id(
    source: SourceManifestEntry,
    structural_id: str,
) -> str:
    """Combine exact source-version identity with structural identity."""

    return f"{source.source_identity}::{structural_id}"


def _resolve_verified_source_path(
    source: SourceManifestEntry,
    project_root: str | Path,
) -> Path:
    """Resolve and reverify exactly one manifest-listed frozen source."""

    resolved_project_root = Path(project_root).resolve()
    approved_source_root = (
        resolved_project_root / "data" / "rag_ready"
    ).resolve()
    relative_path = Path(*PurePosixPath(source.local_path).parts)
    source_path = (resolved_project_root / relative_path).resolve()

    try:
        source_path.relative_to(approved_source_root)
    except ValueError as exc:
        raise MdcgLoaderError(
            f"Source resolves outside data/rag_ready: "
            f"{source.local_path!r}."
        ) from exc

    if not source_path.is_file():
        raise MdcgLoaderError(
            "Approved Marker source is missing or is not a file: "
            f"{source.local_path!r}."
        )

    actual_sha256 = calculate_sha256(source_path)

    if actual_sha256.lower() != source.sha256.lower():
        raise MdcgLoaderError(
            f"Checksum mismatch for {source.document_id!r}: "
            f"expected {source.sha256.lower()}, "
            f"found {actual_sha256.lower()}."
        )

    return source_path


def _read_marker_json(source_path: Path) -> dict[str, Any]:
    """Read one UTF-8 Marker JSON object without duplicate keys."""

    try:
        with source_path.open("r", encoding="utf-8") as source_file:
            data = json.load(
                source_file,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except UnicodeDecodeError as exc:
        raise MdcgLoaderError(
            f"Marker source is not valid UTF-8: {source_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MdcgLoaderError(
            "Marker source contains invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise MdcgLoaderError(
            f"Marker source could not be read: {source_path}"
        ) from exc

    if not isinstance(data, dict):
        raise MdcgLoaderError(
            "The Marker source root must be a JSON object."
        )

    return data


def _validate_root(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the Marker document root and return its page nodes."""

    _require_exact_keys(data, _ROOT_KEYS, "Marker root")

    if data["block_type"] != "Document":
        raise MdcgLoaderError(
            "Marker root block_type must be 'Document'."
        )

    if not isinstance(data["metadata"], dict):
        raise MdcgLoaderError("Marker root metadata must be an object.")

    children = data["children"]

    if not isinstance(children, list) or not children:
        raise MdcgLoaderError(
            "Marker root children must be a non-empty array."
        )

    if not all(isinstance(child, dict) for child in children):
        raise MdcgLoaderError(
            "Every Marker root child must be an object."
        )

    return children


def _validate_node(node: dict[str, Any], location: str) -> str:
    """Validate one Marker node and return its source block ID."""

    block_type = _require_non_empty_string(
        node.get("block_type"),
        f"{location}.block_type",
    )
    expected_keys = (
        _NODE_KEYS | _TABLE_EXTRA_KEYS
        if block_type == "Table"
        else _NODE_KEYS
    )
    _require_exact_keys(node, expected_keys, location)

    source_block_id = _require_non_empty_string(
        node["id"],
        f"{location}.id",
    )
    _require_string(node["html"], f"{location}.html")

    children = node["children"]
    if children is not None and not isinstance(children, list):
        raise MdcgLoaderError(
            f"{location}.children must be an array or null."
        )
    if isinstance(children, list) and not all(
        isinstance(child, dict) for child in children
    ):
        raise MdcgLoaderError(
            f"Every child of {location} must be an object."
        )

    if not isinstance(node["section_hierarchy"], dict):
        raise MdcgLoaderError(
            f"{location}.section_hierarchy must be an object."
        )

    if node["images"] is not None and not isinstance(
        node["images"], dict
    ):
        raise MdcgLoaderError(
            f"{location}.images must be an object or null."
        )

    for geometry_name in ("polygon", "bbox"):
        geometry = node[geometry_name]
        if geometry is not None and not isinstance(geometry, list):
            raise MdcgLoaderError(
                f"{location}.{geometry_name} must be an array or null."
            )

    return source_block_id


def _collect_nodes(
    root_children: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str | None]]:
    """Collect nodes in pre-order while validating unique source IDs."""

    records: list[tuple[dict[str, Any], str | None]] = []
    seen_source_ids: set[str] = set()

    def visit(
        node: dict[str, Any],
        parent_source_id: str | None,
        location: str,
    ) -> None:
        source_block_id = _validate_node(node, location)

        if source_block_id in seen_source_ids:
            raise MdcgLoaderError(
                f"Duplicate Marker block ID: {source_block_id!r}."
            )

        if _PAGE_ID_PATTERN.match(source_block_id) is None:
            raise MdcgLoaderError(
                "Marker block ID does not contain a valid page index: "
                f"{source_block_id!r}."
            )

        seen_source_ids.add(source_block_id)
        records.append((node, parent_source_id))

        children = node["children"]
        if isinstance(children, list):
            for child_index, child in enumerate(children):
                visit(
                    child,
                    source_block_id,
                    f"{location}.children[{child_index}]",
                )

    for child_index, child in enumerate(root_children):
        visit(child, None, f"Marker root.children[{child_index}]")

    return records


def _page_identity(source_block_id: str) -> PageIdentity:
    """Derive zero-based and human-facing page identity from a Marker ID."""

    match = _PAGE_ID_PATTERN.match(source_block_id)

    if match is None:
        raise MdcgLoaderError(
            f"Cannot derive page identity from {source_block_id!r}."
        )

    source_index = int(match.group(1))
    return PageIdentity(
        source_index=source_index,
        pdf_page_number=source_index + 1,
    )


def _validate_top_level_pages(
    root_children: list[dict[str, Any]],
) -> None:
    """Require one ordered top-level Page node per page index."""

    for expected_index, page_node in enumerate(root_children):
        if page_node["block_type"] != "Page":
            raise MdcgLoaderError(
                "Every Marker root child must be a Page block."
            )

        page = _page_identity(page_node["id"])
        if page.source_index != expected_index:
            raise MdcgLoaderError(
                "Top-level Marker pages must be contiguous and ordered: "
                f"expected page index {expected_index}, "
                f"found {page.source_index}."
            )


def _build_hierarchy(
    node: dict[str, Any],
    nodes_by_source_id: dict[str, dict[str, Any]],
    source: SourceManifestEntry,
) -> tuple[HierarchyEntry, ...]:
    """Resolve Marker heading IDs, including cross-page references."""

    resolved: list[HierarchyEntry] = []

    for raw_level, target_source_id in node["section_hierarchy"].items():
        if not isinstance(raw_level, str) or not raw_level.isdecimal():
            raise MdcgLoaderError(
                "section_hierarchy levels must be positive integer "
                f"strings; found {raw_level!r}."
            )

        level = int(raw_level)
        if level < 1:
            raise MdcgLoaderError(
                "section_hierarchy levels must be positive."
            )

        if not isinstance(target_source_id, str):
            raise MdcgLoaderError(
                "section_hierarchy targets must be Marker block IDs."
            )

        target = nodes_by_source_id.get(target_source_id)
        if target is None:
            raise MdcgLoaderError(
                "Unknown section_hierarchy target: "
                f"{target_source_id!r}."
            )
        if target["block_type"] != "SectionHeader":
            raise MdcgLoaderError(
                "section_hierarchy target is not a SectionHeader: "
                f"{target_source_id!r}."
            )

        resolved.append(
            HierarchyEntry(
                level=level,
                kind="SectionHeader",
                block_id=_canonical_block_id(
                    source,
                    target_source_id,
                ),
                title=_normalized_text(target["html"]),
            )
        )

    resolved.sort(key=lambda entry: entry.level)
    return tuple(resolved)


def _require_table_cells(
    value: Any,
    location: str,
) -> tuple[str | None, ...]:
    """Validate and freeze one table header or row."""

    if not isinstance(value, list):
        raise MdcgLoaderError(f"{location} must be an array.")
    if not all(cell is None or isinstance(cell, str) for cell in value):
        raise MdcgLoaderError(
            f"{location} may contain only strings or null values."
        )

    return tuple(value)


def _build_table(
    node: dict[str, Any],
    page: PageIdentity,
) -> TableData | None:
    """Build structured data only for an enhanced Marker Table block."""

    if node["block_type"] != "Table":
        return None

    extraction_method = _require_non_empty_string(
        node["extraction_method"],
        "Table.extraction_method",
    )
    if extraction_method != "pdfplumber":
        raise MdcgLoaderError(
            "Enhanced guidance tables must use pdfplumber."
        )

    source_page = _require_integer(
        node["source_page"],
        "Table.source_page",
    )
    if source_page != page.pdf_page_number:
        raise MdcgLoaderError(
            "Table.source_page does not match its Marker page: "
            f"expected {page.pdf_page_number}, found {source_page}."
        )

    table_number = _require_integer(
        node["table_number"],
        "Table.table_number",
    )
    if table_number < 1:
        raise MdcgLoaderError(
            "Table.table_number must be a positive integer."
        )

    table_data = node["table_data"]
    if not isinstance(table_data, dict):
        raise MdcgLoaderError("Table.table_data must be an object.")

    required_table_data_keys = frozenset(
        {
            "headers",
            "rows",
            "pdfplumber_bbox",
        }
    )

    # Some validated extraction outputs record whether a header was
    # detected, while others omit that optional extraction metadata.
    # Required table content remains identical in both variants.
    if "header_detected" in table_data:
        expected_table_data_keys = (
            required_table_data_keys
            | frozenset({"header_detected"})
        )
    else:
        expected_table_data_keys = required_table_data_keys

    _require_exact_keys(
        table_data,
        expected_table_data_keys,
        "Table.table_data",
    )

    headers = _require_table_cells(
        table_data["headers"],
        "Table.table_data.headers",
    )

    raw_rows = table_data["rows"]
    if not isinstance(raw_rows, list):
        raise MdcgLoaderError("Table.table_data.rows must be an array.")
    rows = tuple(
        _require_table_cells(
            row,
            f"Table.table_data.rows[{row_index}]",
        )
        for row_index, row in enumerate(raw_rows)
    )

    raw_bbox = table_data["pdfplumber_bbox"]
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise MdcgLoaderError(
            "Table.table_data.pdfplumber_bbox must contain four "
            "coordinates."
        )
    if not all(
        isinstance(coordinate, (int, float))
        and not isinstance(coordinate, bool)
        for coordinate in raw_bbox
    ):
        raise MdcgLoaderError(
            "Table.table_data.pdfplumber_bbox coordinates must be "
            "numeric."
        )

    if "header_detected" in table_data:
        header_detected = table_data["header_detected"]

        if not isinstance(header_detected, bool):
            raise MdcgLoaderError(
                "Table.table_data.header_detected must be a boolean."
            )
    else:
        # The normalized model represents unavailable extraction
        # metadata explicitly rather than inferring True or False.
        header_detected = None

    return TableData(
        headers=headers,
        rows=rows,
        extraction_method=extraction_method,
        source_page=source_page,
        table_number=table_number,
        pdfplumber_bbox=tuple(raw_bbox),
        header_detected=header_detected,
    )


def _build_images(node: dict[str, Any]) -> Mapping[str, object] | None:
    """Detach and preserve a Marker image mapping."""

    raw_images = node["images"]
    if raw_images is None:
        return None

    # The deep copy prevents later work on normalized records from changing
    # the JSON object used during this load. It cannot alter the frozen file.
    return copy.deepcopy(raw_images)


def _normalize_document(
    data: dict[str, Any],
    source: SourceManifestEntry,
) -> LoadedSource:
    """Normalize a validated Marker document in deterministic order."""

    root_children = _validate_root(data)
    records = _collect_nodes(root_children)
    _validate_top_level_pages(root_children)

    nodes_by_source_id = {
        node["id"]: node
        for node, _ in records
    }
    root_block_id = _canonical_block_id(
        source,
        _ROOT_STRUCTURAL_ID,
    )

    blocks: list[SourceBlock] = [
        SourceBlock(
            block_id=root_block_id,
            source_block_id=None,
            block_type="Document",
            order=0,
            parent_id=None,
            child_ids=tuple(
                _canonical_block_id(source, child["id"])
                for child in root_children
            ),
            hierarchy=(),
            page=None,
            legal_locator=(),
            raw_html="",
            searchable_text="",
            table=None,
            annotations=(),
            images=None,
        )
    ]

    for order, (node, parent_source_id) in enumerate(records, start=1):
        source_block_id = node["id"]
        page = _page_identity(source_block_id)
        raw_children = node["children"] or []

        parent_id = (
            root_block_id
            if parent_source_id is None
            else _canonical_block_id(source, parent_source_id)
        )

        blocks.append(
            SourceBlock(
                block_id=_canonical_block_id(
                    source,
                    source_block_id,
                ),
                source_block_id=source_block_id,
                block_type=node["block_type"],
                order=order,
                parent_id=parent_id,
                child_ids=tuple(
                    _canonical_block_id(source, child["id"])
                    for child in raw_children
                ),
                hierarchy=_build_hierarchy(
                    node,
                    nodes_by_source_id,
                    source,
                ),
                page=page,
                legal_locator=(),
                raw_html=node["html"],
                searchable_text=_normalized_text(node["html"]),
                table=_build_table(node, page),
                annotations=(),
                images=_build_images(node),
            )
        )

    return LoadedSource(
        source=source,
        blocks=tuple(blocks),
    )


def load_mdcg_source(
    manifest: CorpusManifest,
    document_id: str,
    *,
    project_root: str | Path,
) -> LoadedSource:
    """Load one manifest-listed, validated Marker JSON source.

    The caller supplies the authoritative manifest rather than an arbitrary
    source path. The selected file is independently rechecked immediately
    before reading, protecting against changes after manifest loading.
    """

    if not isinstance(manifest, CorpusManifest):
        raise MdcgLoaderError(
            "manifest must be a CorpusManifest."
        )
    if not isinstance(document_id, str) or not document_id.strip():
        raise MdcgLoaderError(
            "document_id must be a non-empty string."
        )

    try:
        source = manifest.get_source(document_id)
    except KeyError as exc:
        raise MdcgLoaderError(
            f"Source is not listed in the manifest: {document_id!r}."
        ) from exc

    if source.validation_status is not ValidationStatus.VALIDATED:
        raise MdcgLoaderError(
            f"Source is not validated: {document_id!r}."
        )
    if source.content_format is not ContentFormat.MARKER_JSON:
        raise MdcgLoaderError(
            f"Source does not use marker_json: {document_id!r}."
        )
    if source.source_type not in _MARKER_SOURCE_TYPES:
        raise MdcgLoaderError(
            f"Unsupported Marker source type: {source.source_type.value!r}."
        )

    source_path = _resolve_verified_source_path(source, project_root)
    data = _read_marker_json(source_path)
    return _normalize_document(data, source)


__all__ = [
    "MdcgLoaderError",
    "load_mdcg_source",
]
