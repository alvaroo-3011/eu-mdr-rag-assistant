"""Load the validated consolidated EUR-Lex MDR HTML source.

The loader accepts only the regulation entry from an authoritative typed
manifest. It verifies the selected file again, parses only the legal-content
container, and normalizes legal structure without modifying the frozen HTML.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

try:
    from lxml import etree, html
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "src.ingestion.mdr_loader requires the 'lxml' package."
    ) from exc

from .manifest import calculate_sha256
from .models import (
    AnnotationKind,
    ContentFormat,
    CorpusManifest,
    HierarchyEntry,
    LegalLocatorPart,
    LoadedSource,
    SourceAnnotation,
    SourceBlock,
    SourceManifestEntry,
    SourceSchemaError,
    SourceType,
    TableData,
    ValidationStatus,
)


_CLASS_TOKEN_XPATH = (
    'contains(concat(" ", normalize-space(@class), " "), " {token} ")'
)
_ARTICLE_ID_PATTERN = re.compile(r"^art_([0-9]+[a-z]?)$")
_CHAPTER_ID_PATTERN = re.compile(r"^cpt_([IVXLCDM]+)$")
_SECTION_ID_PATTERN = re.compile(
    r"^cpt_([IVXLCDM]+)\.sct_([0-9]+[a-z]?)$"
)
_ANNEX_ID_PATTERN = re.compile(r"^anx_([IVXLCDM]+)$")
_ANNEX_HEADING_CLASS_PATTERN = re.compile(
    r"(?:^|\s)title-gr-seq-level-([1-5])(?:\s|$)"
)
_NUMBERED_PARAGRAPH_PATTERN = re.compile(
    r"^([0-9]+[a-z]?(?:\.[0-9]+[a-z]?)*)(?:\.)?$",
    re.IGNORECASE,
)
_AMENDMENT_TITLE_PATTERN = re.compile(
    r"^(?P<source>[^:]+):\s*(?P<action>INSERTED|REPLACED|DELETED)$"
)
_POINT_WRAPPER = "grid-container"
_POINT_LIST = "grid-list"
_ROOT_STRUCTURAL_ID = "document"


class MdrLoaderError(SourceSchemaError):
    """Raised when the approved consolidated MDR HTML cannot load safely."""


def _has_class(element: Any, class_name: str) -> bool:
    """Return whether one element contains an exact whitespace class token."""

    return class_name in (element.get("class") or "").split()


def _normalized_text(value: str) -> str:
    """Normalize source whitespace without changing visible characters."""

    return " ".join(value.replace("\xa0", " ").split())


def _element_text(element: Any) -> str:
    """Return normalized visible text for one parsed element."""

    return _normalized_text(element.text_content())


def _raw_html(element: Any) -> str:
    """Serialize one source element without including its following tail."""

    return etree.tostring(
        element,
        encoding="unicode",
        method="html",
        with_tail=False,
    )


def _raw_html_group(elements: list[Any]) -> str:
    """Serialize an ordered group of adjacent source elements."""

    return "\n".join(_raw_html(element) for element in elements)


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
    """Resolve and reverify the exact manifest-listed regulation source."""

    resolved_project_root = Path(project_root).resolve()
    approved_source_root = (
        resolved_project_root / "data" / "rag_ready"
    ).resolve()
    relative_path = Path(*PurePosixPath(source.local_path).parts)
    source_path = (resolved_project_root / relative_path).resolve()

    try:
        source_path.relative_to(approved_source_root)
    except ValueError as exc:
        raise MdrLoaderError(
            "Regulation source resolves outside data/rag_ready: "
            f"{source.local_path!r}."
        ) from exc

    if not source_path.is_file():
        raise MdrLoaderError(
            "Approved regulation source is missing or is not a file: "
            f"{source.local_path!r}."
        )

    actual_sha256 = calculate_sha256(source_path)
    if actual_sha256.lower() != source.sha256.lower():
        raise MdrLoaderError(
            f"Checksum mismatch for {source.document_id!r}: "
            f"expected {source.sha256.lower()}, "
            f"found {actual_sha256.lower()}."
        )

    return source_path


def _parse_html(source_path: Path) -> Any:
    """Parse checksum-verified local HTML without network access."""

    parser = html.HTMLParser(
        encoding="utf-8",
        no_network=True,
        # The saved EUR-Lex page contains valid HTML5 elements such as
        # ``nav``. Some libxml2 versions report these as XML validity
        # errors. Recovery is safe because the file is checksum-verified
        # before parsing and the required EUR-Lex structure is validated
        # afterwards.
        recover=True,
        remove_comments=False,
    )

    try:
        tree = html.parse(str(source_path), parser=parser)
    except (OSError, etree.Error) as exc:
        raise MdrLoaderError(
            f"Regulation HTML could not be parsed: {source_path}"
        ) from exc

    return tree.getroot()


def _single_element(
    root: Any,
    xpath: str,
    description: str,
) -> Any:
    """Require exactly one structural element for a trusted selector."""

    matches = root.xpath(xpath)
    if len(matches) != 1:
        raise MdrLoaderError(
            f"Expected exactly one {description}; found {len(matches)}."
        )
    return matches[0]


def _validate_unique_ids(document: Any) -> None:
    """Reject duplicate source IDs before resolving any relationships."""

    identifiers = [
        element.get("id")
        for element in document.xpath('.//*[@id]')
    ]
    duplicates = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        raise MdrLoaderError(
            f"Regulation HTML contains duplicate IDs: {duplicates}"
        )


def _drop_element(element: Any) -> None:
    """Remove one copied element while preserving its following tail text."""

    if hasattr(element, "drop_tree"):
        element.drop_tree()
    else:  # pragma: no cover - lxml.html elements provide drop_tree
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)


def _is_grid_item(element: Any) -> bool:
    """Recognize EUR-Lex's two-column list-item wrapper."""

    return (
        element.tag == "div"
        and _has_class(element, _POINT_WRAPPER)
        and _has_class(element, _POINT_LIST)
    )


def _outer_grid_items(container: Any) -> list[Any]:
    """Return only the outermost list items below one content container."""

    candidates = container.xpath(
        './/div['
        + _CLASS_TOKEN_XPATH.format(token=_POINT_WRAPPER)
        + " and "
        + _CLASS_TOKEN_XPATH.format(token=_POINT_LIST)
        + "]"
    )
    result: list[Any] = []

    for candidate in candidates:
        ancestor = candidate.getparent()
        nested = False
        while ancestor is not None and ancestor is not container:
            # A nested grid item belongs to its closest list item. A grid item
            # inside a nested layout table belongs to that annex item instead.
            if _is_grid_item(ancestor) or ancestor.tag == "table":
                nested = True
                break
            ancestor = ancestor.getparent()
        if not nested:
            result.append(candidate)

    return result


def _outer_tables(container: Any) -> list[Any]:
    """Return tables that are not nested inside another table below a node."""

    result: list[Any] = []
    for candidate in container.xpath('.//table'):
        ancestor = candidate.getparent()
        nested = False
        while ancestor is not None and ancestor is not container:
            if ancestor.tag == "table":
                nested = True
                break
            ancestor = ancestor.getparent()
        if not nested:
            result.append(candidate)
    return result


def _content_copy(
    element: Any,
    *,
    exclude_grid_items: bool,
    exclude_tables: bool = False,
) -> Any:
    """Copy content and remove UI or separately normalized descendants."""

    cloned = copy.deepcopy(element)

    for unwanted in cloned.xpath('.//script | .//style'):
        _drop_element(unwanted)

    for anchor in cloned.xpath('.//a'):
        title = anchor.get("title") or ""
        if _has_class(anchor, "anchorarrow") or _AMENDMENT_TITLE_PATTERN.match(
            title
        ):
            _drop_element(anchor)

    if exclude_grid_items:
        for grid_item in _outer_grid_items(cloned):
            _drop_element(grid_item)

    if exclude_tables:
        for table_element in _outer_tables(cloned):
            _drop_element(table_element)

    return cloned


def _searchable_text(
    element: Any,
    *,
    exclude_grid_items: bool = False,
    exclude_tables: bool = False,
) -> str:
    """Build clean retrieval text while preserving annotations separately."""

    cloned = _content_copy(
        element,
        exclude_grid_items=exclude_grid_items,
        exclude_tables=exclude_tables,
    )
    return _element_text(cloned)


def _annotation_elements(
    element: Any,
    *,
    exclude_grid_items: bool,
    exclude_tables: bool,
) -> Any:
    """Copy one semantic unit, optionally excluding nested list units."""

    cloned = copy.deepcopy(element)
    if exclude_grid_items:
        for grid_item in _outer_grid_items(cloned):
            _drop_element(grid_item)
    if exclude_tables:
        for table_element in _outer_tables(cloned):
            _drop_element(table_element)
    return cloned


def _annotations_for(
    element: Any,
    source: SourceManifestEntry,
    *,
    exclude_grid_items: bool = False,
    exclude_tables: bool = False,
) -> tuple[SourceAnnotation, ...]:
    """Preserve footnote references and amendment provenance."""

    scanned = _annotation_elements(
        element,
        exclude_grid_items=exclude_grid_items,
        exclude_tables=exclude_tables,
    )
    annotations: list[SourceAnnotation] = []

    for anchor in scanned.xpath('.//a'):
        anchor_id = anchor.get("id")
        title = anchor.get("title") or ""

        if anchor_id and anchor_id.startswith("src.E"):
            definition_id = anchor_id.removeprefix("src.")
            annotation = SourceAnnotation(
                kind=AnnotationKind.FOOTNOTE_REFERENCE,
                identifier=definition_id,
                target_id=_canonical_block_id(
                    source,
                    f"footnotes/{definition_id}",
                ),
                raw_html=_raw_html(anchor),
                text=_element_text(anchor),
            )
        else:
            amendment_match = _AMENDMENT_TITLE_PATTERN.match(title)
            if amendment_match is None:
                continue

            marker = _element_text(anchor).lstrip("►▼")
            annotation = SourceAnnotation(
                kind=AnnotationKind.AMENDMENT,
                identifier=marker or amendment_match.group("source"),
                target_id=None,
                raw_html=_raw_html(anchor),
                text=(
                    f"{_element_text(anchor)} — "
                    f"{amendment_match.group('source')}: "
                    f"{amendment_match.group('action')}"
                ),
            )

        # Repeated identical markers are distinct source occurrences and
        # must not be collapsed merely because their serialized anchors match.
        annotations.append(annotation)

    return tuple(annotations)


def _locator(
    existing: tuple[LegalLocatorPart, ...],
    kind: str,
    value: str,
) -> tuple[LegalLocatorPart, ...]:
    """Append one normalized legal-locator component."""

    normalized_value = _normalized_text(value).strip(" ")
    if not normalized_value:
        raise MdrLoaderError(f"Cannot create an empty {kind} locator.")
    return existing + (LegalLocatorPart(kind=kind, value=normalized_value),)


def _clean_number_label(label: str) -> str:
    """Normalize a displayed paragraph or annex-item number."""

    normalized = _normalized_text(label).rstrip(".")
    match = _NUMBERED_PARAGRAPH_PATTERN.fullmatch(normalized)
    if match is None:
        raise MdrLoaderError(
            f"Unsupported numbered legal label: {label!r}."
        )
    return match.group(1)


def _clean_point_label(label: str) -> tuple[str, str]:
    """Return locator kind and value for one EUR-Lex list label."""

    normalized = _normalized_text(label)
    if normalized in {"—", "–", "-"}:
        return "indent", normalized

    value = normalized
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    value = value.rstrip(".").strip()

    if not value:
        raise MdrLoaderError(f"Unsupported empty list label: {label!r}.")
    return "point", value


@dataclass(slots=True)
class _DraftBlock:
    """Mutable relationship shell finalized as an immutable SourceBlock."""

    structural_id: str
    source_block_id: str | None
    block_type: str
    parent_structural_id: str | None
    hierarchy: tuple[HierarchyEntry, ...]
    legal_locator: tuple[LegalLocatorPart, ...]
    raw_html: str
    searchable_text: str
    table: TableData | None
    annotations: tuple[SourceAnnotation, ...]
    child_structural_ids: list[str] = field(default_factory=list)


class _SourceBuilder:
    """Build deterministic block order and symmetric relationships."""

    def __init__(self, source: SourceManifestEntry) -> None:
        self.source = source
        self.drafts: list[_DraftBlock] = []
        self.by_structural_id: dict[str, _DraftBlock] = {}
        self.source_ids: set[str] = set()
        self.counters: Counter[str] = Counter()

    def canonical(self, structural_id: str) -> str:
        """Return one canonical block ID."""

        return _canonical_block_id(self.source, structural_id)

    def derived_id(self, parent_id: str, block_type: str) -> str:
        """Create a deterministic positional ID within an exact source."""

        counter_key = f"{parent_id}/{block_type.lower()}"
        self.counters[counter_key] += 1
        return f"{counter_key}/{self.counters[counter_key]:04d}"

    def hierarchy_entry(
        self,
        *,
        level: int,
        kind: str,
        structural_id: str,
        title: str,
    ) -> HierarchyEntry:
        """Build one canonical hierarchy reference."""

        return HierarchyEntry(
            level=level,
            kind=kind,
            block_id=self.canonical(structural_id),
            title=title,
        )

    def add(
        self,
        *,
        structural_id: str,
        source_block_id: str | None,
        block_type: str,
        parent_structural_id: str | None,
        hierarchy: tuple[HierarchyEntry, ...],
        legal_locator: tuple[LegalLocatorPart, ...],
        raw_html: str,
        searchable_text: str,
        table: TableData | None = None,
        annotations: tuple[SourceAnnotation, ...] = (),
    ) -> str:
        """Add one block after validating local identity constraints."""

        if structural_id in self.by_structural_id:
            raise MdrLoaderError(
                f"Duplicate derived regulation block ID: {structural_id!r}."
            )
        if source_block_id is not None:
            if source_block_id in self.source_ids:
                raise MdrLoaderError(
                    "A source HTML ID was assigned to multiple blocks: "
                    f"{source_block_id!r}."
                )
            self.source_ids.add(source_block_id)
        if (
            parent_structural_id is not None
            and parent_structural_id not in self.by_structural_id
        ):
            raise MdrLoaderError(
                f"Unknown draft parent: {parent_structural_id!r}."
            )

        draft = _DraftBlock(
            structural_id=structural_id,
            source_block_id=source_block_id,
            block_type=block_type,
            parent_structural_id=parent_structural_id,
            hierarchy=hierarchy,
            legal_locator=legal_locator,
            raw_html=raw_html,
            searchable_text=searchable_text,
            table=table,
            annotations=annotations,
        )
        self.drafts.append(draft)
        self.by_structural_id[structural_id] = draft

        if parent_structural_id is not None:
            self.by_structural_id[
                parent_structural_id
            ].child_structural_ids.append(structural_id)
        return structural_id

    def finalize(self) -> LoadedSource:
        """Freeze all drafts and invoke document-wide model validation."""

        blocks = tuple(
            SourceBlock(
                block_id=self.canonical(draft.structural_id),
                source_block_id=draft.source_block_id,
                block_type=draft.block_type,
                order=order,
                parent_id=(
                    self.canonical(draft.parent_structural_id)
                    if draft.parent_structural_id is not None
                    else None
                ),
                child_ids=tuple(
                    self.canonical(child_id)
                    for child_id in draft.child_structural_ids
                ),
                hierarchy=draft.hierarchy,
                page=None,
                legal_locator=draft.legal_locator,
                raw_html=draft.raw_html,
                searchable_text=draft.searchable_text,
                table=draft.table,
                annotations=draft.annotations,
                images=None,
            )
            for order, draft in enumerate(self.drafts)
        )
        return LoadedSource(source=self.source, blocks=blocks)


def _header_elements(
    container: Any,
    first_class: str,
    second_class: str,
) -> tuple[Any, Any]:
    """Require the two heading lines used by chapters and sections."""

    significant = [
        child
        for child in container
        if _element_text(child) or child.get("id")
    ]
    if len(significant) < 2:
        raise MdrLoaderError(
            f"{container.get('id')!r} does not contain two heading lines."
        )
    first, second = significant[:2]
    if not _has_class(first, first_class) or not _has_class(
        second,
        second_class,
    ):
        raise MdrLoaderError(
            f"{container.get('id')!r} has an unexpected heading structure."
        )
    return first, second


def _emit_grid_item(
    builder: _SourceBuilder,
    element: Any,
    *,
    parent_id: str,
    hierarchy: tuple[HierarchyEntry, ...],
    legal_locator: tuple[LegalLocatorPart, ...],
) -> str:
    """Normalize one possibly nested EUR-Lex list item."""

    direct_divs = [child for child in element if child.tag == "div"]
    if len(direct_divs) != 2:
        raise MdrLoaderError(
            "A grid-list item must contain exactly two direct columns."
        )
    label_column, content_column = direct_divs
    if not _has_class(label_column, "grid-list-column-1") or not _has_class(
        content_column,
        "grid-list-column-2",
    ):
        raise MdrLoaderError("A grid-list item has unexpected columns.")

    label = _element_text(label_column)
    point_kind, point_value = _clean_point_label(label)
    item_locator = _locator(legal_locator, point_kind, point_value)
    structural_id = builder.derived_id(parent_id, "point")
    own_text = _searchable_text(
        content_column,
        exclude_grid_items=True,
    )
    searchable_text = _normalized_text(f"{label} {own_text}")

    builder.add(
        structural_id=structural_id,
        source_block_id=element.get("id"),
        block_type="Indent" if point_kind == "indent" else "Point",
        parent_structural_id=parent_id,
        hierarchy=hierarchy,
        legal_locator=item_locator,
        raw_html=_raw_html(element),
        searchable_text=searchable_text,
        annotations=_annotations_for(
            content_column,
            builder.source,
            exclude_grid_items=True,
        ),
    )

    for child_item in _outer_grid_items(content_column):
        _emit_grid_item(
            builder,
            child_item,
            parent_id=structural_id,
            hierarchy=hierarchy,
            legal_locator=item_locator,
        )
    return structural_id


def _numbered_paragraph_label(element: Any) -> str | None:
    """Return the direct EUR-Lex paragraph number when present."""

    direct_labels = element.xpath(
        './span['
        + _CLASS_TOKEN_XPATH.format(token="no-parag")
        + "]"
    )
    if not direct_labels:
        return None
    if len(direct_labels) != 1:
        raise MdrLoaderError("A paragraph contains multiple number labels.")
    raw_label = _element_text(direct_labels[0])
    # Article 119 quotes paragraph “4.” of another regulation. EUR-Lex uses
    # no-parag styling for that quotation, but it is not an MDR paragraph.
    if raw_label.startswith(("‘", "’", "\"", "'")):
        return None
    return _clean_number_label(raw_label)


def _emit_content(
    builder: _SourceBuilder,
    element: Any,
    *,
    parent_id: str,
    hierarchy: tuple[HierarchyEntry, ...],
    legal_locator: tuple[LegalLocatorPart, ...],
    block_type: str,
) -> str | None:
    """Normalize one prose unit and its nested points."""

    searchable_text = _searchable_text(
        element,
        exclude_grid_items=True,
    )
    annotations = _annotations_for(
        element,
        builder.source,
        exclude_grid_items=True,
    )
    nested_items = _outer_grid_items(element)

    if not searchable_text and not annotations and not nested_items:
        return None

    structural_id = builder.derived_id(parent_id, block_type)
    builder.add(
        structural_id=structural_id,
        source_block_id=element.get("id"),
        block_type=block_type,
        parent_structural_id=parent_id,
        hierarchy=hierarchy,
        legal_locator=legal_locator,
        raw_html=_raw_html(element),
        searchable_text=searchable_text,
        annotations=annotations,
    )

    for nested_item in nested_items:
        _emit_grid_item(
            builder,
            nested_item,
            parent_id=structural_id,
            hierarchy=hierarchy,
            legal_locator=legal_locator,
        )
    return structural_id


def _table_rows(table_element: Any) -> list[list[str]]:
    """Extract ordered visible cell text from a genuine legal table."""

    rows: list[list[str]] = []
    for row in table_element.xpath('./thead/tr | ./tbody/tr | ./tr'):
        cells = row.xpath('./th | ./td')
        if not cells:
            raise MdrLoaderError("A genuine legal table contains an empty row.")
        rows.append([_element_text(cell) for cell in cells])
    if not rows:
        raise MdrLoaderError("A genuine legal table contains no rows.")
    return rows


def _is_genuine_table(table_element: Any) -> bool:
    """Distinguish data tables from EUR-Lex two-column layout wrappers."""

    rows = table_element.xpath('./thead/tr | ./tbody/tr | ./tr')
    return (
        _has_class(table_element, "borderOj")
        or bool(
            table_element.xpath(
                './/*['
                + _CLASS_TOKEN_XPATH.format(token="tbl-norm")
                + "]"
            )
        )
        or len(rows) > 1
    )


def _emit_genuine_table(
    builder: _SourceBuilder,
    table_element: Any,
    *,
    parent_id: str,
    hierarchy: tuple[HierarchyEntry, ...],
    legal_locator: tuple[LegalLocatorPart, ...],
) -> str:
    """Normalize a genuine EUR-Lex table without pdfplumber metadata."""

    rows = _table_rows(table_element)
    first_row = table_element.xpath('./thead/tr | ./tbody/tr | ./tr')[0]
    first_cells = first_row.xpath('./th | ./td')
    header_detected = bool(first_row.xpath('./th')) or all(
        "font-weight" in (cell.get("style") or "")
        and "bold" in (cell.get("style") or "").lower()
        for cell in first_cells
    )
    headers = tuple(rows[0]) if header_detected else ()
    data_rows = rows[1:] if header_detected else rows
    table_data = TableData(
        headers=headers,
        rows=tuple(tuple(cell for cell in row) for row in data_rows),
        extraction_method="eurlex_html",
        source_page=None,
        table_number=None,
        pdfplumber_bbox=None,
        header_detected=header_detected,
    )
    structural_id = builder.derived_id(parent_id, "table")
    searchable_text = "\n".join(
        " | ".join(row)
        for row in rows
    )
    builder.add(
        structural_id=structural_id,
        source_block_id=table_element.get("id"),
        block_type="Table",
        parent_structural_id=parent_id,
        hierarchy=hierarchy,
        legal_locator=legal_locator,
        raw_html=_raw_html(table_element),
        searchable_text=searchable_text,
        table=table_data,
        annotations=_annotations_for(
            table_element,
            builder.source,
        ),
    )
    return structural_id


def _build_article(
    builder: _SourceBuilder,
    article: Any,
    *,
    parent_id: str,
    outer_hierarchy: tuple[HierarchyEntry, ...],
    outer_locator: tuple[LegalLocatorPart, ...],
) -> None:
    """Normalize one article and its numbered internal content."""

    article_id = article.get("id")
    match = _ARTICLE_ID_PATTERN.fullmatch(article_id or "")
    if match is None or not _has_class(article, "eli-subdivision"):
        raise MdrLoaderError(f"Invalid article container: {article_id!r}.")

    children = list(article)
    if len(children) < 2:
        raise MdrLoaderError(f"Article {article_id!r} has no heading.")
    label_element, title_element = children[:2]
    if not _has_class(label_element, "title-article-norm") or not (
        title_element.get("id") == f"{article_id}.tit_1"
        and _has_class(title_element, "eli-title")
    ):
        raise MdrLoaderError(
            f"Article {article_id!r} has an unexpected heading structure."
        )

    article_value = match.group(1)
    article_title = _normalized_text(
        f"{_element_text(label_element)} — {_element_text(title_element)}"
    )
    article_locator = _locator(outer_locator, "article", article_value)
    builder.add(
        structural_id=article_id,
        source_block_id=article_id,
        block_type="Article",
        parent_structural_id=parent_id,
        hierarchy=outer_hierarchy,
        legal_locator=article_locator,
        raw_html=_raw_html_group([label_element, title_element]),
        searchable_text=article_title,
        annotations=_annotations_for(
            label_element,
            builder.source,
        )
        + _annotations_for(title_element, builder.source),
    )
    article_entry = builder.hierarchy_entry(
        level=3,
        kind="Article",
        structural_id=article_id,
        title=article_title,
    )
    content_hierarchy = outer_hierarchy + (article_entry,)

    current_paragraph_id: str | None = None
    current_paragraph_locator = article_locator
    subparagraph_number = 0

    for child in children[2:]:
        if child.tag == "table" and _is_genuine_table(child):
            _emit_genuine_table(
                builder,
                child,
                parent_id=current_paragraph_id or article_id,
                hierarchy=content_hierarchy,
                legal_locator=current_paragraph_locator,
            )
            continue
        if _is_grid_item(child):
            _emit_grid_item(
                builder,
                child,
                parent_id=current_paragraph_id or article_id,
                hierarchy=content_hierarchy,
                legal_locator=current_paragraph_locator,
            )
            continue

        paragraph_label = _numbered_paragraph_label(child)
        if paragraph_label is not None:
            paragraph_locator = _locator(
                article_locator,
                "paragraph",
                paragraph_label,
            )
            emitted_id = _emit_content(
                builder,
                child,
                parent_id=article_id,
                hierarchy=content_hierarchy,
                legal_locator=paragraph_locator,
                block_type="Paragraph",
            )
            if emitted_id is None:
                raise MdrLoaderError(
                    f"Article {article_id} paragraph {paragraph_label} is empty."
                )
            current_paragraph_id = emitted_id
            current_paragraph_locator = paragraph_locator
            subparagraph_number = 0
            continue

        is_marker = _has_class(child, "modref") or _has_class(
            child,
            "arrow",
        )
        parent = current_paragraph_id or article_id
        locator = current_paragraph_locator
        block_type = "AmendmentMarker" if is_marker else "Subparagraph"
        if not is_marker:
            subparagraph_number += 1
            locator = _locator(
                current_paragraph_locator,
                "subparagraph",
                str(subparagraph_number),
            )
        _emit_content(
            builder,
            child,
            parent_id=parent,
            hierarchy=content_hierarchy,
            legal_locator=locator,
            block_type=block_type,
        )


def _build_section(
    builder: _SourceBuilder,
    section: Any,
    *,
    chapter_id: str,
    chapter_hierarchy: tuple[HierarchyEntry, ...],
    chapter_locator: tuple[LegalLocatorPart, ...],
) -> None:
    """Normalize one optional chapter section and its articles."""

    section_id = section.get("id")
    match = _SECTION_ID_PATTERN.fullmatch(section_id or "")
    if match is None:
        raise MdrLoaderError(f"Invalid section container: {section_id!r}.")
    first, second = _header_elements(
        section,
        "title-division-1",
        "title-division-2",
    )
    section_value = match.group(2)
    section_title = _normalized_text(
        f"{_element_text(first)} — {_element_text(second)}"
    )
    section_locator = _locator(chapter_locator, "section", section_value)
    builder.add(
        structural_id=section_id,
        source_block_id=section_id,
        block_type="Section",
        parent_structural_id=chapter_id,
        hierarchy=chapter_hierarchy,
        legal_locator=section_locator,
        raw_html=_raw_html_group([first, second]),
        searchable_text=section_title,
        annotations=_annotations_for(first, builder.source)
        + _annotations_for(second, builder.source),
    )
    section_entry = builder.hierarchy_entry(
        level=2,
        kind="Section",
        structural_id=section_id,
        title=section_title,
    )
    hierarchy = chapter_hierarchy + (section_entry,)

    significant = [
        child
        for child in section
        if _element_text(child) or child.get("id")
    ]
    for child in significant[2:]:
        if _ARTICLE_ID_PATTERN.fullmatch(child.get("id") or ""):
            _build_article(
                builder,
                child,
                parent_id=section_id,
                outer_hierarchy=hierarchy,
                outer_locator=section_locator,
            )
        else:
            raise MdrLoaderError(
                f"Unexpected substantive child in {section_id!r}."
            )


def _build_chapter(
    builder: _SourceBuilder,
    chapter: Any,
    *,
    enacting_id: str,
) -> None:
    """Normalize one regulation chapter, optional sections, and articles."""

    chapter_id = chapter.get("id")
    match = _CHAPTER_ID_PATTERN.fullmatch(chapter_id or "")
    if match is None:
        raise MdrLoaderError(f"Invalid chapter container: {chapter_id!r}.")
    first, second = _header_elements(
        chapter,
        "title-division-1",
        "title-division-2",
    )
    chapter_value = match.group(1)
    chapter_title = _normalized_text(
        f"{_element_text(first)} — {_element_text(second)}"
    )
    chapter_locator = _locator((), "chapter", chapter_value)
    builder.add(
        structural_id=chapter_id,
        source_block_id=chapter_id,
        block_type="Chapter",
        parent_structural_id=enacting_id,
        hierarchy=(),
        legal_locator=chapter_locator,
        raw_html=_raw_html_group([first, second]),
        searchable_text=chapter_title,
        annotations=_annotations_for(first, builder.source)
        + _annotations_for(second, builder.source),
    )
    chapter_entry = builder.hierarchy_entry(
        level=1,
        kind="Chapter",
        structural_id=chapter_id,
        title=chapter_title,
    )
    chapter_hierarchy = (chapter_entry,)

    significant = [
        child
        for child in chapter
        if _element_text(child) or child.get("id")
    ]
    for child in significant[2:]:
        child_id = child.get("id") or ""
        if _SECTION_ID_PATTERN.fullmatch(child_id):
            _build_section(
                builder,
                child,
                chapter_id=chapter_id,
                chapter_hierarchy=chapter_hierarchy,
                chapter_locator=chapter_locator,
            )
        elif _ARTICLE_ID_PATTERN.fullmatch(child_id):
            _build_article(
                builder,
                child,
                parent_id=chapter_id,
                outer_hierarchy=chapter_hierarchy,
                outer_locator=chapter_locator,
            )
        elif _has_class(child, "modref") or _has_class(child, "arrow"):
            _emit_content(
                builder,
                child,
                parent_id=chapter_id,
                hierarchy=chapter_hierarchy,
                legal_locator=chapter_locator,
                block_type="AmendmentMarker",
            )
        else:
            raise MdrLoaderError(
                f"Unexpected substantive child in {chapter_id!r}."
            )


def _annex_heading_level(element: Any) -> int | None:
    """Return the declared EUR-Lex annex heading level."""

    match = _ANNEX_HEADING_CLASS_PATTERN.search(element.get("class") or "")
    return int(match.group(1)) if match is not None else None


def _annex_heading_locator(
    text: str,
) -> tuple[str, str]:
    """Classify one combined annex heading without inventing semantics."""

    chapter_match = re.match(r"^CHAPTER\s+([^\s—]+)", text, re.IGNORECASE)
    if chapter_match:
        return "annex_chapter", chapter_match.group(1)
    part_match = re.match(r"^PART\s+([^\s—]+)", text, re.IGNORECASE)
    if part_match:
        return "annex_part", part_match.group(1)
    number_match = re.match(
        r"^([0-9]+(?:\.[0-9]+)*[a-z]?)(?:\.)?(?:\s|$)",
        text,
        re.IGNORECASE,
    )
    if number_match:
        return "annex_section", number_match.group(1)
    return "annex_heading", text


def _emit_annex_item(
    builder: _SourceBuilder,
    table_element: Any,
    *,
    parent_id: str,
    hierarchy: tuple[HierarchyEntry, ...],
    legal_locator: tuple[LegalLocatorPart, ...],
) -> tuple[str, tuple[LegalLocatorPart, ...]]:
    """Normalize one two-column numbered annex layout table."""

    rows = table_element.xpath('./tbody/tr | ./tr')
    if len(rows) != 1:
        raise MdrLoaderError("An annex layout item must contain one row.")
    cells = rows[0].xpath('./th | ./td')
    if len(cells) != 2:
        raise MdrLoaderError("An annex layout item must contain two cells.")

    label = _element_text(cells[0])
    item_value = _clean_number_label(label)
    item_locator = _locator(legal_locator, "annex_point", item_value)
    own_text = _searchable_text(
        cells[1],
        exclude_grid_items=True,
        exclude_tables=True,
    )
    structural_id = builder.derived_id(parent_id, "annex-item")
    builder.add(
        structural_id=structural_id,
        source_block_id=table_element.get("id"),
        block_type="AnnexPoint",
        parent_structural_id=parent_id,
        hierarchy=hierarchy,
        legal_locator=item_locator,
        raw_html=_raw_html(table_element),
        searchable_text=_normalized_text(f"{label} {own_text}"),
        annotations=_annotations_for(
            cells[1],
            builder.source,
            exclude_grid_items=True,
            exclude_tables=True,
        ),
    )
    for grid_item in _outer_grid_items(cells[1]):
        _emit_grid_item(
            builder,
            grid_item,
            parent_id=structural_id,
            hierarchy=hierarchy,
            legal_locator=item_locator,
        )
    for nested_table in _outer_tables(cells[1]):
        if _is_genuine_table(nested_table):
            _emit_genuine_table(
                builder,
                nested_table,
                parent_id=structural_id,
                hierarchy=hierarchy,
                legal_locator=item_locator,
            )
        else:
            _emit_annex_item(
                builder,
                nested_table,
                parent_id=structural_id,
                hierarchy=hierarchy,
                legal_locator=item_locator,
            )
    return structural_id, item_locator


def _build_annex(
    builder: _SourceBuilder,
    annex: Any,
    *,
    annexes_id: str,
) -> None:
    """Normalize one annex, its heading path, points, and legal tables."""

    annex_id = annex.get("id")
    match = _ANNEX_ID_PATTERN.fullmatch(annex_id or "")
    if match is None:
        raise MdrLoaderError(f"Invalid annex container: {annex_id!r}.")
    annex_value = match.group(1)
    children = list(annex)
    annex_label = next(
        (
            child
            for child in children
            if _has_class(child, "title-annex-1")
        ),
        None,
    )
    annex_title = next(
        (
            child
            for child in children
            if _annex_heading_level(child) == 1
        ),
        None,
    )
    if annex_label is None or annex_title is None:
        raise MdrLoaderError(f"Annex {annex_id!r} has no complete title.")

    title_elements = [annex_label, annex_title]
    title = _normalized_text(
        f"{_element_text(annex_label)} — {_element_text(annex_title)}"
    )
    annex_locator = _locator((), "annex", annex_value)
    builder.add(
        structural_id=annex_id,
        source_block_id=annex_id,
        block_type="Annex",
        parent_structural_id=annexes_id,
        hierarchy=(),
        legal_locator=annex_locator,
        raw_html=_raw_html_group(title_elements),
        searchable_text=title,
        annotations=_annotations_for(annex_title, builder.source),
    )
    annex_entry = builder.hierarchy_entry(
        level=1,
        kind="Annex",
        structural_id=annex_id,
        title=title,
    )

    start_index = children.index(annex_title) + 1
    active_headings: dict[int, tuple[str, HierarchyEntry, LegalLocatorPart]] = {}
    current_item_id: str | None = None
    current_item_locator = annex_locator
    continuation_number = 0
    index = start_index

    while index < len(children):
        child = children[index]
        heading_level = _annex_heading_level(child)

        if heading_level is not None:
            group = [child]
            next_index = index + 1
            while (
                next_index < len(children)
                and _annex_heading_level(children[next_index]) == heading_level
            ):
                group.append(children[next_index])
                next_index += 1

            heading_text = " — ".join(_element_text(item) for item in group)
            previous_levels = sorted(
                level
                for level in active_headings
                if level < heading_level
            )
            parent_id = (
                active_headings[previous_levels[-1]][0]
                if previous_levels
                else annex_id
            )
            outer_entries = tuple(
                active_headings[level][1]
                for level in previous_levels
            )
            outer_locator = annex_locator + tuple(
                active_headings[level][2]
                for level in previous_levels
            )
            locator_kind, locator_value = _annex_heading_locator(
                heading_text
            )
            heading_locator_part = LegalLocatorPart(
                kind=locator_kind,
                value=locator_value,
            )
            structural_id = builder.derived_id(
                parent_id,
                f"annex-heading-{heading_level}",
            )
            hierarchy = (annex_entry,) + outer_entries
            builder.add(
                structural_id=structural_id,
                source_block_id=None,
                block_type="AnnexHeading",
                parent_structural_id=parent_id,
                hierarchy=hierarchy,
                legal_locator=outer_locator + (heading_locator_part,),
                raw_html=_raw_html_group(group),
                searchable_text=heading_text,
                annotations=tuple(
                    annotation
                    for item in group
                    for annotation in _annotations_for(item, builder.source)
                ),
            )
            entry = builder.hierarchy_entry(
                level=heading_level + 1,
                kind="AnnexHeading",
                structural_id=structural_id,
                title=heading_text,
            )
            for level in tuple(active_headings):
                if level >= heading_level:
                    del active_headings[level]
            active_headings[heading_level] = (
                structural_id,
                entry,
                heading_locator_part,
            )
            current_item_id = None
            current_item_locator = annex_locator
            continuation_number = 0
            index = next_index
            continue

        if child.tag == "hr" or (
            not _element_text(child)
            and not _annotations_for(child, builder.source)
        ):
            index += 1
            continue

        active_levels = sorted(active_headings)
        content_parent = (
            active_headings[active_levels[-1]][0]
            if active_levels
            else annex_id
        )
        hierarchy = (annex_entry,) + tuple(
            active_headings[level][1]
            for level in active_levels
        )
        content_locator = annex_locator + tuple(
            active_headings[level][2]
            for level in active_levels
        )

        if child.tag == "table":
            if _is_genuine_table(child):
                _emit_genuine_table(
                    builder,
                    child,
                    parent_id=content_parent,
                    hierarchy=hierarchy,
                    legal_locator=content_locator,
                )
                current_item_id = None
                current_item_locator = content_locator
            else:
                current_item_id, current_item_locator = _emit_annex_item(
                    builder,
                    child,
                    parent_id=content_parent,
                    hierarchy=hierarchy,
                    legal_locator=content_locator,
                )
            continuation_number = 0
            index += 1
            continue

        if _is_grid_item(child):
            _emit_grid_item(
                builder,
                child,
                parent_id=current_item_id or content_parent,
                hierarchy=hierarchy,
                legal_locator=current_item_locator,
            )
            index += 1
            continue

        nested_tables = _outer_tables(child)
        if nested_tables:
            wrapper_text = _searchable_text(
                child,
                exclude_grid_items=True,
                exclude_tables=True,
            )
            wrapper_annotations = _annotations_for(
                child,
                builder.source,
                exclude_grid_items=True,
                exclude_tables=True,
            )
            table_parent = content_parent
            if wrapper_text or wrapper_annotations:
                continuation_number += 1
                wrapper_locator = _locator(
                    content_locator,
                    "subparagraph",
                    str(continuation_number),
                )
                table_parent = builder.derived_id(
                    content_parent,
                    "table-container",
                )
                builder.add(
                    structural_id=table_parent,
                    source_block_id=child.get("id"),
                    block_type="TableContainer",
                    parent_structural_id=content_parent,
                    hierarchy=hierarchy,
                    legal_locator=wrapper_locator,
                    raw_html=_raw_html(child),
                    searchable_text=wrapper_text,
                    annotations=wrapper_annotations,
                )

            for nested_table in nested_tables:
                if _is_genuine_table(nested_table):
                    _emit_genuine_table(
                        builder,
                        nested_table,
                        parent_id=table_parent,
                        hierarchy=hierarchy,
                        legal_locator=content_locator,
                    )
                else:
                    current_item_id, current_item_locator = (
                        _emit_annex_item(
                            builder,
                            nested_table,
                            parent_id=table_parent,
                            hierarchy=hierarchy,
                            legal_locator=content_locator,
                        )
                    )
            index += 1
            continue

        is_marker = _has_class(child, "modref") or _has_class(
            child,
            "arrow",
        )
        locator = current_item_locator
        block_type = "AmendmentMarker" if is_marker else "Subparagraph"
        if not is_marker:
            continuation_number += 1
            locator = _locator(
                current_item_locator,
                "subparagraph",
                str(continuation_number),
            )
        _emit_content(
            builder,
            child,
            parent_id=current_item_id or content_parent,
            hierarchy=hierarchy,
            legal_locator=locator,
            block_type=block_type,
        )
        index += 1


def _validate_footnote_links(legal_container: Any, document: Any) -> list[Any]:
    """Resolve all footnote references before building canonical targets."""

    definitions = document.xpath('./p[@class="footnote"]')
    definition_ids: set[str] = set()
    for definition in definitions:
        anchors = definition.xpath('./a[starts-with(@id,"E")]')
        if len(anchors) != 1:
            raise MdrLoaderError(
                "Every footnote definition must have one E-prefixed anchor."
            )
        identifier = anchors[0].get("id")
        if identifier in definition_ids:
            raise MdrLoaderError(
                f"Duplicate footnote definition: {identifier!r}."
            )
        definition_ids.add(identifier)

    for reference in legal_container.xpath('.//a[starts-with(@id,"src.E")]'):
        expected = reference.get("id").removeprefix("src.")
        fragment = urlparse(reference.get("href") or "").fragment
        if fragment != expected or expected not in definition_ids:
            raise MdrLoaderError(
                f"Unresolved footnote reference: {reference.get('id')!r}."
            )
    return definitions


def _build_document(
    root: Any,
    source: SourceManifestEntry,
) -> LoadedSource:
    """Normalize the consolidated regulation into canonical source blocks."""

    document = _single_element(
        root,
        '//*[@id="docHtml"]',
        "#docHtml legal document",
    )
    _validate_unique_ids(document)
    legal_container = _single_element(
        document,
        './div['
        + _CLASS_TOKEN_XPATH.format(token="eli-container")
        + "]",
        ".eli-container legal-content element",
    )
    footnote_definitions = _validate_footnote_links(
        legal_container,
        document,
    )

    title = _single_element(
        legal_container,
        './div[@id="tit_1"]',
        "regulation title",
    )
    enacting = _single_element(
        legal_container,
        './div[@id="enc_1"]',
        "enacting-terms container",
    )
    concluding = _single_element(
        legal_container,
        './div[@id="fnp_1"]',
        "concluding-formula container",
    )
    annexes = [
        child
        for child in legal_container
        if _ANNEX_ID_PATTERN.fullmatch(child.get("id") or "")
    ]
    if not annexes:
        raise MdrLoaderError("No substantive MDR annexes were found.")

    builder = _SourceBuilder(source)
    builder.add(
        structural_id=_ROOT_STRUCTURAL_ID,
        source_block_id=None,
        block_type="Document",
        parent_structural_id=None,
        hierarchy=(),
        legal_locator=(),
        raw_html="",
        searchable_text="",
    )
    builder.add(
        structural_id="tit_1",
        source_block_id="tit_1",
        block_type="DocumentTitle",
        parent_structural_id=_ROOT_STRUCTURAL_ID,
        hierarchy=(),
        legal_locator=(),
        raw_html=_raw_html(title),
        searchable_text=_searchable_text(title),
        annotations=_annotations_for(title, source),
    )
    builder.add(
        structural_id="enc_1",
        source_block_id="enc_1",
        block_type="EnactingTerms",
        parent_structural_id=_ROOT_STRUCTURAL_ID,
        hierarchy=(),
        legal_locator=(),
        raw_html="",
        searchable_text="",
    )

    chapters = [
        child
        for child in enacting
        if _CHAPTER_ID_PATTERN.fullmatch(child.get("id") or "")
    ]
    if not chapters:
        raise MdrLoaderError("No MDR chapters were found.")
    for chapter in chapters:
        _build_chapter(builder, chapter, enacting_id="enc_1")

    builder.add(
        structural_id="fnp_1",
        source_block_id="fnp_1",
        block_type="ConcludingFormula",
        parent_structural_id=_ROOT_STRUCTURAL_ID,
        hierarchy=(),
        legal_locator=(),
        raw_html=_raw_html(concluding),
        searchable_text=_searchable_text(concluding),
        annotations=_annotations_for(concluding, source),
    )
    builder.add(
        structural_id="annexes",
        source_block_id=None,
        block_type="Annexes",
        parent_structural_id=_ROOT_STRUCTURAL_ID,
        hierarchy=(),
        legal_locator=(),
        raw_html="",
        searchable_text="",
    )
    for annex in annexes:
        _build_annex(builder, annex, annexes_id="annexes")

    if footnote_definitions:
        builder.add(
            structural_id="footnotes",
            source_block_id=None,
            block_type="Footnotes",
            parent_structural_id=_ROOT_STRUCTURAL_ID,
            hierarchy=(),
            legal_locator=(),
            raw_html="",
            searchable_text="",
        )
        for definition in footnote_definitions:
            anchor = definition.xpath('./a[starts-with(@id,"E")]')[0]
            identifier = anchor.get("id")
            structural_id = f"footnotes/{identifier}"
            text = _searchable_text(definition)
            builder.add(
                structural_id=structural_id,
                source_block_id=identifier,
                block_type="FootnoteDefinition",
                parent_structural_id="footnotes",
                hierarchy=(),
                legal_locator=_locator((), "footnote", identifier),
                raw_html=_raw_html(definition),
                searchable_text=text,
                annotations=(
                    SourceAnnotation(
                        kind=AnnotationKind.FOOTNOTE_DEFINITION,
                        identifier=identifier,
                        target_id=None,
                        raw_html=_raw_html(definition),
                        text=text,
                    ),
                ),
            )

    return builder.finalize()


def load_mdr_source(
    manifest: CorpusManifest,
    document_id: str,
    *,
    project_root: str | Path,
) -> LoadedSource:
    """Load one manifest-listed, validated consolidated MDR HTML source."""

    if not isinstance(manifest, CorpusManifest):
        raise MdrLoaderError("manifest must be a CorpusManifest.")
    if not isinstance(document_id, str) or not document_id.strip():
        raise MdrLoaderError("document_id must be a non-empty string.")

    try:
        source = manifest.get_source(document_id)
    except KeyError as exc:
        raise MdrLoaderError(
            f"Source is not listed in the manifest: {document_id!r}."
        ) from exc

    if source.validation_status is not ValidationStatus.VALIDATED:
        raise MdrLoaderError(f"Source is not validated: {document_id!r}.")
    if source.source_type is not SourceType.REGULATION:
        raise MdrLoaderError(
            f"Source is not the regulation: {document_id!r}."
        )
    if source.content_format is not ContentFormat.EURLEX_HTML:
        raise MdrLoaderError(
            f"Regulation does not use eurlex_html: {document_id!r}."
        )

    source_path = _resolve_verified_source_path(source, project_root)
    root = _parse_html(source_path)
    return _build_document(root, source)


__all__ = [
    "MdrLoaderError",
    "load_mdr_source",
]
