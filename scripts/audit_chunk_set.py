"""Build and audit the complete EU MDR ChunkSet without persisting it.

This script is intentionally read-only. It verifies the frozen manifest and
source files, loads all normalized sources, builds one in-memory ChunkSet, and
checks corpus-wide traceability before indexing work begins.

The current corpus has no source block or table that requires splitting. The
temporary REJECT policy therefore makes any unexpected oversized block fail
closed. It is not the final approved sentence-first/whitespace split policy.
"""

from __future__ import annotations

from collections import Counter
from itertools import groupby
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)


from src.ingestion.chunk_models import (  # noqa: E402
    ChunkingConfig,
    OverlapPolicy,
    OversizedTextPolicy,
    SizeUnit,
    TableSplitPolicy,
)
from src.ingestion.chunk_set_builder import build_chunk_set  # noqa: E402
from src.ingestion.chunker import (  # noqa: E402
    BlockRoute,
    route_blocks,
)
from src.ingestion.corpus_loader import load_corpus  # noqa: E402
from src.ingestion.manifest import load_manifest  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"

# This version label is deliberately audit-only. It prevents the resulting
# identity from being mistaken for a final production chunking configuration.
AUDIT_CONFIG = ChunkingConfig(
    schema_version="1.0",
    algorithm_version="deterministic-grouping-audit-v1",
    target_size=4_000,
    hard_max_size=6_000,
    size_unit=SizeUnit.UNICODE_CHARACTERS,
    overlap_policy=OverlapPolicy.NONE,
    table_split_policy=(
        TableSplitPolicy.RETAIN_WHOLE_OR_ROW_GROUPS_WITH_REPEATED_HEADERS
    ),
    oversized_text_policy=OversizedTextPolicy.REJECT,
    include_hierarchy_in_text=True,
)

RETRIEVABLE_ROUTES = frozenset(
    {
        BlockRoute.BODY,
        BlockRoute.TABLE,
        BlockRoute.CAPTION,
    }
)


class ChunkSetAuditError(RuntimeError):
    """Raised when the built ChunkSet violates an audit invariant."""


def require(condition: bool, message: str) -> None:
    """Fail the audit with an actionable message when condition is false."""

    if not condition:
        raise ChunkSetAuditError(message)


def annotation_counts(annotations) -> Counter[str]:
    """Count annotation values without relying on their hash implementation."""

    return Counter(repr(annotation) for annotation in annotations)


def main() -> None:
    """Build the in-memory ChunkSet, verify it, and print compact metrics."""

    manifest = load_manifest(
        MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        verify_files=True,
    )
    loaded_sources = load_corpus(
        manifest,
        project_root=PROJECT_ROOT,
    )
    chunk_set = build_chunk_set(
        manifest,
        loaded_sources,
        AUDIT_CONFIG,
    )

    chunks = chunk_set.chunks
    chunks_by_document = {
        source.document_id: tuple(
            chunk
            for chunk in chunks
            if chunk.source.document_id == source.document_id
        )
        for source in manifest.sources
    }

    expected_document_order = tuple(
        source.document_id
        for source in manifest.sources
    )
    actual_document_order = tuple(
        document_id
        for document_id, _ in groupby(
            chunk.source.document_id
            for chunk in chunks
        )
    )
    require(
        actual_document_order == expected_document_order,
        "Chunk source order does not match authoritative manifest order.",
    )

    total_expected_blocks = 0
    total_selected_blocks = 0
    total_source_annotations = 0
    total_chunk_annotations = 0
    total_source_tables = 0
    total_table_segments = 0
    marker_blocks = 0
    annotated_marker_blocks = 0
    marker_annotations = 0

    source_rows: list[str] = []

    for loaded_source in loaded_sources:
        document_id = loaded_source.source.document_id
        source_chunks = chunks_by_document[document_id]
        routed = route_blocks(loaded_source.blocks)

        expected_blocks = tuple(
            item.block.block_id
            for item in routed
            if (
                item.route in RETRIEVABLE_ROUTES
                and item.rendered_text is not None
            )
        )
        selected_blocks = tuple(
            selection.block_id
            for chunk in source_chunks
            for selection in chunk.block_selections
        )
        require(
            selected_blocks == expected_blocks,
            "Selected block coverage or order differs from routed evidence: "
            f"document_id={document_id!r}.",
        )

        expected_tables = tuple(
            item.block.block_id
            for item in routed
            if (
                item.route is BlockRoute.TABLE
                and item.rendered_text is not None
            )
        )
        table_segments = tuple(
            segment
            for chunk in source_chunks
            for segment in chunk.table_segments
        )
        require(
            tuple(segment.block_id for segment in table_segments)
            == expected_tables,
            "Table segment coverage or order differs from routed tables: "
            f"document_id={document_id!r}.",
        )
        require(
            all(
                segment.is_complete_table
                and segment.source_row_start == 0
                and segment.source_row_end == len(segment.table.rows)
                for segment in table_segments
            ),
            "The unsplit audit requires every table segment to be complete: "
            f"document_id={document_id!r}.",
        )

        source_annotations = tuple(
            annotation
            for block in loaded_source.blocks
            for annotation in block.annotations
        )
        chunk_annotations = tuple(
            annotation
            for chunk in source_chunks
            for annotation in chunk.annotations
        )
        require(
            annotation_counts(chunk_annotations)
            == annotation_counts(source_annotations),
            "Annotations were lost, duplicated, or changed: "
            f"document_id={document_id!r}.",
        )

        require(
            tuple(chunk.order for chunk in source_chunks)
            == tuple(range(len(source_chunks))),
            "Per-source chunk order is not contiguous: "
            f"document_id={document_id!r}.",
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
            require(
                chunk.previous_chunk_id == expected_previous
                and chunk.next_chunk_id == expected_next,
                "Neighbor links are not source-local and contiguous: "
                f"chunk_id={chunk.chunk_id!r}.",
            )

        source_marker_blocks = tuple(
            block
            for block in loaded_source.blocks
            if block.block_type == "AmendmentMarker"
        )
        selected_types = {
            item.block.block_id: item.block.block_type
            for item in routed
            if item.block.block_id in selected_blocks
        }
        require(
            "AmendmentMarker" not in selected_types.values(),
            "An AmendmentMarker was incorrectly selected as evidence: "
            f"document_id={document_id!r}.",
        )

        total_expected_blocks += len(expected_blocks)
        total_selected_blocks += len(selected_blocks)
        total_source_annotations += len(source_annotations)
        total_chunk_annotations += len(chunk_annotations)
        total_source_tables += len(expected_tables)
        total_table_segments += len(table_segments)
        marker_blocks += len(source_marker_blocks)
        annotated_marker_blocks += sum(
            bool(block.annotations)
            for block in source_marker_blocks
        )
        marker_annotations += sum(
            len(block.annotations)
            for block in source_marker_blocks
        )

        source_rows.append(
            f"{document_id}: blocks={len(loaded_source.blocks)}, "
            f"selected={len(selected_blocks)}, chunks={len(source_chunks)}, "
            f"tables={len(table_segments)}, "
            f"annotations={len(chunk_annotations)}"
        )

    require(
        len({chunk.chunk_id for chunk in chunks}) == len(chunks),
        "Chunk IDs are not globally unique.",
    )
    require(
        all(len(chunk.text) <= AUDIT_CONFIG.hard_max_size for chunk in chunks),
        "At least one chunk exceeds the hard maximum size.",
    )
    require(
        all(chunk.text.strip() for chunk in chunks),
        "At least one chunk has empty text.",
    )
    require(
        all(
            selection.start_char is None
            and selection.end_char is None
            and selection.table_row_start is None
            and selection.table_row_end is None
            for chunk in chunks
            for selection in chunk.block_selections
        ),
        "The no-splitting audit unexpectedly produced a partial selection.",
    )

    sizes = tuple(len(chunk.text) for chunk in chunks)
    largest_chunk = max(chunks, key=lambda chunk: len(chunk.text))
    above_target = sum(
        size > AUDIT_CONFIG.target_size
        for size in sizes
    )
    chunks_with_pages = sum(bool(chunk.pages) for chunk in chunks)
    chunks_with_hierarchy = sum(bool(chunk.hierarchies) for chunk in chunks)
    chunks_with_locators = sum(bool(chunk.legal_locators) for chunk in chunks)
    chunks_with_annotations = sum(bool(chunk.annotations) for chunk in chunks)

    print("CONFIGURATION")
    print(f"algorithm_version: {AUDIT_CONFIG.algorithm_version}")
    print(f"target_size: {AUDIT_CONFIG.target_size}")
    print(f"hard_max_size: {AUDIT_CONFIG.hard_max_size}")
    print(
        "oversized_text_policy: reject "
        "(temporary audit behavior; production splitter not yet implemented)"
    )
    print()

    print("IDENTITIES")
    print(f"corpus_identity: {chunk_set.corpus_identity}")
    print(f"chunking_config_id: {chunk_set.chunking_config_id}")
    print(f"chunk_set_id: {chunk_set.chunk_set_id}")
    print()

    print("PER SOURCE")
    for row in source_rows:
        print(row)
    print()

    print("CORPUS SUMMARY")
    print(f"sources: {len(loaded_sources)}")
    print(f"chunks: {len(chunks)}")
    print(f"selected_blocks: {total_selected_blocks}/{total_expected_blocks}")
    print(f"table_segments: {total_table_segments}/{total_source_tables}")
    print(
        "annotations: "
        f"{total_chunk_annotations}/{total_source_annotations}"
    )
    print(
        "amendment_markers_with_annotations: "
        f"{annotated_marker_blocks}/{marker_blocks}"
    )
    print(f"amendment_annotations_attached: {marker_annotations}")
    print(f"minimum_chunk_characters: {min(sizes)}")
    print(f"maximum_chunk_characters: {max(sizes)}")
    print(f"chunks_above_target: {above_target}")
    print(f"chunks_above_hard_max: {sum(size > 6_000 for size in sizes)}")
    print(f"chunks_with_pages: {chunks_with_pages}")
    print(f"chunks_with_hierarchy: {chunks_with_hierarchy}")
    print(f"chunks_with_legal_locators: {chunks_with_locators}")
    print(f"chunks_with_annotations: {chunks_with_annotations}")
    print(
        "largest_chunk: "
        f"document_id={largest_chunk.source.document_id!r}, "
        f"order={largest_chunk.order}, "
        f"characters={len(largest_chunk.text)}"
    )
    print()

    print("AUDIT PASSED")
    print("Manifest order, complete evidence-block coverage, table coverage,")
    print("annotation preservation, chunk sizes, and neighbor links are valid.")


if __name__ == "__main__":
    main()
