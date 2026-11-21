"""
Read-only audit of block types that need chunk-routing decisions.

This script loads the validated corpus through the approved ingestion
boundary. It does not modify source or repository files.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow standalone scripts to import the local src package when executed
# as `python scripts/audit_chunk_routing.py`.
project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)


from src.ingestion.corpus_loader import load_corpus
from src.ingestion.manifest import load_manifest


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"

SPECIAL_BLOCK_TYPES = (
    "Caption",
    "Footnote",
    "FootnoteDefinition",
    "AmendmentMarker",
    "ConcludingFormula",
    "DocumentTitle",
    "Chapter",
    "Section",
    "Article",
    "Annex",
    "AnnexHeading",
    "SectionHeader",
    "TableOfContents",
)

EXAMPLES_PER_TYPE = 3
SNIPPET_LENGTH = 180


def compact_text(value: str, limit: int = SNIPPET_LENGTH) -> str:
    """Collapse whitespace and truncate text for safe terminal output."""

    compact = " ".join(value.split())

    if len(compact) <= limit:
        return compact

    return f"{compact[:limit]}…"


def short_block_id(block_id: str | None) -> str:
    """Return the structural suffix of a canonical block ID."""

    if block_id is None:
        return "none"

    return block_id.rsplit("::", maxsplit=1)[-1]


def hierarchy_text(block) -> str:
    """Render non-empty hierarchy titles for inspection."""

    titles = [
        compact_text(entry.title, 100)
        for entry in block.hierarchy
        if entry.title.strip()
    ]

    return " > ".join(titles) if titles else "none"


def locator_text(block) -> str:
    """Render the ordered legal locator."""

    if not block.legal_locator:
        return "none"

    return " > ".join(
        f"{part.kind}={part.value}"
        for part in block.legal_locator
    )


def annotation_text(block) -> str:
    """Render compact annotation information."""

    if not block.annotations:
        return "none"

    return " | ".join(
        (
            f"{annotation.kind.value}"
            f"(id={annotation.identifier!r}, "
            f"target={short_block_id(annotation.target_id)}, "
            f"text={compact_text(annotation.text, 100)!r})"
        )
        for annotation in block.annotations
    )


def main() -> None:
    manifest = load_manifest(
        MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        verify_files=True,
    )
    loaded_sources = load_corpus(
        manifest,
        project_root=PROJECT_ROOT,
    )

    blocks_by_type = {
        block_type: []
        for block_type in SPECIAL_BLOCK_TYPES
    }

    all_type_counts: Counter[str] = Counter()

    for loaded_source in loaded_sources:
        for block in loaded_source.blocks:
            all_type_counts[block.block_type] += 1

            if block.block_type in blocks_by_type:
                blocks_by_type[block.block_type].append(
                    (loaded_source.source.document_id, block)
                )

    print("CORPUS")
    print(f"corpus_id: {manifest.corpus_id}")
    print(f"sources: {len(loaded_sources)}")
    print()

    print("ALL BLOCK TYPES")
    for block_type, count in sorted(all_type_counts.items()):
        print(f"{block_type}: {count}")
    print()

    for block_type in SPECIAL_BLOCK_TYPES:
        examples = blocks_by_type[block_type]

        print("=" * 80)
        print(f"{block_type}: {len(examples)} blocks")
        print("=" * 80)

        for document_id, block in examples[:EXAMPLES_PER_TYPE]:
            print(
                f"source={document_id} "
                f"order={block.order} "
                f"text_chars={len(block.searchable_text)}"
            )
            print(
                f"block_id={short_block_id(block.block_id)} "
                f"parent={short_block_id(block.parent_id)} "
                f"children={len(block.child_ids)}"
            )
            print(f"hierarchy={hierarchy_text(block)}")
            print(f"locator={locator_text(block)}")
            print(f"annotations={annotation_text(block)}")
            print(
                f"has_table={block.table is not None} "
                f"text={compact_text(block.searchable_text)!r}"
            )
            print()

        if not examples:
            print("No examples found.")
            print()

    amendment_markers = blocks_by_type["AmendmentMarker"]
    markers_with_annotations = sum(
        bool(block.annotations)
        for _, block in amendment_markers
    )

    captions = blocks_by_type["Caption"]
    captions_with_text = sum(
        bool(block.searchable_text.strip())
        for _, block in captions
    )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        "amendment_markers_with_annotations: "
        f"{markers_with_annotations}/{len(amendment_markers)}"
    )
    print(
        "captions_with_searchable_text: "
        f"{captions_with_text}/{len(captions)}"
    )


if __name__ == "__main__":
    main()