"""
Audit all RAG-ready MDCG JSON files.

This script checks whether the prepared JSON documents follow a consistent
tree structure before we create the production ingestion loader.

The audit is READ-ONLY:
- It does not modify the JSON files.
- It does not rename files.
- It does not create chunks.
- It does not create embeddings.

Run from the project root with:

    python scripts/audit_rag_ready_json.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# __file__ is this Python script.
# .parent is the "scripts" folder.
# .parent.parent is therefore the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the prepared JSON source documents.
RAG_READY_DIR = PROJECT_ROOT / "data" / "rag_ready"

# Maximum number of example problems printed for each category.
# This prevents the terminal output from becoming enormous.
MAX_EXAMPLES = 5


# Marker block IDs often include a path similar to:
#     /page/12/Text/5
#
# We only detect and report this value. We do not yet decide whether it is
# zero-based or whether it matches the visible PDF page number.
PAGE_INDEX_PATTERN = re.compile(r"/page/(\d+)(?:/|$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def walk_nodes(
    node: dict[str, Any],
    location: str = "root",
) -> Iterator[tuple[dict[str, Any], str]]:
    """
    Yield every dictionary node in a Marker JSON tree.

    Parameters
    ----------
    node:
        Current JSON node.

    location:
        Internal diagnostic path showing where the node was found.
        This is not a regulatory page or section locator.

    Yields
    ------
    tuple[dict, str]
        The node and its diagnostic location.

    Notes
    -----
    Marker JSON is hierarchical. Each node can contain a "children" list,
    and each child can contain more children. This function walks through
    that complete hierarchy.
    """

    yield node, location

    children = node.get("children")

    # A missing or null children field means that this node has no
    # traversable children. Its validity is assessed elsewhere.
    if not isinstance(children, list):
        return

    for child_index, child in enumerate(children):
        child_location = f"{location}.children[{child_index}]"

        # Invalid child values are reported by the audit function.
        # We cannot traverse them as nodes.
        if not isinstance(child, dict):
            continue

        yield from walk_nodes(child, child_location)


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def clean_block_type(value: Any) -> str:
    """
    Return a readable block-type label.

    Missing, null, or blank block types are grouped under "<missing>".
    """

    if not isinstance(value, str) or not value.strip():
        return "<missing>"

    return value.strip()


def shorten(value: str, maximum_length: int = 120) -> str:
    """Shorten long values so diagnostic output remains readable."""

    normalized = " ".join(value.split())

    if len(normalized) <= maximum_length:
        return normalized

    return normalized[: maximum_length - 3] + "..."


def add_example(examples: list[str], message: str) -> None:
    """Add an example only while the configured limit has not been reached."""

    if len(examples) < MAX_EXAMPLES:
        examples.append(message)


def count_image_references(images: Any) -> int:
    """
    Count image references without assuming one exact image schema.

    Marker may represent images using a list or a dictionary. We report both
    formats without interpreting the image content.
    """

    if isinstance(images, list):
        return len(images)

    if isinstance(images, dict):
        return len(images)

    return 0


def extract_page_index(block_id: Any) -> int | None:
    """
    Extract a raw page index from a Marker block ID when available.

    No conversion is performed because we have not yet confirmed whether the
    stored index is zero-based or how it relates to displayed PDF page numbers.
    """

    if not isinstance(block_id, str):
        return None

    match = PAGE_INDEX_PATTERN.search(block_id)

    if match is None:
        return None

    return int(match.group(1))


# ---------------------------------------------------------------------------
# Per-document audit
# ---------------------------------------------------------------------------

def audit_json_file(file_path: Path) -> dict[str, Any]:
    """
    Audit one JSON file and return structured statistics.

    Loading errors are captured and returned instead of stopping the complete
    corpus audit. This allows one malformed file to be identified while the
    remaining documents are still inspected.
    """

    result: dict[str, Any] = {
        "filename": file_path.name,
        "load_error": None,
        "structural_issues": [],
        "issue_examples": [],
    }

    try:
        with file_path.open("r", encoding="utf-8") as file:
            document = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result["load_error"] = f"{type(error).__name__}: {error}"
        return result

    if not isinstance(document, dict):
        result["structural_issues"].append(
            "The top-level JSON value is not an object."
        )
        result["top_level_type"] = type(document).__name__
        return result

    result["root_keys"] = sorted(document.keys())
    result["root_block_type"] = clean_block_type(
        document.get("block_type")
    )

    metadata = document.get("metadata")

    if isinstance(metadata, dict):
        result["metadata_keys"] = sorted(metadata.keys())
    else:
        result["metadata_keys"] = []
        result["structural_issues"].append(
            "The root metadata field is missing or is not an object."
        )

    root_children = document.get("children")

    if not isinstance(root_children, list):
        result["structural_issues"].append(
            "The root children field is missing or is not a list."
        )

    # Counters used to understand the complete document structure.
    block_types: Counter[str] = Counter()
    node_key_schemas: Counter[tuple[str, ...]] = Counter()
    hierarchy_types: Counter[str] = Counter()

    block_ids: Counter[str] = Counter()
    raw_page_indexes: set[int] = set()

    total_nodes = 0
    nodes_with_id = 0
    nodes_without_id = 0
    nodes_with_html = 0
    nodes_with_nonempty_html = 0
    nodes_with_empty_html = 0
    nodes_without_html_field = 0
    nodes_with_section_hierarchy = 0
    nodes_with_images = 0
    total_image_references = 0
    invalid_children_fields = 0
    invalid_child_values = 0

    missing_id_examples: list[str] = []
    empty_html_examples: list[str] = []
    invalid_children_examples: list[str] = []
    invalid_child_examples: list[str] = []

    # The root represents the complete document and does not necessarily need
    # its own Marker block ID. Therefore, ID statistics are collected only for
    # the descendants inside root.children.
    if isinstance(root_children, list):
        for top_level_index, top_level_node in enumerate(root_children):
            top_level_location = f"root.children[{top_level_index}]"

            if not isinstance(top_level_node, dict):
                invalid_child_values += 1
                add_example(
                    invalid_child_examples,
                    (
                        f"{top_level_location}: expected object, found "
                        f"{type(top_level_node).__name__}"
                    ),
                )
                continue

            for node, location in walk_nodes(
                top_level_node,
                top_level_location,
            ):
                total_nodes += 1

                # Record which keys occur together on each node.
                # This helps detect schema variants across the corpus.
                node_key_schemas[tuple(sorted(node.keys()))] += 1

                block_type = clean_block_type(node.get("block_type"))
                block_types[block_type] += 1

                # -----------------------------------------------------------
                # Block ID checks
                # -----------------------------------------------------------

                block_id = node.get("id")

                if isinstance(block_id, str) and block_id.strip():
                    normalized_id = block_id.strip()
                    nodes_with_id += 1
                    block_ids[normalized_id] += 1

                    page_index = extract_page_index(normalized_id)

                    if page_index is not None:
                        raw_page_indexes.add(page_index)
                else:
                    nodes_without_id += 1
                    add_example(
                        missing_id_examples,
                        f"{location}: block_type={block_type}",
                    )

                # -----------------------------------------------------------
                # HTML checks
                # -----------------------------------------------------------

                if "html" not in node:
                    nodes_without_html_field += 1
                else:
                    nodes_with_html += 1
                    html = node.get("html")

                    if isinstance(html, str) and html.strip():
                        nodes_with_nonempty_html += 1
                    else:
                        nodes_with_empty_html += 1
                        add_example(
                            empty_html_examples,
                            (
                                f"{location}: block_type={block_type}, "
                                f"id={shorten(str(block_id))}"
                            ),
                        )

                # -----------------------------------------------------------
                # Section hierarchy checks
                # -----------------------------------------------------------

                if "section_hierarchy" in node:
                    hierarchy = node.get("section_hierarchy")

                    if hierarchy not in (None, {}, []):
                        nodes_with_section_hierarchy += 1

                    hierarchy_types[type(hierarchy).__name__] += 1

                # -----------------------------------------------------------
                # Image checks
                # -----------------------------------------------------------

                image_count = count_image_references(node.get("images"))

                if image_count > 0:
                    nodes_with_images += 1
                    total_image_references += image_count

                # -----------------------------------------------------------
                # Children checks
                # -----------------------------------------------------------

                if "children" in node:
                    children = node.get("children")

                    # Marker uses null/None for leaf nodes. This is valid and simply
                    # means that the current block has no child blocks.
                    if children is None:
                        pass

                    # A populated children field must be a list.
                    elif not isinstance(children, list):
                        invalid_children_fields += 1
                        add_example(
                            invalid_children_examples,
                            (
                                f"{location}: children is "
                                f"{type(children).__name__}"
                            ),
                        )
                    else:
                        for child_index, child in enumerate(children):
                            if not isinstance(child, dict):
                                invalid_child_values += 1
                                add_example(
                                    invalid_child_examples,
                                    (
                                        f"{location}.children[{child_index}]: "
                                        f"expected object, found "
                                        f"{type(child).__name__}"
                                    ),
                                )

    duplicate_ids = {
        block_id: count
        for block_id, count in block_ids.items()
        if count > 1
    }

    # These are observations rather than automatic failures.
    # For example, structural container nodes may legitimately have blank HTML.
    result.update(
        {
            "total_nodes": total_nodes,
            "block_types": block_types,
            "node_key_schemas": node_key_schemas,
            "hierarchy_types": hierarchy_types,
            "nodes_with_id": nodes_with_id,
            "nodes_without_id": nodes_without_id,
            "duplicate_ids": duplicate_ids,
            "nodes_with_html": nodes_with_html,
            "nodes_with_nonempty_html": nodes_with_nonempty_html,
            "nodes_with_empty_html": nodes_with_empty_html,
            "nodes_without_html_field": nodes_without_html_field,
            "nodes_with_section_hierarchy": (
                nodes_with_section_hierarchy
            ),
            "nodes_with_images": nodes_with_images,
            "total_image_references": total_image_references,
            "invalid_children_fields": invalid_children_fields,
            "invalid_child_values": invalid_child_values,
            "raw_page_indexes": sorted(raw_page_indexes),
            "missing_id_examples": missing_id_examples,
            "empty_html_examples": empty_html_examples,
            "invalid_children_examples": invalid_children_examples,
            "invalid_child_examples": invalid_child_examples,
        }
    )

    # Conditions that would prevent us from safely building a loader.
    if duplicate_ids:
        result["structural_issues"].append(
            f"{len(duplicate_ids)} duplicated block ID(s) detected."
        )

    if invalid_children_fields:
        result["structural_issues"].append(
            f"{invalid_children_fields} invalid children field(s) detected."
        )

    if invalid_child_values:
        result["structural_issues"].append(
            f"{invalid_child_values} non-object child value(s) detected."
        )

    if nodes_without_id:
        result["structural_issues"].append(
            f"{nodes_without_id} descendant node(s) have no usable ID."
        )

    return result


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------

def print_counter(counter: Counter[Any], indent: str = "  ") -> None:
    """Print a Counter in descending frequency order."""

    if not counter:
        print(f"{indent}(none)")
        return

    for key, count in counter.most_common():
        print(f"{indent}{key}: {count}")


def print_examples(title: str, examples: list[str]) -> None:
    """Print a limited group of diagnostic examples."""

    if not examples:
        return

    print(f"\n  {title}:")

    for example in examples:
        print(f"    - {example}")


def print_document_report(result: dict[str, Any]) -> None:
    """Print the audit report for one document."""

    print("\n" + "=" * 88)
    print(f"FILE: {result['filename']}")
    print("=" * 88)

    if result["load_error"] is not None:
        print(f"STATUS: LOAD ERROR")
        print(f"ERROR: {result['load_error']}")
        return

    if "top_level_type" in result:
        print("STATUS: INVALID TOP-LEVEL STRUCTURE")
        print(f"Top-level type: {result['top_level_type']}")
        return

    status = (
        "REVIEW REQUIRED"
        if result["structural_issues"]
        else "NO CRITICAL STRUCTURAL ISSUES"
    )

    print(f"STATUS: {status}")
    print(f"Root block type: {result['root_block_type']}")
    print(f"Root keys: {result['root_keys']}")
    print(f"Metadata keys: {result['metadata_keys']}")
    print(f"Total descendant nodes: {result['total_nodes']}")

    print("\nBlock types:")
    print_counter(result["block_types"], indent="  ")

    print("\nBlock IDs:")
    print(f"  Nodes with ID: {result['nodes_with_id']}")
    print(f"  Nodes without ID: {result['nodes_without_id']}")
    print(f"  Distinct duplicated IDs: {len(result['duplicate_ids'])}")

    print("\nHTML:")
    print(f"  Nodes with HTML field: {result['nodes_with_html']}")
    print(
        "  Nodes with non-empty HTML: "
        f"{result['nodes_with_nonempty_html']}"
    )
    print(f"  Nodes with blank HTML: {result['nodes_with_empty_html']}")
    print(
        "  Nodes without HTML field: "
        f"{result['nodes_without_html_field']}"
    )

    print("\nSection hierarchy:")
    print(
        "  Nodes with populated hierarchy: "
        f"{result['nodes_with_section_hierarchy']}"
    )
    print("  Observed hierarchy value types:")
    print_counter(result["hierarchy_types"], indent="    ")

    print("\nImages:")
    print(f"  Nodes containing images: {result['nodes_with_images']}")
    print(
        "  Total image references: "
        f"{result['total_image_references']}"
    )

    print("\nChildren:")
    print(
        "  Invalid children fields: "
        f"{result['invalid_children_fields']}"
    )
    print(
        "  Non-object child values: "
        f"{result['invalid_child_values']}"
    )

    page_indexes = result["raw_page_indexes"]

    print("\nPage identity found in block IDs:")

    if page_indexes:
        print(f"  Distinct raw page indexes: {len(page_indexes)}")
        print(f"  Minimum raw page index: {min(page_indexes)}")
        print(f"  Maximum raw page index: {max(page_indexes)}")
    else:
        print("  No /page/<number>/ patterns detected.")

    print("\nMost common node-key schemas:")

    for keys, count in result["node_key_schemas"].most_common(5):
        print(f"  {count} node(s): {list(keys)}")

    if result["structural_issues"]:
        print("\nStructural issues:")

        for issue in result["structural_issues"]:
            print(f"  - {issue}")

    print_examples(
        "Examples of nodes without IDs",
        result["missing_id_examples"],
    )
    print_examples(
        "Examples of nodes with blank HTML",
        result["empty_html_examples"],
    )
    print_examples(
        "Examples of invalid children fields",
        result["invalid_children_examples"],
    )
    print_examples(
        "Examples of invalid child values",
        result["invalid_child_examples"],
    )


def main() -> int:
    """
    Audit every JSON file and print a corpus summary.

    Returns
    -------
    int
        0 when every file loads and no critical structural issue is found.
        1 when at least one file requires review.
    """

    print(f"Project root: {PROJECT_ROOT}")
    print(f"RAG-ready directory: {RAG_READY_DIR}")

    if not RAG_READY_DIR.is_dir():
        print("\nERROR: The RAG-ready directory does not exist.")
        return 1

    json_files = sorted(
        RAG_READY_DIR.glob("*.json"),
        key=lambda path: path.name.lower(),
    )

    if not json_files:
        print("\nERROR: No JSON files were found.")
        return 1

    print(f"JSON files found: {len(json_files)}")

    results = [
        audit_json_file(file_path)
        for file_path in json_files
    ]

    for result in results:
        print_document_report(result)

    load_errors = sum(
        result["load_error"] is not None
        for result in results
    )

    invalid_top_levels = sum(
        "top_level_type" in result
        for result in results
    )

    files_requiring_review = sum(
        bool(result.get("structural_issues"))
        for result in results
    )

    combined_block_types: Counter[str] = Counter()

    for result in results:
        combined_block_types.update(result.get("block_types", {}))

    print("\n" + "#" * 88)
    print("CORPUS SUMMARY")
    print("#" * 88)
    print(f"JSON files audited: {len(results)}")
    print(f"Files with load errors: {load_errors}")
    print(f"Files with invalid top-level values: {invalid_top_levels}")
    print(f"Files requiring structural review: {files_requiring_review}")

    print("\nCombined block types:")
    print_counter(combined_block_types, indent="  ")

    audit_failed = (
        load_errors > 0
        or invalid_top_levels > 0
        or files_requiring_review > 0
    )

    if audit_failed:
        print(
            "\nAUDIT RESULT: Review is required before production ingestion."
        )
        return 1

    print(
        "\nAUDIT RESULT: No critical structural issues were detected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
