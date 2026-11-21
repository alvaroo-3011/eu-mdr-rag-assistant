"""
Quick, read-only inspection of representative RAG input files.

Purpose
-------
This script helps us understand the structure of the JSON and HTML files
before writing the real document loaders.

It does not modify, rename, or generate any project data.
"""

import json
from pathlib import Path


# Path to the folder containing the prepared source documents.
RAG_READY_DIR = Path("data/rag_ready")


# Representative JSON files that may use different structures.
JSON_FILES_TO_INSPECT = [
    # Representative MDCG document
    "md_mdcg_2021_26_en_RAG.json",

    # Document with tables and more complicated structure
    "MDCG 2023-5 Annex XVI_RAG (2).json",
]

# Official consolidated MDR HTML.
HTML_FILE_TO_INSPECT = "Regulation_MDR_Consolidated_19_07.html"


def inspect_json(file_path: Path) -> None:
    """Print the top-level structure of a JSON file."""

    print(f"\n{'=' * 80}")
    print(f"JSON FILE: {file_path.name}")
    print(f"{'=' * 80}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Top-level Python type: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")

        # Show the type of value stored under each top-level key.
        for key, value in data.items():
            print(f"  - {key}: {type(value).__name__}")

            if isinstance(value, list):
                print(f"    Number of items: {len(value)}")

                # Show the structure of the first item without printing
                # the complete document text.
                if value:
                    first_item = value[0]
                    print(
                        f"    First item type: "
                        f"{type(first_item).__name__}"
                    )

                    if isinstance(first_item, dict):
                        print(
                            f"    First item keys: "
                            f"{list(first_item.keys())}"
                        )

    elif isinstance(data, list):
        print(f"Number of top-level items: {len(data)}")

        if data:
            first_item = data[0]
            print(f"First item type: {type(first_item).__name__}")

            if isinstance(first_item, dict):
                print(f"First item keys: {list(first_item.keys())}")


def inspect_html(file_path: Path) -> None:
    """Print basic information about the official MDR HTML."""

    print(f"\n{'=' * 80}")
    print(f"HTML FILE: {file_path.name}")
    print(f"{'=' * 80}")

    html = file_path.read_text(encoding="utf-8")

    print(f"File size: {len(html):,} characters")
    print(f"Starts with: {html[:500]!r}")


def main() -> None:
    """Run all read-only inspections."""

    for filename in JSON_FILES_TO_INSPECT:
        inspect_json(RAG_READY_DIR / filename)

    inspect_html(RAG_READY_DIR / HTML_FILE_TO_INSPECT)


if __name__ == "__main__":
    main()