"""
Audit the official consolidated MDR HTML before ingestion.

This script is READ-ONLY:
- It does not modify the HTML.
- It does not create chunks.
- It does not modify the JSON files.

It checks whether the HTML contains enough reliable structure to build
legal citations such as:

    Regulation (EU) 2017/745, Article 15(3)
    Regulation (EU) 2017/745, Annex I, Section 23.2

Dependency
----------
BeautifulSoup is used because the MDR must be analysed as an HTML document,
not as plain text.

Install it, if necessary, with:

    python -m pip install beautifulsoup4

Run from the project root with:

    python scripts/audit_mdr_html.py
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable
import sys

from bs4 import BeautifulSoup, Comment, Tag

# Force UTF-8 terminal output on Windows.
# EUR-Lex uses symbols such as ▼ and ► for amendment markers,
# which cannot be encoded using the default Windows cp1252 encoding.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_READY_DIR = PROJECT_ROOT / "data" / "rag_ready"


# ---------------------------------------------------------------------------
# Expected MDR structure
# ---------------------------------------------------------------------------

# MDR contains numbered Articles 1–123.
# Lettered insertions, such as Article 10a, are checked separately.
EXPECTED_ARTICLE_NUMBERS = set(range(1, 124))

EXPECTED_CHAPTERS = {
    "I", "II", "III", "IV", "V",
    "VI", "VII", "VIII", "IX", "X",
}

EXPECTED_ANNEXES = {
    "I", "II", "III", "IV", "V", "VI",
    "VII", "VIII", "IX", "X", "XI", "XII",
    "XIII", "XIV", "XV", "XVI", "XVII",
}


# Exact heading patterns.
ARTICLE_PATTERN = re.compile(
    r"^Article\s+(\d+)([a-z]?)$",
    re.IGNORECASE,
)

CHAPTER_PATTERN = re.compile(
    r"^CHAPTER\s+([IVXLCDM]+)$",
    re.IGNORECASE,
)

ANNEX_PATTERN = re.compile(
    r"^ANNEX\s+([IVXLCDM]+)$",
    re.IGNORECASE,
)

SECTION_PATTERN = re.compile(
    r"^Section\s+(\d+[a-z]?)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """
    Normalize whitespace without changing the legal wording.

    This is used only for inspection and comparison. It does not replace
    the original HTML text.
    """

    return re.sub(r"\s+", " ", value).strip()


def shorten(value: str, maximum_length: int = 180) -> str:
    """Shorten terminal examples while keeping them readable."""

    value = normalize_text(value)

    if len(value) <= maximum_length:
        return value

    return value[: maximum_length - 3] + "..."


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the source-file checksum.

    The checksum will later allow the corpus manifest to detect whether
    the official HTML has changed.
    """

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for data_block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(data_block)

    return digest.hexdigest()


def element_signature(element: Tag) -> str:
    """
    Create a compact description of an HTML element.

    Example:
        p#article-15.oj-ti-art
    """

    signature = element.name or "<unknown>"

    element_id = element.get("id")

    if element_id:
        signature += f"#{element_id}"

    classes = element.get("class", [])

    if isinstance(classes, str):
        classes = classes.split()

    if classes:
        signature += "." + ".".join(classes[:4])

    return signature


def get_element_text(element: Tag) -> str:
    """Extract readable text from one HTML element."""

    return normalize_text(
        element.get_text(separator=" ", strip=True)
    )


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def find_exact_headings(
    soup: BeautifulSoup,
    pattern: re.Pattern[str],
) -> list[tuple[Tag, re.Match[str]]]:
    """
    Find elements whose complete visible text matches a legal heading.

    Long container elements are excluded because their text may include an
    entire article or annex. We are looking for small heading elements.
    """

    candidates: list[tuple[Tag, re.Match[str]]] = []

    candidate_tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "div", "span", "td",
    ]

    for element in soup.find_all(candidate_tags):
        text = get_element_text(element)

        if not text or len(text) > 80:
            continue

        match = pattern.fullmatch(text)

        if match is not None:
            candidates.append((element, match))

    return candidates


def count_heading_labels(
    headings: Iterable[tuple[Tag, re.Match[str]]],
) -> Counter[str]:
    """Count normalized labels detected as legal headings."""

    labels: Counter[str] = Counter()

    for element, _ in headings:
        labels[get_element_text(element)] += 1

    return labels


def print_heading_examples(
    title: str,
    headings: list[tuple[Tag, re.Match[str]]],
    maximum_examples: int = 8,
) -> None:
    """Print sample detected headings and their HTML selectors."""

    print(f"\n{title} examples:")

    if not headings:
        print("  (none detected)")
        return

    for element, _ in headings[:maximum_examples]:
        print(
            f"  {get_element_text(element)!r}"
            f" -> {element_signature(element)}"
        )


# ---------------------------------------------------------------------------
# Structure diagnostics
# ---------------------------------------------------------------------------

def find_duplicate_ids(soup: BeautifulSoup) -> dict[str, int]:
    """Return duplicated HTML id attributes."""

    id_counts: Counter[str] = Counter()

    for element in soup.find_all(attrs={"id": True}):
        element_id = element.get("id")

        if isinstance(element_id, str) and element_id.strip():
            id_counts[element_id.strip()] += 1

    return {
        element_id: count
        for element_id, count in id_counts.items()
        if count > 1
    }


def find_article_ancestor_signatures(
    article_headings: list[tuple[Tag, re.Match[str]]],
) -> Counter[str]:
    """
    Find common containers surrounding detected Article headings.

    This helps identify the element that contains the main legislation
    rather than website navigation or document metadata.
    """

    ancestors: Counter[str] = Counter()

    for element, _ in article_headings:
        levels_checked = 0

        for parent in element.parents:
            if not isinstance(parent, Tag):
                continue

            if parent.name in {"main", "article", "section", "div", "body"}:
                ancestors[element_signature(parent)] += 1
                levels_checked += 1

            if levels_checked >= 6:
                break

    return ancestors


def find_suspicious_classes(soup: BeautifulSoup) -> Counter[str]:
    """
    Find classes potentially related to amendments, notes or pages.

    These are not automatically removed. They are reported so we can
    understand their role before ingestion.
    """

    suspicious_words = (
        "amend",
        "mod",
        "foot",
        "note",
        "page",
        "annotation",
    )

    matches: Counter[str] = Counter()

    for element in soup.find_all(class_=True):
        classes = element.get("class", [])

        if isinstance(classes, str):
            classes = classes.split()

        for class_name in classes:
            lower_name = class_name.lower()

            if any(word in lower_name for word in suspicious_words):
                matches[class_name] += 1

    return matches


def find_duplicate_paragraphs(
    soup: BeautifulSoup,
) -> list[tuple[str, int]]:
    """
    Detect repeated substantial paragraphs.

    Repeated short labels are normal, so only text with at least 100
    characters is considered.
    """

    paragraph_counts: Counter[str] = Counter()

    for element in soup.find_all(["p", "li"]):
        text = get_element_text(element)

        if len(text) >= 100:
            paragraph_counts[text] += 1

    duplicates = [
        (text, count)
        for text, count in paragraph_counts.items()
        if count > 1
    ]

    return sorted(
        duplicates,
        key=lambda item: item[1],
        reverse=True,
    )


def print_counter(
    counter: Counter[str],
    maximum_items: int = 20,
) -> None:
    """Print the most common values in a Counter."""

    if not counter:
        print("  (none)")
        return

    for value, count in counter.most_common(maximum_items):
        print(f"  {value}: {count}")


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the complete read-only MDR HTML audit."""

    print(f"Project root: {PROJECT_ROOT}")
    print(f"RAG-ready directory: {RAG_READY_DIR}")

    if not RAG_READY_DIR.is_dir():
        print("\nERROR: The RAG-ready directory does not exist.")
        return 1

    html_files = sorted(RAG_READY_DIR.glob("*.html"))

    if len(html_files) != 1:
        print(
            "\nERROR: Expected exactly one HTML file in data/rag_ready, "
            f"but found {len(html_files)}."
        )

        for file_path in html_files:
            print(f"  - {file_path.name}")

        return 1

    html_path = html_files[0]
    raw_html = html_path.read_text(encoding="utf-8")

    # html.parser is included with Python, so BeautifulSoup does not require
    # lxml for this initial audit.
    soup = BeautifulSoup(raw_html, "html.parser")

    print("\n" + "=" * 88)
    print("SOURCE IDENTITY")
    print("=" * 88)
    print(f"Filename: {html_path.name}")
    print(f"File size: {html_path.stat().st_size:,} bytes")
    print(f"SHA-256: {calculate_sha256(html_path)}")

    title = (
        normalize_text(soup.title.get_text())
        if soup.title
        else "<missing>"
    )

    print(f"HTML title: {title}")

    # Comments can contain CONVEX generation metadata.
    relevant_comments = [
        normalize_text(str(comment))
        for comment in soup.find_all(
            string=lambda value: isinstance(value, Comment)
        )
        if "convex" in str(comment).lower()
        or "generated" in str(comment).lower()
    ]

    print("\nRelevant HTML comments:")

    if relevant_comments:
        for comment in relevant_comments[:10]:
            print(f"  - {shorten(comment)}")
    else:
        print("  (none detected)")

    visible_text = normalize_text(
        soup.get_text(separator=" ", strip=True)
    )

    regulation_identity_found = bool(
        re.search(
            r"Regulation\s*\(EU\)\s*2017/745",
            visible_text,
            re.IGNORECASE,
        )
    )

    celex_identity_found = "02017R0745" in raw_html.upper()

    print("\nIdentity checks:")
    print(
        "  Regulation (EU) 2017/745 text detected: "
        f"{regulation_identity_found}"
    )
    print(
        "  Consolidated CELEX identifier 02017R0745 detected: "
        f"{celex_identity_found}"
    )

    # -----------------------------------------------------------------------
    # General HTML structure
    # -----------------------------------------------------------------------

    tag_counts = Counter(
        element.name
        for element in soup.find_all(True)
        if element.name
    )

    class_counts: Counter[str] = Counter()

    for element in soup.find_all(class_=True):
        classes = element.get("class", [])

        if isinstance(classes, str):
            classes = classes.split()

        class_counts.update(classes)

    print("\n" + "=" * 88)
    print("GENERAL HTML STRUCTURE")
    print("=" * 88)
    print(f"Visible text length: {len(visible_text):,} characters")
    print(f"Total HTML elements: {sum(tag_counts.values()):,}")

    print("\nMost common tags:")
    print_counter(tag_counts, maximum_items=20)

    print("\nMost common classes:")
    print_counter(class_counts, maximum_items=30)

    # -----------------------------------------------------------------------
    # Legal hierarchy
    # -----------------------------------------------------------------------

    article_headings = find_exact_headings(
        soup,
        ARTICLE_PATTERN,
    )
    chapter_headings = find_exact_headings(
        soup,
        CHAPTER_PATTERN,
    )
    annex_headings = find_exact_headings(
        soup,
        ANNEX_PATTERN,
    )
    section_headings = find_exact_headings(
        soup,
        SECTION_PATTERN,
    )

    article_numbers: set[int] = set()
    lettered_articles: set[str] = set()

    for _, match in article_headings:
        article_number = int(match.group(1))
        article_suffix = match.group(2).lower()

        article_numbers.add(article_number)

        if article_suffix:
            lettered_articles.add(
                f"{article_number}{article_suffix}"
            )

    chapters = {
        match.group(1).upper()
        for _, match in chapter_headings
    }

    annexes = {
        match.group(1).upper()
        for _, match in annex_headings
    }

    missing_articles = sorted(
        EXPECTED_ARTICLE_NUMBERS - article_numbers
    )
    unexpected_articles = sorted(
        article_numbers - EXPECTED_ARTICLE_NUMBERS
    )
    missing_chapters = sorted(
        EXPECTED_CHAPTERS - chapters
    )
    missing_annexes = sorted(
        EXPECTED_ANNEXES - annexes
    )

    print("\n" + "=" * 88)
    print("LEGAL HIERARCHY")
    print("=" * 88)

    print(f"Article heading elements detected: {len(article_headings)}")
    print(f"Unique numbered articles detected: {len(article_numbers)}")
    print(f"Lettered articles detected: {sorted(lettered_articles)}")
    print(f"Missing Articles 1–123: {missing_articles}")
    print(f"Unexpected numeric articles: {unexpected_articles}")

    print(f"\nChapter heading elements detected: {len(chapter_headings)}")
    print(f"Unique chapters detected: {sorted(chapters)}")
    print(f"Missing Chapters I–X: {missing_chapters}")

    print(f"\nAnnex heading elements detected: {len(annex_headings)}")
    print(f"Unique annexes detected: {sorted(annexes)}")
    print(f"Missing Annexes I–XVII: {missing_annexes}")

    print(f"\nSection heading elements detected: {len(section_headings)}")

    print_heading_examples(
        "Article heading",
        article_headings,
    )
    print_heading_examples(
        "Chapter heading",
        chapter_headings,
    )
    print_heading_examples(
        "Annex heading",
        annex_headings,
    )

    # Repeated headings may indicate a table of contents plus the legal body,
    # nested HTML wrappers, or duplicated website content.
    article_label_counts = count_heading_labels(article_headings)
    repeated_article_labels = Counter(
        {
            label: count
            for label, count in article_label_counts.items()
            if count > 1
        }
    )

    print("\nRepeated Article heading labels:")
    print_counter(repeated_article_labels, maximum_items=20)

    print("\nMost common containers surrounding Article headings:")

    article_ancestors = find_article_ancestor_signatures(
        article_headings
    )
    print_counter(article_ancestors, maximum_items=15)

    # -----------------------------------------------------------------------
    # Tables, lists, anchors and editorial structures
    # -----------------------------------------------------------------------

    tables = soup.find_all("table")
    ordered_lists = soup.find_all("ol")
    unordered_lists = soup.find_all("ul")
    list_items = soup.find_all("li")

    all_ids = [
        element.get("id")
        for element in soup.find_all(attrs={"id": True})
        if isinstance(element.get("id"), str)
        and element.get("id").strip()
    ]

    duplicate_ids = find_duplicate_ids(soup)
    suspicious_classes = find_suspicious_classes(soup)

    print("\n" + "=" * 88)
    print("TABLES, LISTS AND LOCATORS")
    print("=" * 88)
    print(f"Tables: {len(tables)}")
    print(f"Ordered lists: {len(ordered_lists)}")
    print(f"Unordered lists: {len(unordered_lists)}")
    print(f"List items: {len(list_items)}")
    print(f"Elements with IDs: {len(all_ids)}")
    print(f"Duplicated IDs: {len(duplicate_ids)}")

    if duplicate_ids:
        print("\nDuplicated ID examples:")

        for element_id, count in list(duplicate_ids.items())[:10]:
            print(f"  {element_id}: {count}")

    print("\nPotential editorial/page-related classes:")
    print_counter(suspicious_classes, maximum_items=30)

    # Consolidated EU legislation can contain amendment markers such as:
    #     ▼B
    #     ▼M1
    #     ►M2
    amendment_markers = Counter(
        re.findall(
            r"[▼►][A-Z]\d*",
            visible_text,
        )
    )

    print("\nPossible consolidation amendment markers:")
    print_counter(amendment_markers, maximum_items=30)

    # -----------------------------------------------------------------------
    # Encoding and duplicated text
    # -----------------------------------------------------------------------

    replacement_characters = raw_html.count("\ufffd")
    null_characters = raw_html.count("\x00")
    non_breaking_spaces = raw_html.count("\u00a0")
    soft_hyphens = raw_html.count("\u00ad")

    duplicate_paragraphs = find_duplicate_paragraphs(soup)

    print("\n" + "=" * 88)
    print("TEXT QUALITY")
    print("=" * 88)
    print(
        f"Unicode replacement characters: "
        f"{replacement_characters}"
    )
    print(f"Null characters: {null_characters}")
    print(f"Non-breaking spaces: {non_breaking_spaces}")
    print(f"Soft hyphens: {soft_hyphens}")
    print(
        "Repeated substantial paragraph values: "
        f"{len(duplicate_paragraphs)}"
    )

    if duplicate_paragraphs:
        print("\nRepeated paragraph examples:")

        for text, count in duplicate_paragraphs[:10]:
            print(f"  {count} times: {shorten(text)}")

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    blockers: list[str] = []

    if soup.body is None:
        blockers.append("No HTML body element was detected.")

    if not regulation_identity_found:
        blockers.append(
            "Regulation (EU) 2017/745 identity was not detected."
        )

    if not article_headings:
        blockers.append("No Article headings were detected.")

    if not annex_headings:
        blockers.append("No Annex headings were detected.")

    if missing_articles:
        blockers.append(
            "One or more expected numbered Articles were not detected."
        )

    if missing_annexes:
        blockers.append(
            "One or more expected Annexes were not detected."
        )

    if replacement_characters:
        blockers.append(
            "Unicode replacement characters were detected."
        )

    if null_characters:
        blockers.append("Null characters were detected.")

    print("\n" + "#" * 88)
    print("HTML AUDIT RESULT")
    print("#" * 88)

    if blockers:
        print("RESULT: REVIEW REQUIRED")

        for blocker in blockers:
            print(f"  - {blocker}")

        print(
            "\nThis does not automatically mean the HTML is incomplete. "
            "The heading selectors may need adjustment after inspecting "
            "the document's real structure."
        )
        return 1

    print("RESULT: NO CRITICAL STRUCTURAL ISSUES DETECTED")
    print(
        "\nThe HTML appears suitable for designing the MDR ingestion "
        "adapter. Its detected selectors and hierarchy must still be "
        "reviewed before production ingestion."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())