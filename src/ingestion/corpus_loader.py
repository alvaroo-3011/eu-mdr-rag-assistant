"""Load every manifest-approved source into one validated corpus collection."""

from pathlib import Path

from src.ingestion.mdcg_loader import load_mdcg_source
from src.ingestion.mdr_loader import load_mdr_source
from src.ingestion.models import (
    ContentFormat,
    CorpusManifest,
    LoadedSource,
)


# Construct these from their serialized values so dispatch remains explicitly
# aligned with the authoritative manifest representation.
_MARKER_JSON = ContentFormat("marker_json")
_EURLEX_HTML = ContentFormat("eurlex_html")


def load_corpus(
    manifest: CorpusManifest,
    *,
    project_root: str | Path,
) -> tuple[LoadedSource, ...]:
    """Load every source in a validated manifest, preserving manifest order.

    The format-specific loaders remain responsible for validating paths,
    checksums, source structures, and normalized blocks. This function provides
    the corpus-wide boundary: it dispatches every source exactly once and
    confirms that the combined output matches the manifest one-to-one.

    Parameters
    ----------
    manifest:
        Authoritative typed corpus manifest.
    project_root:
        Repository root used by the format-specific loaders to resolve paths.

    Returns
    -------
    tuple[LoadedSource, ...]
        Immutable loaded sources in the same order as ``manifest.sources``.

    Raises
    ------
    ValueError
        If the manifest count is inconsistent, a format is unsupported, a
        loader returns no blocks, or source identities are missing, duplicated,
        reordered, or otherwise inconsistent.
    TypeError
        If a format-specific loader does not return a ``LoadedSource``.
    """
    if len(manifest.sources) != manifest.expected_source_count:
        raise ValueError(
            "Manifest source count mismatch: "
            f"expected_source_count={manifest.expected_source_count}, "
            f"actual_sources={len(manifest.sources)}."
        )

    expected_identities = tuple(
        source.source_identity
        for source in manifest.sources
    )

    if len(set(expected_identities)) != len(expected_identities):
        raise ValueError(
            "Manifest contains duplicate source identities."
        )

    loaded_sources: list[LoadedSource] = []
    loaded_identities: set[str] = set()
    root = Path(project_root)

    for expected_source in manifest.sources:
        content_format = expected_source.content_format

        if content_format == _MARKER_JSON:
            loaded_source = load_mdcg_source(
                manifest,
                expected_source.document_id,
                project_root=root,
            )
        elif content_format == _EURLEX_HTML:
            loaded_source = load_mdr_source(
                manifest,
                expected_source.document_id,
                project_root=root,
            )
        else:
            format_label = getattr(
                content_format,
                "value",
                repr(content_format),
            )
            raise ValueError(
                "Unsupported content format "
                f"{format_label!r} for source "
                f"{expected_source.document_id!r}."
            )

        if not isinstance(loaded_source, LoadedSource):
            raise TypeError(
                "Source loader returned an invalid result for "
                f"{expected_source.document_id!r}: expected "
                f"LoadedSource, received "
                f"{type(loaded_source).__name__}."
            )

        # Equality checks the complete manifest entry, including version,
        # checksum, format, regimes, authority metadata, and local path.
        if loaded_source.source != expected_source:
            raise ValueError(
                "Loaded source identity or metadata does not match the "
                f"requested manifest entry "
                f"{expected_source.document_id!r}."
            )

        if not loaded_source.blocks:
            raise ValueError(
                "Source loader returned no normalized blocks for "
                f"{expected_source.document_id!r}."
            )

        source_identity = loaded_source.source.source_identity

        if source_identity in loaded_identities:
            raise ValueError(
                "Duplicate loaded source identity: "
                f"{source_identity!r}."
            )

        loaded_identities.add(source_identity)
        loaded_sources.append(loaded_source)

    result = tuple(loaded_sources)
    actual_identities = tuple(
        loaded.source.source_identity
        for loaded in result
    )

    # This final ordered comparison provides the corpus-wide one-to-one gate.
    if actual_identities != expected_identities:
        raise ValueError(
            "Loaded corpus does not match the manifest source sequence."
        )

    return result
