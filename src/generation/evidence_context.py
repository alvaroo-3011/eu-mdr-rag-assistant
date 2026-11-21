"""Build deterministic, unbudgeted evidence JSON and trusted citation mappings.

Only the explicit evidence projection enters model_context_json. The original
selection retains complete objects and retrieval provenance for local audit.
This module has no query, prompt, SDK, token counter, API call, or sufficiency
decision. The package must pass a future request-level budget gate before use.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import re

from src.ingestion.chunk_models import BlockSelection, Chunk, ChunkSet, ChunkTableSegment
from src.ingestion.models import (
    HierarchyEntry, LegalLocatorPart, PageIdentity, RegulatoryRegime,
    SourceAnnotation, SourceManifestEntry, TableData,
)
from src.retrieval.evidence_selection import EvidenceSelection, SelectedEvidence


EVIDENCE_CONTEXT_FORMAT = "mdr-evidence-context/v2"


class EvidenceContextError(ValueError):
    """Raised when evidence cannot form a faithful context package."""


# Explicit schema fields: adding a domain-model field must not silently change
# model context or expose new operational metadata under an unchanged version.
_STRUCTURE_FIELDS = {
    BlockSelection: (
        "block_id", "source_block_id", "block_order", "start_char", "end_char",
        "table_row_start", "table_row_end",
    ),
    HierarchyEntry: ("level", "kind", "block_id", "title"),
    PageIdentity: ("source_index", "pdf_page_number"),
    LegalLocatorPart: ("kind", "value"),
    ChunkTableSegment: (
        "block_id", "table", "source_row_start", "source_row_end", "is_complete_table",
    ),
    TableData: (
        "headers", "rows", "extraction_method", "source_page", "table_number",
        "pdfplumber_bbox", "header_detected",
    ),
    SourceAnnotation: ("kind", "identifier", "target_id", "raw_html", "text"),
}
_MODEL_SOURCE_FIELDS = (
    "document_id", "official_title", "short_title", "version", "source_type",
    "authority", "authority_tier", "official_url", "sha256",
    "regulatory_regimes", "permitted_regimes",
)
_MODEL_CHUNK_FIELDS = (
    "text", "hierarchies", "pages", "legal_locators",
    "table_segments", "annotations",
)


def _primitive(value: object) -> object:
    """Convert only schema-approved records, preserving arrays and raw strings."""

    if isinstance(value, Enum):
        return _primitive(value.value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is tuple:
        return [_primitive(item) for item in value]
    names = _STRUCTURE_FIELDS.get(type(value))
    if names is None:
        raise EvidenceContextError(f"Unsupported context value: {type(value).__name__}")
    # Revalidate each nested record, including tables absent from the dense
    # metadata projection. Keep the original objects, not validation copies.
    replace(value)
    return {name: _primitive(getattr(value, name)) for name in names}


def _source_payload(source: SourceManifestEntry) -> dict[str, object]:
    replace(source)
    return {
        "source_identity": source.source_identity,
        **{name: _primitive(getattr(source, name)) for name in _MODEL_SOURCE_FIELDS},
    }


def _validate_selection(selection: EvidenceSelection) -> None:
    if type(selection) is not EvidenceSelection:
        raise EvidenceContextError("selection must be EvidenceSelection.")
    replace(selection)
    for item in selection.selected:
        if type(item) is not SelectedEvidence:
            raise EvidenceContextError("Selected records must be SelectedEvidence.")
        replace(item)


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    """Package-local citation number mapped to a complete trusted Chunk.

    Numbers are strings, not global identities. The builder uses the actual
    matching Chunk object from its caller-owned authoritative ChunkSet.
    """

    citation_id: str
    chunk: Chunk

    def __post_init__(self) -> None:
        try:
            if type(self.citation_id) is not str or re.fullmatch(r"[1-9][0-9]*", self.citation_id) is None:
                raise EvidenceContextError("citation_id must be a canonical positive ASCII integer string.")
            if not isinstance(self.chunk, Chunk):
                raise EvidenceContextError("Citation chunk must be Chunk.")
            replace(self.chunk)
            replace(self.chunk.source)
            if RegulatoryRegime.MDR not in self.chunk.source.permitted_regimes:
                raise EvidenceContextError("Citation source must permit MDR.")
        except EvidenceContextError:
            raise
        except Exception as exc:
            raise EvidenceContextError("Invalid evidence citation.") from exc


def _serialize_context(chunk_set_id: str, citations: tuple[EvidenceCitation, ...]) -> str:
    evidence = [
        {
            "citation_id": citation.citation_id,
            "chunk_id": citation.chunk.chunk_id,
            "source": _source_payload(citation.chunk.source),
            **{name: _primitive(getattr(citation.chunk, name)) for name in _MODEL_CHUNK_FIELDS},
        }
        for citation in citations
    ]
    result = json.dumps(
        {
            "format": EVIDENCE_CONTEXT_FORMAT,
            "budget_status": "unbudgeted",
            "chunk_set_id": chunk_set_id,
            "evidence": evidence,
        },
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    # Reject invalid Unicode rather than replacing source text on transmission.
    # This is an encoding check, not token counting or a byte-budget policy.
    result.encode("utf-8")
    return result


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """An unbudgeted model projection plus a separate local audit envelope.

    selection retains original candidates, full chunks, scores, ranks, count
    limit, and omissions. Only model_context_json is the model projection.
    Direct construction checks local consistency; use build_evidence_context()
    for authoritative ChunkSet membership and original-record validation.
    """

    selection: EvidenceSelection
    citations: tuple[EvidenceCitation, ...]

    def __post_init__(self) -> None:
        try:
            _validate_selection(self.selection)
            if type(self.citations) is not tuple or len(self.citations) != len(self.selection.selected):
                raise EvidenceContextError("Citations must be a tuple aligned with selected evidence.")
            for index, (citation, item) in enumerate(zip(self.citations, self.selection.selected), 1):
                if type(citation) is not EvidenceCitation:
                    raise EvidenceContextError("citations must contain EvidenceCitation.")
                replace(citation)
                if citation.citation_id != str(index):
                    raise EvidenceContextError("Citation IDs must equal selected positions 1..N.")
                if citation.chunk != item.chunk:
                    raise EvidenceContextError("Citation chunk differs from selected original record.")
            # Fail during construction, rather than defer malformed JSON/UTF-8
            # failures to a later property access or future API request.
            _serialize_context(self.selection.chunk_set_id, self.citations)
        except EvidenceContextError:
            raise
        except Exception as exc:
            raise EvidenceContextError("Invalid evidence context.") from exc

    @property
    def budget_status(self) -> str:
        """No model-token or request-size validation has been performed."""

        return "unbudgeted"

    @property
    def model_context_json(self) -> str:
        """Return the versioned model projection, excluding the audit envelope."""

        try:
            return _serialize_context(self.selection.chunk_set_id, self.citations)
        except EvidenceContextError:
            raise
        except Exception as exc:
            raise EvidenceContextError("Evidence context serialization failed.") from exc

    def resolve_citation(self, citation_id: str) -> Chunk:
        """Resolve an exact package-local ID; never coerce or guess an ID."""

        if type(citation_id) is not str:
            raise EvidenceContextError("Citation lookup requires an exact string ID.")
        for citation in self.citations:
            if citation.citation_id == citation_id:
                return citation.chunk
        raise EvidenceContextError(f"Unknown citation ID in this package: {citation_id!r}")


def build_evidence_context(*, selection: EvidenceSelection, chunk_set: ChunkSet) -> EvidenceContext:
    """Validate against original records, then construct deterministic context.

    Valid direct-constructed selections are accepted after the same checks;
    Python dataclasses cannot prove which function created an object. Selected
    full records must equal authoritative originals, not merely projections.
    Omitted IDs must exist and permit MDR, but are not added to model context.

    This does not reconstruct withheld candidates, prove query/backend identity,
    or repeat validation of an unavailable full fused union. It does not reload
    source files, audit an index, select a model, or enforce a context budget.
    """

    try:
        if not isinstance(chunk_set, ChunkSet):
            raise EvidenceContextError("chunk_set must be a validated ChunkSet.")
        _validate_selection(selection)
        if selection.chunk_set_id != chunk_set.chunk_set_id:
            raise EvidenceContextError("Selection has an unexpected ChunkSet identity.")
        originals = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
        positions = {chunk.chunk_id: i for i, chunk in enumerate(chunk_set.chunks)}
        citations: list[EvidenceCitation] = []
        previous_key: tuple[float, int] | None = None
        for index, item in enumerate(selection.selected, 1):
            chunk_id = item.chunk.chunk_id
            if chunk_id not in originals:
                raise EvidenceContextError(f"Selected chunk is absent from ChunkSet: {chunk_id}")
            original = originals[chunk_id]
            if item.chunk != original:
                raise EvidenceContextError("Selected full record differs from authoritative original.")
            key = (-float(item.candidate.rrf_score), positions[chunk_id])
            if previous_key is not None and key < previous_key:
                raise EvidenceContextError("Selected fused score/tie ordering is invalid.")
            previous_key = key
            citations.append(EvidenceCitation(citation_id=str(index), chunk=original))
        for chunk_id in selection.omitted_chunk_ids:
            if chunk_id not in originals:
                raise EvidenceContextError(f"Omitted chunk is absent from ChunkSet: {chunk_id}")
            if RegulatoryRegime.MDR not in originals[chunk_id].source.permitted_regimes:
                raise EvidenceContextError("Omitted candidate source must permit MDR.")
        return EvidenceContext(selection=selection, citations=tuple(citations))
    except EvidenceContextError:
        raise
    except Exception as exc:
        raise EvidenceContextError("Evidence context construction failed.") from exc


__all__ = [
    "EVIDENCE_CONTEXT_FORMAT", "EvidenceContextError", "EvidenceCitation",
    "EvidenceContext", "build_evidence_context",
]
