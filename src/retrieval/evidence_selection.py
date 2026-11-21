"""Select a rank-preserving prefix while retaining complete original chunks.

The caller supplies the full fused union and its validated original ChunkSet.
All supplied records are checked before applying the explicit count limit.
Selected records retain the ORIGINAL candidate and Chunk objects, including
structured tables, annotations and branch provenance omitted by projections.

This is a mechanical selection baseline, not a relevance or sufficiency test.
It does not retrieve, rerank, expand neighbors, trim text, assign citation IDs,
read evaluation judgments, or enforce a token budget. No production limit is
chosen here. An empty selection is not a finding that the corpus lacks support.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.ingestion.chunk_models import Chunk, ChunkSet
from src.ingestion.models import RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import deserialize_chunk_metadata
from src.retrieval.fusion import FusedCandidate


class EvidenceSelectionError(ValueError):
    """Raised when a selection cannot preserve trustworthy input records."""


def _positive_count(value: object) -> None:
    if type(value) is not int or value < 1:
        raise EvidenceSelectionError("max_chunks must be a positive nonboolean integer.")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceSelectionConfig:
    """Reusable explicit validation for the mechanical evidence count limit."""

    max_chunks: int

    def __post_init__(self) -> None:
        _positive_count(self.max_chunks)


@dataclass(frozen=True, slots=True)
class SelectedEvidence:
    """An unchanged fused candidate paired with its complete original Chunk.

    Direct construction checks pair consistency, not membership in a specific
    ChunkSet. Use select_evidence() for membership and full-input validation.
    """

    candidate: FusedCandidate
    chunk: Chunk

    def __post_init__(self) -> None:
        try:
            if type(self.candidate) is not FusedCandidate:
                raise EvidenceSelectionError("candidate must be FusedCandidate.")
            if not isinstance(self.chunk, Chunk):
                raise EvidenceSelectionError("chunk must be an original Chunk.")
            # Invoke existing field checks, including nested branch records,
            # and the Chunk's complete canonical evidence-identity checks.
            # Validation copies are discarded; original objects are retained.
            replace(self.candidate)
            replace(self.chunk)
            if RegulatoryRegime.MDR not in self.chunk.source.permitted_regimes:
                raise EvidenceSelectionError("Original source must permit MDR.")
            if self.candidate.chunk_id != self.chunk.chunk_id:
                raise EvidenceSelectionError("Candidate and original chunk IDs differ.")
            if self.candidate.document != self.chunk.text:
                raise EvidenceSelectionError("Candidate text differs from original chunk.")
            expected = deserialize_chunk_metadata(chunk_metadata(self.chunk))
            if self.candidate.metadata != expected:
                raise EvidenceSelectionError("Candidate metadata differs from original chunk.")
        except EvidenceSelectionError:
            raise
        except Exception as exc:
            raise EvidenceSelectionError("Invalid selected evidence pair.") from exc


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    """Count-limited context with explicit omissions and no sufficiency flag.

    Omitted IDs follow the remaining fused order. They are not judgments of
    irrelevance. Their records remain available in the caller-owned union.
    """

    chunk_set_id: str
    max_chunks: int
    selected: tuple[SelectedEvidence, ...]
    omitted_chunk_ids: tuple[str, ...]
    strategy: str = "ranked_prefix"

    def __post_init__(self) -> None:
        try:
            if type(self.chunk_set_id) is not str or not self.chunk_set_id.strip():
                raise EvidenceSelectionError("chunk_set_id must be a nonblank string.")
            _positive_count(self.max_chunks)
            if type(self.selected) is not tuple or type(self.omitted_chunk_ids) is not tuple:
                raise EvidenceSelectionError("Selected evidence and omitted IDs must be tuples.")
            selected_ids: list[str] = []
            if self.strategy not in ("ranked_prefix", "coverage"):
                raise EvidenceSelectionError("Unknown selection strategy.")
            previous_rank = 0
            for rank, item in enumerate(self.selected, 1):
                if type(item) is not SelectedEvidence:
                    raise EvidenceSelectionError("selected must contain SelectedEvidence.")
                if self.strategy == "ranked_prefix" and item.candidate.rank != rank:
                    raise EvidenceSelectionError("Selected ranks must form the original prefix.")
                if self.strategy == "coverage" and item.candidate.rank <= previous_rank:
                    raise EvidenceSelectionError("Coverage selection must retain fused order.")
                previous_rank = item.candidate.rank
                if item.chunk.chunk_set_id != self.chunk_set_id:
                    raise EvidenceSelectionError("Selected chunk has an unexpected ChunkSet identity.")
                selected_ids.append(item.chunk.chunk_id)
            if any(type(value) is not str or not value.strip() for value in self.omitted_chunk_ids):
                raise EvidenceSelectionError("Omitted chunk IDs must be nonblank strings.")
            ids = (*selected_ids, *self.omitted_chunk_ids)
            if len(set(ids)) != len(ids):
                raise EvidenceSelectionError("Selected and omitted chunk IDs must be unique.")
            if len(self.selected) != min(self.max_chunks, len(ids)):
                raise EvidenceSelectionError("Selected length does not match the explicit limit.")
        except EvidenceSelectionError:
            raise
        except Exception as exc:
            raise EvidenceSelectionError("Invalid evidence selection result.") from exc

    @property
    def has_omitted_candidates(self) -> bool:
        """Whether the count limit omitted any supplied candidates."""

        return bool(self.omitted_chunk_ids)


def select_evidence(
    *,
    chunk_set: ChunkSet,
    fused_candidates: tuple[FusedCandidate, ...],
    max_chunks: int,
) -> EvidenceSelection:
    """Validate the complete fused input, then retain its first max_chunks.

    Empty input is valid; the context and count are still required and checked.
    Ranks must be contiguous, scores nonincreasing, and exact ties follow global
    ChunkSet position. Present branch ranks must also be contiguous in the union.

    This cannot prove the input is the complete backend union: a caller could
    withhold records without exposing rank gaps. It cannot prove query/backend
    identity or recompute RRF without its constant. Call it directly on the full
    retrieve_hybrid() result, rather than on a manually truncated tuple.

    Ordinary failures retain their causes. BaseException is not swallowed.
    """

    try:
        if not isinstance(chunk_set, ChunkSet):
            raise EvidenceSelectionError("chunk_set must be a validated ChunkSet.")
        max_chunks = EvidenceSelectionConfig(max_chunks=max_chunks).max_chunks
        if type(fused_candidates) is not tuple:
            raise EvidenceSelectionError("fused_candidates must be an exact tuple.")

        originals = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
        positions = {chunk.chunk_id: i for i, chunk in enumerate(chunk_set.chunks)}
        seen: set[str] = set()
        branch_ranks: tuple[list[int], list[int]] = ([], [])
        pairs: list[SelectedEvidence] = []
        previous_key: tuple[float, int] | None = None
        for rank, candidate in enumerate(fused_candidates, 1):
            if type(candidate) is not FusedCandidate:
                raise EvidenceSelectionError("Every candidate must be FusedCandidate.")
            # Validate before dictionary lookup: malformed IDs must fail as a
            # bad record, not trigger accidental coercion or partial output.
            checked = replace(candidate)
            if candidate.rank != rank:
                raise EvidenceSelectionError("Fused ranks must equal tuple positions 1..N.")
            chunk_id = candidate.chunk_id
            if chunk_id in seen:
                raise EvidenceSelectionError(f"Duplicate fused chunk ID: {chunk_id}")
            if chunk_id not in originals:
                raise EvidenceSelectionError(f"Candidate ID is absent from ChunkSet: {chunk_id}")
            if candidate.metadata.chunk_set_id != chunk_set.chunk_set_id:
                raise EvidenceSelectionError("Candidate has an unexpected ChunkSet identity.")

            pair = SelectedEvidence(candidate=candidate, chunk=originals[chunk_id])
            key = (-checked.rrf_score, positions[chunk_id])
            if previous_key is not None and key < previous_key:
                raise EvidenceSelectionError("Fused score/tie ordering is invalid.")
            previous_key = key
            for ranks, branch in zip(branch_ranks, (candidate.dense_candidate, candidate.lexical_candidate)):
                if branch is not None:
                    ranks.append(branch.rank)
            seen.add(chunk_id)
            pairs.append(pair)

        # Do not recompute fusion or silently renumber incomplete branch input.
        for ranks in branch_ranks:
            if sorted(ranks) != list(range(1, len(ranks) + 1)):
                raise EvidenceSelectionError("Present branch ranks must be unique and contiguous.")

        # Only after every row passes may the requested count omit candidates.
        return EvidenceSelection(
            chunk_set_id=chunk_set.chunk_set_id,
            max_chunks=max_chunks,
            selected=tuple(pairs[:max_chunks]),
            omitted_chunk_ids=tuple(item.candidate.chunk_id for item in pairs[max_chunks:]),
        )
    except EvidenceSelectionError:
        raise
    except Exception as exc:
        raise EvidenceSelectionError("Evidence selection failed.") from exc


def select_coverage_evidence(*, chunk_set: ChunkSet, fused_candidates: tuple[FusedCandidate, ...],
                             max_chunks: int, priority_ids: tuple[str, ...]) -> EvidenceSelection:
    """Prioritize retrieved coverage IDs while retaining original fusion provenance.

    Validate the entire union with the existing selector first. Priority IDs must
    be distinct members of that union; selection remains in original fused order.
    """
    if (type(priority_ids) is not tuple or any(type(value) is not str or not value for value in priority_ids)
            or len(set(priority_ids)) != len(priority_ids)):
        raise EvidenceSelectionError("Priority IDs must be a unique tuple.")
    full = select_evidence(chunk_set=chunk_set, fused_candidates=fused_candidates,
                           max_chunks=max(1, len(fused_candidates)))
    by_id = {item.chunk.chunk_id: item for item in full.selected}
    if any(value not in by_id for value in priority_ids):
        raise EvidenceSelectionError("Priority evidence must be in the validated fused union.")
    limit = EvidenceSelectionConfig(max_chunks=max_chunks).max_chunks
    chosen = set(priority_ids[:limit])
    for item in full.selected:
        if len(chosen) >= limit:
            break
        chosen.add(item.chunk.chunk_id)
    selected = tuple(item for item in full.selected if item.chunk.chunk_id in chosen)
    omitted = tuple(item.chunk.chunk_id for item in full.selected if item.chunk.chunk_id not in chosen)
    return EvidenceSelection(chunk_set_id=chunk_set.chunk_set_id, max_chunks=limit,
                             selected=selected, omitted_chunk_ids=omitted, strategy="coverage")


__all__ = [
    "EvidenceSelectionError", "EvidenceSelectionConfig", "SelectedEvidence",
    "EvidenceSelection", "select_evidence", "select_coverage_evidence",
]
