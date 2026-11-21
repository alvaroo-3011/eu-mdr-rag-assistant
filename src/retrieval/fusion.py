"""Fuse already-retrieved candidates, without retrieval or backend access.

The caller must supply dense and lexical results for the SAME QUERY. Current
candidate interfaces contain no query identity, so this boundary cannot prove
that requirement. Backend failures must not be disguised as empty branches;
orchestration and query/index envelopes are intentionally outside this module.

Unweighted RRF sums 1 / (rank_constant + original_branch_rank) for each branch
containing a chunk. The constant is explicit, finite, and nonnegative, with no
production default. Raw cosine distances and BM25 scores are preserved but
never used for ranking or thresholds. Return the entire unique union, sorted
by descending RRF score and then global ChunkSet position for exact ties.

Every input must match its original chunk's complete text and decoded metadata.
Within-branch duplicates fail; cross-branch duplicates merge only if identical.
No input is repaired, renumbered, mutated, or silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real

from src.ingestion.chunk_models import ChunkSet
from src.ingestion.models import RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import (
    DenseCandidate, TrustedChunkMetadata, deserialize_chunk_metadata,
)
from src.retrieval.lexical_query import LexicalCandidate


class FusionError(ValueError):
    """Raised when inputs or calculations cannot produce trustworthy fusion."""


def _finite_real(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FusionError(f"{field}: must be a finite nonboolean real number.")
    # Check the original sign: a tiny negative Fraction can normalize to -0.0.
    # Only rank_constant opts in; raw retrieval scores have no sign constraint.
    if nonnegative and value < 0:
        raise FusionError(f"{field}: must be nonnegative.")
    result = float(value)
    if not math.isfinite(result):
        raise FusionError(f"{field}: must be finite.")
    return result


def _validate_metadata(metadata: TrustedChunkMetadata) -> None:
    if type(metadata) is not TrustedChunkMetadata:
        raise FusionError("metadata: must be TrustedChunkMetadata.")
    # Reuse exact nested type/invariant checks, not just dataclass equality
    # (for example, bool and int compare equal but are not interchangeable).
    replace(metadata)
    if not metadata.permits_mdr:
        raise FusionError("metadata: MDR retrieval must be permitted.")


def _validate_candidate(candidate: object, kind: type) -> None:
    if type(candidate) is not kind:
        raise FusionError(f"candidate: must be {kind.__name__}.")
    # A fresh validation copy invokes the existing field checks. Keep the
    # ORIGINAL object as provenance; do not return or substitute this copy.
    replace(candidate)
    _validate_metadata(candidate.metadata)


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """Immutable fused result retaining each original branch candidate."""

    rank: int
    chunk_id: str
    document: str
    metadata: TrustedChunkMetadata
    rrf_score: float
    dense_candidate: DenseCandidate | None
    lexical_candidate: LexicalCandidate | None

    def __post_init__(self) -> None:
        try:
            if type(self.rank) is not int or self.rank < 1:
                raise FusionError("rank: must be a positive nonboolean integer.")
            for field in ("chunk_id", "document"):
                value = getattr(self, field)
                if type(value) is not str or not value.strip():
                    raise FusionError(f"{field}: must be a nonblank string.")
            _validate_metadata(self.metadata)
            if self.chunk_id != self.metadata.chunk_id:
                raise FusionError("chunk_id: must match trusted metadata.")
            score = _finite_real(self.rrf_score, "rrf_score")
            if score <= 0:
                raise FusionError("rrf_score: must be positive.")
            if self.dense_candidate is None and self.lexical_candidate is None:
                raise FusionError("At least one original candidate is required.")
            for candidate, kind in (
                (self.dense_candidate, DenseCandidate),
                (self.lexical_candidate, LexicalCandidate),
            ):
                if candidate is None:
                    continue
                _validate_candidate(candidate, kind)
                if (
                    candidate.chunk_id != self.chunk_id
                    or candidate.document != self.document
                    or candidate.metadata != self.metadata
                ):
                    raise FusionError("Original candidate conflicts with fused record.")
            object.__setattr__(self, "rrf_score", score)
        except FusionError:
            raise
        except Exception as exc:
            raise FusionError("Invalid fused candidate.") from exc


def _contribution(rank_constant: float, rank: int) -> float:
    denominator = _finite_real(rank_constant + rank, "RRF denominator")
    if denominator <= 0:
        raise FusionError("RRF denominator must be positive.")
    result = _finite_real(1 / denominator, "RRF contribution")
    if result <= 0:
        raise FusionError("RRF contribution must be positive.")
    return result


def fuse_candidates(
    *,
    chunk_set: ChunkSet,
    dense_candidates: tuple[DenseCandidate, ...],
    lexical_candidates: tuple[LexicalCandidate, ...],
    rank_constant: float,
) -> tuple[FusedCandidate, ...]:
    """Validate all candidates, then fuse the full union using original ranks.

    Empty tuples are valid, including both empty. Context and parameters are
    still checked. Ordinary failures retain their causes; BaseException escapes.
    """

    try:
        if not isinstance(chunk_set, ChunkSet):
            raise FusionError("chunk_set must be a validated ChunkSet.")
        constant = _finite_real(rank_constant, "rank_constant", nonnegative=True)
        if type(dense_candidates) is not tuple or type(lexical_candidates) is not tuple:
            raise FusionError("Both candidate branches must be tuples.")
        originals = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
        positions = {chunk.chunk_id: i for i, chunk in enumerate(chunk_set.chunks)}
        expected_metadata: dict[str, TrustedChunkMetadata] = {}
        branches: list[dict[str, DenseCandidate | LexicalCandidate]] = []
        for candidates, kind in (
            (dense_candidates, DenseCandidate),
            (lexical_candidates, LexicalCandidate),
        ):
            branch: dict[str, DenseCandidate | LexicalCandidate] = {}
            for position, candidate in enumerate(candidates, start=1):
                _validate_candidate(candidate, kind)
                if candidate.rank != position:
                    raise FusionError("Branch ranks must equal tuple positions 1..N.")
                chunk_id = candidate.chunk_id
                if chunk_id in branch:
                    raise FusionError(f"Duplicate ID within branch: {chunk_id}")
                if chunk_id not in originals:
                    raise FusionError(f"Candidate ID is absent from ChunkSet: {chunk_id}")
                original = originals[chunk_id]
                if (
                    candidate.metadata.chunk_set_id != chunk_set.chunk_set_id
                    or candidate.metadata.chunk_id != chunk_id
                ):
                    raise FusionError("Candidate metadata has an unexpected identity.")
                if RegulatoryRegime.MDR not in original.source.permitted_regimes:
                    raise FusionError("Original source does not permit MDR retrieval.")
                if candidate.document != original.text:
                    raise FusionError("Candidate text differs from the original chunk.")
                if chunk_id not in expected_metadata:
                    expected_metadata[chunk_id] = deserialize_chunk_metadata(
                        chunk_metadata(original),
                    )
                if candidate.metadata != expected_metadata[chunk_id]:
                    raise FusionError("Candidate metadata differs from the original chunk.")
                branch[chunk_id] = candidate
            branches.append(branch)

        dense, lexical = branches
        scored: list[tuple[float, str]] = []
        # Both branches have now passed every validation, including later rows.
        for chunk_id in dict.fromkeys((*dense, *lexical)):
            left, right = dense.get(chunk_id), lexical.get(chunk_id)
            if left is not None and right is not None and (
                left.document != right.document or left.metadata != right.metadata
            ):
                raise FusionError("Cross-branch records conflict.")
            contributions: list[float] = []
            for candidate in (left, right):
                if candidate is not None:
                    contribution = _finite_real(
                        _contribution(constant, candidate.rank), "RRF contribution",
                    )
                    if contribution <= 0:
                        raise FusionError("Computed RRF contribution must be positive.")
                    contributions.append(contribution)
            score = _finite_real(math.fsum(contributions), "rrf_score")
            if score <= 0:
                raise FusionError("Computed RRF score must be positive.")
            scored.append((score, chunk_id))
        # Global tuple position, NOT source-local chunk.order or branch priority.
        scored.sort(key=lambda item: (-item[0], positions[item[1]]))
        return tuple(
            FusedCandidate(
                rank, chunk_id, originals[chunk_id].text, expected_metadata[chunk_id],
                score, dense.get(chunk_id), lexical.get(chunk_id),
            )
            for rank, (score, chunk_id) in enumerate(scored, start=1)
        )
    except FusionError:
        raise
    except Exception as exc:
        raise FusionError("Candidate fusion failed.") from exc


__all__ = ["FusionError", "FusedCandidate", "fuse_candidates"]
