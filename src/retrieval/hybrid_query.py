"""Sequential hybrid retrieval over caller-owned retrievers and context.

Lexical's public chunk_set_id is checked before retrieval. DenseRetriever has
no public backend identity, so this boundary cannot independently verify it;
its construction-time collection checks remain the caller's responsibility.
Fusion validates all returned evidence against the supplied ChunkSet. That
does not prove which backend was searched, or independently identify a query.
This boundary forwards the same unchanged query to both branches instead.

No backend construction, retries, fallback, loading, or output limit lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

from src.ingestion.chunk_models import ChunkSet
from src.retrieval.dense_results import DenseCandidate
from src.retrieval.fusion import FusedCandidate, fuse_candidates
from src.retrieval.lexical_query import LexicalCandidate


class HybridError(ValueError):
    """Raised when hybrid preflight, a branch, or fusion fails."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridRetrievalConfig:
    """Reusable explicit validation for hybrid query parameters."""

    dense_candidate_count: int
    lexical_candidate_count: int
    rank_constant: float

    def __post_init__(self) -> None:
        for name in ("dense_candidate_count", "lexical_candidate_count"):
            count = getattr(self, name)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise HybridError(f"{name} must be a positive nonboolean integer.")
        value = self.rank_constant
        if isinstance(value, bool) or not isinstance(value, Real):
            raise HybridError("rank_constant must be a finite nonboolean real.")
        if value < 0:
            raise HybridError("rank_constant must be nonnegative.")
        value = float(value)
        if not math.isfinite(value):
            raise HybridError("rank_constant must be finite.")
        object.__setattr__(self, "rank_constant", value)


class _DenseRetriever(Protocol):
    def retrieve(
        self, query: str, *, candidate_count: int,
    ) -> tuple[DenseCandidate, ...]:
        """Return dense candidates without accepting an implicit count."""
        ...


class _LexicalRetriever(Protocol):
    @property
    def chunk_set_id(self) -> str:
        """Expose the existing lexical context identity."""
        ...

    def retrieve(
        self, query: str, *, candidate_count: int,
    ) -> tuple[LexicalCandidate, ...]:
        """Return lexical candidates, including a successful empty tuple."""
        ...


def _validate_envelope(
    candidates: object, *, candidate_count: int, allow_empty: bool,
) -> None:
    if type(candidates) is not tuple:
        raise ValueError("Results must be an exact tuple.")
    minimum = 0 if allow_empty else 1
    if not minimum <= len(candidates) <= candidate_count:
        raise ValueError(
            f"Result count must be between {minimum} and {candidate_count}."
        )


def retrieve_hybrid(
    query: str,
    *,
    chunk_set: ChunkSet,
    dense_retriever: _DenseRetriever,
    lexical_retriever: _LexicalRetriever,
    dense_candidate_count: int,
    lexical_candidate_count: int,
    rank_constant: float,
) -> tuple[FusedCandidate, ...]:
    """Preflight all inputs, then call dense, lexical, and fusion once each.

    Ordinary failures retain their causes and identify the failing stage.
    BaseException escapes. Dense results must be nonempty; lexical may be empty.
    Candidate-level validation belongs to fusion, whose full tuple is returned
    directly, retaining original branch candidate objects and their raw scores.
    """

    try:
        if not isinstance(chunk_set, ChunkSet):
            raise ValueError("chunk_set must be a validated ChunkSet.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string.")
        settings = HybridRetrievalConfig(
            dense_candidate_count=dense_candidate_count,
            lexical_candidate_count=lexical_candidate_count,
            rank_constant=rank_constant,
        )
        constant = settings.rank_constant

        # Resolve each method exactly once, before either branch is invoked.
        dense_retrieve = dense_retriever.retrieve
        lexical_retrieve = lexical_retriever.retrieve
        if not callable(dense_retrieve) or not callable(lexical_retrieve):
            raise ValueError("Both retrievers must expose callable retrieve methods.")
        lexical_identity = lexical_retriever.chunk_set_id
        if type(lexical_identity) is not str or not lexical_identity.strip():
            raise ValueError("lexical chunk_set_id must be a nonblank string.")
        if lexical_identity != chunk_set.chunk_set_id:
            raise ValueError("lexical chunk_set_id does not match chunk_set.")
    except Exception as exc:
        raise HybridError(f"Hybrid preflight failed: {exc}") from exc

    try:
        dense = dense_retrieve(query, candidate_count=dense_candidate_count)
        _validate_envelope(
            dense, candidate_count=dense_candidate_count, allow_empty=False,
        )
    except Exception as exc:
        raise HybridError("Hybrid dense stage failed.") from exc

    try:
        lexical = lexical_retrieve(query, candidate_count=lexical_candidate_count)
        _validate_envelope(
            lexical, candidate_count=lexical_candidate_count, allow_empty=True,
        )
    except Exception as exc:
        raise HybridError("Hybrid lexical stage failed.") from exc

    try:
        return fuse_candidates(
            chunk_set=chunk_set, dense_candidates=dense,
            lexical_candidates=lexical, rank_constant=constant,
        )
    except Exception as exc:
        raise HybridError("Hybrid fusion stage failed.") from exc


__all__ = ["HybridError", "HybridRetrievalConfig", "retrieve_hybrid"]
