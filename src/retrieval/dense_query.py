"""Query an injected dense collection and return immutable trusted candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.ingestion.chunk_models import ChunkSet
from src.retrieval.dense_index import open_dense_collection
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.embeddings import EmbeddingModelConfig


class DenseQueryError(ValueError):
    """Raised when a dense query cannot return trustworthy candidates."""


class _QueryEmbedder(Protocol):
    @property
    def config(self) -> EmbeddingModelConfig:
        """Return the exact identity of the injected query model."""

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed one query without changing its text."""


class _QueryCollection(Protocol):
    @property
    def metadata(self) -> object:
        """Return collection identity metadata for the shared opener."""

    def count(self) -> int:
        """Return the stored record count."""

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object],
        include: list[str],
    ) -> object:
        """Query using explicit embeddings and return nested result columns."""


class _QueryClient(Protocol):
    def get_collection(
        self,
        *,
        name: str,
        embedding_function: None,
    ) -> _QueryCollection:
        """Open an existing collection with managed embeddings disabled."""


def _query_vector(raw: object, dimension: int) -> list[float]:
    """Validate without reshaping, coercing text, or repairing model output."""

    vector = np.asarray(raw)
    if vector.shape != (dimension,):
        raise DenseQueryError("Query vector has an unexpected shape or dimension.")
    # Only real numeric arrays are suitable for cosine search. In particular,
    # strings, booleans, objects, and complex values must not be coerced.
    if vector.dtype.kind not in "iuf":
        raise DenseQueryError("Query vector must contain real numeric values.")
    if not np.isfinite(vector).all():
        raise DenseQueryError("Query vector contains a non-finite value.")
    return vector.tolist()


def _candidates(
    response: object,
    *,
    chunk_set_id: str,
    candidate_count: int,
) -> tuple[DenseCandidate, ...]:
    """Validate the complete response before exposing any candidate."""

    if not isinstance(response, Mapping):
        raise DenseQueryError("Dense query response must be a mapping.")
    rows: dict[str, list[object]] = {}
    for field in ("ids", "documents", "metadatas", "distances"):
        column = response.get(field)
        if type(column) is not list or len(column) != 1:
            raise DenseQueryError(f"{field}: expected exactly one query row.")
        if type(column[0]) is not list:
            raise DenseQueryError(f"{field}: query row must be a list.")
        rows[field] = column[0]

    size = len(rows["ids"])
    if any(len(row) != size for row in rows.values()):
        raise DenseQueryError("Dense query result columns are misaligned.")
    if not 1 <= size <= candidate_count:
        raise DenseQueryError("Dense query result count is empty or excessive.")

    seen: set[str] = set()
    candidates: list[DenseCandidate] = []
    for position in range(size):
        chunk_id = rows["ids"][position]
        if type(chunk_id) is not str or not chunk_id.strip():
            raise DenseQueryError("Result ID must be a nonblank string.")
        if chunk_id in seen:
            raise DenseQueryError("Dense query returned duplicate result IDs.")
        seen.add(chunk_id)
        metadata = deserialize_chunk_metadata(rows["metadatas"][position])
        if metadata.chunk_set_id != chunk_set_id:
            raise DenseQueryError("Result metadata has an unexpected ChunkSet ID.")
        # The server filter is not a trust boundary: validate permission again.
        # Dual MDR/IVDR permission is intentionally retained.
        if not metadata.permits_mdr:
            raise DenseQueryError("Result metadata does not permit MDR retrieval.")
        candidates.append(DenseCandidate(
            rank=position + 1,
            chunk_id=chunk_id,
            document=rows["documents"][position],
            metadata=metadata,
            cosine_distance=rows["distances"][position],
        ))
    return tuple(candidates)


class DenseRetriever:
    """Open and query an identity-bound collection using caller-owned objects."""

    def __init__(
        self,
        *,
        client: _QueryClient,
        chunk_set: ChunkSet,
        embedder: _QueryEmbedder,
    ) -> None:
        try:
            config = embedder.config
            # The existing boundary validates ChunkSet/config types and exact
            # collection identity/count; no independent model config is accepted.
            collection = open_dense_collection(
                client=client,
                chunk_set=chunk_set,
                embedding_config=config,
            )
        except Exception as exc:
            raise DenseQueryError("Unable to construct the dense retriever.") from exc
        self._collection = collection
        self._embedder = embedder
        self._dimension = config.dimension
        self._chunk_set_id = chunk_set.chunk_set_id

    def retrieve(
        self,
        query: str,
        *,
        candidate_count: int,
    ) -> tuple[DenseCandidate, ...]:
        """Return validated candidates in Chroma order with an explicit count."""

        if not isinstance(query, str) or not query.strip():
            raise DenseQueryError("query must be a nonblank string.")
        if (
            not isinstance(candidate_count, int)
            or isinstance(candidate_count, bool)
            or candidate_count < 1
        ):
            raise DenseQueryError("candidate_count must be a positive integer.")
        try:
            vector = _query_vector(
                self._embedder.embed_query(query), self._dimension,
            )
        except DenseQueryError:
            raise
        except Exception as exc:
            raise DenseQueryError("Query embedding failed or is invalid.") from exc
        try:
            response = self._collection.query(
                query_embeddings=[vector],
                n_results=candidate_count,
                where={"permits_mdr": True},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise DenseQueryError("Dense collection query failed.") from exc
        try:
            return _candidates(
                response,
                chunk_set_id=self._chunk_set_id,
                candidate_count=candidate_count,
            )
        except DenseQueryError:
            raise
        except Exception as exc:
            raise DenseQueryError(f"Invalid dense query response: {exc}") from exc


__all__ = ["DenseQueryError", "DenseRetriever"]
