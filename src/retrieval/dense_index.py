"""Build or validate opening of an identity-bound Chroma dense index.

The builder accepts precomputed embeddings and writes a fresh collection. It
does not load source files, create chunks, run an embedding model, query the
index, or replace an existing collection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.ingestion.chunk_models import Chunk, ChunkSet
from src.ingestion.models import RegulatoryRegime
from src.retrieval.embeddings import EmbeddingModelConfig
from src.retrieval.dense_results import CHUNK_METADATA_KEYS


class DenseIndexError(ValueError):
    """Raised when a dense index cannot be built or opened trustworthily."""


@dataclass(frozen=True, slots=True)
class DenseIndexDescriptor:
    """Identity and size of one successfully built dense collection."""

    collection_name: str
    chunk_set_id: str
    embedding_model_id: str
    embedding_revision: str
    embedding_dimension: int
    record_count: int


class _StoredCollection(Protocol):
    @property
    def metadata(self) -> object:
        """Return stored metadata for identity validation."""

    def count(self) -> int:
        """Return the stored record count."""


_CollectionT = TypeVar("_CollectionT", bound=_StoredCollection, covariant=True)


class _CollectionReader(Protocol[_CollectionT]):
    def get_collection(
        self,
        *,
        name: str,
        embedding_function: None,
    ) -> _CollectionT:
        """Open an existing collection with managed embeddings disabled."""


class _Collection(_StoredCollection, Protocol):
    def add(
        self,
        *,
        ids: list[str],
        embeddings: NDArray[np.float32],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        """Add one ordered batch of complete records."""


class _Client(Protocol):
    def create_collection(
        self,
        *,
        name: str,
        configuration: dict[str, object],
        metadata: dict[str, object],
        embedding_function: None,
        get_or_create: bool,
    ) -> _Collection:
        """Create a fresh collection."""

    def delete_collection(self, name: str) -> None:
        """Delete a collection created by this build attempt."""

    def get_max_batch_size(self) -> int:
        """Return the storage engine's current safe write-batch size."""


def _json(value: object) -> str:
    """Serialize structured metadata canonically for scalar-only storage."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hierarchies(chunk: Chunk) -> list[list[dict[str, object]]]:
    """Return ordered hierarchy paths as JSON-ready values."""

    return [
        [
            {
                "level": entry.level,
                "kind": entry.kind,
                "block_id": entry.block_id,
                "title": entry.title,
            }
            for entry in path
        ]
        for path in chunk.hierarchies
    ]


def _legal_locators(chunk: Chunk) -> list[list[dict[str, str]]]:
    """Return ordered legal-locator paths as JSON-ready values."""

    return [
        [
            {
                "kind": part.kind,
                "value": part.value,
            }
            for part in path
        ]
        for path in chunk.legal_locators
    ]


def _annotations(chunk: Chunk) -> list[dict[str, object]]:
    """Return annotation provenance without adding marker text to evidence."""

    return [
        {
            "kind": annotation.kind.value,
            "identifier": annotation.identifier,
            "target_id": annotation.target_id,
            "text": annotation.text,
        }
        for annotation in chunk.annotations
    ]


def chunk_metadata(chunk: Chunk) -> dict[str, object]:
    """Return validated Chroma-compatible scalar metadata for one chunk."""

    if not isinstance(chunk, Chunk):
        raise DenseIndexError(
            "chunk must be a Chunk."
        )

    source = chunk.source
    regulatory_regimes = tuple(
        regime.value
        for regime in source.regulatory_regimes
    )
    permitted_regimes = tuple(
        regime.value
        for regime in source.permitted_regimes
    )

    metadata = {
        "schema_version": "1.0",
        "chunk_id": chunk.chunk_id,
        "chunk_set_id": chunk.chunk_set_id,
        "chunk_order": chunk.order,
        "document_id": source.document_id,
        "official_title": source.official_title,
        "short_title": source.short_title,
        "version": source.version,
        "source_type": source.source_type.value,
        "authority": source.authority,
        "authority_tier": source.authority_tier.value,
        "official_url": source.official_url,
        "source_sha256": source.sha256,
        "regulatory_regimes_json": _json(regulatory_regimes),
        "permitted_regimes_json": _json(permitted_regimes),
        "permits_mdr": RegulatoryRegime.MDR in source.permitted_regimes,
        "permits_ivdr": RegulatoryRegime.IVDR in source.permitted_regimes,
        "pages_json": _json(
            [
                {
                    "source_index": page.source_index,
                    "pdf_page_number": page.pdf_page_number,
                }
                for page in chunk.pages
            ]
        ),
        "hierarchies_json": _json(_hierarchies(chunk)),
        "legal_locators_json": _json(_legal_locators(chunk)),
        "block_ids_json": _json(
            [
                selection.block_id
                for selection in chunk.block_selections
            ]
        ),
        "annotations_json": _json(_annotations(chunk)),
        "has_table": bool(chunk.table_segments),
        "previous_chunk_id": chunk.previous_chunk_id or "",
        "next_chunk_id": chunk.next_chunk_id or "",
    }
    if metadata.keys() != CHUNK_METADATA_KEYS:
        raise DenseIndexError("Chunk metadata keys do not match the stored schema.")
    return metadata


def collection_metadata(
    chunk_set: ChunkSet,
    embedding_config: EmbeddingModelConfig,
) -> dict[str, object]:
    """Return the complete immutable identity contract for a collection."""

    if not isinstance(chunk_set, ChunkSet):
        raise DenseIndexError(
            "chunk_set must be a ChunkSet."
        )
    if not isinstance(embedding_config, EmbeddingModelConfig):
        raise DenseIndexError(
            "embedding_config must be an EmbeddingModelConfig."
        )

    return {
        "schema_version": "1.0",
        "chunk_set_id": chunk_set.chunk_set_id,
        "corpus_identity": chunk_set.corpus_identity,
        "chunking_config_id": chunk_set.chunking_config_id,
        "embedding_model_id": embedding_config.model_id,
        "embedding_revision": embedding_config.revision,
        "embedding_dimension": embedding_config.dimension,
        "distance_metric": "cosine",
    }


def dense_collection_name(
    chunk_set: ChunkSet,
    embedding_config: EmbeddingModelConfig,
) -> str:
    """Derive one Chroma-safe name from all index-defining identities."""

    if not isinstance(chunk_set, ChunkSet):
        raise DenseIndexError("chunk_set must be a ChunkSet.")
    if not isinstance(embedding_config, EmbeddingModelConfig):
        raise DenseIndexError(
            "embedding_config must be an EmbeddingModelConfig."
        )

    identity = {
        "chunk_set_id": chunk_set.chunk_set_id,
        "embedding_model_id": embedding_config.model_id,
        "embedding_revision": embedding_config.revision,
        "embedding_dimension": embedding_config.dimension,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    return f"mdr-dense-{digest}"


def open_dense_collection(
    *,
    client: _CollectionReader[_CollectionT],
    chunk_set: ChunkSet,
    embedding_config: EmbeddingModelConfig,
) -> _CollectionT:
    """Open an existing collection only after identity and count validation.

    Client construction belongs to the caller. This check does not replace
    a full audit of stored IDs, documents, chunk metadata, or vectors.
    """

    expected_metadata = collection_metadata(chunk_set, embedding_config)
    collection_name = dense_collection_name(chunk_set, embedding_config)

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception as exc:
        raise DenseIndexError(
            "Unable to open the expected dense collection; it may be missing "
            f"or unavailable: collection_name={collection_name!r}."
        ) from exc

    try:
        metadata = collection.metadata
    except Exception as exc:
        raise DenseIndexError(
            "Unable to read dense collection identity metadata."
        ) from exc

    if not isinstance(metadata, dict):
        raise DenseIndexError(
            "Dense collection identity metadata is missing or malformed."
        )
    if metadata.keys() != expected_metadata.keys():
        raise DenseIndexError(
            "Dense collection identity metadata keys do not match "
            "the expected contract."
        )
    for field, expected_value in expected_metadata.items():
        actual_value = metadata[field]
        if (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            raise DenseIndexError(
                f"Dense collection identity metadata mismatch: field={field}."
            )

    try:
        actual_count = collection.count()
    except Exception as exc:
        raise DenseIndexError(
            "Unable to read dense collection record count."
        ) from exc

    expected_count = len(chunk_set.chunks)
    if not isinstance(actual_count, int) or isinstance(actual_count, bool):
        raise DenseIndexError("Dense collection returned an invalid record count.")
    if actual_count != expected_count:
        raise DenseIndexError(
            "Dense collection record count mismatch: "
            f"expected={expected_count}, actual={actual_count}."
        )

    return collection


def _validated_embeddings(
    embeddings: object,
    *,
    record_count: int,
    dimension: int,
) -> NDArray[np.float32]:
    """Return a finite float32 matrix with exact expected dimensions."""

    try:
        vectors = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise DenseIndexError(
            "embeddings must be convertible to a numeric matrix."
        ) from exc

    expected_shape = (record_count, dimension)
    if vectors.shape != expected_shape:
        raise DenseIndexError(
            "Embedding matrix has an unexpected shape: "
            f"expected={expected_shape}, actual={vectors.shape}."
        )
    if not np.isfinite(vectors).all():
        raise DenseIndexError(
            "Embedding matrix contains a non-finite value."
        )

    return vectors


def build_dense_index(
    *,
    client: _Client,
    collection_name: str,
    chunk_set: ChunkSet,
    embedding_config: EmbeddingModelConfig,
    embeddings: object,
) -> DenseIndexDescriptor:
    """Create and completely populate one fresh cosine collection."""

    if not isinstance(collection_name, str) or not collection_name.strip():
        raise DenseIndexError(
            "collection_name must be a non-empty string."
        )
    if not isinstance(chunk_set, ChunkSet):
        raise DenseIndexError(
            "chunk_set must be a ChunkSet."
        )
    if not isinstance(embedding_config, EmbeddingModelConfig):
        raise DenseIndexError(
            "embedding_config must be an EmbeddingModelConfig."
        )

    chunks = chunk_set.chunks
    vectors = _validated_embeddings(
        embeddings,
        record_count=len(chunks),
        dimension=embedding_config.dimension,
    )
    metadata = collection_metadata(chunk_set, embedding_config)
    created = False

    try:
        collection = client.create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=metadata,
            embedding_function=None,
            get_or_create=False,
        )
        created = True

        max_batch_size = client.get_max_batch_size()
        if (
            not isinstance(max_batch_size, int)
            or isinstance(max_batch_size, bool)
            or max_batch_size < 1
        ):
            raise DenseIndexError(
                "Chroma returned an invalid maximum batch size."
            )

        for start in range(0, len(chunks), max_batch_size):
            stop = min(start + max_batch_size, len(chunks))
            batch = chunks[start:stop]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=vectors[start:stop],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk_metadata(chunk) for chunk in batch],
            )

        actual_count = collection.count()
        if actual_count != len(chunks):
            raise DenseIndexError(
                "Dense collection record count is incomplete: "
                f"expected={len(chunks)}, actual={actual_count}."
            )
    except Exception as exc:
        if created:
            try:
                client.delete_collection(collection_name)
            except Exception as cleanup_exc:
                raise DenseIndexError(
                    "Dense-index construction failed and rollback also failed: "
                    f"collection_name={collection_name!r}."
                ) from cleanup_exc

        if isinstance(exc, DenseIndexError):
            raise
        raise DenseIndexError(
            "Dense-index construction failed: "
            f"collection_name={collection_name!r}."
        ) from exc

    return DenseIndexDescriptor(
        collection_name=collection_name,
        chunk_set_id=chunk_set.chunk_set_id,
        embedding_model_id=embedding_config.model_id,
        embedding_revision=embedding_config.revision,
        embedding_dimension=embedding_config.dimension,
        record_count=len(chunks),
    )


__all__ = [
    "DenseIndexDescriptor",
    "DenseIndexError",
    "build_dense_index",
    "chunk_metadata",
    "collection_metadata",
    "dense_collection_name",
    "open_dense_collection",
]
