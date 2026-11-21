"""Build or independently audit the persistent EU MDR dense index.

Run ``build`` once to validate the frozen corpus, construct its ChunkSet,
embed every chunk, and create a fresh identity-bound Chroma collection.
Run ``audit`` in a separate Python process to reopen that collection and
verify its identity, records, citation metadata, documents, and vectors.

The current frozen corpus contains no retrievable block that requires
splitting. The baseline therefore rejects unexpected oversized text. When
the approved sentence-first/whitespace splitter is implemented, its changed
algorithm version will deliberately produce a new ChunkSet and index identity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)


from src.ingestion.chunking_profiles import (  # noqa: E402
    BASELINE_CHUNKING_CONFIG,
)
from src.ingestion.chunk_set_builder import build_chunk_set  # noqa: E402
from src.ingestion.corpus_loader import load_corpus  # noqa: E402
from src.ingestion.manifest import load_manifest  # noqa: E402
from src.retrieval.dense_index import (  # noqa: E402
    build_dense_index,
    chunk_metadata,
    dense_collection_name,
    open_dense_collection,
)
from src.retrieval.embeddings import (  # noqa: E402
    GTE_MODERNBERT_BASE,
    TransformerEmbedder,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"
INDEX_DIRECTORY = PROJECT_ROOT / "data" / "indexes" / "chroma"


class PersistentIndexAuditError(RuntimeError):
    """Raised when a persisted collection violates its build contract."""


def require(condition: bool, message: str) -> None:
    """Fail with one actionable audit message when condition is false."""

    if not condition:
        raise PersistentIndexAuditError(message)


def load_baseline_chunk_set():
    """Load the exact manifest-approved corpus and build its ChunkSet."""

    manifest = load_manifest(
        MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        verify_files=True,
    )
    loaded_sources = load_corpus(
        manifest,
        project_root=PROJECT_ROOT,
    )
    return build_chunk_set(
        manifest,
        loaded_sources,
        BASELINE_CHUNKING_CONFIG,
    )


def _chroma_module():
    """Import Chroma only when a persistent-index command requires it."""

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is required to build or audit the dense index."
        ) from exc

    return chromadb


def _existing_collection_names(client) -> set[str]:
    """Normalize collection listings across supported Chroma return forms."""

    names: set[str] = set()
    for item in client.list_collections():
        name = item if isinstance(item, str) else getattr(item, "name", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                "Chroma returned an unsupported collection listing."
            )
        names.add(name)
    return names


def build_command(*, batch_size: int) -> None:
    """Build one fresh persistent collection for the validated corpus."""

    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    chunk_set = load_baseline_chunk_set()
    collection_name = dense_collection_name(chunk_set, GTE_MODERNBERT_BASE)

    INDEX_DIRECTORY.mkdir(parents=True, exist_ok=True)
    chromadb = _chroma_module()
    client = chromadb.PersistentClient(path=str(INDEX_DIRECTORY))

    if collection_name in _existing_collection_names(client):
        raise RuntimeError(
            "The exact identity-bound collection already exists; refusing "
            f"to overwrite it: {collection_name!r}."
        )

    chunks = chunk_set.chunks
    vectors = np.empty(
        (len(chunks), GTE_MODERNBERT_BASE.dimension),
        dtype=np.float32,
    )
    embedder = TransformerEmbedder(GTE_MODERNBERT_BASE)

    print("BUILD INPUT")
    print(f"sources: {len(chunk_set.manifest.sources)}")
    print(f"chunks: {len(chunks)}")
    print(f"chunk_set_id: {chunk_set.chunk_set_id}")
    print(f"collection_name: {collection_name}")
    print(f"index_directory: {INDEX_DIRECTORY}")
    print(f"embedding_batch_size: {batch_size}")
    print()
    print("Embedding chunks on CPU...")

    for start in range(0, len(chunks), batch_size):
        stop = min(start + batch_size, len(chunks))
        vectors[start:stop] = embedder.embed_documents(
            (chunk.text for chunk in chunks[start:stop]),
            batch_size=batch_size,
        )
        print(f"embedded: {stop}/{len(chunks)}", flush=True)

    descriptor = build_dense_index(
        client=client,
        collection_name=collection_name,
        chunk_set=chunk_set,
        embedding_config=GTE_MODERNBERT_BASE,
        embeddings=vectors,
    )

    print()
    print("BUILD PASSED")
    print(f"collection_name: {descriptor.collection_name}")
    print(f"records: {descriptor.record_count}")
    print(f"embedding_dimension: {descriptor.embedding_dimension}")
    print("Run the audit command in a new Python process before committing.")


def _read_all_records(collection, *, record_count: int, batch_size: int):
    """Read every stored record in bounded batches without assuming order."""

    records: dict[str, tuple[str, dict[str, object], np.ndarray]] = {}

    for offset in range(0, record_count, batch_size):
        result = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=min(batch_size, record_count - offset),
            offset=offset,
        )
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        embeddings = result.get("embeddings")

        require(isinstance(ids, list), "Chroma did not return record IDs.")
        require(documents is not None, "Chroma did not return documents.")
        require(metadatas is not None, "Chroma did not return metadata.")
        require(embeddings is not None, "Chroma did not return embeddings.")
        require(
            len(ids) == len(documents) == len(metadatas) == len(embeddings),
            "A persisted read returned misaligned record columns.",
        )

        for chunk_id, document, metadata, vector in zip(
            ids,
            documents,
            metadatas,
            embeddings,
        ):
            require(
                chunk_id not in records,
                f"Duplicate persisted chunk ID: {chunk_id!r}.",
            )
            require(
                isinstance(document, str),
                f"Persisted document is not text: chunk_id={chunk_id!r}.",
            )
            require(
                isinstance(metadata, dict),
                f"Persisted metadata is invalid: chunk_id={chunk_id!r}.",
            )
            records[chunk_id] = (
                document,
                metadata,
                np.asarray(vector, dtype=np.float32),
            )

    return records


def audit_command(*, read_batch_size: int) -> None:
    """Reopen and fully audit the expected persistent collection."""

    if read_batch_size < 1:
        raise ValueError("read_batch_size must be a positive integer.")

    require(
        INDEX_DIRECTORY.is_dir(),
        f"Dense-index directory does not exist: {INDEX_DIRECTORY}.",
    )

    chunk_set = load_baseline_chunk_set()
    collection_name = dense_collection_name(chunk_set, GTE_MODERNBERT_BASE)
    chromadb = _chroma_module()
    client = chromadb.PersistentClient(path=str(INDEX_DIRECTORY))

    collection = open_dense_collection(
        client=client,
        chunk_set=chunk_set,
        embedding_config=GTE_MODERNBERT_BASE,
    )

    expected_chunks = {
        chunk.chunk_id: chunk
        for chunk in chunk_set.chunks
    }
    actual_count = collection.count()

    records = _read_all_records(
        collection,
        record_count=actual_count,
        batch_size=read_batch_size,
    )
    require(
        records.keys() == expected_chunks.keys(),
        "Persisted chunk IDs do not exactly match the current ChunkSet.",
    )

    minimum_norm = float("inf")
    maximum_norm = 0.0

    for chunk_id, chunk in expected_chunks.items():
        document, metadata, vector = records[chunk_id]
        require(
            document == chunk.text,
            f"Persisted document text changed: chunk_id={chunk_id!r}.",
        )
        require(
            metadata == chunk_metadata(chunk),
            f"Persisted metadata changed: chunk_id={chunk_id!r}.",
        )
        require(
            vector.shape == (GTE_MODERNBERT_BASE.dimension,),
            "Persisted vector dimension is invalid: "
            f"chunk_id={chunk_id!r}, shape={vector.shape}.",
        )
        require(
            bool(np.isfinite(vector).all()),
            f"Persisted vector is non-finite: chunk_id={chunk_id!r}.",
        )

        norm = float(np.linalg.norm(vector))
        require(
            bool(np.isclose(norm, 1.0, rtol=0.01, atol=0.01)),
            "Persisted vector is not normalized: "
            f"chunk_id={chunk_id!r}, norm={norm}.",
        )
        minimum_norm = min(minimum_norm, norm)
        maximum_norm = max(maximum_norm, norm)

    print("AUDIT PASSED")
    print(f"collection_name: {collection_name}")
    print(f"sources: {len(chunk_set.manifest.sources)}")
    print(f"records: {actual_count}")
    print(f"embedding_dimension: {GTE_MODERNBERT_BASE.dimension}")
    print(f"minimum_vector_norm: {minimum_norm:.6f}")
    print(f"maximum_vector_norm: {maximum_norm:.6f}")
    print("Collection identity, IDs, text, citation metadata, and vectors match.")


def positive_integer(value: str) -> int:
    """Parse one positive command-line integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )

    return parsed


def parse_args() -> argparse.Namespace:
    """Return the explicitly selected build or audit command."""

    parser = argparse.ArgumentParser(
        description="Build or audit the persistent EU MDR dense index.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Embed the corpus and create a fresh persistent collection.",
    )
    build_parser.add_argument(
        "--batch-size",
        required=True,
        type=positive_integer,
        help="Explicit CPU embedding batch size.",
    )

    audit_parser = subparsers.add_parser(
        "audit",
        help="Reopen and fully validate the persistent collection.",
    )
    audit_parser.add_argument(
        "--read-batch-size",
        type=positive_integer,
        default=500,
        help="Number of stored records read per audit batch (default: 500).",
    )

    return parser.parse_args()


def main() -> None:
    """Execute exactly one requested persistent-index operation."""

    args = parse_args()

    if args.command == "build":
        build_command(batch_size=args.batch_size)
    else:
        audit_command(read_batch_size=args.read_batch_size)


if __name__ == "__main__":
    main()
