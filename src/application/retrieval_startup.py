"""Construct corpus-bound retrievers from explicit caller-owned inputs.

The caller owns dense-client construction, configuration, lifetime, and
closure, as well as the process policy that keeps the embedding model offline.
Constructing :class:`TransformerEmbedder` is lazy: this initializer performs no
embedding or model load. Dense collection opening validates identity metadata
and record count, but it is not a full audit of stored IDs, documents, chunk
metadata, or vectors.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunk_models import ChunkSet, ChunkingConfig
from src.ingestion.chunk_set_builder import build_chunk_set
from src.ingestion.corpus_loader import load_corpus
from src.ingestion.manifest import load_manifest
from src.retrieval.dense_query import DenseRetriever
from src.retrieval.embeddings import EmbeddingModelConfig, TransformerEmbedder
from src.retrieval.lexical_query import BM25Config, BM25Retriever


def initialize_retrieval(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingModelConfig,
    bm25_config: BM25Config,
    dense_client: object,
) -> tuple[ChunkSet, DenseRetriever, BM25Retriever]:
    """Load the authoritative corpus and construct its two retrievers.

    Existing validation and constructor exceptions propagate unchanged. A
    failure stops the flow before any subsequent stage. The supplied dense
    client is only passed to :class:`DenseRetriever`; it is never created,
    configured, replaced, or closed here.
    """

    manifest = load_manifest(
        manifest_path,
        project_root=project_root,
        verify_files=True,
    )
    loaded_sources = load_corpus(manifest, project_root=project_root)
    chunk_set = build_chunk_set(
        manifest,
        loaded_sources,
        chunking_config,
    )
    lexical_retriever = BM25Retriever(
        chunk_set=chunk_set,
        config=bm25_config,
    )
    embedder = TransformerEmbedder(embedding_config)
    dense_retriever = DenseRetriever(
        client=dense_client,
        chunk_set=chunk_set,
        embedder=embedder,
    )
    return chunk_set, dense_retriever, lexical_retriever


__all__ = ["initialize_retrieval"]
