"""Submission-scoped construction and cleanup for the local MDR application."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import sys

import httpx

from src.application.chroma_startup import open_existing_chroma_client
from src.application.local_config import LocalAppConfig, validate_local_app_paths
from src.application.question_answering import QuestionAnsweringResult, answer_question
from src.application.retrieval_startup import initialize_retrieval
from src.generation.gemini_transport import (
    GeminiRESTTransport,
    GeminiTransportError,
    validate_gemini_api_key,
)


class LocalCredentialError(ValueError):
    """The execution-boundary Gemini credential is absent or invalid."""


class LocalRuntimeError(RuntimeError):
    """A caller-owned local runtime resource violates its close contract."""


def load_gemini_api_key(environment: Mapping[str, str]) -> str:
    """Read only GEMINI_API_KEY from an explicitly supplied environment mapping."""

    try:
        value = environment.get("GEMINI_API_KEY")
        return validate_gemini_api_key(value)
    except Exception:
        raise LocalCredentialError("GEMINI_API_KEY is missing or invalid.") from None


def _default_http_transport_factory(*, verify: bool, retries: int) -> httpx.BaseTransport:
    return httpx.HTTPTransport(verify=verify, trust_env=False, retries=retries)


def _close_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        raise LocalRuntimeError("Constructed resource does not expose close().")
    close()


def _close_without_masking(resource: object) -> None:
    failure_active = sys.exc_info()[0] is not None
    try:
        _close_resource(resource)
    except Exception:
        if not failure_active:
            raise


def run_local_question(
    *,
    config: LocalAppConfig,
    question: str,
    api_key: str,
    chroma_client_factory: Callable[..., object] = open_existing_chroma_client,
    retrieval_initializer: Callable[..., tuple[object, object, object]] = initialize_retrieval,
    http_transport_factory: Callable[..., httpx.BaseTransport] = _default_http_transport_factory,
    adapter_factory: Callable[..., GeminiRESTTransport] = GeminiRESTTransport,
    answerer: Callable[..., QuestionAnsweringResult] = answer_question,
) -> QuestionAnsweringResult:
    """Construct, run once, and close all resources for one explicit question.

    A live call with selected evidence performs the normal fresh
    complete-request count and, only when the existing application budget gate
    approves it, one generation. A validated empty selection returns the fixed
    local non-answer without either provider operation. There is no retry,
    fallback, model switch, cached approval, or resource cache.
    Ordinary Chroma startup is not genuinely read-only and may write only to
    the explicitly configured persistence directory.
    """

    if type(question) is not str or not question.strip():
        raise LocalRuntimeError("question must be a nonblank string.")
    validate_local_app_paths(config)
    try:
        api_key = validate_gemini_api_key(api_key)
    except GeminiTransportError:
        raise LocalCredentialError("GEMINI_API_KEY is missing or invalid.") from None

    client = chroma_client_factory(
        persistence_directory=config.persistence_directory,
        tenant=config.chroma_tenant,
        database=config.chroma_database,
        migration_hash_algorithm=config.migration_hash_algorithm,
    )
    try:
        chunk_set, dense_retriever, lexical_retriever = retrieval_initializer(
            project_root=config.project_root,
            manifest_path=config.manifest_path,
            chunking_config=config.chunking_config,
            embedding_config=config.embedding_config,
            bm25_config=config.bm25_config,
            dense_client=client,
        )
        transport = http_transport_factory(
            verify=config.http_backend_config.verify_tls,
            retries=config.http_backend_config.retries,
        )
        try:
            adapter = adapter_factory(
                api_key=api_key,
                config=config.transport_config,
                transport=transport,
            )
        except BaseException:
            _close_without_masking(transport)
            raise
        try:
            retrieval = config.retrieval_config
            return answerer(
                question=question,
                chunk_set=chunk_set,
                dense_retriever=dense_retriever,
                lexical_retriever=lexical_retriever,
                dense_candidate_count=retrieval.dense_candidate_count,
                lexical_candidate_count=retrieval.lexical_candidate_count,
                rank_constant=retrieval.rank_constant,
                max_chunks=retrieval.max_chunks,
                generation_config=config.generation_config,
                budget=config.budget,
                adapter=adapter,
            )
        finally:
            _close_without_masking(adapter)
    finally:
        _close_without_masking(client)


__all__ = [
    "LocalCredentialError", "LocalRuntimeError", "load_gemini_api_key",
    "run_local_question",
]
