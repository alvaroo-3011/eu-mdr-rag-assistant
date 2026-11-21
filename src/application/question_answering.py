"""Compose one explicit question-to-validated-answer application flow.

All corpus, retrieval, generation, budget, credential, and transport choices are
caller-owned. A validated empty selection returns one fixed trusted local
non-answer. A narrow, explicit MDR device question may first receive a fixed
local clarification. Otherwise, invoking :func:`answer_question` with a live adapter
performs one complete-request token count and, if the existing budget gate
approves it, one generation operation. This module adds no retry, fallback,
general semantic routing or sufficiency policy, persistence, or adapter lifecycle
management.
"""

from __future__ import annotations

from src.application.answer_policy import (
    LocalInsufficientEvidence,
    LocalNeedsClarification,
    local_device_clarification,
    local_empty_evidence_result,
)
from src.generation.evidence_context import build_evidence_context
from src.generation.gemini_request import GeminiGenerationConfig, prepare_gemini_request
from src.generation.gemini_transport import (
    GeminiRESTTransport,
    GeminiTransportRun,
    run_gemini_request,
)
from src.generation.request_budget import RequestBudget
from src.ingestion.chunk_models import ChunkSet
from src.retrieval.broad_coverage import BroadLexicalRetriever
from src.retrieval.evidence_selection import select_evidence, select_coverage_evidence
from src.retrieval.hybrid_query import retrieve_hybrid


QuestionAnsweringResult = GeminiTransportRun | LocalInsufficientEvidence | LocalNeedsClarification


def answer_question(
    *,
    question: str,
    chunk_set: ChunkSet,
    dense_retriever: object,
    lexical_retriever: object,
    dense_candidate_count: int,
    lexical_candidate_count: int,
    rank_constant: float,
    max_chunks: int,
    generation_config: GeminiGenerationConfig,
    budget: RequestBudget,
    adapter: GeminiRESTTransport,
) -> QuestionAnsweringResult:
    """Retrieve, select, prepare, budget, generate, and validate one answer.

    The retrievers must satisfy the structural interfaces enforced by
    ``retrieve_hybrid``. The caller owns their construction and the supplied
    adapter's construction and closure. Narrow device clarification precedes
    retrieval. A successfully validated empty
    selection returns locally before context or provider-request preparation.
    Existing stage exceptions propagate unchanged, and each later stage is
    skipped after a failure.
    """

    clarification = local_device_clarification(question)
    if clarification is not None:
        return clarification

    broad_lexical = BroadLexicalRetriever(lexical_retriever, chunk_set)
    fused_candidates = retrieve_hybrid(
        question,
        chunk_set=chunk_set,
        dense_retriever=dense_retriever,
        lexical_retriever=broad_lexical,
        dense_candidate_count=dense_candidate_count,
        lexical_candidate_count=lexical_candidate_count,
        rank_constant=rank_constant,
    )
    selector = select_coverage_evidence if broad_lexical.priority_ids else select_evidence
    options = {"priority_ids": broad_lexical.priority_ids} if broad_lexical.priority_ids else {}
    selection = selector(
        chunk_set=chunk_set,
        fused_candidates=fused_candidates,
        max_chunks=max_chunks,
        **options,
    )
    local_result = local_empty_evidence_result(selection)
    if local_result is not None:
        return local_result
    context = build_evidence_context(selection=selection, chunk_set=chunk_set)
    request = prepare_gemini_request(
        question=question,
        context=context,
        chunk_set=chunk_set,
        config=generation_config,
    )
    return run_gemini_request(
        request=request,
        chunk_set=chunk_set,
        budget=budget,
        adapter=adapter,
    )


__all__ = ["QuestionAnsweringResult", "answer_question"]
