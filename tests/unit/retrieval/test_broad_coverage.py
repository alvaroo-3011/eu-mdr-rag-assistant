"""Focused offline checks for broad-topic routing and validated selection."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.retrieval.broad_coverage import BroadLexicalRetriever, broad_facets
from src.retrieval.evidence_selection import EvidenceSelectionError, select_coverage_evidence
from src.generation.evidence_context import build_evidence_context
from src.ingestion.models import LegalLocatorPart
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.hybrid_query import retrieve_hybrid
from src.retrieval.lexical_query import BM25Config, BM25Retriever
from tests.unit.ingestion.test_chunk_models import _chunk, _chunk_set, _chunk_set_id, _manifest, _mdr_source, _selection
from tests.unit.generation import test_evidence_context as context_fixtures


@pytest.mark.parametrize(("question", "expected"), [
    ("Who is covered by the MDR and what is its scope?", ["1"]),
    ("Explain the device classification framework", ["51", "VIII"]),
    ("Which devices require notified bodies?", ["52", "52", "52"]),
    ("What does the manufacturer's QMS need to address?", ["10"]),
    ("When are device incidents reported?", ["87", "88"]),
    ("Explain the PRRC's duties", ["15"]),
    ("Describe post-market surveillance", ["83", "87", "88"]),
    ("What does a manufacturer do after placing a device on the market?", ["83", "87", "88"]),
    ("How should we monitor a product once it is sold?", ["83", "87", "88"]),
    ("What steps follow a medical device recall?", ["10", "2", "87", "89"]),
    ("How must a manufacturer manage withdrawal of a device?", ["10", "2", "87", "89"]),
])
def test_topic_facets_apply_to_paraphrases(question, expected):
    assert [facet.locator for facet in broad_facets(question)] == expected


def test_explicit_provision_and_ivdr_questions_are_not_expanded():
    assert broad_facets("Under Article 87 what must I report?") == ()
    assert broad_facets("What are IVDR device classifications?") == ()
    assert broad_facets("How do I classify an in vitro device?") == ()
    assert broad_facets("How do I classify an IVD?") == ()
    assert broad_facets("How does FDA device classification work?") == ()
    assert broad_facets("What is the difference between the MDR and the IVDR?") == ()
    assert broad_facets("Compare the two regulations for medical devices and in vitro diagnostics") == ()


def test_missing_facet_hit_does_not_invent_or_reuse_evidence():
    chunk_set = SimpleNamespace(chunks=())
    retriever = SimpleNamespace(chunk_set_id="synthetic", retrieve=Mock(return_value=()))
    wrapper = BroadLexicalRetriever(retriever, chunk_set)
    assert wrapper.retrieve("Explain the device classification framework", candidate_count=5) == ()
    assert wrapper.priority_ids == ()
    assert retriever.retrieve.call_count == 3
    assert all(row["match"] is None for row in wrapper.facet_results)


def test_coverage_selection_preserves_original_fused_ranks_and_rejects_unknown_ids():
    fixture = context_fixtures.EvidenceContextTests(methodName="test_versioned_compact_json_is_deterministic_and_unicode_faithful")
    fixture.setUp()
    try:
        fused = fixture.fused
        result = select_coverage_evidence(chunk_set=fixture.chunk_set, fused_candidates=fused,
                                          max_chunks=2, priority_ids=(fused[-1].chunk_id,))
        assert [x.candidate.rank for x in result.selected] == [1, len(fused)]
        assert result.selected[-1].candidate is fused[-1]
        assert result.selected[-1].chunk is fixture.chunks[1]
        context = build_evidence_context(selection=result, chunk_set=fixture.chunk_set)
        assert context.resolve_citation("2") is fixture.chunks[1]
        assert context.resolve_citation("2").text == fixture.chunks[1].text
        capped = select_coverage_evidence(chunk_set=fixture.chunk_set, fused_candidates=fused,
                                          max_chunks=1, priority_ids=(fused[-1].chunk_id, fused[0].chunk_id))
        assert [x.chunk.chunk_id for x in capped.selected] == [fused[-1].chunk_id]
        with pytest.raises(EvidenceSelectionError):
            select_coverage_evidence(chunk_set=fixture.chunk_set, fused_candidates=fused,
                                     max_chunks=2, priority_ids=("not-retrieved",))
        with pytest.raises(EvidenceSelectionError):
            select_coverage_evidence(chunk_set=fixture.chunk_set, fused_candidates=fused,
                                     max_chunks=2, priority_ids=(fused[0].chunk_id, fused[0].chunk_id))
    finally:
        fixture.doCleanups()


def test_facet_retrieval_and_selection_keep_both_incident_provisions_verbatim():
    source = replace(_mdr_source(), document_id="eu-regulation-2017-745")
    manifest = _manifest(source)
    identity = _chunk_set_id(manifest)
    texts = (
        "Article 87 serious incidents corrective action reporting deadlines unless excluded.",
        "Article 88 trend reporting incidents statistically significant if conditions are met.",
        "General medical device report information.",
    )
    articles = ("87", "88", "2")
    chunks = tuple(_chunk(identity, source, order=i, text=text,
                   block_selections=(_selection(source, f"block-{i}", block_order=i),),
                   legal_locators=((LegalLocatorPart("article", articles[i]), LegalLocatorPart("paragraph", "1")),))
                   for i, text in enumerate(texts))
    chunks = tuple(replace(chunk, previous_chunk_id=chunks[i-1].chunk_id if i else None,
                           next_chunk_id=chunks[i+1].chunk_id if i+1 < len(chunks) else None)
                   for i, chunk in enumerate(chunks))
    chunk_set = _chunk_set(manifest, chunks)
    bm25 = BM25Retriever(chunk_set=chunk_set, config=BM25Config(k1=1.2, b=0.75))
    broad = BroadLexicalRetriever(bm25, chunk_set)
    unrelated = chunks[2]
    dense_candidate = DenseCandidate(1, unrelated.chunk_id, unrelated.text,
                                     deserialize_chunk_metadata(chunk_metadata(unrelated)), 0.1)
    dense = SimpleNamespace(retrieve=lambda query, candidate_count: (dense_candidate,))
    fused = retrieve_hybrid("When should a manufacturer report a problem with a device?",
                            chunk_set=chunk_set, dense_retriever=dense, lexical_retriever=broad,
                            dense_candidate_count=1, lexical_candidate_count=3, rank_constant=2)
    selection = select_coverage_evidence(chunk_set=chunk_set, fused_candidates=fused,
                                         max_chunks=2, priority_ids=broad.priority_ids)
    context = build_evidence_context(selection=selection, chunk_set=chunk_set)
    assert {x.chunk.chunk_id for x in selection.selected} == {chunks[0].chunk_id, chunks[1].chunk_id}
    assert {x.chunk.text for x in context.citations} == {texts[0], texts[1]}
    assert all(context.resolve_citation(c.citation_id) is c.chunk for c in context.citations)
