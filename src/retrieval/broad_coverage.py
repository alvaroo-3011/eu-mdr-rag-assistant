"""Bounded lexical expansion for common broad MDR topics.

Topic cues nominate searches, never answers. Each nominated passage must be
returned by BM25, match a trusted regulation locator, and pass normal fusion
and selection validation. The original dense search remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

BROAD_RETRIEVAL_VERSION = "broad-mdr-facets/v3"


@dataclass(frozen=True)
class Facet:
    query: str
    kind: str
    locator: str
    paragraph: str | None = None
    point: str | None = None


def broad_facets(question: str) -> tuple[Facet, ...]:
    q = question.casefold()
    # Avoid altering explicit provision questions and mixed/IVDR scope handling.
    if re.search(r"\b(?:ivdr|ivds?|in[ -]vitro|fda|article\s+\d+|annex\s+[ivx]+)\b", q):
        return ()
    if re.search(r"\b(?:classif\w*|risk class(?:es)?)\b", q):
        return (Facet("article 51 classification of devices intended purpose inherent risks", "article", "51", "1"),
                Facet("annex VIII implementing classification rules intended purpose duration invasive", "annex", "VIII"))
    if re.search(r"\b(?:notified bod(?:y|ies)|conformity assessment bod(?:y|ies))\b", q):
        return (Facet("article 52 class III conformity assessment annex IX X notified body", "article", "52", "3"),
                Facet("article 52 class IIa IIb notified body conformity assessment", "article", "52", "7"),
                Facet("article 52 class IIb implantable devices exceptions sutures staples dental implants", "article", "52", "4", "a"))
    if re.search(r"\b(?:quality management|quality system|qms)\b", q):
        return (Facet("article 10 9 quality management system manufacturer appropriate proportionate", "article", "10", "9"),)
    if re.search(r"\b(?:recall\w*|withdraw\w*|withdrawal\w*)\b", q):
        return (Facet("article 10 paragraph 12 manufacturer nonconforming recall distributors competent authorities serious risk", "article", "10", "12"),
                Facet("article 2 field safety corrective action definition recall return device user", "article", "2", None, "68"),
                Facet("article 87 reporting field safety corrective action manufacturer", "article", "87", "1"),
                Facet("article 89 field safety corrective action field safety notice users", "article", "89", "8"))
    if (re.search(r"\b(?:report\w*|notif\w*|vigilance)\b", q)
            and re.search(r"\b(?:problem\w*|incident\w*|adverse|event\w*|device\w*|safety|vigilance)\b", q)):
        return (Facet("article 87 serious incidents corrective action reporting deadlines", "article", "87", "1"),
                Facet("article 88 trend reporting incidents statistically significant", "article", "88", "1"))
    if (re.search(r"\b(?:regulation|mdr)\b", q)
            and re.search(r"\b(?:scope|appl\w*|covered|who|overview|what is)\b", q)):
        return (Facet("subject matter scope regulation article 1 devices placing market", "article", "1", "1"),)
    if re.search(r"\b(?:responsible for regulatory compliance|prrc)\b", q):
        return (Facet("article 15 person responsible regulatory compliance qualifications responsibilities", "article", "15", "1"),)
    if (re.search(r"\b(?:post[ -]?market|pms)\b", q)
            or (re.search(r"\b(?:after|once|following)\b", q)
                and re.search(r"\b(?:market|sold|sale|placed|selling)\b", q)
                and re.search(r"\b(?:device|product|manufacturer)\w*\b", q))):
        return (Facet("article 83 post market surveillance system manufacturers collect record analyse data", "article", "83", "1"),
                Facet("article 87 serious incidents field safety corrective action reporting", "article", "87", "1"),
                Facet("article 88 trend reporting incidents statistically significant", "article", "88", "1"))
    return ()


def _matches(chunk, facet: Facet) -> bool:
    if chunk.source.document_id != "eu-regulation-2017-745":
        return False
    return any(any(p.kind == facet.kind and p.value == facet.locator for p in path)
               and (facet.paragraph is None or any(p.kind == "paragraph" and p.value == facet.paragraph for p in path))
               and (facet.point is None or any(p.kind == "point" and p.value == facet.point for p in path))
               for path in chunk.legal_locators)


class BroadLexicalRetriever:
    """Merge original BM25 results with at most four topic-facet BM25 searches.

    Ranks in the returned branch are composite ranks; BM25 scores remain those
    of the query that returned each candidate and are not used by RRF. Missing
    facet hits do not invent evidence or a sufficiency judgment.
    """
    def __init__(self, retriever, chunk_set):
        self._retriever = retriever
        self._chunks = {chunk.chunk_id: chunk for chunk in chunk_set.chunks}
        self.priority_ids: tuple[str, ...] = ()
        self.facet_results: list[dict] = []

    @property
    def chunk_set_id(self):
        return self._retriever.chunk_set_id

    def retrieve(self, query, *, candidate_count):
        base = self._retriever.retrieve(query, candidate_count=candidate_count)
        preferred = []
        self.facet_results = []
        for facet in broad_facets(query):
            rows = self._retriever.retrieve(facet.query, candidate_count=candidate_count)
            match = next((row for row in rows if row.chunk_id in self._chunks
                          and _matches(self._chunks[row.chunk_id], facet)), None)
            self.facet_results.append({"query": facet.query, "kind": facet.kind,
                                       "locator": facet.locator, "paragraph": facet.paragraph,
                                       "point": facet.point,
                                       "candidate_ids": [row.chunk_id for row in rows],
                                       "match": match.chunk_id if match else None,
                                       "match_rank": match.rank if match else None})
            if match is not None and all(row.chunk_id != match.chunk_id for row in preferred):
                preferred.append(match)
        self.priority_ids = tuple(row.chunk_id for row in preferred[:candidate_count])
        merged = list(preferred)
        seen = {row.chunk_id for row in merged}
        for row in base:
            if row.chunk_id not in seen:
                seen.add(row.chunk_id)
                merged.append(row)
        return tuple(replace(row, rank=i) for i, row in enumerate(merged[:candidate_count], 1))
