"""In-memory, identity-bound BM25 over original MDR-permitted chunk text.

Documents and queries use NFKC, casefold, and Unicode alphanumeric runs.
Punctuation splits identifiers; numbers, single characters, and stopwords stay.
No stemming, accent removal, phrase matching, or identifier/acronym expansion
is performed: partial identifier matches and unmatched inflections are expected.
Only document repetitions affect frequency; query tokens are sorted and unique.

The exact score is sum over overlapping unique query terms of
    log(1 + (N - df + 0.5) / (df + 0.5))
    * f * (k1 + 1) / (f + k1 * (1 - b + b * L / average_length)).
N, df, and average_length include only MDR-permitted chunks. This positive-IDF,
k1-plus-one variant is not a claim of Lucene-identical scoring. Match eligibility
uses token overlap, never a score threshold. k1 and b are explicit experimental
inputs, not tuned production defaults. There is no persistence, inverted index,
corpus loading, model execution, or database access here.

Tokenless eligible documents and queries fail; no eligible documents or no
overlap return an empty tuple. Fixed behavior versions and policies below form
part of index identity and must change when their implemented semantics change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from types import MappingProxyType
import unicodedata

from src.ingestion.chunk_models import Chunk, ChunkSet
from src.ingestion.models import RegulatoryRegime
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import (
    TrustedChunkMetadata, deserialize_chunk_metadata,
)


_TOKENIZER_VERSION = "unicode-nfkc-casefold-alnum-v1"
_SCORER_VERSION = "bm25-positive-idf-k1-plus-one-v1"
_POLICIES = (
    ("filter", "permitted-regimes-mdr-v1"),
    ("text", "original-chunk-text-v1"),
    ("query_tokens", "sorted-unique-v1"),
    ("tie_order", "original-chunkset-position-v1"),
    ("tokenless_query", "error-v1"),
    ("tokenless_document", "error-v1"),
)
_TOKEN_PATTERN = re.compile(r"[^\W_]+")


class LexicalError(ValueError):
    """Raised when lexical configuration, indexing, or retrieval is invalid."""


def _finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LexicalError(f"{field}: must be a finite real nonboolean number.")
    try:
        result = float(value)
    except Exception as exc:
        raise LexicalError(f"{field}: cannot be represented as float.") from exc
    if not math.isfinite(result):
        raise LexicalError(f"{field}: must be finite.")
    # Equivalent zero configurations must have the same canonical JSON.
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True, kw_only=True)
class BM25Config:
    """Explicit finite parameters: k1 > 0 and 0 <= b <= 1, without defaults."""

    k1: float
    b: float

    def __post_init__(self) -> None:
        k1 = _finite_real(self.k1, "k1")
        b = _finite_real(self.b, "b")
        if k1 <= 0:
            raise LexicalError("k1: must be positive.")
        if not 0 <= b <= 1:
            raise LexicalError("b: must be between zero and one inclusive.")
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "b", b)


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    """One immutable trusted lexical result; score is not cosine distance."""

    rank: int
    chunk_id: str
    document: str
    metadata: TrustedChunkMetadata
    bm25_score: float

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise LexicalError("rank: must be a positive nonboolean integer.")
        for field in ("chunk_id", "document"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise LexicalError(f"{field}: must be a nonblank string.")
        if not isinstance(self.metadata, TrustedChunkMetadata):
            raise LexicalError("metadata: must be TrustedChunkMetadata.")
        if self.chunk_id != self.metadata.chunk_id:
            raise LexicalError("chunk_id: must match trusted metadata.")
        # No score sign threshold: overlap is the retriever's match boundary.
        object.__setattr__(
            self, "bm25_score", _finite_real(self.bm25_score, "bm25_score"),
        )


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold()))


def _index_identity(
    chunk_set_id: str, chunk_ids: tuple[str, ...], config: BM25Config,
) -> str:
    payload = {
        "schema_version": "1.0",
        "chunk_set_id": chunk_set_id,
        "eligible_chunk_ids": chunk_ids,
        "k1": config.k1,
        "b": config.b,
        "tokenizer_version": _TOKENIZER_VERSION,
        "scorer_version": _SCORER_VERSION,
        "unicode_database_version": unicodedata.unidata_version,
        "policies": dict(_POLICIES),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return "lexical-index:sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_metadata(
    metadata: TrustedChunkMetadata, chunk: Chunk, chunk_set_id: str,
) -> None:
    if not isinstance(metadata, TrustedChunkMetadata):
        raise LexicalError("metadata: must be TrustedChunkMetadata.")
    if metadata.chunk_set_id != chunk_set_id:
        raise LexicalError("metadata: unexpected ChunkSet identity.")
    if metadata.chunk_id != chunk.chunk_id:
        raise LexicalError("metadata: chunk ID does not match the original chunk.")
    if not metadata.permits_mdr:
        raise LexicalError("metadata: MDR retrieval is not permitted.")


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    position: int
    chunk: Chunk
    metadata: TrustedChunkMetadata
    frequencies: Mapping[str, int]
    length: int


class BM25Retriever:
    """Build a local index without mutating the supplied validated ChunkSet.

    Construction is O(total eligible tokens). Queries scan O(N * unique query
    tokens), then sort overlapping chunks. No external backend is injected.
    """

    __slots__ = (
        "_config", "_chunk_set_id", "_index_id", "_records", "_idf",
        "_average_length",
    )

    def __init__(self, *, chunk_set: ChunkSet, config: BM25Config) -> None:
        try:
            if not isinstance(chunk_set, ChunkSet):
                raise LexicalError("chunk_set must be a validated ChunkSet.")
            if not isinstance(config, BM25Config):
                raise LexicalError("config must be BM25Config.")
            records: list[_IndexedChunk] = []
            document_frequencies: Counter[str] = Counter()
            for position, chunk in enumerate(chunk_set.chunks):
                # Filter BEFORE tokenization, metadata work, and all statistics.
                if RegulatoryRegime.MDR not in chunk.source.permitted_regimes:
                    continue
                tokens = _tokenize(chunk.text)
                if not tokens:
                    raise LexicalError(f"Eligible chunk has no tokens: {chunk.chunk_id}")
                metadata = deserialize_chunk_metadata(chunk_metadata(chunk))
                _validate_metadata(metadata, chunk, chunk_set.chunk_set_id)
                frequencies = Counter(tokens)
                document_frequencies.update(frequencies.keys())
                records.append(_IndexedChunk(
                    position, chunk, metadata, MappingProxyType(dict(frequencies)),
                    len(tokens),
                ))
            size = len(records)
            average_length = sum(record.length for record in records) / size if size else 0.0
            idf = {
                token: _finite_real(
                    math.log1p((size - frequency + 0.5) / (frequency + 0.5)),
                    "IDF",
                )
                for token, frequency in document_frequencies.items()
            }
            index_id = _index_identity(
                chunk_set.chunk_set_id,
                tuple(record.chunk.chunk_id for record in records), config,
            )
        except LexicalError:
            raise
        except Exception as exc:
            raise LexicalError("Unable to construct the lexical retriever.") from exc
        self._config = config
        self._chunk_set_id = chunk_set.chunk_set_id
        self._index_id = index_id
        self._records = tuple(records)
        self._idf = MappingProxyType(idf)
        self._average_length = average_length

    @property
    def config(self) -> BM25Config:
        return self._config

    @property
    def chunk_set_id(self) -> str:
        return self._chunk_set_id

    @property
    def index_id(self) -> str:
        return self._index_id

    def _score(self, record: _IndexedChunk, query_tokens: tuple[str, ...]) -> float:
        normalization = _finite_real(
            self.config.k1 * (
                1 - self.config.b + self.config.b * record.length / self._average_length
            ), "length normalization",
        )
        contributions: list[float] = []
        for token in query_tokens:
            frequency = record.frequencies.get(token, 0)
            if not frequency:
                continue
            numerator = _finite_real(frequency * (self.config.k1 + 1), "TF numerator")
            denominator = _finite_real(frequency + normalization, "TF denominator")
            contributions.append(_finite_real(
                self._idf[token] * numerator / denominator, "term score",
            ))
        return _finite_real(math.fsum(contributions), "bm25_score")

    def retrieve(self, query: str, *, candidate_count: int) -> tuple[LexicalCandidate, ...]:
        """Return overlap matches in descending score / original-position order."""

        try:
            if not isinstance(query, str) or not query.strip():
                raise LexicalError("query must be a nonblank string.")
            if (
                not isinstance(candidate_count, int)
                or isinstance(candidate_count, bool)
                or candidate_count < 1
            ):
                raise LexicalError("candidate_count must be a positive integer.")
            query_tokens = tuple(sorted(set(_tokenize(query))))
            if not query_tokens:
                raise LexicalError("query must contain at least one lexical token.")
            scored: list[tuple[float, _IndexedChunk]] = []
            for record in self._records:
                if any(token in record.frequencies for token in query_tokens):
                    score = _finite_real(self._score(record, query_tokens), "bm25_score")
                    scored.append((score, record))
            scored.sort(key=lambda item: (-item[0], item[1].position))
            candidates: list[LexicalCandidate] = []
            for rank, (score, record) in enumerate(scored[:candidate_count], start=1):
                _validate_metadata(record.metadata, record.chunk, self.chunk_set_id)
                candidates.append(LexicalCandidate(
                    rank, record.chunk.chunk_id, record.chunk.text, record.metadata, score,
                ))
            return tuple(candidates)
        except LexicalError:
            raise
        except Exception as exc:
            raise LexicalError("Lexical retrieval failed.") from exc


__all__ = ["LexicalError", "BM25Config", "LexicalCandidate", "BM25Retriever"]
