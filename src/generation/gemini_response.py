"""Validate the requested JSON answer and bind citations to trusted evidence.

This is an offline application-payload boundary, not a provider HTTP parser.
Known citation IDs establish identity, not entailment, completeness, or truth.
The caller supplies the originating request ID; the model cannot supply it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json

from src.generation.evidence_context import EvidenceCitation, build_evidence_context
from src.generation.gemini_request import PreparedGeminiRequest, RESPONSE_STATUSES
from src.ingestion.chunk_models import ChunkSet


class GeminiResponseError(ValueError):
    """The payload or its request/citation binding is invalid; no answer returned."""


def _text(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise GeminiResponseError(f"{name} must be a nonblank string.")
    # json.loads accepts escaped lone surrogates. They cannot be faithfully
    # encoded as UTF-8, so do not pass them on to rendering or storage.
    value.encode("utf-8")


def _optional_text(value: object, name: str) -> None:
    if value is not None:
        _text(value, name)


def _status_fields(status, blocks, limitation, clarification_question) -> None:
    if type(status) is not str or status not in RESPONSE_STATUSES:
        raise GeminiResponseError("Unknown response status.")
    _optional_text(limitation, "limitation")
    _optional_text(clarification_question, "clarification_question")
    if status == "answered":
        if not blocks or clarification_question is not None:
            raise GeminiResponseError("answered requires blocks and no clarification question.")
    elif blocks:
        raise GeminiResponseError("Non-answer statuses require empty answer_blocks.")
    if status == "needs_clarification":
        _text(clarification_question, "clarification_question")
    elif clarification_question is not None:
        raise GeminiResponseError("Only needs_clarification may include a clarification question.")
    if status in ("insufficient_evidence", "out_of_scope"):
        _text(limitation, "limitation")


@dataclass(frozen=True, slots=True, kw_only=True)
class CitedAnswerBlock:
    """Unmodified model text with ordered, resolved evidence references.

    Text remains untrusted display data. Do not execute or render it as HTML.
    Repeating an ID across different blocks is valid; within a block it is not.
    """

    text: str
    citations: tuple[EvidenceCitation, ...]

    def __post_init__(self) -> None:
        try:
            _text(self.text, "answer block text")
            if type(self.citations) is not tuple or not self.citations:
                raise GeminiResponseError("Each answer block needs a nonempty citation tuple.")
            seen: set[str] = set()
            for citation in self.citations:
                if type(citation) is not EvidenceCitation:
                    raise GeminiResponseError("Resolved citations must be EvidenceCitation.")
                replace(citation)
                if citation.citation_id in seen:
                    raise GeminiResponseError("Duplicate citation ID within answer block.")
                seen.add(citation.citation_id)
        except GeminiResponseError:
            raise
        except Exception as exc:
            raise GeminiResponseError("Invalid cited answer block.") from exc

    @property
    def citation_ids(self) -> tuple[str, ...]:
        return tuple(citation.citation_id for citation in self.citations)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedGeminiResponse:
    """Immutable local result; this does not certify substantive grounding.

    Direct construction checks structure only. parse_gemini_response() also
    validates against an authoritative ChunkSet and the exact prepared request.
    """

    request_id: str
    model: str
    status: str
    answer_blocks: tuple[CitedAnswerBlock, ...]
    limitation: str | None
    clarification_question: str | None

    def __post_init__(self) -> None:
        try:
            _text(self.request_id, "request_id")
            _text(self.model, "model")
            if type(self.answer_blocks) is not tuple:
                raise GeminiResponseError("answer_blocks must be a tuple.")
            for block in self.answer_blocks:
                if type(block) is not CitedAnswerBlock:
                    raise GeminiResponseError("answer_blocks must contain CitedAnswerBlock.")
                replace(block)
            _status_fields(self.status, self.answer_blocks, self.limitation, self.clarification_question)
        except GeminiResponseError:
            raise
        except Exception as exc:
            raise GeminiResponseError("Invalid parsed response.") from exc


def _object(pairs):
    """Reject duplicate keys at every depth, including identical duplicates."""

    result = {}
    for key, value in pairs:
        if key in result:
            raise GeminiResponseError("Duplicate JSON object key.")
        result[key] = value
    return result


def _constant(value):
    raise GeminiResponseError("Nonstandard JSON numeric constant.")


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != expected:
        raise GeminiResponseError(f"{name} has missing, extra, or invalid fields.")


def validate_response_binding(
    *, response: ParsedGeminiResponse, request: PreparedGeminiRequest,
) -> None:
    """Recheck structure and request-local IDs before later reuse/display.

    This checks complete cited chunks, not the lossy dense metadata projection.
    It does not establish how an HTTP response was obtained or independently
    recheck source files/ChunkSet membership. Use the parser at initial ingress.
    """

    try:
        if type(request) is not PreparedGeminiRequest:
            raise GeminiResponseError("request must be PreparedGeminiRequest.")
        replace(request)
        if type(response) is not ParsedGeminiResponse:
            raise GeminiResponseError("response must be ParsedGeminiResponse.")
        replace(response)
        if response.request_id != request.request_id or response.model != request.model:
            raise GeminiResponseError("Response does not match this prepared request/model.")
        originals = {c.citation_id: c.chunk for c in request.context.citations}
        for block in response.answer_blocks:
            for citation in block.citations:
                if citation.citation_id not in originals or citation.chunk != originals[citation.citation_id]:
                    raise GeminiResponseError("Response citation differs from this request's evidence.")
    except GeminiResponseError:
        raise
    except Exception as exc:
        raise GeminiResponseError("Response binding validation failed.") from exc


def parse_gemini_response(
    *, response_json: str, request_id: str, request: PreparedGeminiRequest,
    chunk_set: ChunkSet,
) -> ParsedGeminiResponse:
    """Validate one application JSON payload without coercion or repair.

    request_id is caller-owned provenance, not a field added to model output.
    The future transport must capture it from the request actually sent. A
    falsely tagged payload cannot be identified from its text alone.
    No HTTP, retries, token counts, generation, or semantic support check.
    Ordinary errors retain their causes; BaseException propagates.
    """

    try:
        if type(request) is not PreparedGeminiRequest:
            raise GeminiResponseError("request must be PreparedGeminiRequest.")
        replace(request)
        _text(request_id, "request_id")
        snapshot_id, snapshot_json = request.request_id, request.request_json
        if request_id != snapshot_id:
            raise GeminiResponseError("Payload was tagged for another prepared request.")
        context = build_evidence_context(selection=request.context.selection, chunk_set=chunk_set)
        if context.model_context_json != request.context.model_context_json:
            raise GeminiResponseError("Request evidence differs from authoritative records.")
        _text(response_json, "response_json")
        payload = json.loads(response_json, object_pairs_hook=_object, parse_constant=_constant)
        _keys(payload, {"status", "answer_blocks", "limitation", "clarification_question"}, "Response")
        raw_blocks = payload["answer_blocks"]
        if type(raw_blocks) is not list:
            raise GeminiResponseError("answer_blocks must be a JSON array.")
        _status_fields(payload["status"], raw_blocks, payload["limitation"], payload["clarification_question"])
        citations = {c.citation_id: c for c in context.citations}
        blocks: list[CitedAnswerBlock] = []
        for raw in raw_blocks:
            _keys(raw, {"text", "citation_ids"}, "Answer block")
            ids = raw["citation_ids"]
            if type(ids) is not list or not ids:
                raise GeminiResponseError("citation_ids must be a nonempty JSON array.")
            resolved: list[EvidenceCitation] = []
            for citation_id in ids:
                if type(citation_id) is not str or citation_id not in citations:
                    raise GeminiResponseError("Unknown or noncanonical request-local citation ID.")
                resolved.append(citations[citation_id])
            blocks.append(CitedAnswerBlock(text=raw["text"], citations=tuple(resolved)))
        if request.request_id != snapshot_id or request.request_json != snapshot_json:
            raise GeminiResponseError("Prepared request changed during response parsing.")
        result = ParsedGeminiResponse(
            request_id=snapshot_id, model=request.model, status=payload["status"],
            answer_blocks=tuple(blocks), limitation=payload["limitation"],
            clarification_question=payload["clarification_question"],
        )
        validate_response_binding(response=result, request=request)
        return result
    except GeminiResponseError:
        raise
    except Exception as exc:
        raise GeminiResponseError("Gemini application response validation failed.") from exc


__all__ = [
    "GeminiResponseError", "CitedAnswerBlock", "ParsedGeminiResponse",
    "parse_gemini_response", "validate_response_binding",
]
