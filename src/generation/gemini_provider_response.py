"""Offline parsing of non-streaming Gemini Developer REST response envelopes.

Caller-supplied status, body, and request tag are not authenticated transport
provenance. No HTTP client, credentials, counting call, generation, or retry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math

from src.generation.gemini_request import PreparedGeminiRequest
from src.generation.gemini_response import ParsedGeminiResponse, parse_gemini_response
from src.generation.request_budget import RequestTokenCount
from src.ingestion.chunk_models import ChunkSet


class GeminiProviderResponseError(ValueError):
    """A provider envelope cannot safely supply a count or application answer."""


class GeminiHTTPStatusError(GeminiProviderResponseError):
    """A non-200 operation failed. The provider body is not put in the message."""

    def __init__(self, status_code: int, *, diagnostic=None) -> None:
        self.status_code = status_code
        self.diagnostic = diagnostic
        super().__init__(f"Gemini operation returned HTTP {status_code}; no retry or fallback.")


def _text(value, name, *, allow_empty=False):
    if type(value) is not str or (not allow_empty and not value.strip()):
        raise GeminiProviderResponseError(f"{name} must be a string of the required shape.")
    value.encode("utf-8")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise GeminiProviderResponseError(f"{name} must be a nonnegative integer excluding bool.")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GeminiProviderResponseError("Duplicate JSON object key in provider body.")
        result[key] = value
    return result


def _constant(value):
    raise GeminiProviderResponseError("Nonstandard JSON numeric constant in provider body.")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise GeminiProviderResponseError("Nonfinite JSON number in provider body.")
    return result


def _strings(value):
    """Check escaped Unicode even in audit metadata that is not interpreted."""

    if type(value) is str:
        value.encode("utf-8")
    elif type(value) is dict:
        for key, item in value.items():
            key.encode("utf-8")
            _strings(item)
    elif type(value) is list:
        for item in value:
            _strings(item)


def _json_object(body):
    _text(body, "body_json")
    value = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    if type(value) is not dict:
        raise GeminiProviderResponseError("Provider response must be a JSON object.")
    _strings(value)
    return value


def _fields(value, required, allowed, name):
    if type(value) is not dict or not required <= set(value) or not set(value) <= allowed:
        raise GeminiProviderResponseError(f"Unsupported or malformed {name} fields.")


def _http_body(status_code, body_json):
    if type(status_code) is not int or not 100 <= status_code <= 599:
        raise GeminiProviderResponseError("status_code must be an integer HTTP status excluding bool.")
    if status_code != 200:
        # Do not parse, log, or echo a potentially sensitive error body.
        raise GeminiHTTPStatusError(status_code)
    value = _json_object(body_json)
    if "error" in value:
        raise GeminiProviderResponseError("Provider returned an error envelope with HTTP 200.")
    return value


def _request_snapshot(request, request_id):
    if type(request) is not PreparedGeminiRequest:
        raise GeminiProviderResponseError("request must be PreparedGeminiRequest.")
    replace(request)
    _text(request_id, "request_id")
    snapshot = request.request_id, request.model, request.request_json
    if request_id != snapshot[0]:
        raise GeminiProviderResponseError("Provider response tag differs from the prepared request.")
    return snapshot


def _unchanged(request, snapshot):
    if (request.request_id, request.model, request.request_json) != snapshot:
        raise GeminiProviderResponseError("Prepared request changed during provider response parsing.")


def _ratings(container):
    if "safetyRatings" not in container:
        return
    ratings = container["safetyRatings"]
    if type(ratings) is not list:
        raise GeminiProviderResponseError("safetyRatings must be an array.")
    categories = set()
    for rating in ratings:
        _fields(rating, {"category", "probability"}, {"category", "probability", "blocked"}, "safety rating")
        _text(rating["category"], "safety category")
        _text(rating["probability"], "safety probability")
        if rating["category"] in categories:
            raise GeminiProviderResponseError("Repeated safety category.")
        categories.add(rating["category"])
        if "blocked" in rating and (type(rating["blocked"]) is not bool or rating["blocked"]):
            raise GeminiProviderResponseError("Blocked or malformed safety rating.")


def _generation_text(payload):
    _fields(payload, {"candidates"}, {
        "candidates", "promptFeedback", "usageMetadata", "modelVersion", "responseId", "modelStatus",
    }, "generation response")
    for name in ("modelVersion", "responseId"):
        if name in payload:
            _text(payload[name], name)
    for name in ("usageMetadata", "modelStatus"):
        if name in payload and type(payload[name]) is not dict:
            raise GeminiProviderResponseError("Provider audit metadata must be objects when supplied.")
    usage = payload.get("usageMetadata", {})
    for name in ("promptTokenCount", "cachedContentTokenCount", "candidatesTokenCount",
                 "toolUsePromptTokenCount", "thoughtsTokenCount", "totalTokenCount"):
        if name in usage:
            _integer(usage[name], name)
    if "promptFeedback" in payload:
        feedback = payload["promptFeedback"]
        _fields(feedback, set(), {"blockReason", "safetyRatings"}, "prompt feedback")
        if "blockReason" in feedback:
            raise GeminiProviderResponseError("Prompt block reason present; no application answer accepted.")
        _ratings(feedback)
    candidates = payload["candidates"]
    if type(candidates) is not list or len(candidates) != 1:
        raise GeminiProviderResponseError("Exactly one completed candidate is required.")
    candidate = candidates[0]
    _fields(candidate, {"content", "finishReason"}, {
        "content", "finishReason", "safetyRatings", "index", "tokenCount",
        "avgLogprobs", "logprobsResult", "finishMessage", "citationMetadata",
    }, "candidate")
    if type(candidate["finishReason"]) is not str or candidate["finishReason"] != "STOP":
        raise GeminiProviderResponseError("Only STOP completions are accepted; no continuation or retry.")
    if "tokenCount" in candidate:
        _integer(candidate["tokenCount"], "candidate tokenCount")
    if "finishMessage" in candidate:
        _text(candidate["finishMessage"], "finishMessage", allow_empty=True)
    for name in ("citationMetadata", "logprobsResult"):
        if name in candidate and type(candidate[name]) is not dict:
            raise GeminiProviderResponseError("Candidate audit metadata must be objects when supplied.")
    if "avgLogprobs" in candidate and type(candidate["avgLogprobs"]) not in (int, float):
        raise GeminiProviderResponseError("avgLogprobs must be a JSON number excluding bool.")
    if "index" in candidate:
        _integer(candidate["index"], "candidate index")
        if candidate["index"] != 0:
            raise GeminiProviderResponseError("Single candidate must have index zero when supplied.")
    _ratings(candidate)
    content = candidate["content"]
    _fields(content, {"parts"}, {"parts", "role"}, "candidate content")
    if "role" in content and content["role"] != "model":
        raise GeminiProviderResponseError("Candidate content role must be model when supplied.")
    parts = content["parts"]
    if type(parts) is not list or not parts:
        raise GeminiProviderResponseError("Candidate must contain nonempty text parts.")
    texts = []
    for part in parts:
        _fields(part, {"text"}, {"text", "thought", "thoughtSignature"}, "text part")
        _text(part["text"], "part text", allow_empty=True)
        if "thought" in part and (type(part["thought"]) is not bool or part["thought"]):
            raise GeminiProviderResponseError("Thought output is not accepted as an answer part.")
        if "thoughtSignature" in part:
            _text(part["thoughtSignature"], "thought signature", allow_empty=True)
        texts.append(part["text"])
    text = "".join(texts)  # Preserve exact part boundaries; do not inject whitespace.
    _text(text, "combined application text")
    return text


def parse_count_tokens_response(
    *, status_code: int, body_json: str, request_id: str, request: PreparedGeminiRequest,
) -> RequestTokenCount:
    """Decode one tagged complete-request count; no inference for missing totals.

    The caller must have counted request.count_tokens_body_json. Matching tags
    cannot prove that the provider received that exact body or model.
    """

    try:
        snapshot = _request_snapshot(request, request_id)
        payload = _http_body(status_code, body_json)
        _fields(payload, {"totalTokens"}, {
            "totalTokens", "cachedContentTokenCount", "promptTokensDetails", "cacheTokensDetails",
        }, "count response")
        _integer(payload["totalTokens"], "totalTokens")
        if "cachedContentTokenCount" in payload:
            _integer(payload["cachedContentTokenCount"], "cachedContentTokenCount")
        for name in ("promptTokensDetails", "cacheTokensDetails"):
            if name in payload and type(payload[name]) is not list:
                raise GeminiProviderResponseError("Count modality details must be arrays when supplied.")
        _unchanged(request, snapshot)
        return RequestTokenCount(request_id=snapshot[0], model=snapshot[1], input_tokens=payload["totalTokens"])
    except GeminiProviderResponseError:
        raise
    except Exception as exc:
        raise GeminiProviderResponseError("Gemini count response validation failed.") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class GeminiProviderGeneration:
    """Application result plus exact raw provider JSON for caller-owned audit.

    No automatic persistence. Direct construction only checks local consistency;
    authoritative request/chunk binding comes from the parser below.
    """

    response: ParsedGeminiResponse
    provider_body_json: str

    def __post_init__(self):
        try:
            if type(self.response) is not ParsedGeminiResponse:
                raise GeminiProviderResponseError("response must be ParsedGeminiResponse.")
            replace(self.response)
            text = _generation_text(_json_object(self.provider_body_json))
            actual = _json_object(text)
            expected = {
                "status": self.response.status,
                "answer_blocks": [{"text": b.text, "citation_ids": list(b.citation_ids)} for b in self.response.answer_blocks],
                "limitation": self.response.limitation,
                "clarification_question": self.response.clarification_question,
            }
            if actual != expected:
                raise GeminiProviderResponseError("Retained provider body differs from parsed application output.")
        except GeminiProviderResponseError:
            raise
        except Exception as exc:
            raise GeminiProviderResponseError("Invalid retained provider generation.") from exc


def parse_generate_content_response(
    *, status_code: int, body_json: str, request_id: str,
    request: PreparedGeminiRequest, chunk_set: ChunkSet,
) -> GeminiProviderGeneration:
    """Check the complete envelope, then call the existing answer parser once.

    Stop on errors, prompt/candidate blocks, unfinished/continuation responses,
    mixed content, thought output, or invalid application JSON/citations. The
    returned raw body preserves metadata without certifying its semantics.
    """

    try:
        snapshot = _request_snapshot(request, request_id)
        payload = _http_body(status_code, body_json)
        text = _generation_text(payload)
        _unchanged(request, snapshot)
        response = parse_gemini_response(
            response_json=text, request_id=snapshot[0], request=request, chunk_set=chunk_set,
        )
        _unchanged(request, snapshot)
        return GeminiProviderGeneration(response=response, provider_body_json=body_json)
    except GeminiProviderResponseError:
        raise
    except Exception as exc:
        raise GeminiProviderResponseError("Gemini generation response validation failed.") from exc


__all__ = [
    "GeminiProviderResponseError", "GeminiHTTPStatusError", "GeminiProviderGeneration",
    "parse_count_tokens_response", "parse_generate_content_response",
]
