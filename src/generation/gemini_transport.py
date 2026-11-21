"""Explicit synchronous Gemini REST transport, exercised with MockTransport.

No default backend, model, API version, limits, environment lookup, or SDK.
Supplying a live HTTP transport can perform real calls; that remains a separate
authorization. Token-limit approval does not establish free-tier billing state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from email.utils import parsedate_to_datetime, format_datetime
import json
import math
from numbers import Real
import re
import sys

import httpx

from src.generation.gemini_request import PreparedGeminiRequest, prepare_gemini_request
from src.generation.gemini_provider_response import (
    GeminiHTTPStatusError, GeminiProviderGeneration,
    parse_count_tokens_response, parse_generate_content_response,
)
from src.generation.request_budget import (
    BudgetApproval, RequestBudget, RequestTokenCount,
    check_request_budget, validate_budget_approval,
)
from src.ingestion.chunk_models import ChunkSet


class GeminiTransportError(ValueError):
    """Transport/configuration failure with a fixed, credential-free message."""


_CANONICAL_STATUS = frozenset({"OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
    "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION",
    "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED"})
_SAFE_REASONS = frozenset({"SERVICE_UNAVAILABLE", "MODEL_CAPACITY_EXHAUSTED", "BACKEND_ERROR",
    "RATE_LIMIT_EXCEEDED", "RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED", "SERVICE_DISABLED",
    "API_KEY_INVALID", "BILLING_DISABLED", "INTERNAL_ERROR"})
_ERROR_BODY_LIMIT = 8192


def _safe_error_diagnostic(response, api_key):
    """Extract only allowlisted metadata; never retain provider messages or headers."""
    result = {"provider_code": None, "provider_status": None, "provider_reasons": (),
              "retry_after": None, "request_id": None}
    try:
        retry = response.headers.get("retry-after", "")
        if re.fullmatch(r"[0-9]{1,8}", retry):
            result["retry_after"] = retry + " seconds"
        elif len(retry) <= 40 and re.fullmatch(r"[A-Za-z0-9,: ]+", retry):
            try:
                date = parsedate_to_datetime(retry)
                if date.tzinfo is not None and date.utcoffset().total_seconds() == 0:
                    result["retry_after"] = format_datetime(date, usegmt=True)
            except (TypeError, ValueError, OverflowError):
                pass
        for name in ("x-request-id", "x-goog-request-id", "x-cloud-trace-context"):
            value = response.headers.get(name, "")
            if name == "x-cloud-trace-context":
                value = value.split("/", 1)[0]
            if (len(value) <= 64 and api_key not in value and
                    re.fullmatch(r"(?:[0-9a-fA-F]{16,64}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", value)):
                result["request_id"] = value
                break
        content_type = response.headers.get("content-type", "").strip()
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if not re.fullmatch(r'application/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?', content_type, re.I) or encoding != "identity":
            return result
        declared = response.headers.get("content-length")
        if declared is not None and (not re.fullmatch(r"[0-9]{1,9}", declared) or int(declared) > _ERROR_BODY_LIMIT):
            return result
        data = bytearray()
        for part in response.iter_bytes(chunk_size=1024):
            if len(data) + len(part) > _ERROR_BODY_LIMIT:
                return result
            data.extend(part)
        body = json.loads(data.decode("utf-8"))
        error = body.get("error") if type(body) is dict else None
        if type(error) is not dict:
            return result
        code = error.get("code")
        if type(code) is int and 100 <= code <= 599:
            result["provider_code"] = code
        status = error.get("status")
        if type(status) is str and status in _CANONICAL_STATUS:
            result["provider_status"] = status
        details = error.get("details")
        if type(details) is list:
            result["provider_reasons"] = tuple(dict.fromkeys(
                item["reason"] for item in details[:20]
                if type(item) is dict and item.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"
                and type(item.get("reason")) is str and item["reason"] in _SAFE_REASONS
            ))
    except Exception:
        # Diagnostics must never change the original HTTP failure.
        pass
    return result


def validate_gemini_api_key(api_key: str) -> str:
    """Validate an explicit header value without retaining or displaying it."""

    if type(api_key) is not str or re.fullmatch(r"[!-~]+", api_key) is None:
        raise GeminiTransportError(
            "api_key must be an explicitly supplied nonblank ASCII header value."
        )
    return api_key


@dataclass(frozen=True, slots=True, kw_only=True)
class GeminiTransportConfig:
    """Required settings, not verified provider capabilities or live defaults.

    timeout_seconds applies to each HTTPX connect/read/write/pool operation,
    not a total wall-clock deadline. max_response_bytes is not a token budget.
    """

    api_version: str
    timeout_seconds: float
    max_response_bytes: int

    def __post_init__(self):
        if type(self.api_version) is not str or re.fullmatch(r"v[1-9][0-9]*(?:beta[0-9]*)?", self.api_version) is None:
            raise GeminiTransportError("api_version must be an explicit safe version path segment.")
        value = self.timeout_seconds
        if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
            raise GeminiTransportError("timeout_seconds must be a finite positive nonboolean real.")
        try:
            value = float(value)
        except Exception:
            raise GeminiTransportError("Invalid timeout_seconds.") from None
        if not math.isfinite(value) or value <= 0:
            raise GeminiTransportError("timeout_seconds must remain finite and positive after conversion.")
        object.__setattr__(self, "timeout_seconds", value)
        if type(self.max_response_bytes) is not int or self.max_response_bytes < 1:
            raise GeminiTransportError("max_response_bytes must be a positive integer excluding bool.")


def _snapshot(request):
    if type(request) is not PreparedGeminiRequest:
        raise GeminiTransportError("request must be PreparedGeminiRequest.")
    replace(request)
    return request.request_id, request.model, request.request_json


def _unchanged(request, snapshot):
    if (request.request_id, request.model, request.request_json) != snapshot:
        raise GeminiTransportError("Prepared request changed during the HTTP operation.")


def _authoritative(request, chunk_set):
    _snapshot(request)
    prepare_gemini_request(
        question=request.question, context=request.context,
        config=request.config, chunk_set=chunk_set,
    )


class GeminiRESTTransport:
    """Own one HTTPX client and the explicitly supplied synchronous transport.

    The caller transfers transport-close responsibility to this object. Use a
    context manager or close(). No parallel operations or mutation are supported.
    An injected backend may have its own behavior; only MockTransport is tested
    here. This adapter adds no retry, redirect, continuation, or paid fallback.
    """

    __slots__ = ("_client", "_config", "_key")

    def __init__(self, *, api_key: str, config: GeminiTransportConfig, transport: httpx.BaseTransport):
        api_key = validate_gemini_api_key(api_key)
        if type(config) is not GeminiTransportConfig:
            raise GeminiTransportError("config must be GeminiTransportConfig.")
        replace(config)
        if not isinstance(transport, httpx.BaseTransport):
            raise GeminiTransportError("An explicit synchronous HTTPX transport is required.")
        self._key, self._config = api_key, config
        self._client = httpx.Client(
            transport=transport, timeout=httpx.Timeout(config.timeout_seconds),
            trust_env=False, follow_redirects=False, auth=None,
        )

    def __repr__(self):
        return "GeminiRESTTransport(api_key=<redacted>)"

    def __enter__(self):
        if self._client.is_closed:
            raise GeminiTransportError("Transport is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False

    def close(self):
        try:
            self._client.close()
        except Exception:
            raise GeminiTransportError("Closing HTTP transport failed.") from None

    def _post(self, *, model, operation, body):
        """Send once, bound the response bytes, and decode strict UTF-8 JSON.

        Wire/backend errors suppress displayed cause chains: injected/network
        exceptions can retain credentials or sensitive request bodies. Provider
        validation outside this method retains its ordinary validation causes.
        """

        response = None
        try:
            if self._client.is_closed:
                raise GeminiTransportError("Transport is closed.")
            config = replace(self._config)
            url = f"https://generativelanguage.googleapis.com/{config.api_version}/{model}:{operation}"
            # A direct Request avoids merging the client's cookie jar or other
            # per-client headers into later operations. No key in URL or body.
            wire = httpx.Request("POST", url, content=body.encode("utf-8"), headers={
                "x-goog-api-key": self._key, "Content-Type": "application/json",
                "Accept": "application/json", "Accept-Encoding": "identity",
            })
            response = self._client.send(wire, stream=True, auth=None, follow_redirects=False)
            if response.status_code != 200:
                diagnostic = _safe_error_diagnostic(response, self._key)
                raise GeminiHTTPStatusError(response.status_code, diagnostic=diagnostic)
            content_type = response.headers.get("content-type", "").strip()
            if re.fullmatch(r'application/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?', content_type, re.IGNORECASE) is None:
                raise GeminiTransportError("Expected application/json with optional UTF-8 charset.")
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                raise GeminiTransportError("Compressed response bodies are outside this transport profile.")
            declared = response.headers.get("content-length")
            if declared is not None:
                if re.fullmatch(r"[0-9]+", declared) is None:
                    raise GeminiTransportError("Malformed response Content-Length.")
                declared = int(declared)
                if declared > config.max_response_bytes:
                    raise GeminiTransportError("Declared response exceeds max_response_bytes.")
            chunks, total = [], 0
            for chunk in response.iter_bytes(chunk_size=min(65536, config.max_response_bytes + 1)):
                total += len(chunk)
                if total > config.max_response_bytes:
                    raise GeminiTransportError("Response exceeds max_response_bytes.")
                chunks.append(chunk)
            if declared is not None and declared != total:
                raise GeminiTransportError("Response body length differs from Content-Length.")
            return b"".join(chunks).decode("utf-8", errors="strict")
        except (GeminiTransportError, GeminiHTTPStatusError):
            raise
        except Exception:
            raise GeminiTransportError("HTTP operation or response reading failed; no retry or fallback.") from None
        finally:
            if response is not None:
                failure_active = sys.exc_info()[0] is not None
                try:
                    response.close()
                except Exception:
                    if not failure_active:
                        raise GeminiTransportError("Closing HTTP response failed.") from None

    def count_tokens(self, *, request: PreparedGeminiRequest) -> RequestTokenCount:
        """One complete-request count, compatible with check_request_budget()."""

        snapshot = _snapshot(request)
        body = self._post(model=snapshot[1], operation="countTokens", body=request.count_tokens_body_json)
        _unchanged(request, snapshot)
        return parse_count_tokens_response(
            status_code=200, body_json=body, request_id=snapshot[0], request=request,
        )

    def generate_content(
        self, *, request: PreparedGeminiRequest, approval: BudgetApproval, chunk_set: ChunkSet,
    ) -> GeminiProviderGeneration:
        """Validate evidence and matching budget approval before one generation.

        Approval is an application-limit check, not authenticated provider quota,
        free-tier/billing verification, semantic sufficiency, or live permission.
        """

        _authoritative(request, chunk_set)
        validate_budget_approval(request=request, approval=approval)
        snapshot = _snapshot(request)
        body = self._post(model=snapshot[1], operation="generateContent", body=request.generate_content_body_json)
        _unchanged(request, snapshot)
        return parse_generate_content_response(
            status_code=200, body_json=body, request_id=snapshot[0],
            request=request, chunk_set=chunk_set,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GeminiTransportRun:
    """Matching application budget approval and validated provider generation."""

    approval: BudgetApproval
    generation: GeminiProviderGeneration

    def __post_init__(self):
        if type(self.approval) is not BudgetApproval or type(self.generation) is not GeminiProviderGeneration:
            raise GeminiTransportError("Run requires BudgetApproval and GeminiProviderGeneration.")
        replace(self.approval)
        replace(self.generation)
        response = self.generation.response
        if (self.approval.request_id, self.approval.model) != (response.request_id, response.model):
            raise GeminiTransportError("Run budget approval and response do not match.")


def run_gemini_request(
    *, request: PreparedGeminiRequest, chunk_set: ChunkSet,
    budget: RequestBudget, adapter: GeminiRESTTransport,
) -> GeminiTransportRun:
    """Recheck authoritative evidence, count once, gate, then generate once.

    An output-cap overflow stops before counting; input overflow or any count
    failure stops before generation. No integrated retrieval or live entry point.
    """

    if type(adapter) is not GeminiRESTTransport:
        raise GeminiTransportError("adapter must be GeminiRESTTransport.")
    _authoritative(request, chunk_set)
    approval = check_request_budget(request=request, budget=budget, token_counter=adapter)
    generation = adapter.generate_content(request=request, approval=approval, chunk_set=chunk_set)
    return GeminiTransportRun(approval=approval, generation=generation)


__all__ = [
    "GeminiTransportError", "validate_gemini_api_key", "GeminiTransportConfig",
    "GeminiRESTTransport", "GeminiTransportRun", "run_gemini_request",
]
