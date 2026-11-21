"""Check application token limits using one injected complete-request counter.

No real counter or transport is supplied here. A returned approval binds a count
to an exact prepared request; it is not proof of provider provenance, model
capacity, remaining quota, zero cost, or substantive evidence sufficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from src.generation.gemini_request import GeminiGenerationConfig, PreparedGeminiRequest


class RequestBudgetError(ValueError):
    """Counting failed, returned an invalid receipt, or could not be bound."""


class RequestBudgetExceededError(RequestBudgetError):
    """An explicit application input or output token limit was exceeded."""


def _integer(value: object, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise RequestBudgetError(f"{name} must be a nonboolean integer >= {minimum}.")


def _identity(request_id: object, model: object) -> None:
    for name, value in (("request_id", request_id), ("model", model)):
        if type(value) is not str or not value.strip():
            raise RequestBudgetError(f"{name} must be a nonblank string.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestBudget:
    """Explicit application ceilings; neither is a provider free-quota balance."""

    input_token_limit: int
    output_token_limit: int

    def __post_init__(self) -> None:
        _integer(self.input_token_limit, "input_token_limit", minimum=1)
        _integer(self.output_token_limit, "output_token_limit", minimum=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestTokenCount:
    """A counter receipt tagged with the exact request and model it counted."""

    request_id: str
    model: str
    input_tokens: int

    def __post_init__(self) -> None:
        _identity(self.request_id, self.model)
        _integer(self.input_tokens, "input_tokens", minimum=0)


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetApproval:
    """Application token-limit approval only; no payment or generation grant."""

    request_id: str
    model: str
    input_tokens: int
    max_output_tokens: int
    budget: RequestBudget

    def __post_init__(self) -> None:
        _identity(self.request_id, self.model)
        _integer(self.input_tokens, "input_tokens", minimum=0)
        _integer(self.max_output_tokens, "max_output_tokens", minimum=1)
        if type(self.budget) is not RequestBudget:
            raise RequestBudgetError("budget must be RequestBudget.")
        replace(self.budget)
        if self.max_output_tokens > self.budget.output_token_limit:
            raise RequestBudgetExceededError("Requested output cap exceeds application output limit.")
        if self.input_tokens > self.budget.input_token_limit:
            raise RequestBudgetExceededError("Counted input exceeds application input limit.")


class _RequestCounter(Protocol):
    def count_tokens(self, *, request: PreparedGeminiRequest) -> RequestTokenCount:
        """Count the complete logical request, not just evidence or user text."""
        ...


def _validate_request(request: PreparedGeminiRequest) -> None:
    if type(request) is not PreparedGeminiRequest:
        raise RequestBudgetError("request must be PreparedGeminiRequest.")
    replace(request)


def validate_generation_budget(
    *, generation_config: GeminiGenerationConfig, budget: RequestBudget,
) -> None:
    """Validate the output-cap relationship without counting or resources."""

    if type(generation_config) is not GeminiGenerationConfig:
        raise RequestBudgetError("generation_config must be GeminiGenerationConfig.")
    replace(generation_config)
    if type(budget) is not RequestBudget:
        raise RequestBudgetError("budget must be RequestBudget.")
    replace(budget)
    if generation_config.max_output_tokens > budget.output_token_limit:
        raise RequestBudgetExceededError(
            "Requested output cap exceeds application output limit."
        )


def validate_budget_approval(*, request: PreparedGeminiRequest, approval: BudgetApproval) -> None:
    """Reject approval reuse for a changed request, model, or output cap."""

    try:
        _validate_request(request)
        if type(approval) is not BudgetApproval:
            raise RequestBudgetError("approval must be BudgetApproval.")
        replace(approval)
        if (
            approval.request_id != request.request_id or approval.model != request.model
            or approval.max_output_tokens != request.config.max_output_tokens
        ):
            raise RequestBudgetError("Budget approval does not match this prepared request.")
    except RequestBudgetError:
        raise
    except Exception as exc:
        raise RequestBudgetError("Budget approval validation failed.") from exc


def check_request_budget(
    *, request: PreparedGeminiRequest, budget: RequestBudget, token_counter: _RequestCounter,
) -> BudgetApproval:
    """Preflight, count once, bind the receipt, and enforce explicit limits.

    No retries, character/byte estimates, partial approval, or generation.
    Future real counters require separate authorization and live validation.
    Ordinary errors retain causes; BaseException propagates.
    """

    try:
        _validate_request(request)
        validate_generation_budget(generation_config=request.config, budget=budget)
        output = request.config.max_output_tokens
        count = token_counter.count_tokens
        if not callable(count):
            raise RequestBudgetError("token_counter must expose callable count_tokens.")
        request_id, model, snapshot = request.request_id, request.model, request.request_json
        receipt = count(request=request)
        if type(receipt) is not RequestTokenCount:
            raise RequestBudgetError("Counter must return RequestTokenCount.")
        replace(receipt)
        if request.request_id != request_id or request.request_json != snapshot:
            raise RequestBudgetError("Prepared request changed during token counting.")
        if receipt.request_id != request_id or receipt.model != model:
            raise RequestBudgetError("Token count does not match this prepared request/model.")
        return BudgetApproval(
            request_id=request_id, model=model, input_tokens=receipt.input_tokens,
            max_output_tokens=output, budget=budget,
        )
    except RequestBudgetError:
        raise
    except Exception as exc:
        raise RequestBudgetError("Complete-request budget check failed.") from exc


__all__ = [
    "RequestBudgetError", "RequestBudgetExceededError", "RequestBudget",
    "RequestTokenCount", "BudgetApproval", "validate_generation_budget",
    "check_request_budget", "validate_budget_approval",
]
