"""Pure trusted display projection for local and provider answer results."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.application.answer_policy import LocalInsufficientEvidence, LocalNeedsClarification
from src.application.local_config import LocalConfigError
from src.application.local_runtime import LocalCredentialError, LocalRuntimeError
from src.generation.gemini_provider_response import GeminiProviderResponseError
from src.generation.gemini_request import GeminiRequestError, NoSelectedEvidenceError
from src.generation.gemini_response import GeminiResponseError
from src.generation.gemini_transport import (
    GeminiTransportError,
    GeminiTransportRun,
)
from src.generation.request_budget import (
    RequestBudgetError,
    RequestBudgetExceededError,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CitationView:
    citation_id: str
    official_title: str
    short_title: str
    document_id: str
    version: str
    source_identity: str
    official_url: str
    pages: tuple[int, ...]
    legal_locators: tuple[str, ...]
    locator_summary: tuple[str, ...]
    hierarchies: tuple[str, ...]
    regulatory_regimes: tuple[str, ...]
    permitted_regimes: tuple[str, ...]


_SUMMARY_LOCATOR_KINDS = {"article": "Article", "annex": "Annex"}


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerBlockView:
    text: str
    citations: tuple[CitationView, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerView:
    status: str
    answer_blocks: tuple[AnswerBlockView, ...]
    limitation: str | None
    clarification_question: str | None


def _citation_view(citation) -> CitationView:
    chunk = citation.chunk
    source = chunk.source
    legal_locators = tuple(
        " > ".join(f"{part.kind}: {part.value}" for part in path)
        for path in chunk.legal_locators
    )
    hierarchies = tuple(
        " > ".join(entry.title for entry in path if entry.title)
        for path in chunk.hierarchies
        if any(entry.title for entry in path)
    )
    summary: list[str] = []
    for path in chunk.legal_locators:
        for part in path:
            label = _SUMMARY_LOCATOR_KINDS.get(part.kind)
            if label is not None and part.value:
                entry = f"{label} {part.value}"
                if entry not in summary:
                    summary.append(entry)
    return CitationView(
        citation_id=citation.citation_id,
        official_title=source.official_title,
        short_title=source.short_title,
        document_id=source.document_id,
        version=source.version,
        source_identity=source.source_identity,
        official_url=source.official_url,
        pages=tuple(page.pdf_page_number for page in chunk.pages),
        legal_locators=legal_locators,
        locator_summary=tuple(summary),
        hierarchies=hierarchies,
        regulatory_regimes=tuple(regime.value for regime in source.regulatory_regimes),
        permitted_regimes=tuple(regime.value for regime in source.permitted_regimes),
    )


def build_answer_view(
    run: GeminiTransportRun | LocalInsufficientEvidence | LocalNeedsClarification,
) -> AnswerView:
    """Project only a trusted local non-answer or validated provider data."""

    if type(run) is LocalNeedsClarification:
        checked = replace(run)
        return AnswerView(
            status=checked.status,
            answer_blocks=(),
            limitation=None,
            clarification_question=checked.clarification_question,
        )

    if type(run) is LocalInsufficientEvidence:
        checked = replace(run)
        return AnswerView(
            status=checked.status,
            answer_blocks=(),
            limitation=checked.limitation,
            clarification_question=None,
        )

    if type(run) is not GeminiTransportRun:
        raise GeminiResponseError(
            "Display requires a trusted local result or validated GeminiTransportRun."
        )
    replace(run)
    response = run.generation.response
    blocks = tuple(
        AnswerBlockView(
            text=block.text,
            citations=tuple(_citation_view(citation) for citation in block.citations),
        )
        for block in response.answer_blocks
    )
    return AnswerView(
        status=response.status,
        answer_blocks=blocks,
        limitation=response.limitation,
        clarification_question=response.clarification_question,
    )


def citation_markers(block: AnswerBlockView) -> str:
    """Return inline markers such as "[2][5]" in block order, without duplicates."""

    seen: list[str] = []
    for citation in block.citations:
        if citation.citation_id not in seen:
            seen.append(citation.citation_id)
    return "".join(f"[{citation_id}]" for citation_id in seen)


def _citation_sort_key(citation: CitationView) -> tuple[int, int, str]:
    value = citation.citation_id
    return (0, int(value), value) if value.isdigit() else (1, 0, value)


def cited_sources(view: AnswerView) -> tuple[CitationView, ...]:
    """Return each cited source once, ordered by citation number."""

    unique: dict[str, CitationView] = {}
    for block in view.answer_blocks:
        for citation in block.citations:
            unique.setdefault(citation.citation_id, citation)
    return tuple(sorted(unique.values(), key=_citation_sort_key))


def source_line(citation: CitationView) -> str:
    """Compose one source line only from trusted manifest and chunk fields."""

    name = ", ".join(part for part in (citation.short_title, citation.version) if part)
    line = f"[{citation.citation_id}] {name}".rstrip()
    if citation.locator_summary:
        line += " — " + "; ".join(citation.locator_summary)
    elif citation.pages:
        label = "p." if len(citation.pages) == 1 else "pp."
        line += f" — {label} " + ", ".join(str(page) for page in citation.pages)
    return line


def public_error_message(error: Exception) -> str:
    """Map failures to fixed UI text without bodies, headers, keys, or reprs."""

    if isinstance(error, LocalConfigError):
        return "The local application configuration is missing or invalid."
    if isinstance(error, LocalCredentialError):
        return "The LLM API key is missing or invalid."
    if isinstance(error, RequestBudgetExceededError):
        return "The request exceeds the configured application token budget."
    if isinstance(error, NoSelectedEvidenceError):
        return "No MDR evidence was selected; no provider request was made."
    if isinstance(error, (
        GeminiTransportError, GeminiProviderResponseError, GeminiResponseError,
        GeminiRequestError, RequestBudgetError,
    )):
        return "The provider request or validated response failed; no partial answer is shown."
    if isinstance(error, LocalRuntimeError):
        return "The local application could not complete startup safely."
    return "The submission failed safely; no partial answer is shown."


__all__ = [
    "CitationView", "AnswerBlockView", "AnswerView", "build_answer_view",
    "citation_markers", "cited_sources", "public_error_message", "source_line",
]
