"""Narrow local clarification and validated-empty-evidence policies.

This module does not decide whether nonempty evidence is sufficient and does
not claim that the corpus lacks support. Those semantic decisions remain in
the model-proposed response and require later semantic review.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Literal

from src.retrieval.evidence_selection import EvidenceSelection


EMPTY_EVIDENCE_LIMITATION = (
    "No MDR evidence was selected and no provider request was made."
)

DEVICE_CLARIFICATION_QUESTION = (
    "What is your device and what is its intended purpose?"
)

# Match only short, explicitly MDR-scoped requests about an undescribed device.
# Full matching deliberately leaves described devices, additional context, and
# mixed or IVDR scope to the existing retrieval and provider decision path.
_REGULATION = r"(?:the\s+)?(?:eu\s+)?(?:mdr|medical\s+device\s+regulation)"
_DEVICE = r"(?:my|our)\s+device"
_DEVICE_DETERMINATIONS = (
    re.compile(
        rf"(?:what|which)\s+materials?\s+(?:must|should|can|does)\s+{_DEVICE}"
        rf"\s+(?:be\s+made\s+(?:from|of)|use)\s+to\s+(?:comply\s+with|meet)\s+{_REGULATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:what|which)\s+(?:risk\s+)?class(?:ification)?\s+(?:is|would|should)\s+"
        rf"{_DEVICE}(?:\s+be)?\s+under\s+{_REGULATION}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class LocalNeedsClarification:
    """One fixed local non-answer for a narrowly recognized missing device fact."""

    status: Literal["needs_clarification"] = field(init=False, default="needs_clarification")
    clarification_question: str = field(init=False, default=DEVICE_CLARIFICATION_QUESTION)


def local_device_clarification(question: str) -> LocalNeedsClarification | None:
    """Ask only when the complete question matches an explicitly in-scope form."""
    if type(question) is not str:
        return None
    normalized = " ".join(question.split()).removesuffix("?").strip()
    if any(pattern.fullmatch(normalized) for pattern in _DEVICE_DETERMINATIONS):
        return LocalNeedsClarification()
    return None


@dataclass(frozen=True, slots=True)
class LocalInsufficientEvidence:
    """One trusted local non-answer for a validated empty selection."""

    status: Literal["insufficient_evidence"] = field(
        init=False, default="insufficient_evidence"
    )
    limitation: str = field(init=False, default=EMPTY_EVIDENCE_LIMITATION)


def local_empty_evidence_result(
    selection: EvidenceSelection,
) -> LocalInsufficientEvidence | None:
    """Return the fixed local non-answer only for a valid empty selection."""

    if type(selection) is not EvidenceSelection:
        raise TypeError("selection must be EvidenceSelection.")
    checked = replace(selection)
    if checked.selected:
        return None
    return LocalInsufficientEvidence()


__all__ = [
    "EMPTY_EVIDENCE_LIMITATION",
    "DEVICE_CLARIFICATION_QUESTION",
    "LocalNeedsClarification",
    "local_device_clarification",
    "LocalInsufficientEvidence",
    "local_empty_evidence_result",
]
