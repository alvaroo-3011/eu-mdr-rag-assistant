"""Offline tests for the deterministic validated-empty-selection policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from src.application.answer_policy import (
    EMPTY_EVIDENCE_LIMITATION,
    LocalInsufficientEvidence,
    local_empty_evidence_result,
)
from src.retrieval.evidence_selection import EvidenceSelectionError, select_evidence
from tests.unit.ingestion.test_chunk_models import _guidance_source
from tests.unit.retrieval.test_lexical_query import _synthetic_chunk_set


class AnswerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.chunk_set = _synthetic_chunk_set((
            _guidance_source(),
            ("Synthetic MDR evidence.",),
        ))
        self.empty_selection = select_evidence(
            chunk_set=self.chunk_set,
            fused_candidates=(),
            max_chunks=1,
        )

    def test_validated_empty_selection_returns_only_the_fixed_local_non_answer(self):
        result = local_empty_evidence_result(self.empty_selection)

        self.assertIs(type(result), LocalInsufficientEvidence)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.limitation, EMPTY_EVIDENCE_LIMITATION)
        self.assertEqual(
            result.limitation,
            "No MDR evidence was selected and no provider request was made.",
        )
        self.assertFalse(hasattr(result, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            result.status = "answered"

    def test_selection_is_revalidated_and_non_selection_objects_are_rejected(self):
        damaged = select_evidence(
            chunk_set=self.chunk_set,
            fused_candidates=(),
            max_chunks=1,
        )
        object.__setattr__(damaged, "max_chunks", 0)

        with self.assertRaises(EvidenceSelectionError):
            local_empty_evidence_result(damaged)
        for value in (None, (), object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                local_empty_evidence_result(value)


if __name__ == "__main__":
    unittest.main()
