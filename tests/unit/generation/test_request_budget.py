"""Budget tests use fake counters only; token fixtures are not quota balances."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import Mock, patch

from tests.unit.retrieval.test_lexical_query import _forbid_production_imports

with _forbid_production_imports():
    from src.generation.request_budget import (
        BudgetApproval, RequestBudget, RequestBudgetError, RequestBudgetExceededError,
        RequestTokenCount, check_request_budget, validate_budget_approval,
    )

from tests.unit.generation.test_gemini_request import _fixture, _config
from src.generation.gemini_request import prepare_gemini_request


class RequestBudgetTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.chunk_set, self.context, self.config, self.request = _fixture()
        self.budget = RequestBudget(input_token_limit=100, output_token_limit=256)
        self.counter = Mock()
        self.counter.count_tokens.side_effect = lambda *, request: RequestTokenCount(
            request_id=request.request_id, model=request.model, input_tokens=100,
        )

    def _check(self, **changes):
        return check_request_budget(**{
            "request": self.request, "budget": self.budget, "token_counter": self.counter, **changes,
        })

    def test_exact_limits_are_accepted_and_counter_receives_original_request_once(self):
        approval = self._check()
        self.counter.count_tokens.assert_called_once_with(request=self.request)
        self.assertEqual(approval.input_tokens, 100)
        self.assertEqual(approval.max_output_tokens, 256)
        self.assertEqual(approval.request_id, self.request.request_id)
        self.assertEqual(approval.model, self.request.model)
        self.assertIs(approval.budget, self.budget)
        self.assertIsNone(validate_budget_approval(request=self.request, approval=approval))

    def test_counter_observes_system_question_evidence_schema_and_explicit_settings(self):
        observed = []
        def count(*, request):
            observed.append(json.loads(request.count_tokens_body_json)["generateContentRequest"])
            return RequestTokenCount(request_id=request.request_id, model=request.model, input_tokens=99)
        self.counter.count_tokens.side_effect = count
        self._check()
        self.assertEqual(len(observed), 1)
        logical = observed[0]
        self.assertIn("systemInstruction", logical)
        self.assertIn("responseJsonSchema", logical["generationConfig"])
        self.assertEqual(logical["generationConfig"]["maxOutputTokens"], 256)
        user = json.loads(logical["contents"][0]["parts"][0]["text"])
        self.assertEqual(user["question"], self.request.question)
        self.assertEqual(user["evidence_context"], json.loads(self.context.model_context_json))

    def test_input_overflow_stops_after_one_count_without_partial_approval(self):
        self.counter.count_tokens.return_value = None
        self.counter.count_tokens.side_effect = lambda *, request: RequestTokenCount(
            request_id=request.request_id, model=request.model, input_tokens=101,
        )
        with self.assertRaisesRegex(RequestBudgetExceededError, "input"):
            self._check()
        self.counter.count_tokens.assert_called_once()

    def test_output_overflow_is_rejected_before_resolving_counter_method(self):
        class ForbiddenCounter:
            @property
            def count_tokens(self):
                raise AssertionError("Counter must not be inspected")
        with self.assertRaisesRegex(RequestBudgetExceededError, "output"):
            self._check(budget=RequestBudget(input_token_limit=100, output_token_limit=255), token_counter=ForbiddenCounter())

    def test_invalid_preflight_does_not_count(self):
        for changes in ({"request": None}, {"budget": None}):
            with self.subTest(changes=changes), self.assertRaises(RequestBudgetError):
                self._check(**changes)
        self.counter.count_tokens.assert_not_called()

    def test_explicit_limits_reject_booleans_nonintegers_and_nonpositive_values(self):
        for field in ("input_token_limit", "output_token_limit"):
            for value in (None, True, False, 0, -1, 1.0, "100"):
                with self.subTest(field=field, value=value), self.assertRaises(RequestBudgetError):
                    RequestBudget(**{"input_token_limit": 100, "output_token_limit": 256, field: value})
        with self.assertRaises(TypeError):
            RequestBudget(input_token_limit=100)
        with self.assertRaises(TypeError):
            RequestBudget(100, 256)

    def test_nonnegative_provider_count_including_zero_is_preserved(self):
        for value in (0, 1, 99):
            self.counter.count_tokens.side_effect = lambda *, request: RequestTokenCount(
                request_id=request.request_id, model=request.model, input_tokens=value,
            )
            with self.subTest(value=value):
                self.assertEqual(self._check().input_tokens, value)

    def test_invalid_receipt_envelopes_are_not_coerced(self):
        self.counter.count_tokens.side_effect = None
        for value in (None, True, 100, {"totalTokens": 100}, (), object()):
            self.counter.count_tokens.return_value = value
            with self.subTest(value=type(value)), self.assertRaises(RequestBudgetError):
                self._check()

    def test_receipt_fields_are_revalidated(self):
        for field, value in (("input_tokens", True), ("input_tokens", -1), ("request_id", ""), ("model", None)):
            receipt = RequestTokenCount(request_id=self.request.request_id, model=self.request.model, input_tokens=99)
            object.__setattr__(receipt, field, value)
            self.counter.count_tokens.side_effect = None
            self.counter.count_tokens.return_value = receipt
            with self.subTest(field=field, value=value), self.assertRaises(RequestBudgetError):
                self._check()

    def test_stale_request_or_model_receipts_are_rejected(self):
        for changes in ({"request_id": "another-request"}, {"model": "models/another"}):
            receipt = RequestTokenCount(**{"request_id": self.request.request_id, "model": self.request.model, "input_tokens": 99, **changes})
            self.counter.count_tokens.side_effect = None
            self.counter.count_tokens.return_value = receipt
            with self.subTest(changes=changes), self.assertRaisesRegex(RequestBudgetError, "does not match"):
                self._check()

    def test_request_mutation_during_counting_is_detected_even_with_a_new_receipt(self):
        request = deepcopy(self.request)
        def count(*, request):
            object.__setattr__(request, "question", "Changed after preflight")
            return RequestTokenCount(request_id=request.request_id, model=request.model, input_tokens=99)
        self.counter.count_tokens.side_effect = count
        with self.assertRaisesRegex(RequestBudgetError, "changed during"):
            self._check(request=request)
        self.counter.count_tokens.assert_called_once()

    def test_approval_cannot_be_reused_for_another_question_or_settings(self):
        approval = self._check()
        requests = (
            replace(self.request, question="Another question"),
            replace(self.request, config=_config(max_output_tokens=255)),
            replace(self.request, config=_config(model="models/another-synthetic")),
        )
        for request in requests:
            with self.subTest(request=request.model), self.assertRaisesRegex(RequestBudgetError, "does not match"):
                validate_budget_approval(request=request, approval=approval)
        self.counter.count_tokens.assert_called_once()

    def test_approval_cannot_be_reused_after_prompt_changes(self):
        approval = self._check()
        with patch("src.generation.gemini_request._SYSTEM_INSTRUCTION", "Changed policy for test"):
            with self.assertRaises(RequestBudgetError):
                validate_budget_approval(request=self.request, approval=approval)

    def test_approval_cannot_be_reused_after_candidate_count_changes(self):
        from src.generation import gemini_request
        approval = self._check()
        original = gemini_request._request_payload
        def changed(*args, **kwargs):
            payload = original(*args, **kwargs)
            payload["generationConfig"]["candidateCount"] = 1
            return payload
        with patch.object(gemini_request, "_request_payload", side_effect=changed):
            with self.assertRaises(RequestBudgetError):
                validate_budget_approval(request=self.request, approval=approval)

    def test_counter_exception_including_quota_failure_is_not_retried_or_estimated(self):
        failure = RuntimeError("Synthetic free quota exhausted")
        self.counter.count_tokens.side_effect = failure
        with self.assertRaises(RequestBudgetError) as caught:
            self._check()
        self.assertIs(caught.exception.__cause__, failure)
        self.counter.count_tokens.assert_called_once()

    def test_base_exception_propagates(self):
        failure = KeyboardInterrupt("Synthetic interrupt")
        self.counter.count_tokens.side_effect = failure
        with self.assertRaises(KeyboardInterrupt) as caught:
            self._check()
        self.assertIs(caught.exception, failure)
        self.counter.count_tokens.assert_called_once()

    def test_counter_method_is_resolved_once_and_must_be_callable(self):
        request = self.request
        class Counter:
            def __init__(self):
                self.lookups = 0
                self.calls = 0
            @property
            def count_tokens(self):
                self.lookups += 1
                def count(*, request):
                    self.calls += 1
                    return RequestTokenCount(request_id=request.request_id, model=request.model, input_tokens=1)
                return count
        counter = Counter()
        self._check(token_counter=counter)
        self.assertEqual((counter.lookups, counter.calls), (1, 1))
        for value in (None, object()):
            with self.subTest(value=value), self.assertRaises(RequestBudgetError):
                self._check(token_counter=value)
        with self.assertRaises(RequestBudgetError):
            self._check(token_counter=type("InvalidCounter", (), {"count_tokens": None})())

    def test_records_are_frozen_slotted_and_approval_does_not_change_context_status(self):
        approval = self._check()
        receipt = RequestTokenCount(request_id=self.request.request_id, model=self.request.model, input_tokens=99)
        for instance, field in ((self.budget, "input_token_limit"), (approval, "input_tokens"), (receipt, "input_tokens")):
            with self.assertRaises(FrozenInstanceError):
                setattr(instance, field, None)
            self.assertFalse(hasattr(instance, "__dict__"))
        self.assertEqual(self.context.budget_status, "unbudgeted")
        self.assertEqual(self.request.spending_policy, "free-tier-only")

    def test_direct_approval_construction_checks_both_limits(self):
        values = dict(request_id=self.request.request_id, model=self.request.model, input_tokens=100, max_output_tokens=256, budget=self.budget)
        for changes in ({"input_tokens": 101}, {"max_output_tokens": 257}, {"input_tokens": True}, {"budget": None}):
            with self.subTest(changes=changes), self.assertRaises(RequestBudgetError):
                BudgetApproval(**{**values, **changes})

    def test_invalid_approvals_or_requests_are_revalidated(self):
        approval = self._check()
        damaged = deepcopy(approval)
        object.__setattr__(damaged, "input_tokens", True)
        for changes in ({"approval": None}, {"approval": damaged}, {"request": None}):
            with self.subTest(changes=changes), self.assertRaises(RequestBudgetError):
                validate_budget_approval(**{"request": self.request, "approval": approval, **changes})

    def test_check_arguments_are_required_and_keyword_only(self):
        with self.assertRaises(TypeError):
            check_request_budget(request=self.request, budget=self.budget)
        with self.assertRaises(TypeError):
            check_request_budget(self.request, self.budget, self.counter)

    def test_budget_check_reads_no_files_or_credentials(self):
        with patch("builtins.open", side_effect=AssertionError("No files")), patch("os.getenv", side_effect=AssertionError("No credentials")):
            self.assertEqual(self._check().input_tokens, 100)


if __name__ == "__main__":
    unittest.main()
