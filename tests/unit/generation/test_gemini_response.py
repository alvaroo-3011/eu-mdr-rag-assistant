"""Offline response/citation boundary tests with synthetic source records only."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

from tests.unit.retrieval.test_lexical_query import _forbid_production_imports, _synthetic_chunk_set

with _forbid_production_imports():
    from src.generation.gemini_response import (
        CitedAnswerBlock, GeminiResponseError, ParsedGeminiResponse,
        parse_gemini_response, validate_response_binding,
    )

from src.generation.evidence_context import build_evidence_context
from src.generation.gemini_request import prepare_gemini_request
from tests.unit.generation.test_gemini_request import _fixture
from tests.unit.generation.test_evidence_context import _single_selection
from tests.unit.ingestion.test_chunk_models import _chunk_set, _guidance_source, _manifest
from tests.unit.retrieval.test_dense_results import _representative_chunk


_MODULE = "src.generation.gemini_response"


def _payload(**changes):
    return {
        "status": "answered", "answer_blocks": [{"text": "Synthetic answer.", "citation_ids": ["1"]}],
        "limitation": None, "clarification_question": None, **changes,
    }


def _rich_records():
    chunk = replace(_representative_chunk(), order=0, previous_chunk_id=None, next_chunk_id=None)
    return chunk, _chunk_set(_manifest(chunk.source), (chunk,))


class GeminiResponseTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_forbid_production_imports())
        self.chunk_set, self.context, self.config, self.request = _fixture()

    def _parse(self, payload=None, **changes):
        return parse_gemini_response(**{
            "response_json": json.dumps(_payload() if payload is None else payload, ensure_ascii=False),
            "request_id": self.request.request_id, "request": self.request,
            "chunk_set": self.chunk_set, **changes,
        })

    def _other_request(self, **changes):
        return prepare_gemini_request(**{
            "question": "Another question?", "context": self.context,
            "config": self.config, "chunk_set": self.chunk_set, **changes,
        })

    def test_valid_answer_retains_text_and_original_authoritative_chunks(self):
        text = '  Unicode α\n<script>untrusted data</script>  '
        result = self._parse(_payload(answer_blocks=[{"text": text, "citation_ids": ["2", "1"]}]))
        self.assertEqual(result.request_id, self.request.request_id)
        self.assertEqual(result.model, self.request.model)
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer_blocks[0].text, text)
        self.assertEqual(result.answer_blocks[0].citation_ids, ("2", "1"))
        self.assertIs(result.answer_blocks[0].citations[0].chunk, self.chunk_set.chunks[1])
        self.assertIs(result.answer_blocks[0].citations[1].chunk, self.chunk_set.chunks[0])
        validate_response_binding(response=result, request=self.request)

    def test_all_nonanswer_statuses_accept_the_documented_shape(self):
        for status in ("insufficient_evidence", "needs_clarification", "out_of_scope"):
            with self.subTest(status=status):
                value = _payload(status=status, answer_blocks=[], limitation="Selected evidence is limited.")
                if status == "needs_clarification":
                    value["clarification_question"] = "Which device do you mean?"
                result = self._parse(value)
                self.assertEqual(result.status, status)
                self.assertEqual(result.answer_blocks, ())

    def test_optional_limitations_accept_null_or_nonblank_without_trimming(self):
        for status in ("answered", "needs_clarification"):
            for limitation in (None, "  A qualification.\n"):
                with self.subTest(status=status, limitation=limitation):
                    value = _payload(status=status, limitation=limitation)
                    if status == "needs_clarification":
                        value.update(answer_blocks=[], clarification_question="What scope?")
                    self.assertEqual(self._parse(value).limitation, limitation)

    def test_multiple_blocks_may_reuse_one_citation(self):
        result = self._parse(_payload(answer_blocks=[
            {"text": "First.", "citation_ids": ["1"]},
            {"text": "Second.", "citation_ids": ["1", "2"]},
        ]))
        self.assertEqual(len(result.answer_blocks), 2)
        self.assertEqual(result.answer_blocks[1].citation_ids, ("1", "2"))

    def test_plain_json_whitespace_and_escaped_unicode_are_accepted(self):
        raw = ' \n' + json.dumps(_payload(), ensure_ascii=True) + '\t '
        self.assertEqual(self._parse(response_json=raw).status, "answered")
        value = _payload(answer_blocks=[{"text": "α", "citation_ids": ["1"]}])
        self.assertEqual(self._parse(response_json=json.dumps(value)).answer_blocks[0].text, "α")

    def test_invalid_json_code_fences_trailing_data_and_bom_are_not_repaired(self):
        raw = json.dumps(_payload())
        for value in ("", "   ", "{", "```json\n" + raw + "\n```", raw + raw,
                      raw + " commentary", "\ufeff" + raw, '{"status": /* comment */ "answered"}'):
            with self.subTest(raw=value[:20]), self.assertRaises(GeminiResponseError):
                self._parse(response_json=value)

    def test_raw_json_requires_exact_string_type(self):
        class Text(str):
            pass
        for value in (None, b"{}", {}, [], 1, True, Text("{}")):
            with self.subTest(value=type(value)), self.assertRaises(GeminiResponseError):
                self._parse(response_json=value)

    def test_root_requires_exact_four_fields(self):
        for value in ([], None, True, "text", 3):
            with self.subTest(value=value), self.assertRaises(GeminiResponseError):
                self._parse(response_json=json.dumps(value))
        for key in _payload():
            value = _payload()
            del value[key]
            with self.subTest(missing=key), self.assertRaises(GeminiResponseError):
                self._parse(value)
        with self.assertRaises(GeminiResponseError):
            self._parse(_payload(request_id=self.request.request_id))

    def test_duplicate_root_and_nested_keys_fail_even_with_identical_values(self):
        raw = json.dumps(_payload())
        cases = (
            raw[:-1] + ', "status": "answered"}',
            raw.replace('"text": "Synthetic answer."', '"text": "Synthetic answer.", "text": "Synthetic answer."'),
            raw.replace('"citation_ids": ["1"]', '"citation_ids": ["1"], "citation_ids": ["1"]'),
        )
        for value in cases:
            with self.subTest(raw=value), self.assertRaisesRegex(GeminiResponseError, "Duplicate"):
                self._parse(response_json=value)

    def test_nonstandard_numeric_constants_fail(self):
        raw = json.dumps(_payload())
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaises(GeminiResponseError):
                self._parse(response_json=raw.replace('"limitation": null', '"limitation": ' + constant))

    def test_invalid_unicode_raw_and_escaped_are_rejected(self):
        for value in (json.dumps(_payload()).replace("Synthetic answer.", "\ud800"),
                      json.dumps(_payload()).replace("Synthetic answer.", r"\ud800")):
            with self.subTest(raw=repr(value)), self.assertRaises(GeminiResponseError):
                self._parse(response_json=value)

    def test_unknown_nonstring_or_noncanonical_status_fails(self):
        for value in ("ANSWERED", " answered", "partial", "", None, [], True, 1):
            with self.subTest(status=value), self.assertRaises(GeminiResponseError):
                self._parse(_payload(status=value))

    def test_answer_blocks_require_array_and_nonempty_for_answered(self):
        for value in (None, {}, "text", True, 1, []):
            with self.subTest(value=value), self.assertRaises(GeminiResponseError):
                self._parse(_payload(answer_blocks=value))

    def test_nonanswer_statuses_reject_any_answer_blocks(self):
        for status in ("insufficient_evidence", "needs_clarification", "out_of_scope"):
            value = _payload(status=status, limitation="Not a complete answer.")
            if status == "needs_clarification":
                value["clarification_question"] = "What scope?"
            with self.subTest(status=status), self.assertRaises(GeminiResponseError):
                self._parse(value)

    def test_answer_blocks_reject_missing_extra_or_nonobject_members(self):
        for block in (None, [], "text", 1, {"text": "A"}, {"citation_ids": ["1"]},
                      {"text": "A", "citation_ids": ["1"], "source": "Invented source"}):
            with self.subTest(block=block), self.assertRaises(GeminiResponseError):
                self._parse(_payload(answer_blocks=[block]))

    def test_answer_text_must_be_nonblank_string(self):
        for text in (None, "", " \n\t", [], True, 1):
            with self.subTest(text=text), self.assertRaises(GeminiResponseError):
                self._parse(_payload(answer_blocks=[{"text": text, "citation_ids": ["1"]}]))

    def test_citation_ids_require_nonempty_array(self):
        for ids in (None, [], "1", {"1": True}, True, 1):
            with self.subTest(ids=ids), self.assertRaises(GeminiResponseError):
                self._parse(_payload(answer_blocks=[{"text": "A", "citation_ids": ids}]))

    def test_unknown_nonstring_or_coercible_citation_ids_fail(self):
        for citation_id in (None, 1, True, {}, [], "", "0", "3", "01", " 1", "1 ", "[1]", "١"):
            with self.subTest(citation_id=citation_id), self.assertRaises(GeminiResponseError):
                self._parse(_payload(answer_blocks=[{"text": "A", "citation_ids": [citation_id]}]))

    def test_duplicate_citations_in_one_block_fail(self):
        with self.assertRaisesRegex(GeminiResponseError, "Duplicate citation"):
            self._parse(_payload(answer_blocks=[{"text": "A", "citation_ids": ["1", "1"]}]))

    def test_bad_later_block_rejects_entire_response(self):
        with self.assertRaises(GeminiResponseError):
            self._parse(_payload(answer_blocks=[
                {"text": "First", "citation_ids": ["1"]},
                {"text": "Second", "citation_ids": ["unknown"]},
            ]))

    def test_limitations_reject_blank_or_wrong_type_for_every_status(self):
        for status in ("answered", "insufficient_evidence", "needs_clarification", "out_of_scope"):
            for limitation in ("", " \n", [], True, 1):
                value = _payload(status=status, limitation=limitation)
                if status != "answered":
                    value["answer_blocks"] = []
                if status == "needs_clarification":
                    value["clarification_question"] = "What scope?"
                with self.subTest(status=status, limitation=limitation), self.assertRaises(GeminiResponseError):
                    self._parse(value)

    def test_insufficiency_and_out_of_scope_require_limitation(self):
        for status in ("insufficient_evidence", "out_of_scope"):
            with self.subTest(status=status), self.assertRaises(GeminiResponseError):
                self._parse(_payload(status=status, answer_blocks=[]))

    def test_clarification_is_required_only_for_its_status(self):
        for question in (None, "", " \n", [], 1, True):
            with self.subTest(question=question), self.assertRaises(GeminiResponseError):
                self._parse(_payload(status="needs_clarification", answer_blocks=[], clarification_question=question))
        for status in ("answered", "insufficient_evidence", "out_of_scope"):
            value = _payload(status=status, clarification_question="What scope?", limitation="Scope.")
            if status != "answered":
                value["answer_blocks"] = []
            with self.subTest(status=status), self.assertRaises(GeminiResponseError):
                self._parse(value)

    def test_receipt_tag_is_explicit_and_must_match_before_parsing(self):
        for value in (None, "", " ", 1, True, self._other_request().request_id):
            with self.subTest(value=value), self.assertRaisesRegex(GeminiResponseError, "request_id|another prepared request"):
                self._parse(request_id=value, response_json="invalid JSON")

    def test_stale_output_cannot_be_bound_to_another_question_or_model(self):
        result = self._parse()
        changed = (self._other_request(), self._other_request(
            question=self.request.question, config=replace(self.config, model="models/other-synthetic")))
        for request in changed:
            with self.subTest(model=request.model), self.assertRaises(GeminiResponseError):
                validate_response_binding(response=result, request=request)

    def test_package_local_same_id_does_not_allow_cross_evidence_reuse(self):
        # Reuse a valid selected record in a new one-member package, renumbered 1.
        context = build_evidence_context(selection=_single_selection(self.chunk_set.chunks[1]), chunk_set=self.chunk_set)
        other = self._other_request(question=self.request.question, context=context)
        self.assertNotEqual(self.request.context.resolve_citation("1"), other.context.resolve_citation("1"))
        result = self._parse()
        forged = replace(result, request_id=other.request_id)
        with self.assertRaises(GeminiResponseError):
            validate_response_binding(response=forged, request=other)

    def test_authoritative_chunk_set_is_required_and_rechecked(self):
        foreign = _synthetic_chunk_set((_guidance_source(), ("Foreign record.",)))
        for chunk_set in (None, {}, self.chunk_set.chunks, foreign):
            with self.subTest(chunk_set=type(chunk_set)), self.assertRaises(GeminiResponseError):
                self._parse(chunk_set=chunk_set)

    def test_copied_request_resolves_to_supplied_original_chunk_objects(self):
        request = deepcopy(self.request)
        self.assertIsNot(request.context.resolve_citation("1"), self.chunk_set.chunks[0])
        result = self._parse(request=request)
        self.assertIs(result.answer_blocks[0].citations[0].chunk, self.chunk_set.chunks[0])

    def test_request_change_during_parsing_fails(self):
        request = deepcopy(self.request)
        loads = json.loads
        raw = json.dumps(_payload())
        def mutate(value, *args, **kwargs):
            if value == raw and "object_pairs_hook" in kwargs:
                object.__setattr__(request, "question", "Changed during parsing")
            return loads(value, *args, **kwargs)
        with patch(_MODULE + ".json.loads", side_effect=mutate), self.assertRaises(GeminiResponseError):
            self._parse(request=request, response_json=raw)

    def test_mutated_result_is_revalidated_before_reuse(self):
        result = self._parse()
        object.__setattr__(result.answer_blocks[0], "text", " ")
        with self.assertRaises(GeminiResponseError):
            validate_response_binding(response=result, request=self.request)

    def test_citation_binding_compares_complete_chunks_not_just_dense_projection(self):
        chunk, chunk_set = _rich_records()
        context = build_evidence_context(selection=_single_selection(chunk), chunk_set=chunk_set)
        request = self._other_request(context=context, chunk_set=chunk_set)
        result = self._parse(request=request, request_id=request.request_id, chunk_set=chunk_set)
        citation = result.answer_blocks[0].citations[0]
        # Neighbor IDs do not enter identity hashing or dense metadata, but are
        # part of the complete original record retained for local audit.
        altered = replace(citation.chunk, next_chunk_id="unexpected-neighbor")
        object.__setattr__(citation, "chunk", altered)
        with self.assertRaises(GeminiResponseError):
            validate_response_binding(response=result, request=request)

    def test_rich_dual_applicable_mdr_evidence_remains_intact(self):
        chunk, chunk_set = _rich_records()
        context = build_evidence_context(selection=_single_selection(chunk), chunk_set=chunk_set)
        request = self._other_request(context=context, chunk_set=chunk_set)
        result = self._parse(request=request, request_id=request.request_id, chunk_set=chunk_set)
        cited = result.answer_blocks[0].citations[0].chunk
        self.assertIs(cited, chunk)
        self.assertEqual(cited.table_segments, chunk.table_segments)
        self.assertEqual(cited.annotations, chunk.annotations)
        self.assertEqual(len(cited.source.permitted_regimes), 2)

    def test_output_records_are_frozen_and_inputs_remain_unchanged(self):
        before = deepcopy((self.chunk_set, self.context, self.request))
        result = self._parse()
        with self.assertRaises(FrozenInstanceError):
            result.status = "out_of_scope"
        with self.assertRaises(FrozenInstanceError):
            result.answer_blocks[0].text = "Changed"
        self.assertFalse(hasattr(result, "__dict__"))
        self.assertEqual((self.chunk_set, self.context, self.request), before)

    def test_direct_dataclass_construction_enforces_local_structure(self):
        result = self._parse()
        for changes in ({"answer_blocks": []}, {"status": "unknown"}, {"request_id": " "}, {"model": None}):
            with self.subTest(changes=changes), self.assertRaises(GeminiResponseError):
                replace(result, **changes)
        for changes in ({"text": " "}, {"citations": []}, {"citations": ()}, {"citations": (object(),)}):
            with self.subTest(changes=changes), self.assertRaises(GeminiResponseError):
                replace(result.answer_blocks[0], **changes)

    def test_invalid_request_and_binding_types_fail(self):
        for request in (None, {}, self.context, object()):
            with self.subTest(request=type(request)), self.assertRaises(GeminiResponseError):
                self._parse(request=request)
        for response in (None, {}, object()):
            with self.subTest(response=type(response)), self.assertRaises(GeminiResponseError):
                validate_response_binding(response=response, request=self.request)

    def test_ordinary_errors_retain_causes_and_baseexceptions_propagate(self):
        with self.assertRaises(GeminiResponseError) as caught:
            self._parse(response_json="{")
        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)
        for error in (KeyboardInterrupt(), SystemExit()):
            with self.subTest(error=type(error)), patch(_MODULE + ".build_evidence_context", side_effect=error):
                with self.assertRaises(type(error)):
                    self._parse()

    def test_deeply_nested_json_failure_is_normalized(self):
        raw = '[' * 1500 + '0' + ']' * 1500
        with self.assertRaises(GeminiResponseError):
            self._parse(response_json=raw)

    def test_no_semantic_support_or_answer_completeness_is_claimed(self):
        # A deliberate false statement with a known ID demonstrates the boundary:
        # syntactic/citation-identity acceptance is not regulatory verification.
        text = "This invented synthetic claim is unsupported by its citation."
        result = self._parse(_payload(answer_blocks=[{"text": text, "citation_ids": ["1"]}]))
        self.assertEqual(result.answer_blocks[0].text, text)
        self.assertFalse(hasattr(result, "is_sufficient"))

    def test_arguments_are_required_and_keyword_only(self):
        values = {"response_json": json.dumps(_payload()), "request_id": self.request.request_id,
                  "request": self.request, "chunk_set": self.chunk_set}
        for key in values:
            with self.subTest(missing=key), self.assertRaises(TypeError):
                parse_gemini_response(**{name: value for name, value in values.items() if name != key})
        with self.assertRaises(TypeError):
            parse_gemini_response(*values.values())


if __name__ == "__main__":
    unittest.main()
