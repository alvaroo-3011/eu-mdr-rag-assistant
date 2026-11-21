"""Framework-independent presentation and explicit-submission tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import tempfile
from pathlib import Path
import unittest

import main as streamlit_entry
from src.application.answer_policy import LocalInsufficientEvidence, LocalNeedsClarification
from src.application.presentation import (
    AnswerBlockView,
    AnswerView,
    build_answer_view,
    citation_markers,
    cited_sources,
    public_error_message,
    source_line,
)
from src.generation.gemini_provider_response import (
    GeminiProviderGeneration,
    parse_generate_content_response,
)
from src.generation.gemini_request import GeminiGenerationConfig, prepare_gemini_request
from src.generation.gemini_transport import GeminiTransportRun
from src.generation.request_budget import BudgetApproval, RequestBudget
from src.generation.evidence_context import build_evidence_context
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.evidence_selection import select_evidence
from src.retrieval.fusion import fuse_candidates
from tests.unit.generation.test_gemini_provider_response import _envelope
from tests.unit.ingestion.test_chunk_models import (
    _chunk,
    _chunk_set,
    _chunk_set_id,
    _guidance_source,
    _manifest,
)
from tests.unit.retrieval.test_dense_results import _representative_chunk


def _run(status: str = "answered", *, rich: bool = False) -> GeminiTransportRun:
    if rich:
        original = _representative_chunk()
        manifest = _manifest(original.source)
        chunk_set = _chunk_set(manifest, (
            type(original)(
                chunk_id=original.chunk_id,
                chunk_set_id=original.chunk_set_id,
                source=original.source,
                order=0,
                text=original.text,
                block_selections=original.block_selections,
                hierarchies=original.hierarchies,
                pages=original.pages,
                legal_locators=original.legal_locators,
                table_segments=original.table_segments,
                annotations=original.annotations,
                previous_chunk_id=None,
                next_chunk_id=None,
            ),
        ))
    else:
        source = _guidance_source()
        manifest = _manifest(source)
        identity = _chunk_set_id(manifest)
        chunk = _chunk(identity, source, text="Synthetic evidence without locators.")
        chunk_set = _chunk_set(manifest, (chunk,))
    chunk = chunk_set.chunks[0]
    dense = (DenseCandidate(
        1,
        chunk.chunk_id,
        chunk.text,
        deserialize_chunk_metadata(chunk_metadata(chunk)),
        0.1,
    ),)
    fused = fuse_candidates(
        chunk_set=chunk_set,
        dense_candidates=dense,
        lexical_candidates=(),
        rank_constant=2.0,
    )
    selection = select_evidence(chunk_set=chunk_set, fused_candidates=fused, max_chunks=1)
    context = build_evidence_context(selection=selection, chunk_set=chunk_set)
    generation_config = GeminiGenerationConfig(
        model="models/gemini-synthetic",
        max_output_tokens=256,
        temperature=0.25,
        thinking_budget=32,
        thinking_level=None,
    )
    request = prepare_gemini_request(
        question="Synthetic question?",
        context=context,
        chunk_set=chunk_set,
        config=generation_config,
    )
    payload = {
        "status": status,
        "answer_blocks": [],
        "limitation": None,
        "clarification_question": None,
    }
    if status == "answered":
        payload["answer_blocks"] = [{
            "text": "<script>model text remains text</script>",
            "citation_ids": ["1"],
        }]
    elif status == "needs_clarification":
        payload["clarification_question"] = "Which device scope do you mean?"
    else:
        payload["limitation"] = "The validated response reports a limitation."
    body = json.dumps(_envelope(json.dumps(payload)))
    generation = parse_generate_content_response(
        status_code=200,
        body_json=body,
        request_id=request.request_id,
        request=request,
        chunk_set=chunk_set,
    )
    budget = RequestBudget(input_token_limit=123, output_token_limit=256)
    approval = BudgetApproval(
        request_id=request.request_id,
        model=request.model,
        input_tokens=123,
        max_output_tokens=256,
        budget=budget,
    )
    return GeminiTransportRun(approval=approval, generation=generation)


class _FakeStreamlit:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return call

    @contextmanager
    def expander(self, label):
        self.calls.append(("expander", (label,), {}))
        yield

    @contextmanager
    def container(self, *, border):
        self.calls.append(("container", (), {"border": border}))
        yield


class LocalInterfaceTests(unittest.TestCase):
    def test_trusted_local_clarification_projects_and_renders_one_question(self):
        view = build_answer_view(LocalNeedsClarification())
        self.assertEqual(view.status, "needs_clarification")
        self.assertEqual(view.answer_blocks, ())
        self.assertIsNone(view.limitation)
        self.assertEqual(view.clarification_question,
                         "What is your device and what is its intended purpose?")
        fake = _FakeStreamlit()
        streamlit_entry._render_answer(fake, view)
        self.assertIn(("subheader", ("Clarification needed",), {}), fake.calls)
        self.assertIn(("text", (view.clarification_question,), {}), fake.calls)

    def test_trusted_local_insufficiency_projects_and_renders_fixed_content(self):
        local_result = LocalInsufficientEvidence()
        object.__setattr__(local_result, "limitation", "untrusted mutation")
        view = build_answer_view(local_result)

        self.assertEqual(view.status, "insufficient_evidence")
        self.assertEqual(view.answer_blocks, ())
        self.assertEqual(
            view.limitation,
            "No MDR evidence was selected and no provider request was made.",
        )
        self.assertIsNone(view.clarification_question)
        fake = _FakeStreamlit()
        streamlit_entry._render_answer(fake, view)
        self.assertIn(
            ("subheader", ("Insufficient selected evidence",), {}),
            fake.calls,
        )
        self.assertIn(("container", (), {"border": True}), fake.calls)
        self.assertIn(("text", (view.limitation,), {}), fake.calls)

    def test_all_four_statuses_project_and_render_without_pipeline_calls(self):
        expected_labels = {
            "answered": "Validated answer",
            "insufficient_evidence": "Insufficient selected evidence",
            "needs_clarification": "Clarification needed",
            "out_of_scope": "Outside the EU MDR MVP scope",
        }
        for status, label in expected_labels.items():
            with self.subTest(status=status):
                view = build_answer_view(_run(status))
                self.assertEqual(view.status, status)
                fake = _FakeStreamlit()
                streamlit_entry._render_answer(fake, view)
                self.assertIn(("subheader", (label,), {}), fake.calls)
                if status == "answered":
                    text_calls = [args[0] for name, args, _ in fake.calls if name == "text"]
                    self.assertIn("<script>model text remains text</script> [1]", text_calls)
                else:
                    if status == "needs_clarification":
                        self.assertIn(("text", (view.clarification_question,), {}), fake.calls)
                    else:
                        self.assertIn(("text", (view.limitation,), {}), fake.calls)
                    markdown = [args[0] for name, args, _ in fake.calls if name == "markdown"]
                    self.assertNotIn("##### Answer", markdown)
                    self.assertNotIn("##### Sources", markdown)
                    self.assertFalse(any(name in ("caption", "expander") for name, _, _ in fake.calls))

    def test_answer_renders_inline_markers_then_one_sources_list(self):
        view = build_answer_view(_run(rich=True))
        fake = _FakeStreamlit()
        streamlit_entry._render_answer(fake, view)
        rendered = [(name, args[0] if args else None) for name, args, _ in fake.calls]

        self.assertEqual(rendered, [
            ("container", None),
            ("subheader", "Validated answer"),
            ("markdown", "##### Answer"),
            ("text", "<script>model text remains text</script> [1]"),
            ("markdown", "##### Sources"),
            ("text", "[1] Guidance, 1.0 — Article 15; Annex I"),
            ("expander", "Technical details [1]"),
            ("text", f"Title: {view.answer_blocks[0].citations[0].official_title}"),
            ("text", "Document: " + view.answer_blocks[0].citations[0].document_id),
            ("text", "Version: 1.0"),
            ("text", "Source identity: " + view.answer_blocks[0].citations[0].source_identity),
            ("text", "Permitted regimes: MDR, IVDR"),
            ("text", "PDF page(s): 3, 5"),
            ("text", "Legal locator: article: 15 > paragraph: 3"),
            ("text", "Legal locator: annex: I > section: 23.2"),
            ("text", "Hierarchy: Scope"),
            ("text", "Hierarchy: Scope"),
            ("text", "Official URL: " + view.answer_blocks[0].citations[0].official_url),
            ("caption",
             "Validated citation identity does not by itself prove semantic support or sufficiency."),
        ])
        self.assertFalse(any(
            name == "caption" and str(value).startswith("Citations:") for name, value in rendered
        ))
        # Model and manifest strings are never passed to Markdown or HTML renderers.
        for name, args, kwargs in fake.calls:
            self.assertNotIn("unsafe_allow_html", kwargs)
            if name in ("markdown", "expander"):
                self.assertNotIn("<script>", str(args[0]))
                self.assertNotIn(view.answer_blocks[0].citations[0].official_title, str(args[0]))

    def test_sources_are_deduplicated_and_sorted_by_number(self):
        base = build_answer_view(_run(rich=True)).answer_blocks[0].citations[0]
        c2, c5, c10 = (replace(base, citation_id=n) for n in ("2", "5", "10"))
        view = AnswerView(
            status="answered",
            answer_blocks=(
                AnswerBlockView(text="First.", citations=(c5, c2, c5)),
                AnswerBlockView(text="Second.", citations=(c10, c2)),
            ),
            limitation=None,
            clarification_question=None,
        )
        self.assertEqual(citation_markers(view.answer_blocks[0]), "[5][2]")
        self.assertEqual(citation_markers(view.answer_blocks[1]), "[10][2]")
        self.assertEqual([c.citation_id for c in cited_sources(view)], ["2", "5", "10"])

        fake = _FakeStreamlit()
        streamlit_entry._render_answer(fake, view)
        texts = [args[0] for name, args, _ in fake.calls if name == "text"]
        self.assertIn("First. [5][2]", texts)
        self.assertIn("Second. [10][2]", texts)
        self.assertEqual(
            [args[0] for name, args, _ in fake.calls if name == "expander"],
            ["Technical details [2]", "Technical details [5]", "Technical details [10]"],
        )

    def test_source_line_uses_only_present_fields(self):
        base = build_answer_view(_run(rich=True)).answer_blocks[0].citations[0]
        self.assertEqual(source_line(base), "[1] Guidance, 1.0 — Article 15; Annex I")
        self.assertEqual(
            source_line(replace(base, locator_summary=(), pages=(3,))),
            "[1] Guidance, 1.0 — p. 3",
        )
        self.assertEqual(
            source_line(replace(base, locator_summary=(), pages=(3, 5))),
            "[1] Guidance, 1.0 — pp. 3, 5",
        )
        self.assertEqual(
            source_line(replace(base, locator_summary=(), pages=(), version="")),
            "[1] Guidance",
        )
        minimal = build_answer_view(_run()).answer_blocks[0].citations[0]
        self.assertEqual(source_line(minimal), "[1] Guidance, 1.0")

    def test_citations_use_authoritative_metadata_and_do_not_invent_locators(self):
        rich = build_answer_view(_run(rich=True)).answer_blocks[0].citations[0]
        self.assertEqual(rich.pages, (3, 5))
        self.assertEqual(rich.legal_locators, (
            "article: 15 > paragraph: 3",
            "annex: I > section: 23.2",
        ))
        self.assertEqual(rich.locator_summary, ("Article 15", "Annex I"))
        self.assertEqual(rich.short_title, "Guidance")
        self.assertEqual(rich.hierarchies, ("Scope", "Scope"))
        self.assertIn("MDR", rich.permitted_regimes)
        self.assertIn("IVDR", rich.permitted_regimes)

        minimal = build_answer_view(_run()).answer_blocks[0].citations[0]
        self.assertEqual(minimal.pages, ())
        self.assertEqual(minimal.legal_locators, ())
        self.assertEqual(minimal.locator_summary, ())
        self.assertEqual(minimal.hierarchies, ())

    def test_submission_and_rerun_call_pipeline_once_and_retain_only_session_view(self):
        session = {}
        calls = []
        view = build_answer_view(_run())

        def submit(question):
            calls.append(question)
            return view

        streamlit_entry._apply_submission(
            submitted=True,
            question="  Synthetic question?  ",
            session_state=session,
            submitter=submit,
        )
        streamlit_entry._apply_submission(
            submitted=False,
            question="  Synthetic question?  ",
            session_state=session,
            submitter=submit,
        )
        streamlit_entry._render_answer(_FakeStreamlit(), session[streamlit_entry._RESULT_KEY])

        self.assertEqual(calls, ["  Synthetic question?  "])
        self.assertIs(session[streamlit_entry._RESULT_KEY], view)
        self.assertNotIsInstance(session[streamlit_entry._RESULT_KEY], GeminiProviderGeneration)

    def test_blank_and_failures_make_no_call_or_partial_result_and_leak_no_secret(self):
        session = {streamlit_entry._RESULT_KEY: build_answer_view(_run())}
        calls = []
        streamlit_entry._apply_submission(
            submitted=True,
            question=" \t ",
            session_state=session,
            submitter=lambda question: calls.append(question),
        )
        self.assertEqual(calls, [])
        self.assertNotIn(streamlit_entry._RESULT_KEY, session)

        secret = "SECRET_VALUE_MUST_NOT_LEAK"
        streamlit_entry._apply_submission(
            submitted=True,
            question="Question?",
            session_state=session,
            submitter=lambda _question: (_ for _ in ()).throw(RuntimeError(secret)),
        )
        self.assertNotIn(streamlit_entry._RESULT_KEY, session)
        self.assertNotIn(secret, session[streamlit_entry._ERROR_KEY])
        self.assertNotIn(secret, public_error_message(RuntimeError(secret)))

    def test_no_automatic_conversation_or_output_file_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = tuple(root.iterdir())
            session = {}
            streamlit_entry._apply_submission(
                submitted=True,
                question="Question?",
                session_state=session,
                submitter=lambda _question: build_answer_view(_run()),
            )
            self.assertEqual(tuple(root.iterdir()), before)
            self.assertEqual(set(session), {streamlit_entry._RESULT_KEY})


if __name__ == "__main__":
    unittest.main()
