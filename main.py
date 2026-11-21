"""Thin Streamlit entry point for the local closed-corpus EU MDR assistant."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
import os
from pathlib import Path
import re
import sys

from src.application import local_config, local_runtime, presentation
from src.generation import gemini_provider_response, gemini_response, gemini_transport


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "local_app_config.json"
_RESULT_KEY = "mdr_latest_validated_result"
_ERROR_KEY = "mdr_latest_submission_error"


def _write_temporary_failure_diagnostic(error: Exception) -> None:
    """Print only a bounded stage, exception class, and optional HTTP status."""
    if os.environ.get("MDR_FAILURE_DIAGNOSTICS") != "1":
        return
    stage = "before_provider_or_unknown"
    name = "Exception"
    status = None
    seen = set()
    current = error
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        candidate = type(current).__name__
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", candidate):
            name = candidate
        if isinstance(current, gemini_provider_response.GeminiHTTPStatusError):
            value = current.status_code
            if type(value) is int and 100 <= value <= 599:
                status = value
        tb = current.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if code is gemini_transport.GeminiRESTTransport._post.__code__:
                operation = tb.tb_frame.f_locals.get("operation")
                if operation in ("countTokens", "generateContent"):
                    stage = operation
            elif code is gemini_provider_response.parse_count_tokens_response.__code__:
                stage = "countTokens_response_validation"
            elif code in (gemini_provider_response.parse_generate_content_response.__code__,
                          gemini_response.parse_gemini_response.__code__):
                stage = "generateContent_response_validation"
            tb = tb.tb_next
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    # This is deliberately not logging exception text, traceback, HTTP headers,
    # request/response bodies, or credential values.
    try:
        suffix = f" http_status={status}" if status is not None else ""
        print(f"MDR_FAILURE stage={stage} exception={name}{suffix}", file=sys.stderr, flush=True)
    except Exception:
        pass  # Diagnostics must not alter the browser failure handling.


def _execute_submission(question: str) -> presentation.AnswerView:
    config = local_config.load_local_app_config(CONFIG_PATH)
    api_key = local_runtime.load_gemini_api_key(os.environ)
    run = local_runtime.run_local_question(config=config, question=question, api_key=api_key)
    return presentation.build_answer_view(run)


def _apply_submission(
    *,
    submitted: bool,
    question: str,
    session_state: MutableMapping[str, object],
    submitter: Callable[[str], presentation.AnswerView],
) -> None:
    """Apply at most one explicit submission; ordinary reruns are inert."""

    if not submitted:
        return
    session_state.pop(_ERROR_KEY, None)
    session_state.pop(_RESULT_KEY, None)
    if type(question) is not str or not question.strip():
        session_state[_ERROR_KEY] = "Enter a nonblank EU MDR question."
        return
    try:
        view = submitter(question)
        if type(view) is not presentation.AnswerView:
            raise TypeError("submitter must return AnswerView")
    except Exception as exc:
        _write_temporary_failure_diagnostic(exc)
        session_state[_ERROR_KEY] = presentation.public_error_message(exc)
        return
    session_state[_RESULT_KEY] = view


def _render_citation(st, citation) -> None:
    # The expander label is fixed text plus the validated citation number, because
    # Streamlit interprets expander labels as Markdown. Metadata is plain st.text.
    with st.expander(f"Technical details [{citation.citation_id}]"):
        st.text(f"Title: {citation.official_title}")
        st.text(f"Document: {citation.document_id}")
        st.text(f"Version: {citation.version}")
        st.text(f"Source identity: {citation.source_identity}")
        st.text(f"Permitted regimes: {', '.join(citation.permitted_regimes)}")
        if citation.pages:
            st.text("PDF page(s): " + ", ".join(str(page) for page in citation.pages))
        for locator in citation.legal_locators:
            st.text("Legal locator: " + locator)
        for hierarchy in citation.hierarchies:
            st.text("Hierarchy: " + hierarchy)
        st.text("Official URL: " + citation.official_url)


def _render_answer(st, view: presentation.AnswerView) -> None:
    labels = {
        "answered": "Validated answer",
        "insufficient_evidence": "Insufficient selected evidence",
        "needs_clarification": "Clarification needed",
        "out_of_scope": "Outside the EU MDR MVP scope",
    }
    with st.container(border=True):
        st.subheader(labels[view.status])
        if view.answer_blocks:
            st.markdown("##### Answer")
            # Model text is rendered with st.text only: never Markdown or HTML.
            for block in view.answer_blocks:
                markers = presentation.citation_markers(block)
                st.text(f"{block.text} {markers}" if markers else block.text)
        if view.limitation is not None:
            st.text(view.limitation)
        if view.clarification_question is not None:
            st.text(view.clarification_question)
        sources = presentation.cited_sources(view)
        if sources:
            st.markdown("##### Sources")
            for citation in sources:
                st.text(presentation.source_line(citation))
                _render_citation(st, citation)
            st.caption(
                "Validated citation identity does not by itself prove semantic support or sufficiency."
            )


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        if exc.name == "streamlit":
            raise RuntimeError(
                "Streamlit is not installed. Install the pinned project dependency before launching the UI."
            ) from None
        raise

    st.set_page_config(page_title="EU MDR Assistant", page_icon="⚕️")
    st.title("EU MDR Assistant")
    st.write("Ask a question about the EU Medical Device Regulation.")

    with st.form("mdr_question_form", clear_on_submit=False):
        question = st.text_area("Question", placeholder="Ask one concrete EU MDR question.")
        submitted = st.form_submit_button("Submit question", type="primary")
        st.caption("Uses local retrieval and may call an LLM.")

    if submitted and isinstance(question, str) and question.strip():
        with st.spinner("Loading the local corpus, retrieving evidence, and validating the answer…"):
            _apply_submission(
                submitted=True,
                question=question,
                session_state=st.session_state,
                submitter=_execute_submission,
            )
    else:
        _apply_submission(
            submitted=submitted,
            question=question,
            session_state=st.session_state,
            submitter=_execute_submission,
        )

    error = st.session_state.get(_ERROR_KEY)
    if isinstance(error, str):
        st.error(error)
    result = st.session_state.get(_RESULT_KEY)
    if type(result) is presentation.AnswerView:
        _render_answer(st, result)

    st.divider()
    st.caption("Informational use only. Not legal advice.")
    st.caption("Do not enter confidential, personal, patient or client-identifying information.")
    st.caption("Answers use selected EU MDR documents only. Coverage is limited. IVDR is out of scope.")


if __name__ == "__main__":
    main()
