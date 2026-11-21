# Architecture and implementation sequence

## Component boundaries

- `data/rag_ready/`: prepared, validated JSON accepted by the application.
- `src/ingestion/`: parse validated JSON, normalize records, validate schemas, and construct chunks.
- `src/retrieval/`: local dense retrieval, exact-term/BM25 retrieval, fusion, deterministic filters, reranking if later justified, and evidence selection.
- `src/generation/`: evidence sufficiency, Gemini prompt construction, structured response parsing, grounding, and refusal.
- `src/evaluation/`: versioned test cases, retrieval metrics, answer checks, citation checks, leakage checks, and regression reporting.
- `src/utils/`: shared utilities only when they do not belong to a domain component.
- `tests/`: deterministic unit and integration tests.
- `notebooks/`: experiments, not authoritative production logic.
- Streamlit UI: input, session-only context, warnings, answer rendering, citations, limitations, and explicit test-case saving.

Keep extraction as a separate upstream workflow. Streamlit must never convert PDFs or repair source data.

## Retrieval and generation flow

1. Load and validate the authoritative corpus manifest.
2. Load only manifest-approved validated JSON.
3. Normalize blocks while preserving document structure and traceability.
4. Build chunks with page locators, section paths, source/version identity, and regime tags.
5. Build local dense and BM25/exact-term indexes tied to the same corpus identity.
6. Route or clarify the query where required.
7. Apply deterministic regime, source, and version filters.
8. Retrieve and fuse candidates.
9. Select sufficient evidence or refuse.
10. Send Gemini only the question, permitted session context, and selected evidence.
11. Validate structured output and citation IDs before rendering.

## Recommended implementation order

1. Inspect and document the current repository state.
2. Finalize the manifest and normalized document/chunk schemas.
3. Create representative gold questions early.
4. Implement ingestion and deterministic schema validation.
5. Implement baseline dense retrieval and BM25/exact-term retrieval.
6. Add filters, fusion, evidence sufficiency, and catalogue routing.
7. Add grounded Gemini generation and citation validation.
8. Add the Streamlit interface and session-only context.
9. Run regression evaluation and tune only against versioned evidence.
10. Pilot with a small tester group after acceptance gates pass.

## Engineering approach

Prefer small importable Python modules over notebook-only logic. Keep configuration explicit and versioned. Avoid duplicating business rules across the UI, retriever, and generator. Add abstractions only after a real second use case appears.
