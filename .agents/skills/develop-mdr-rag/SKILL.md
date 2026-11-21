---
name: develop-mdr-rag
description: Develop, review, debug, test, or plan the user's EU MDR regulatory RAG project. Use for work on its RAG-ready JSON ingestion, metadata and chunk schemas, corpus manifest, Chroma and BM25 retrieval, regime filters, Gemini grounding, numbered citations, Streamlit interface, evaluation set, repository structure, Python modules, tests, or approved implementation plan. Preserve established project decisions and surface unresolved parameters instead of guessing.
---

# Develop MDR RAG

Work as the implementation partner for the user's closed-corpus EU MDR RAG. Continue from approved decisions, protect regulatory traceability, and favor the smallest testable implementation.

## Load project context

Read the references selectively before acting:

- Read `references/project-contract.md` for scope, exclusions, collaboration style, or product behavior.
- Read `references/architecture.md` for repository design, component boundaries, or implementation sequencing.
- Read `references/data-contract.md` for ingestion, manifest, metadata, chunks, outputs, citations, or index identity.
- Read `references/validation-checklist.md` for tests, QA, acceptance gates, debugging, or completion claims.

Inspect the current repository and its `AGENTS.md`, configuration, schemas, and tests when available. Treat current user instructions and approved repository artifacts as authoritative when they explicitly supersede a reference. Do not silently reinterpret an unresolved choice as an approved decision.

## Work one checkpoint at a time

1. State the immediate objective and the approved assumptions it depends on.
2. Inspect existing files before proposing or making changes.
3. Explain the next small change in plain language.
4. Implement only the authorized step with simple reusable Python.
5. Run the narrowest relevant validation.
6. Report the observed result and pause at a material decision gate.

Give concrete commands and short checkpoints. Avoid large untested command dumps. Troubleshoot from actual errors or evidence rather than inventing likely failures.

## Protect project invariants

- Accept only prepared, validated JSON into the RAG application. Keep raw-PDF extraction and correction outside the application unless the user explicitly reopens that scope.
- Preserve tables, page identity, section hierarchy, source/version identity, and regulatory regime metadata.
- Keep MDR and IVDR evidence separate. Do not answer IVDR-focused questions in the MDR MVP.
- Require the authoritative corpus manifest to match source files and built indexes.
- Ground substantive regulatory claims only in retrieved, permitted evidence.
- Validate every displayed citation against trusted metadata.
- Refuse when evidence is insufficient; do not let Gemini fill gaps from general knowledge.
- Keep session memory non-persistent unless the user explicitly saves a test case.
- Never treat the tool as legal advice or invite confidential, personal, or client-identifying data.

## Keep the implementation proportionate

Prefer explicit Python modules, typed data models where helpful, deterministic filters, and focused tests. Maintain separation between ingestion, retrieval, generation, evaluation, and UI. Do not introduce a framework, hosted service, agentic workflow, live web search, or database migration merely because it is popular.

When suggesting a dependency, explain the concrete problem it solves, its cost or complexity, and whether the standard library or an existing dependency already suffices. Preserve compatibility with the approved local Windows/CPU MVP and reusable logic that can be imported where relevant.

## Handle decisions safely

For unresolved parameters such as embedding model, chunk size and overlap, fusion weights, candidate counts, evidence counts, sufficiency threshold, session-context limits, or pilot access method:

1. Mark the parameter as unresolved.
2. Present a small set of testable options with tradeoffs.
3. Recommend a provisional experiment when useful.
4. Obtain user approval before making the choice authoritative.
5. Record the decision in the appropriate project artifact and update this skill only when asked.

## Completion standard

Do not call a stage complete because code was written. Require relevant tests or validation output, confirm traceability and scope controls, identify any untested condition, and leave the project in a reproducible state.
