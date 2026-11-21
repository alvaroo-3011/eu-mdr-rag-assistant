# Validation checklist

## Before modifying code

- Confirm the requested stage and current repository state.
- Identify the approved decision or unresolved parameter involved.
- Check for existing user changes and preserve unrelated work.
- Define the smallest observable success condition.

## Corpus and ingestion

- Validate manifest schema, unique IDs, paths, versions, regimes, checksums, and validation status.
- Reject unlisted, unvalidated, missing, duplicated, or changed sources.
- Confirm preservation of page identity, section hierarchy, tables, and source/version metadata.
- Check chunk IDs for stability and chunks for regime purity.
- Test representative regulation prose, numbered lists, cross-references, annexes, tables, and footnotes.

## Indexing and retrieval

- Tie dense and BM25/exact-term indexes to the same corpus identity.
- Detect stale, partial, or mismatched indexes before queries run.
- Test exact article/annex queries, paraphrases, abbreviations, ambiguous questions, source conflicts, and unsupported questions.
- Report dense, lexical, fused, filtered, and selected results separately during evaluation.
- Measure retrieval against versioned gold evidence rather than judging only the final prose.

## Generation and citations

- Confirm Gemini receives only selected evidence and permitted session context.
- Validate structured output before rendering.
- Verify that every citation ID was supplied with the evidence.
- Verify rendered document, version, locator, and page against trusted metadata.
- Verify every substantive regulatory claim has a citation.
- Verify insufficiency produces refusal or clarification rather than a confident answer.
- Verify catalogue suggestions are clearly outside the searched corpus.

## Privacy, scope, and resilience

- Test the no-confidential-data warning and informational-use disclaimer.
- Confirm session state is not automatically persisted.
- Confirm no raw-PDF upload or extraction path exists in Streamlit.
- Confirm no live web search or uncited Gemini knowledge is used.
- Confirm IVDR-only evidence cannot leak into MDR answers.
- Test malformed JSON, missing indexes, API errors, invalid model output, and empty retrieval without unhandled crashes.

## Provisional acceptance gates

Treat these as the approved specification's provisional targets and keep them versioned:

- Recall@5 at least 90%.
- Citation correctness at least 95%.
- Citation coverage 100%.
- Zero confident answers outside the corpus.
- Zero IVDR-only leakage.
- MDR priority applied in 100% of tested source conflicts.
- Required clarification requested in at least 90% of applicable tests.
- No unhandled crashes across the evaluation suite.
- Maintain approximately 45–60 versioned tests for the MVP.

Do not tune exclusively to the current gold set. Record configuration, corpus identity, model identity, date, metrics, and regressions for every meaningful evaluation run.

## Completion report

State what changed, what was tested, the observed result, and what remains unresolved. Do not claim success based solely on code inspection or a command returning without exercising the relevant behavior.
