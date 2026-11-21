# Offline Gemini response and citation validation

## Scope and existing contract

This checkpoint enforces the four-field response contract already requested by
`gemini_request.py` and documented in `gemini_request_design.md`. It introduces
one pure parser, immutable result records, and a reuse/binding validator. It
does not change the request, prompt, response JSON schema, or earlier modules.

There is no SDK, transport, credential access, real token counting, generation,
dependency change, automatic integration, or corpus reconstruction. User policy
remains free tier only, EUR 0 paid spending. Synthetic success is not permission
to call the API, nor proof of model support, free quota, or billing state.

## Application-payload boundary

```python
parse_gemini_response(*, response_json, request_id, request, chunk_set)
    -> ParsedGeminiResponse

validate_response_binding(*, response, request) -> None
```

All arguments are explicit and keyword-only. `response_json` is the extracted
application JSON text, not a Gemini HTTP envelope. `request` is the exact
PreparedGeminiRequest used for that operation. `chunk_set` is the authoritative
original ChunkSet. The parser rechecks the prepared request, reconstructs the
evidence projection from those original records, and requires exact equality.

`request_id` is a separate caller-owned provenance tag. A future transport must
capture it from the request actually sent and carry it alongside the response.
It is not a fifth model-output field. The parser requires it to equal the
current content-bound request ID before parsing application JSON, and checks
that the request has not changed during parsing. This prevents accidental reuse
of a correctly tagged payload for a different question, evidence package, model,
or generation configuration. It cannot detect a caller deliberately or
incorrectly tagging old text with a new ID; model text cannot prove its origin.

## Strict JSON and status rules

The root must contain exactly `status`, `answer_blocks`, `limitation`, and
`clarification_question`. Answer block objects contain exactly `text` and
`citation_ids`. JSON objects, arrays, strings, and nulls must have their exact
expected types. There is no conversion of numeric IDs, automatic key removal,
code-fence stripping, JSON repair, truncation, or partial answer recovery.

Duplicate object keys are rejected at every depth, even if their values match.
NaN and infinity constants, trailing content, comments, BOM-prefixed text,
malformed JSON, invalid UTF-8 strings, and escaped lone surrogates are rejected.
Normal surrounding JSON whitespace is accepted. Accepted text is preserved
exactly, including whitespace, Unicode, and markup; it remains untrusted display
data and must not be executed or rendered as raw HTML by a future UI.

| Status | Answer blocks | Limitation | Clarification question |
| --- | --- | --- | --- |
| answered | One or more nonblank cited blocks | null or nonblank string | null |
| insufficient_evidence | Empty array | Nonblank string | null |
| needs_clarification | Empty array | null or nonblank string | Nonblank string |
| out_of_scope | Empty array | Nonblank string | null |

Optional text, when supplied, must be nonblank. These checks implement the
existing prompt's nonblank and cross-field requirements without changing the
four-field schema. Insufficiency describes the selected evidence; this parser
does not search the corpus or establish that no answer exists elsewhere.

## Trusted citation resolution and result records

Every answer block must supply a nonempty array of exact package-local string
IDs from this request. Unknown IDs, integer/bool IDs, leading zeroes, bracketed
IDs, whitespace-padded IDs, and noncanonical spellings are rejected. A block
cannot repeat the same citation ID; different blocks can cite the same evidence.
Order is preserved. No sorting, deduplication, or inferred citation is applied.

`CitedAnswerBlock` retains the original model text and an immutable tuple of
EvidenceCitation records. Its `citation_ids` property returns their ordered IDs.
The parser resolves each citation to the exact original Chunk object in the
supplied ChunkSet, retaining complete source/version identity, pages, legal
locators, hierarchy, structured tables, annotations, and raw HTML. It does not
reconstruct chunks from dense metadata or model-written bibliographic details.
Dual-applicable guidance remains eligible when MDR is permitted.

`ParsedGeminiResponse` retains request ID, model, status, an immutable tuple of
cited answer blocks, limitation, and clarification question. Both output record
types are frozen/slotted and validate their local structure. Direct construction
does not independently establish authoritative ChunkSet membership; use the
parser at initial ingress. No source display labels or rendered citations are
introduced yet.

`validate_response_binding` rechecks output structure, request/model identity,
and every cited complete chunk against the current request before later reuse or
display. A response bound to a different question or evidence package is rejected
even if both packages use citation ID "1". Matching a lossy metadata projection
is insufficient. This helper does not reload source files, reconstruct a
ChunkSet, or authenticate service provenance. No extra transport request occurs.

## Fail-closed behavior and practical limits

Any malformed block rejects the entire response; no valid prefix is returned.
Ordinary failures raise GeminiResponseError with original causes retained where
wrapped. BaseException propagates. There are no retries, fallback, counter calls,
generation calls, or state mutation. Excessively nested JSON parser failures are
normalized as ordinary errors; transport size limits remain future work.

Passing this boundary establishes structure and citation identity only. It does
not prove entailment, citation coverage within each block, completeness of the
answer, source-conflict resolution, temporal correctness, MDR scope compliance,
or evidence sufficiency. An unsupported sentence paired with a known citation
can pass these structural checks; a focused test makes that limitation explicit.
Model status values are checked for shape, not independently adjudicated.
Unjudged evidence remains unjudged.

The future transport must independently validate HTTP status, response shape,
candidate count, finish reasons, truncation, safety blocks, and content parts
before extracting application JSON. A syntactically valid JSON answer alone
cannot establish a successful provider completion. Usage accounting, request
budget approval, actual model/settings, API version, authentication contract,
timeouts, no-retry quota handling, and authorized free-tier execution remain
separate requirements. No HTTP adapter or end-to-end generation is completed.

## Validation

The new focused suite uses synthetic records and existing pure fixture helpers.
It tests all four statuses, strict types/keys/Unicode, duplicate JSON keys,
invalid constants and JSON, missing or unknown citations, per-block duplicate
IDs, whole-response rejection, exact original chunk references, rich metadata
preservation, dual applicability, authoritative reconstruction checks, changed
requests, copied request objects, cross-package citation collisions, output
corruption, immutable records, error causes, and BaseException propagation.

Only this new suite is run for the checkpoint. Earlier successful suites are
not repeated because no previous implementation or test file changes. Local
Windows review and the same focused test run are required before committing.
