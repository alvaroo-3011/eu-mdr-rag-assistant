# Offline Gemini Developer REST response envelopes

## Checkpoint scope

This module fills the response-envelope boundary deferred by the application
response validator. It interprets caller-supplied HTTP status and JSON body;
it does not send an HTTP request, import an SDK/client, read credentials,
count tokens remotely, generate content, retry, or continue a response.
No previous module, prompt, response schema, dependency, or setting changes.

The user policy remains free tier only, EUR 0 paid spending. Parsing a synthetic
count or answer grants no live-execution permission and proves no billing state,
quota availability, model capability, endpoint acceptance, or monetary balance.

## Interfaces

```python
parse_count_tokens_response(*, status_code, body_json, request_id, request)
    -> RequestTokenCount

parse_generate_content_response(
    *, status_code, body_json, request_id, request, chunk_set
) -> GeminiProviderGeneration
```

Both functions require a PreparedGeminiRequest and its caller-owned request ID.
They revalidate the request, reject mismatched IDs, and check for request changes
during parsing. The future transport must capture this tag from the operation
actually sent; a caller falsely tagging a body cannot be detected from its text.
Input status and JSON are not authenticated service provenance.

Exact integer HTTP 200 is required. Other valid HTTP status codes raise
GeminiHTTPStatusError, retaining status_code but never including or parsing the
provider error body. There is no retry, fallback, or conversion of quota failure
to an empty count or unsupported-answer result. Invalid status types/ranges fail.
An error object inside an HTTP-200 body also fails.

Bodies must be exact strings containing one JSON object. Duplicate keys at all
depths, NaN/infinity, overflowing nonfinite numbers, malformed JSON, invalid
Unicode (including escaped lone surrogates), code fences, BOMs, comments, and
trailing content are rejected. Every decoded string is checked, including audit
metadata. No repair, coercion, prefix recovery, or text trimming is applied.

## Provider contract consulted

Google documents totalTokens as a nonnegative integer, candidate finish reasons,
prompt block feedback, candidate safety ratings, and ordered content parts.
STOP differs from MAX_TOKENS and CONTINUATION; a continuation can require another
request. These provider fields supply the basis for this application's narrower
acceptance policy below. They are not proof that the real endpoint accepts the
prepared request or that a selected model behaves as the fixtures assume.

Official references inspected on 18 October 2026:

- https://ai.google.dev/api/tokens
- https://ai.google.dev/api/generate-content

## Token-count response

The supported envelope contains required totalTokens and optional
cachedContentTokenCount, promptTokensDetails, and cacheTokensDetails. Total and
cached counts, when present, must be nonnegative exact integers excluding bool.
Explicit zero is accepted; a missing total is rejected rather than inferred as
zero from serialization defaults. Modality details must be arrays when present;
their internal semantics are not used for counting or independently certified.

The returned existing RequestTokenCount retains the original request/model tags
and totalTokens as input_tokens. It neither subtracts cached tokens nor substitutes
a generation usage counter. Missing totals are never estimated from text length.
The future counter must send the complete request.count_tokens_body_json, not a
contents-only approximation. Matching tags cannot independently prove what was
sent. Count-response raw-body retention remains caller-owned; the existing receipt
interface is unchanged. The receipt works with the existing injected budget gate.

## Generation acceptance policy

This is a conservative, non-streaming, single-candidate, text-only MVP boundary:

- Exactly one candidate must be present. Missing, empty, or multiple candidates
  fail. Candidate index may be omitted; when supplied it must be exact integer 0.
- finishReason must be exactly STOP. Missing, unknown, truncated, blocked, and
  continuation completions fail even if the text contains valid application JSON.
  Any continuationToken is unsupported; no follow-up request is made.
- Any promptFeedback.blockReason fails, including unspecified or malformed values.
  An absent/empty promptFeedback is permitted. Present safety-rating lists require
  unique nonblank categories and nonblank probability strings. A supplied blocked
  flag must be exactly false; true or wrong types fail. Absence is accepted as an
  omitted optional flag. No new probability threshold or semantic safety claim is
  introduced, and missing ratings do not certify safety.
- Content requires a nonempty parts array. Role may be absent; if supplied it
  must be model. Each part must contain string text and only optional thought
  and thoughtSignature metadata. A thought flag, if supplied, must be false.
  Thought output is rejected rather than silently discarded. Signatures remain
  opaque strings in the retained body, with no replay or verification.
- All text parts are concatenated in original order without inserting whitespace.
  Individual empty pieces are allowed, but the combined application text must
  be nonblank. Tools, media, mixed content, and unsupported part fields fail.
- External grounding/URL-context fields are outside this closed-corpus request
  profile and fail the candidate field allowlist. Provider citationMetadata may
  be retained as audit data; it never becomes an approved local citation.

Root, candidate, content, part, count, prompt-feedback, and safety-rating field
sets are explicit. Unsupported additions fail closed; this parser is not a
universal Gemini response decoder. Future provider changes may require a reviewed
compatibility update rather than silent field dropping. Optional ordinary audit
fields (for example modelVersion, responseId, usageMetadata, tokenCount, and
logprobsResult) are retained. Their principal container/string/count types are
checked; nested audit semantics and accounting arithmetic are not certified.

## Existing application validation and audit

Only after the envelope passes does generation parsing call the existing
parse_gemini_response once with the exact joined text, unchanged request tag,
original prepared request, and authoritative ChunkSet. The existing four response
statuses, strict application fields, and trusted chunk/citation checks all remain.
Any invalid block or citation rejects the whole operation, with the original
application-validation error retained as its cause.

GeminiProviderGeneration is frozen/slotted and retains response (the existing
ParsedGeminiResponse) and provider_body_json (the exact accepted input string).
Its constructor rechecks local consistency between retained envelope text and
parsed output; it cannot independently establish authoritative ChunkSet membership.
Use parse_generate_content_response for initial ingress, and the existing
validate_response_binding on result.response before future reuse/display.

Raw body retention preserves provider-reported model version, response identifier,
usage, safety feedback, signatures, and other supported audit metadata. It does
not automatically persist anything or verify the backend model. The nested
response.model remains the requested model; a provider-reported revision may
differ from an alias and is not treated as an independently verified revision.
Future rendering must use trusted chunk metadata and treat all model/provider
text as untrusted display data.

No answer correctness, evidence sufficiency, completeness, regime compliance,
entailment, or freedom from model background knowledge is established. Known
local citation IDs remain an identity check, not a support judgment.

## Validation and remaining work

Focused synthetic tests cover count receipts and their budget-gate integration,
HTTP failures, quota errors, strict JSON, candidates and finish reasons, safety
flags, text-part assembly, metadata preservation, stale/request mutation checks,
single application-parser invocation, authoritative citation resolution, immutable
audit results, error causes, BaseException propagation, and required arguments.
Existing successful suites are not repeated because their files are unchanged.

Still deferred: a real/injected HTTP adapter; endpoint/API version; key-variable
contract; response-byte and timeout limits; model/free-tier compatibility;
explicit production settings and budgets; HTTP headers/content-type handling;
authorized live counting/generation; actual accounting; and semantic evaluation.
No API execution is authorized by this checkpoint or its synthetic test values.
