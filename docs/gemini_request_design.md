# Offline Gemini request preparation and application token limits

## Authorization and checkpoint scope

Current user policy is **free-tier only; EUR 0 spending**. The user confirmed
that the key's project is on the free tier. No paid usage, billing activation,
credit purchase, automatic paid fallback, or upgrade is authorized. This is
distinct from a request's token limits and is not inferred from token counts.

This checkpoint adds only pure request preparation and application budget
checks using an injected counter. Tests supply fake counters. There is no HTTP
adapter, SDK import, credential access, real counter, generation call, automatic
retry, continuation, dependency change, or automatic application integration.

The planned integration direction is the Gemini Developer REST API using the
already-declared httpx dependency. Local inspection of google-genai 2.31.0
found that its Developer count_tokens converter rejects system_instruction,
tools, and generation_config, although generation supports them. Google
documents a generateContentRequest counting envelope for overall request input.
The proposed REST path avoids relying on a contents-only count. It has not
been tested against the real service; endpoint acceptance and actual token
accounting remain future separately authorized checks.

## Request boundary

```python
prepare_gemini_request(*, question, context, chunk_set, config)
    -> PreparedGeminiRequest
```

The builder revalidates the context, rebuilds its trusted projection against
the supplied original ChunkSet, and checks equality before preparation. It
retains the original context and explicit configuration as local audit data.
Direct PreparedGeminiRequest construction checks local consistency, but does
not establish authoritative ChunkSet membership.

Question text is required and nonblank; its original whitespace and Unicode
are retained as JSON data. Empty selected evidence raises
NoSelectedEvidenceError before any request is returned. This is a local lack
of selected evidence, not a corpus-absence finding or a model-generated refusal.

GeminiGenerationConfig requires all keyword-only fields, with no production
defaults: model, max_output_tokens, temperature, thinking_budget, thinking_level.

- model is an explicit models/<model-id> resource, not an inferred default.
- max_output_tokens is a positive integer excluding bool.
- temperature is a finite nonnegative nonboolean real, normalized to float.
  Model-specific supported ranges are not certified by this generic check.
- Exactly one thinking mode is supplied. thinking_budget is -1 or a nonnegative
  integer excluding bool; thinking_level uses MINIMAL, LOW, MEDIUM, or HIGH.
  The unused field must be explicitly None. Capability/range support is
  model-specific and remains a prerequisite to live use.

No actual model or numeric production settings are selected. Synthetic model
names and numeric fixtures are not recommendations or real model profiles.

The immutable PreparedGeminiRequest derives deterministic compact UTF-8 JSON
and a request ID from its content. The logical GenerateContentRequest includes:

- model identity;
- the versioned system instruction;
- one user content part containing the unchanged question and model-evidence
  projection, serialized together as JSON data;
- generationConfig with one candidate, explicit output cap and temperature,
  explicit thinking mode, includeThoughts=false, JSON response MIME type, and
  the response JSON schema.

No retrieval diagnostics, source local paths, secrets, tools, external search,
session history, cached content, or operational metadata are added. Setting
includeThoughts=false does not assert that internal thinking or its token use
is disabled.

request_json is the complete logical request including model. count_tokens_body_json
wraps that request under generateContentRequest, with no competing root contents
field. generate_content_body_json removes only model because the future
generation URL routes that same model. The two wire bodies intentionally differ
by routing/envelope; their common request content and configuration are identical.

request_id is gemini-request:sha256:<hash> over canonical logical request JSON,
request format mdr-gemini-request/v1, prompt version mdr-grounding/v2, and the
free-tier-only spending intent. It changes with question, evidence projection,
instructions, response schema, model, or generation settings. It is not a
provider model-revision identity or a cryptographic attestation of a token count.
Never serialize the retained entire audit dataclass as an API body.

## Prompt and response contract

The system instruction requests English, MDR-only, evidence-grounded answers;
treats the question and source content, including HTML, as data rather than
overriding instructions; preserves source qualifications and temporal framing;
gives supplied MDR regulation priority over conflicting guidance while
disclosing conflict; distinguishes wholly or inseparably outside-scope requests
from separable mixed-scope requests; sends material ambiguity or unresolved
product/regime facts to clarification; sends partial support or unresolved
material conflict to insufficiency; and prohibits external knowledge and
invented citations. Counts, retrieval scores, citation validity, and nominated
evidence recovery are explicitly not evidence-sufficiency decisions. These are
model instructions, not proofs of compliance or deterministic query routing.
Mocked outputs establish only the response contract; real behavioral and
semantic evaluation remain pending.

The requested response has four required fields:

- status: answered, insufficient_evidence, needs_clarification, or out_of_scope.
- answer_blocks: array of text and citation_ids records. Citation IDs are strings
  from this exact evidence package; the schema enumerates its allowed IDs.
- limitation: string or null.
- clarification_question: string or null.

The prompt requires a complete supported answer and cited substantive claims
for answered. Other statuses require empty answer_blocks. needs_clarification
requires a clarification question; other statuses use null for that field.
These cross-field conditions and nonblank text require a future local response
validator; the JSON schema alone does not enforce all of them. No response
parser, citation renderer, substantive sufficiency verifier, or answer-quality
claim is introduced. Valid syntax and known citation IDs do not prove support.

## Budget boundary

```python
check_request_budget(*, request, budget, token_counter) -> BudgetApproval
validate_budget_approval(*, request, approval) -> None
```

RequestBudget requires separate positive integer application ceilings:
input_token_limit and output_token_limit. They are not the provider's free-tier
allowance, remaining quota, a monetary balance, or verified model capacity.

The request's maximum output tokens is its requested output reservation/cap.
It must not exceed output_token_limit. Actual output and thinking usage remain
unknown; model-specific output/thinking accounting must be verified before live
use. This code does not invent a universal combined input/output context limit.

Preflight validates request and budget before resolving the injected count_tokens
method. If the requested output cap already exceeds its limit, no counting
method is resolved or called. Otherwise the bound method is retained once and
invoked once with the original complete PreparedGeminiRequest.

The counter must return RequestTokenCount containing request_id, model, and
nonnegative integer input_tokens excluding bool. The count must be for the
complete request, not just evidence, question text, or a character/byte estimate.
An HTTP response parser and a real counting adapter are outside this checkpoint.
The function revalidates the receipt, matches its request/model identities, and
detects request changes during counting. It rejects stale counts and malformed
envelopes. A returned receipt's tags prevent accidental cross-request reuse;
they cannot independently prove that an injected counter used the real provider.

Input exactly at its application limit is accepted; higher input raises
RequestBudgetExceededError. No overflow path trims text, splits tables, drops
chunks, silently changes settings, retries, estimates tokens, or returns partial
approval. Quota failures from a future counter likewise stop without retry or
paid fallback. Ordinary failures retain causes; BaseException propagates.

BudgetApproval retains request/model identity, counted input, requested output
cap, and the explicit application budget. validate_budget_approval rejects reuse
after the request, model, or output cap changes. It grants only application
token-limit approval: not generation permission, billing verification, free-quota
availability, provider/model compatibility, or semantic evidence sufficiency.
The original EvidenceContext remains unbudgeted; it is not mutated into a
provider-approved request. Live orchestration must explicitly verify the matching
request-level approval and all remaining prerequisites before generation.

## Financial and live-execution prerequisites

PreparedGeminiRequest.spending_policy is always free-tier-only as an audit intent.
It is not a Google request parameter and cannot enforce Google account billing.
The offline code does not read keys, query billing, infer project tier, purchase
credits, or choose another model. A future live adapter must use the confirmed
free-tier project and a free-eligible model, stop on quota exhaustion, and avoid
automatic retries, paid fallback, billing changes, and continuation calls.

Still unresolved before live execution: actual model and free-tier availability;
model-specific limits, temperature and thinking support; API version; key-variable
contract; explicit numeric input/output settings; timeout; real count and
generation transport behavior; HTTP parsing; accounting acceptance; response
validation; and final generation authorization. No paid or live execution is
authorized by a successful synthetic test or BudgetApproval.

## Validation and references

Focused synthetic tests verify request/body identity, evidence preservation,
response/citation shape, explicit settings, empty-context blocking, counting
once, exact budget boundaries, stale/malformed receipts, changes during counting,
no fallback, and error behavior. Existing successful suites need not be rerun
because their code is unchanged. These tests do not validate the real service,
model performance, remaining quotas, charges, or regulatory answer quality.

Provider references consulted on 17 October 2026:

- https://ai.google.dev/api/tokens
- https://ai.google.dev/api/generate-content
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/gemini-api/docs/billing

Provider contracts may change; verify the exact selected model/API behavior at
the later live checkpoint rather than treating these offline fixtures as proof.
