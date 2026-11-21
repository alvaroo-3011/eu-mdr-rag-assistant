# Explicit Gemini REST transport and mocked request flow

## Scope and authorization

This checkpoint implements a synchronous HTTPX adapter and a small orchestration
function. It uses the already-declared `httpx==0.28.1`; requirements and prior
modules do not change. All checkpoint executions use HTTPX MockTransport, a fake
key, synthetic records, and socket guards. No Gemini operation, real key lookup,
SDK import, source loading, retrieval, model loading, or Chroma access occurs.

User policy remains free tier only, EUR 0 paid spending. This adapter cannot
verify remote billing or quota state. Providing it a real key and live backend
can make real calls; doing so remains separately unauthorized for this checkpoint.
There is no automatic live transport, CLI, environment-key loader, application
integration, or production model/parameter profile.

## Public interfaces

```python
GeminiTransportConfig(*, api_version, timeout_seconds, max_response_bytes)
GeminiRESTTransport(*, api_key, config, transport)
adapter.count_tokens(*, request) -> RequestTokenCount
adapter.generate_content(*, request, approval, chunk_set) -> GeminiProviderGeneration
run_gemini_request(*, request, chunk_set, budget, adapter) -> GeminiTransportRun
```

Configuration fields are required, without defaults. API version must be a safe
version path segment; syntax does not certify endpoint support. Timeout is a
finite positive nonboolean real. It applies to HTTPX connect/read/write/pool
operations individually, not a complete-operation wall-clock deadline. The byte
limit is a positive nonboolean integer per response and is not a model-token
budget. Fixture values (`v1beta`, 1.25 seconds, 20000 bytes) are synthetic test
settings, not approved production choices. Tests also demonstrate explicit `v1`
routing without asserting real-service support.

The caller explicitly supplies the API key and a synchronous HTTPX BaseTransport.
There is no key-variable choice, default backend, arbitrary base URL, SDK, or
implicit environment/proxy configuration. Tests supply only a fake ASCII key
and MockTransport. A future live backend requires separate review and approval.

The adapter owns its HTTPX client and the injected backend's close lifecycle.
Use it as a context manager or call close(); closed adapters reject reuse. It is
not designed for concurrent operations or externally mutated objects. The adapter
adds no retries. A custom injected backend can have its own retry, TLS, proxy,
or logging behavior; the adapter cannot independently prove those internals.
Only MockTransport behavior is validated here. A live backend profile must
explicitly establish no retries and appropriate transport settings later.

## Exact wire requests

The host is fixed to `https://generativelanguage.googleapis.com`. Each endpoint
uses the explicit version and the validated request model, followed by
`:countTokens` or `:generateContent`. There are no query parameters. API key
authentication uses the `x-goog-api-key` header, never the URL or model JSON.
The adapter does not read, print, persist, or include the key in its repr.

Count and generation use the existing request's exact UTF-8 encoded body JSON.
The count body wraps the complete logical request under generateContentRequest;
generation removes only the model routed through its URL. The adapter does not
reserialize, trim evidence, change settings, add tools, or modify the prompt.

An owned HTTPX client uses trust_env=false, follow_redirects=false, and auth=None.
Each operation builds a direct Request, avoiding automatic cookie-jar or default
header merging. Provider cookies are not replayed into subsequent requests.
Redirects are returned as failures; credentials are not forwarded to a redirect
target. The application adds one POST per count or generation, with no retry,
continuation, endpoint switching, or paid fallback.

## Response reading and existing validators

Non-200 responses raise the existing GeminiHTTPStatusError immediately, before
reading a body. This includes quota errors and redirects. No error-body text is
included in the displayed error. Accepted responses require application/json,
optionally with UTF-8 charset. Other media types, charsets, or parameters fail
this narrow profile. Accept-Encoding requests identity and compressed responses
are rejected rather than decompressed.

If Content-Length is supplied, it must be a nonnegative decimal integer, fit
the explicit byte limit, and equal the number of bytes received. The adapter
also counts streamed bytes, independently of a missing or incorrect declared
length, rejecting overflow before returning a body. Exact-limit bodies pass.
Bytes are decoded using strict UTF-8 without replacement. Empty/invalid JSON
then fails in the existing provider parser. This bounds application accumulation
of response bytes; it does not certify allocations inside an arbitrary injected
backend or establish a total elapsed-time limit.

Responses are closed on success and ordinary failure, including errors before
body reading. Request identity, model, and canonical JSON are snapshotted before
an operation and rechecked after it, preventing a returned answer/count from
being accepted after mutation during a backend callback.

count_tokens calls the existing count-response parser and returns its tagged
RequestTokenCount. generate_content revalidates the authoritative evidence and
matching BudgetApproval before sending, then invokes the existing provider
generation parser. HTTP success cannot bypass completion, safety, application
JSON, citation membership, or original-ChunkSet checks.

## Ordered application flow

run_gemini_request first revalidates the request's evidence against the supplied
ChunkSet. It uses the existing check_request_budget with this same adapter:
an output-cap overflow stops before HTTP; otherwise one count is sent, and an
input overflow or any count failure stops before generation. On success the
adapter checks the matching approval and sends exactly one generation request.

GeminiTransportRun is frozen/slotted and retains the BudgetApproval and
GeminiProviderGeneration, requiring matching request/model tags. The latter
retains the exact provider JSON and original trusted citation chunks. Direct
record construction checks local consistency, not authenticated provider origin.
No files or session data are automatically persisted. Use existing response
binding validation before future reuse/display.

The direct generate_content interface accepts an existing matching approval;
it does not independently authenticate how its count was obtained. Ordinary
Python construction of an approval is not provider attestation. The ordered
runner obtains one count through the chosen adapter but still cannot verify
remote quota or monetary cost from that receipt.

## Error and credential boundaries

Budget and provider-validation errors retain their existing types and causes.
Wire/backend/UTF-8 reading failures become fixed GeminiTransportError messages,
without printing the original network exception chain: those exceptions can
carry request headers or sensitive body text. Suppressed exception context is
not a secure erasure mechanism; credentials and request objects remain ordinary
in-process data. No logging hooks or error-report serialization are introduced.
External debuggers or arbitrary backend logging remain caller-owned behavior.

BaseException propagates. An ordinary cleanup failure does not mask an already
active exception or interrupt. There are no retries, JSON repair, partial run
results, alternative model calls, or fallback token estimates.

## Validation and remaining decisions

The focused suite uses real HTTPX Request/Response handling with MockTransport,
fake key/evidence, blocked sockets, and a guard against implicit HTTPTransport
construction. It checks exact wire bodies/routing/authentication/timeout fields,
count-then-generate order, budget failures before the appropriate HTTP stage,
quota/server failures, redirect rejection, timeout/error sanitization, byte/UTF-8
limits, content headers, request mutation, trusted citations, cookie isolation,
resource cleanup, immutable run records, and required explicit inputs.

Earlier successful suites need not be rerun: no previous implementation or test
changes. Local Windows review and the new focused test are still required.

The scratch test environment provisioned the repository's already-pinned HTTPX
version solely to execute this suite; no repository dependency file changed and
local Codex must not install anything for application of the patch.

Still unresolved before live execution: API version/model capability and free
eligibility, key-variable contract, actual timeout/byte limits and generation/
token-budget values, approved live backend/TLS/no-retry profile, real complete-
request count acceptance, actual usage/accounting, and authorization for a
limited live checkpoint. Semantic support, evidence sufficiency, answer quality,
Streamlit, and end-to-end acceptance remain separate work.

Primary references inspected on 18 October 2026:

- https://www.python-httpx.org/advanced/transports/
- https://www.python-httpx.org/advanced/timeouts/
- https://ai.google.dev/api/tokens
- https://ai.google.dev/gemini-api/docs/api-key

HTTPX 0.28.1 installed source was also inspected for synchronous send/timeout
behavior. Documentation and mocked success do not establish real API acceptance.
