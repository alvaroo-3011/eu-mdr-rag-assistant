# Deterministic evidence context, version 1

## Approved boundary

```python
build_evidence_context(*, selection, chunk_set) -> EvidenceContext
```

The builder accepts an `EvidenceSelection` and its original validated
`ChunkSet`. It constructs a deterministic model-evidence projection and a
separate local audit envelope. It does not call Gemini, build a prompt or
request, count tokens, select a model, enforce a budget, or decide sufficiency.
All packages are explicitly `unbudgeted` and require a future request-level
budget gate before API use. No automatic generation path is introduced.

## Input validation and provenance

The builder revalidates the selection envelope, selected pairs, candidates,
nested branch provenance, original chunks, and selected source metadata. It
checks the ChunkSet identity and full selected-record equality with matching
authoritative originals. A matching chunk ID and dense metadata projection
alone are insufficient: nonprojected source fields must also agree.

Selected fused ranks must form the existing prefix, scores must be
nonincreasing, and exact score ties must follow global ChunkSet position.
Omitted IDs must exist in the same original set and permit MDR, but their
records are not included in model context. Their identity checks do not turn
omissions into relevance judgments. Dual-applicable guidance remains eligible
when MDR is permitted.

Valid directly constructed or copied selections are accepted after these
checks; Python dataclasses cannot certify which function created an object.
Each citation binds to the actual authoritative Chunk object from the supplied
ChunkSet. The original supplied selection, candidates, and their object
references remain in the audit envelope.

The builder cannot reconstruct or independently validate an unavailable full
fused union, prove that no candidate was withheld, prove query/backend identity,
or recompute RRF without its constant. It does not reload source files or audit
an index. Callers remain responsible for supplying the validated original
ChunkSet and the selection from the appropriate retrieval flow.

## Model JSON contract

`context.model_context_json` returns a JSON string with exactly these root fields:

- `format`: `mdr-evidence-context/v1`.
- `budget_status`: `unbudgeted`.
- `chunk_set_id`: the original ChunkSet identity.
- `evidence`: an ordered array corresponding to selected fused order.

Each evidence entry contains `citation_id`, `chunk_id`, `source`, `text`,
`block_selections`, `hierarchies`, `pages`, `legal_locators`, `table_segments`,
and `annotations`.

The source projection contains `source_identity`, `document_id`,
`official_title`, `short_title`, `version`, `source_type`, `authority`,
`authority_tier`, `official_url`, `sha256`, `regulatory_regimes`, and
`permitted_regimes`.

Nested structure is retained explicitly:

- Block selections preserve canonical/source block IDs, source ordering, and
  half-open character or table-row ranges, including absent values.
- Hierarchy and legal-locator paths preserve nesting, ordering, and empty titles.
- Pages preserve both zero-based source indices and physical PDF page numbers.
  Regulation HTML without pages keeps an empty page array; no page is invented.
- Table segments preserve block ID, source-row range, completeness, and the
  entire table: headers, rows, extraction method, source page, table number,
  bounding box, and header-detection state. Null cells and empty strings remain
  distinct. No table or row is split or removed.
- Annotations preserve kind, identifier, target, raw HTML, and text. Raw HTML
  remains a JSON string; it is not executed or rendered by this component.

Enums become string values, tuples become ordered arrays, and optional values
become JSON null. Keys are sorted, separators are compact, Unicode is retained,
and there is no trailing newline. Quotes, backslashes, control characters, and
source line breaks receive standard JSON escaping. No Unicode normalization,
whitespace normalization, string replacement, or source repair is performed.
Nonfinite numeric values and invalid UTF-8 output are rejected. The UTF-8
encoding check is not token counting or a byte-budget limit.

Field lists are explicit rather than automatic dataclass dumps. New domain
fields must not silently enter the model payload under the same context version.
This is a versioned context projection, not a complete serialized Chunk export.

## Local audit envelope and citations

The immutable, slotted `EvidenceContext` retains the original `selection` and
an immutable ordered tuple of `EvidenceCitation` records. The selection keeps
complete original objects, retrieval scores and ranks, branch provenance,
selection limit, and omitted IDs. Source local paths, content format, validation
and operational index status, neighbor IDs, and chunk source-local order also
remain available through those original objects. These operational fields and
retrieval diagnostics are excluded from model JSON.

Never serialize the entire audit object as the model context; use the explicit
`model_context_json` projection. Query text, instructions, session context,
response schema, and output-token reservation are outside this component.

Citation IDs are exact ASCII strings `"1"` through `"N"`, assigned in selected
order. The same ordered evidence produces the same numbering, but a different
package may assign `"1"` to a different chunk. These are package-local references,
not permanent chunk or source identities. The durable evidence identity remains
`chunk_id`, with source/version identity and locators retained separately.

`context.resolve_citation(citation_id)` returns the authoritative original
Chunk. Unknown IDs and non-string values fail; integers, booleans, whitespace,
leading zeros, bracketed display forms, and other guessed variants are not
coerced. Resolution proves membership in this package, not that an answer's
claim is supported. Citation display, answer validation, and rendering remain
future work; mappings from different packages must not be mixed.

Direct `EvidenceCitation` and `EvidenceContext` construction checks local
record consistency and serializability. Only the builder also checks membership
and full-record equality against the supplied authoritative ChunkSet.

## Empty contexts, errors, and deferred decisions

An empty selection produces an empty evidence array and no citations, while
retaining the ChunkSet identity and unbudgeted status. This is not a finding of
corpus absence, an automatic refusal, or a sufficiency judgment. Omissions,
nonempty context, or successful serialization likewise do not establish support.

Ordinary failures raise `EvidenceContextError(ValueError)` and preserve their
underlying causes. No partial package or silent fallback is returned.
`BaseException` propagates.

Synthetic tests cover deterministic serialization, exact structural values,
Unicode/escaping, nulls, package-local citation mapping, original-record checks,
audit exclusion, invalid inputs, and error handling. No real retrieval or API
call is required for this checkpoint.

Model selection, SDK dependency/configuration, exact token accounting over the
complete request, prompt overhead, output reservation, budget values, overflow
policy, sufficiency/refusal behavior, and API integration remain unresolved.
The builder does not trim or drop evidence to fit a size limit. Existing
retrieval, selection, and evaluation entry points remain unchanged.
