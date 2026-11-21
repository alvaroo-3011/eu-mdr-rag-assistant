# Rank-preserving evidence selection baseline

## Purpose and boundary

The approved first selection step is a deterministic count-limited prefix of
the existing fused ranking. It retains the original complete chunks for later
context construction. It does not establish relevance, answer completeness,
semantic sufficiency, or readiness to call Gemini.

The public boundary is:

```python
select_evidence(*, chunk_set, fused_candidates, max_chunks) -> EvidenceSelection
```

Supply the validated original `ChunkSet`, the complete tuple returned by
`retrieve_hybrid()`, and an explicit positive integer `max_chunks` excluding
booleans. There is no default or approved production value. Values larger than
the supplied union retain the whole union without padding. Empty tuples are
valid, but context and parameter validation still apply.

## Validation and preservation

Every supplied row is validated before the count limit is applied, including
rows that will be omitted. Validation rejects:

- Non-tuple envelopes, incorrect candidate types, malformed candidate fields,
  and invalid nested branch records or metadata.
- Duplicate IDs, IDs absent from the original ChunkSet, different ChunkSet
  identities, different original text, or different complete decoded metadata.
- Original chunks that do not permit MDR, or fail their canonical chunk checks.
- Noncontiguous fused ranks, increasing RRF scores, and exact score ties that
  disagree with original global ChunkSet order.
- Duplicate or gapped ranks among the branch candidates present in the union.

The selector validates records; it does not repair them, sort them again, or
recompute fusion. The existing `FusedCandidate`, `Chunk`, metadata serializer,
and metadata decoder supply the corresponding record checks. Ordinary errors
raise `EvidenceSelectionError(ValueError)` and preserve underlying causes;
`BaseException` propagates.

Each immutable, slotted `SelectedEvidence` retains the exact original
`FusedCandidate` object and its matching original `Chunk` object. This preserves
all nested branch provenance, original text, source/version and regime data,
pages, hierarchy, legal locators, structured tables, annotations including raw
HTML, block selections, and neighbor IDs. Neighbor IDs are retained as metadata;
neighbors are not fetched or added. Dual-applicable guidance remains eligible
whenever its original source permits MDR.

`EvidenceSelection` contains:

- `chunk_set_id`: the supplied original ChunkSet identity.
- `max_chunks`: the explicitly requested limit.
- `selected`: a tuple of the first selected pairs, in original fused order.
- `omitted_chunk_ids`: the remaining IDs in original fused order.
- `has_omitted_candidates`: a derived boolean indicating count truncation only.

Pair construction checks local consistency. Result construction checks its
structural envelope. The `select_evidence()` function is the boundary that also
checks membership, the entire input, and ordering. Constructing a result directly
does not provide those function-level guarantees.

Omitted records remain in the caller-owned fused tuple. Omission means only
that the explicit count limit excluded a supplied candidate. It is not a
non-supporting judgment. Neither a nonempty result nor an absence of omissions
means sufficient evidence. Empty input does not establish corpus absence.

## Explicit limits of this checkpoint

The caller must supply the full fused union. Rank checks can detect some
incomplete inputs, but cannot prove that no backend record was withheld. Query
and backend identities are likewise not independently provable from candidates.
The RRF constant is not an input, so score arithmetic is not independently
recomputed. The supplied ChunkSet is a validated, immutable ingestion object;
this function does not reload or checksum the source files or audit an index.

No retrieval, model, network, Chroma, evaluation seed, or judgment file is used.
No raw-score threshold, reranking, semantic deduplication, authority reordering,
neighbor expansion, text trimming, citation assignment, prompt construction,
token counting, token budget, or refusal behavior is introduced.

The saved development experiments at `815ab47` show why sufficiency remains a
separate issue: C6's approved combination occupies fused ranks 1, 6, and 7 in
the depth-10 union. A limit of five would retain only one of those approved
members. This implementation deliberately preserves that ranking and exposes
omissions; it does not claim to fix C6 or approve a limit of seven. Runtime
selection never consults approved chunk IDs or support combinations. All
non-nominated chunks remain unjudged.

## Validation and follow-up

Focused synthetic tests exercise prefix selection, explicit limits, empty
input, complete-input rejection, original-object preservation, structured
context, regime handling, ordering, branch provenance, and error behavior.
They do not establish relevance or answer sufficiency. No real retrieval is
needed to verify this mechanical boundary.

Context budgeting, serialization of complete structured context, selection
quality, conflicting or incomplete support, runtime query scope, clarification,
refusal behavior, Gemini integration, and production parameters remain separate
unresolved decisions. The existing hybrid and evaluation entry points are
unchanged; nothing automatically invokes this selector yet.
