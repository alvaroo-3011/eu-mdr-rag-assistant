# Data, evidence, and answer contract

## Authoritative manifest

Maintain one manifest entry per source with, at minimum:

- stable document identifier
- official title and short title
- document version or revision
- regulatory regime
- source type and authority priority
- official URL
- local validated-JSON path
- source checksum
- extraction-validation status
- index-build identity or status

Fail closed when a file is missing, duplicated, unvalidated, has an unexpected checksum, or conflicts with the manifest. Rebuild both dense and lexical indexes whenever corpus identity changes.

## Chunk requirements

Each chunk must retain enough trusted metadata to reconstruct its source without model invention:

- stable chunk identifier
- document identifier, title, and version
- regulatory regime
- source type and authority priority
- human-readable section hierarchy or path
- article, annex, section, or heading locator when present
- page number or page range
- ordered chunk text, including table content where applicable
- parent or neighboring context identifiers when used
- corpus/index build identity

Do not discard page numbers, tables, hierarchy, or source/version identity for token convenience. Do not mix MDR and IVDR material in a chunk.

## Query and evidence controls

- Filter by regime and current source/version rules deterministically.
- Favor authoritative regulation text when guidance conflicts or merely interprets it.
- Preserve exact terms, article numbers, annex references, abbreviations, and device-class expressions through lexical retrieval.
- Keep retrieved candidates distinguishable from the final evidence selected for generation.
- Refuse or request clarification when selected evidence is insufficient or the question is materially ambiguous.

## Generation contract

Provide Gemini only:

- the current user question
- permitted session context
- selected evidence with stable citation IDs
- the response schema and grounding instructions

Expect structured output containing answer blocks, citation IDs, a limitation or insufficiency indicator, an optional clarification question, and optional catalogue suggestions. Reject unknown citation IDs and claims that lack permitted evidence.

Render citations from trusted metadata as document/version plus article, annex, section, and page where available. Never ask Gemini to invent or format source identity from memory.

## Unresolved parameters

Do not hard-code these as approved until the user decides after testing:

- embedding model
- chunk size and overlap
- dense/lexical fusion method and weights
- retrieval candidate and selected-evidence counts
- evidence-sufficiency threshold
- maximum session turns or context size
- tester access, hosting, and authentication after local validation
