# Checkpoint 5 evidence-selection and context-budget review

## Status and scope

Checkpoint 5's bounded real-corpus comparison completed successfully. The run reconstructed the manifest-approved frozen corpus once, validated the frozen judgments against that authoritative `ChunkSet`, reconstructed candidates from the two stored depth-10 ranking reports, required exact agreement with their recorded fusion, and measured the production selector and trusted context builder at fused prefixes 5, 7, and 9.

The run covered C1–C6 and P1–P4: ten cases and thirty completed case/option measurements. It did not open Chroma, perform retrieval, load an embedding or generation model, download anything, access credentials, prepare a Gemini request, count provider tokens, generate an answer, or conduct semantic review. The network guard remained active.

This remains an exploratory development-set comparison. C6 and P2 retrieval failures informed the choice of prefixes being compared.

## Report and execution provenance

- Report: `data/evaluation/runs/mdr-evidence-selection-context-81e5401-attempt01.json`
- Report SHA-256: `d405f01154aea835ed13047d7d5e327011f2c442db6e7b6373b144662dba7b1e`
- Report format: `evidence-selection-context-comparison/v1`
- Status: `completed`
- Failure: none
- Execution HEAD: `81e54012c23016f86b6bddcb52fc01540bb0e745`
- Started: `2026-10-31T17:09:09.029830+00:00`
- Finished: `2026-10-31T17:09:18.509424+00:00`
- Python: 3.12.10
- Platform: Windows 11 (`Windows-11-10.0.26100-SP0`)
- Comparison script SHA-256: `30f346851eccc92f4dc5ce24d340f37f3ef8d77f8fa8858b63c9ebd09c78c512`

The single execution command was:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_evidence_selection_context.py --project-root "<project-root>" --output-path "<project-root>\data\evaluation\runs\mdr-evidence-selection-context-81e5401-attempt01.json"
```

## Frozen inputs and identities

The run verified these hash-pinned inputs:

| Input | SHA-256 |
|---|---|
| Original C1–C6 depth-10 report | `08d853838d3cc6fa493b84ac0645048b411c4b9da16cbf3ac5c8c428922f5bf0` |
| Prospective P1–P4 depth-10 report | `3c6adf4769535223e6326128209f2f89ba2eb92a06d24de759bf70f31f832b7e` |
| Original judgments | `b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258` |
| Prospective judgments | `b24050f555c31507a4f9521fbb8d2cf50a9235a08375f1fccbb5111dc2b6b758` |

Expected and reconstructed identities agreed:

- Sources: 18
- Chunks: 938
- Corpus: `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab`
- ChunkSet: `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b`
- Chunking configuration: `chunking-config:sha256:84eab6ff2f1bbd9cfb55f20cd63ac21d0cf29125d7e521fa7c9d7fa94e3ff634`
- Recorded dense collection: `mdr-dense-623682d019d735ec03ccc2b4`
- Recorded lexical index: `lexical-index:sha256:81c6503252c5879bc1aae552ffe41f025212509d8b118fb1d817b951b1427204`

The existing original-seed and prospective evidence-binding validators passed against the reconstructed authoritative chunks. Candidate reconstruction and fusion correspondence passed for every case before context measurement.

## Nominated-evidence and complete-combination recovery

The nominated denominator is the sum of distinct nominated chunk IDs within each case; the same chunk nominated in two cases counts twice. A case counts as combination-complete when the selected prefix contains every member of at least one accepted substantive combination.

| Group | Prefix | Nominated occurrences recovered | Complete-combination cases |
|---|---:|---:|---:|
| Original C1–C6 | 5 | 6/8 | 5/6 |
| Original C1–C6 | 7 | 8/8 | 6/6 |
| Original C1–C6 | 9 | 8/8 | 6/6 |
| Prospective P1–P4 | 5 | 7/8 | 3/4 |
| Prospective P1–P4 | 7 | 7/8 | 3/4 |
| Prospective P1–P4 | 9 | 8/8 | 4/4 |

Case-level interpretation:

- C1–C5 were complete at all three prefixes.
- C6 recovered 1/3 nominated chunks and no complete combination at prefix 5; it recovered 3/3 and a complete combination at prefixes 7 and 9.
- P1, P3, and P4 recovered 2/2 nominated chunks and a complete combination at all three prefixes.
- P2 recovered 1/2 and no complete combination at prefixes 5 and 7. At prefix 9 it recovered 2/2 and a complete combination. This is consistent with the reviewed prospective result that P2's importer chunk was fused rank 9.

No unselected chunk was treated as negative evidence; it remained unjudged.

## Context size comparison

The following totals sum the per-case contexts within each group. They are useful for comparing the three deterministic settings but are not the size of a single request. Deltas and percentages are relative to prefix 5 in the same group.

### Original C1–C6 totals

| Prefix | Raw characters | Change | Raw UTF-8 bytes | Change | Serialized characters | Change | Serialized UTF-8 bytes | Change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 56,330 | — | 56,428 | — | 155,267 | — | 155,405 | — |
| 7 | 79,593 | +23,263 (+41.3%) | 79,725 | +23,297 (+41.3%) | 221,618 | +66,351 (+42.7%) | 221,814 | +66,409 (+42.7%) |
| 9 | 99,864 | +43,534 (+77.3%) | 100,052 | +43,624 (+77.3%) | 281,644 | +126,377 (+81.4%) | 281,908 | +126,503 (+81.4%) |

### Prospective P1–P4 totals

| Prefix | Raw characters | Change | Raw UTF-8 bytes | Change | Serialized characters | Change | Serialized UTF-8 bytes | Change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 30,958 | — | 31,010 | — | 92,044 | — | 92,140 | — |
| 7 | 42,890 | +11,932 (+38.5%) | 42,954 | +11,944 (+38.5%) | 127,174 | +35,130 (+38.2%) | 127,296 | +35,156 (+38.2%) |
| 9 | 57,492 | +26,534 (+85.7%) | 57,578 | +26,568 (+85.7%) | 172,130 | +80,086 (+87.0%) | 172,296 | +80,156 (+87.0%) |

Per-case serialized-context distributions show the request-level spread:

| Group | Prefix | Characters min / median / max | UTF-8 bytes min / median / max |
|---|---:|---:|---:|
| Original | 5 | 18,915 / 21,273 / 37,196 | 18,929 / 21,276 / 37,252 |
| Original | 7 | 26,759 / 32,769 / 51,783 | 26,761 / 32,785 / 51,863 |
| Original | 9 | 34,204 / 41,016.5 / 65,429 | 34,206 / 41,036.5 / 65,541 |
| Prospective | 5 | 17,149 / 21,890.5 / 31,114 | 17,153 / 21,920.5 / 31,146 |
| Prospective | 7 | 27,550 / 30,837.5 / 37,949 | 27,606 / 30,849.5 / 37,991 |
| Prospective | 9 | 38,816 / 42,873 / 47,568 | 38,884 / 42,891 / 47,630 |

Serialization expansion is substantial because the trusted context carries citation and provenance fields in addition to raw chunk text. Character counts and UTF-8 byte counts are not provider token counts and must not be used as substitutes for them. The historical 7,067-token observation was not reused.

## Tables, annotations, and separately reviewed evidence

Table and annotation counts below are summed occurrences across case contexts, not unique corpus objects:

| Group | Prefix | Table segments | Chunks with tables | Annotations | Chunks with annotations |
|---|---:|---:|---:|---:|---:|
| Original | 5 | 2 | 2 | 0 | 0 |
| Original | 7 | 5 | 5 | 0 | 0 |
| Original | 9 | 8 | 8 | 0 | 0 |
| Prospective | 5 | 3 | 3 | 0 | 0 |
| Prospective | 7 | 3 | 3 | 1 | 1 |
| Prospective | 9 | 4 | 4 | 1 | 1 |

P3's supporting-context chunk `chunk:sha256:472ed78a939bbd328e7186255d236b20c4feb560964810040d61342f5cc7f5bf` was not selected at prefixes 5, 7, or 9. P4's answer-review qualification chunk `chunk:sha256:7341bbb7969a67f67052c21255255f3175295698c864560397078fddd155bcdf` was also not selected at any tested prefix. Both remained separately reported and were excluded from substantive nominated-evidence and complete-combination denominators.

Their absence does not invalidate the substantive recovery calculation, but it is relevant to later answer-quality review: this comparison cannot establish whether those separately identified chunks are needed for an adequate P3 or appropriately qualified P4 answer.

## Approved provisional setting and checkpoint closure

Prefix 7 is the smallest tested setting that recovers every original C1–C6 nominated occurrence and complete combination. It does not improve the prospective group over prefix 5 because P2 remains incomplete. Prefix 9 is the only tested setting that recovers all nominated substantive occurrences and at least one accepted substantive combination in all ten positive development cases.

For the next bounded experiment, the user approved 10 dense and 10 lexical
candidates with fused prefix 9 as the provisional evidence-selection setting.
Checkpoint 5 is complete. This approval is limited to that next bounded
experiment: it does not approve prefix 9 for production, establish a production
context limit, or change any application default. Its observed cost on this
development set is roughly 81% more serialized characters/bytes for the
original group and 87% more for the prospective group than prefix 5.

Provider-specific token counting and generation each require a separately
reviewed scope and explicit authorization. Answer-quality and semantic review
also remain necessary, particularly for P3 context and P4 qualification. Until
those validations are completed, this comparison does not establish semantic
sufficiency or acceptable provider request cost.

## Limitations

- The development set contains only ten positive cases and was used to choose the compared prefixes after observing C6 and P2 failures; the analysis is exploratory.
- Recovery measures frozen nominated chunks and accepted substantive combinations, not answer correctness, completeness, citation quality, or semantic sufficiency.
- P3 context and P4 qualification evidence were assessed separately and were absent from every tested selected prefix.
- Character and UTF-8 byte measurements are deterministic storage/serialization measurements, not provider token counts or pricing estimates.
- No Gemini request, provider operation, generation, answer-behavior evaluation, or semantic review was performed.
- Prefix 9 is approved only as the provisional selection setting for the next
  bounded 10/10 retrieval experiment; no production cutoff or selection size is
  approved by this report.
