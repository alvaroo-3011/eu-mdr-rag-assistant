# Provider-free selected-evidence evaluation review

## Execution and provenance

This review documents the completed, separately authorized six-case
provider-free experiment. The experiment was not rerun for this document.

- Execution HEAD: `f007f214223e0f6cdc744188c3f834d3ce50ffc6`.
- Report: [`data/evaluation/runs/mdr-retrieval-selected-v2-f007f21-attempt01.json`](../data/evaluation/runs/mdr-retrieval-selected-v2-f007f21-attempt01.json).
- Report SHA-256, verified for this review:
  `d2606de7c1af2d530c95dc485bcf6fdbcbb1de836d6b8d91365b8d135cced61c`.
- Report format: `retrieval-evidence-run/v2`; status: `completed`.
- Seed: `mdr-retrieval-seed-v1`, SHA-256
  `b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258`.
- The report records the nine protected staged entries and its own reserved
  untracked report path in the execution-time Git status. No changes were made
  to those protected entries during this documentation checkpoint.

The recorded execution began at `2026-10-31T11:23:48.779376+00:00` and ended
at `2026-10-31T11:24:53.834988+00:00`. Its settings are experimental values,
not approved production defaults.

## Configuration and identity checks

| Setting | Recorded value |
| --- | --- |
| Dense / lexical candidates | 5 / 5 per case |
| BM25 `k1` / `b` | 1.2 / 0.75 |
| RRF rank constant | 2.0 |
| Assessment cutoff / selected chunks | 5 / 5 |
| Chroma tenant / database | `default_tenant` / `default_database` |
| Migration hash algorithm | `md5` |
| Chunking profile | `deterministic-grouping-baseline-v1`; 4,000-character target, 6,000-character hard maximum, no overlap |
| Embedding model / revision | `Alibaba-NLP/gte-modernbert-base` / `e7f32e3c00f91d699e8c43b53106206bcc72bb22` |
| Embedding settings | CPU, 768 dimensions, 8,192 maximum sequence length, normalized vectors |
| Recorded package versions | Chroma 1.5.9, NumPy 2.5.2, PyTorch 2.14.1, Transformers 5.18.0, huggingface-hub 1.33.0 |

The local-only embedding loader was used. The report records that the
writer-stopped acknowledgement was supplied and that the client was opened on
a disposable copy. The acknowledgement and matching fingerprints do not
independently prove that no writer was active during the copy.

The report's observed identity fields matched the expected fields:

| Identity | Expected and observed value |
| --- | --- |
| Sources / chunks | 18 / 938 |
| Corpus | `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab` |
| ChunkSet | `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b` |
| Chunking configuration | `chunking-config:sha256:84eab6ff2f1bbd9cfb55f20cd63ac21d0cf29125d7e521fa7c9d7fa94e3ff634` |
| Dense collection | `mdr-dense-623682d019d735ec03ccc2b4` |
| Lexical index | `lexical-index:sha256:81c6503252c5879bc1aae552ffe41f025212509d8b118fb1d817b951b1427204` |

Collection opening checks identity metadata and record count; it is not a full
audit of stored IDs, text, metadata, or vectors. Dual-applicable guidance is
eligible where MDR is permitted.

## Operations and cleanup

| Operation | Attempted | Completed |
| --- | ---: | ---: |
| Client open | 1 | 1 |
| Retrieval initialization | 1 | 1 |
| Hybrid retrieval | 6 | 6 |
| Dense branch | 6 | 6 |
| Lexical branch | 6 | 6 |
| Evidence selection | 6 | 6 |

The report records zero provider calls, no primary failure, and no cleanup
failures. Client closure and disposable-copy removal both passed. The original
index fingerprint after cleanup matched the pre-copy fingerprint. The
disposable copy was reported removed; after execution its recorded copy path
was also observed absent.

## Approved-evidence recovery

Each cell is approved supporting chunks found / total approved for that case;
the branch and fused columns assess their first five results. The union column
assesses the complete fused union; selected assesses the selector's actual
output. A complete combination requires all jointly required members of at
least one approved combination.

| Case | Dense top 5 | Lexical top 5 | Fused top 5 | Full fused union | Selected | Complete combination: union / selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | Yes / Yes |
| C2 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | Yes / Yes |
| C3 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | Yes / Yes |
| C4 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | Yes / Yes |
| C5 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | Yes / Yes |
| C6 | 2/3 | 1/3 | 1/3 | 2/3 | 1/3 | No / No |

All six case records have execution status `ok`. Incomplete recovery is an
evaluation result, not an execution failure.

C6's only approved combination, C6-S1, jointly requires E5, E6 and E7:

| Member | Dense rank | Lexical rank | Fused rank | Selected rank |
| --- | ---: | ---: | ---: | ---: |
| E5 | absent | absent | absent | absent |
| E6 | 5 | absent | 8 | absent |
| E7 | 2 | 1 | 1 | 1 |

Thus the union contains E6 and E7 but not E5; selection contains E7 only.
Neither the full union nor selected evidence recovers the approved combination.

## Comparison with the historical depth-10 run

The historical [depth-10 report](../data/evaluation/runs/mdr-retrieval-depth10-815ab47.json)
has SHA-256 `b8c6fc9cc7cbfb00517c63a8054a2289fdaa9a65d2fb18ee3022d255d44f01b2`.
It records execution at HEAD `815ab47db2f748cfb47707a3a686bdd0e19df04c`
on 16 October 2026, using `retrieval-evidence-run/v1` and the historical
retrieval evaluation runner. Its recorded Git status included the two
initializer working-tree edits. The existing [baseline comparison](retrieval_seed_baseline_comparison.md)
documents that run alongside a historical 5/5 run.

The historical run used 10 dense and 10 lexical candidates, BM25 `k1=1.2`
and `b=0.75`, RRF constant `2.0`, and reporting cutoff 5. It recorded the same
seed hash, corpus and ChunkSet identities, counts, chunking configuration ID,
dense collection, lexical index ID, embedding model/revision/settings and
listed package versions as attempt01. Both reports record six successful
hybrid calls and six `ok` cases.

Their **recorded top-five approved-evidence counts** are directly comparable
as observations for the same seed and cutoff: C1–C5 each recovered 1/1 in
dense, lexical and fused top five in both reports; C6 recovered 2/3 dense,
1/3 lexical and 1/3 fused in both. In the historical depth-10 full fused union,
C6's E5, E6 and E7 appeared at ranks 7, 6 and 1, respectively, so the complete
combination was present. In attempt01's 5/5 union, E5 was absent, E6 ranked 8
and E7 ranked 1, so the combination was incomplete. In both reports only E7
was in the fused top five.

These are observations across two distinct runs, not an isolated causal
estimate of changing candidate depth. The execution commits, runner and startup
paths differ; recorded hashes of `hybrid_query.py`, `embeddings.py` and
`retrieval_seed.py` also differ. Other matching identities and package versions
do not prove the runs were otherwise equivalent. The historical v1 report did
not run or record the later evidence selector, so its fused top five must not be
reported as an observed historical selected result.

## Interpretation limits

The denominator is the case's nominated approved supporting chunks, not all
potentially relevant corpus evidence. All other chunks remain unjudged; their
absence of a supporting judgment does not mean irrelevance. A missing nominated
chunk does not prove that no alternative support exists.

The seed contains six questions across five topics, with C1 and C2 sharing an
evidence topic. It is a small development seed, not a representative benchmark.
This experiment does not establish semantic sufficiency, answer quality,
citation entailment or completeness, regulatory correctness, or production
readiness. It does not approve a routing policy, sufficiency threshold,
retrieval parameter, or selection limit for production.
