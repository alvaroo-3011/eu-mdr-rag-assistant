# Prospective retrieval evaluation review

## Execution and provenance

This review documents the completed, separately authorized prospective
retrieval experiment. The experiment was not rerun for this document.

- Execution HEAD: `cbcdb2527e1ac4dc461fd47c882d43a8153ddae5`.
- Report: [`data/evaluation/runs/mdr-retrieval-prospective-v1-cbcdb25-attempt01.json`](../data/evaluation/runs/mdr-retrieval-prospective-v1-cbcdb25-attempt01.json).
- Report SHA-256, verified for this review:
  `3c6adf4769535223e6326128209f2f89ba2eb92a06d24de759bf70f31f832b7e`.
- Report format: `prospective-retrieval-run/v1`; status: `completed`.
- Frozen artifact: `prospective-cases/v1` at
  [`data/evaluation/prospective_cases_v1.json`](../data/evaluation/prospective_cases_v1.json),
  SHA-256
  `b24050f555c31507a4f9521fbb8d2cf50a9235a08375f1fccbb5111dc2b6b758`.
- The report records the nine protected staged entries and its own reserved
  untracked report path in the execution-time Git status. Those staged entries
  were preserved during this documentation checkpoint.

The recorded execution began at `2026-10-31T15:44:21.167122+00:00` and ended
at `2026-10-31T15:44:48.105055+00:00`. It executed P1-P4 once each. P5-P8
remained unexecuted. The settings and results are experimental development
observations, not approved production defaults or acceptance thresholds.

## Experimental settings and identities

| Setting | Recorded value |
| --- | --- |
| Dense / lexical candidates | 10 / 10 per executed case |
| BM25 `k1` / `b` | 1.2 / 0.75 |
| RRF rank constant | 2.0 |
| Assessment cutoff / selected chunks | 5 / 5 |
| Chroma tenant / database | `default_tenant` / `default_database` |
| Migration hash algorithm | `md5` |
| Chunking profile | `deterministic-grouping-baseline-v1`; 4,000-character target, 6,000-character hard maximum, no overlap |
| Embedding model / revision | `Alibaba-NLP/gte-modernbert-base` / `e7f32e3c00f91d699e8c43b53106206bcc72bb22` |
| Embedding settings | CPU, 768 dimensions, 8,192 maximum sequence length, normalized vectors |
| Recorded package versions | Chroma 1.5.9, NumPy 2.5.2, PyTorch 2.14.1, Transformers 5.18.0, huggingface-hub 1.33.0 |

The run used the cached pinned model under offline and network-blocked
conditions. It records zero provider calls. The caller acknowledged that index
writers were stopped, and the client opened only the disposable index copy.

Every observed identity matched its expected value:

| Identity | Expected and observed value |
| --- | --- |
| Sources / chunks | 18 / 938 |
| Corpus | `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab` |
| ChunkSet | `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b` |
| Chunking configuration | `chunking-config:sha256:84eab6ff2f1bbd9cfb55f20cd63ac21d0cf29125d7e521fa7c9d7fa94e3ff634` |
| Dense collection | `mdr-dense-623682d019d735ec03ccc2b4` |
| Lexical index | `lexical-index:sha256:81c6503252c5879bc1aae552ffe41f025212509d8b118fb1d817b951b1427204` |

Collection opening validates identity metadata and record count; it is not a
full audit of every stored ID, text, metadata value, or vector.

## Nominated-evidence recovery

Each recovery cell is nominated supporting chunks found / total nominated for
the case. Dense, lexical, and fused columns assess their first five results.
The union column assesses the complete fused candidate union, and selected
assesses the selector's output. A complete combination requires every member
of at least one accepted substantive combination.

| Case | Dense top 5 | Lexical top 5 | Fused top 5 | Full fused union | Selected | Complete combination: union / selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P1 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | Yes / Yes |
| P2 | 1/2 | 1/2 | 1/2 | 2/2 | 1/2 | Yes / No |
| P3 | 1/2 | 1/2 | 2/2 | 2/2 | 2/2 | Yes / Yes |
| P4 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | Yes / Yes |

All four executed case records have status `ok`. Incomplete nominated-evidence
recovery is an evaluation result, not an operational failure.

P1 recovered both nominated chunks and its complete combination in each branch
top five, the fused top five, the full union, and selected evidence.

P2's accepted combination requires distributor-sampling chunk `3592de24` and
importer-identification chunk `c0e3ce70`. The sampling chunk ranked third in
the dense branch and second in both the lexical and fused rankings. The
importer chunk ranked sixth in the dense branch, was absent from the lexical
branch, ranked ninth in the fused union, and was absent from selection.
Consequently, the full union recovered P2's combination, but no branch top
five, the fused top five, or selected evidence did.

P3's accepted combination requires Q10 chunk `a3519082` and Table 1 chunk
`fbd8c151`. Dense top five recovered only `a3519082` at rank 2; lexical top
five recovered only `a3519082` at rank 1. The table chunk ranked seventh in the dense branch and eighth in the lexical branch. After fusion, it ranked fourth, while the Q10 chunk ranked first. The fused top five and selected evidence therefore recovered the complete substantive combination despite both individual branch top fives being incomplete.

P4 recovered its two substantive chunks, `0bf1e179` and `21cf16a6`, in both
branch top fives, the fused top five, the full union, and selected evidence.

## Separately scored P3 and P4 evidence

The following evidence is recorded separately from each case's nominated
substantive combination and does not contribute to the recovery denominator.

| Evidence | Dense | Lexical | Fused top 5 | Full fused union | Selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3 supporting context `472ed78a` | Absent | Rank 10, outside top 5 | Rank 16, outside top 5 | Present at rank 16 | Absent |
| P4 answer-review qualification `7341bbb7` | Rank 9, outside top 5 | Absent | Rank 12, outside top 5 | Present at rank 12 | Absent |

P3's Q4 introduction is supporting context, not a mandatory member of the
approved substantive combination. P4's non-binding-guidance qualification is
reserved for separate answer review and is likewise outside the substantive
retrieval denominator. Their presence in the full union does not establish
that a generated answer would use them correctly.

## Operations and cleanup

| Operation | Attempted | Completed |
| --- | ---: | ---: |
| Client open | 1 | 1 |
| Retrieval initialization | 1 | 1 |
| Hybrid retrieval | 4 | 4 |
| Dense branch | 4 | 4 |
| Lexical branch | 4 | 4 |
| Evidence selection | 4 | 4 |

The report records zero provider calls, no primary failure, and no cleanup
failures. Client closure and disposable-copy removal passed. The original
index fingerprint after cleanup exactly matched its pre-copy fingerprint. A
post-execution observation also found the recorded disposable-copy path absent.
No retry or fallback was performed.

## Unexecuted cases and evaluation boundary

P5-P8 have status `not_executed`. Their retrieval, refusal, clarification, and
scope behavior was not tested by this run. For every case P1-P8, answer
behavior is `not_evaluated` and semantic review is `not_evaluated`. The run
therefore does not assess generated answers, refusal quality, clarification,
scope routing, citation entailment or coverage, or regulatory correctness.

The denominator contains only the frozen artifact's nominated supporting
chunks. Accepted combinations are non-exhaustive, and all other chunks remain
unjudged rather than irrelevant. Failure to recover a nominated chunk does not
prove that no alternative sufficient evidence exists; recovery does not prove
that retrieved text is semantically sufficient for an answer.

This is a small, assistant-assisted development set of four executed positive
cases, not an independent or representative benchmark. It is unsuitable for
estimating general retrieval quality or selecting a production threshold.
Candidate depth, fusion, cutoff, and selection remain experimental parameters.
No production cutoff is recommended, and semantic sufficiency, answer quality,
or production readiness is not inferred.

Within this bounded nominated-evidence assessment, the complete candidate
unions recovered all four accepted substantive combinations, while selection
retained three: P1, P3, and P4. P2's accepted combination was present only in
the full union because the importer chunk ranked ninth after fusion.
