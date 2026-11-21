# Provider-free candidate-depth comparison

## Provenance and scope

This document compares two completed, separately authorized provider-free
experiments against the same six-case development seed. Neither experiment was
rerun for this review.

| Run | Execution HEAD | Report | Verified SHA-256 |
| --- | --- | --- | --- |
| 5/5 | `f007f214223e0f6cdc744188c3f834d3ce50ffc6` | [5/5 attempt01](../data/evaluation/runs/mdr-retrieval-selected-v2-f007f21-attempt01.json) | `d2606de7c1af2d530c95dc485bcf6fdbcbb1de836d6b8d91365b8d135cced61c` |
| 10/10 | `bcc0cdb6591df3df2856c305809709295757c051` | [10/10 attempt01](../data/evaluation/runs/mdr-retrieval-selected-depth10-v2-bcc0cdb-attempt01.json) | `08d853838d3cc6fa493b84ac0645048b411c4b9da16cbf3ac5c8c428922f5bf0` |

Both reports use `retrieval-evidence-run/v2`, record status `completed`, and
contain cases C1–C6 in the same order with case status `ok`. Their seed is
`mdr-retrieval-seed-v1`, SHA-256
`b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258`.

## Configuration and comparability

| Setting | 5/5 run | 10/10 run |
| --- | --- | --- |
| Dense / lexical candidate counts | 5 / 5 | 10 / 10 |
| BM25 `k1` / `b` | 1.2 / 0.75 | 1.2 / 0.75 |
| RRF rank constant | 2.0 | 2.0 |
| Assessment cutoff / selection limit | 5 / 5 | 5 / 5 |
| Tenant / database / migration hash | `default_tenant` / `default_database` / `md5` | Same |
| Embedding | `Alibaba-NLP/gte-modernbert-base`, revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, CPU, 768 dimensions, 8,192 maximum sequence length, normalized | Same |
| Chunking | `deterministic-grouping-baseline-v1`, 4,000-character target, 6,000-character maximum, no overlap | Same |

Both reports record Chroma 1.5.9, NumPy 2.5.2, PyTorch 2.14.1,
Transformers 5.18.0 and huggingface-hub 1.33.0. Their expected and observed
identity fields match between runs:

- 18 sources and 938 chunks;
- corpus `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab`;
- ChunkSet `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b`;
- chunking configuration `chunking-config:sha256:84eab6ff2f1bbd9cfb55f20cd63ac21d0cf29125d7e521fa7c9d7fa94e3ff634`;
- dense collection `mdr-dense-623682d019d735ec03ccc2b4`; and
- lexical index `lexical-index:sha256:81c6503252c5879bc1aae552ffe41f025212509d8b118fb1d817b951b1427204`.

The original-index fingerprints recorded before the runs match across all six
recorded paths. The runner is the only differing recorded source hash: the
second HEAD parameterized candidate-count input, forwarding and reporting.
The runs occurred at different times and HEADs, so matching recorded fields do
not prove every runtime condition was identical. The settings are experimental,
not production defaults. Collection identity and count checks are not a full
audit of stored records or vectors.

## Approved-evidence recovery

Each value is approved supporting chunks found / total approved for the case;
branch and fused columns assess their first five results. The union column
assesses all returned fused candidates; selected reflects the selector's
actual five-chunk limit. In each cell `Y` or `N` indicates whether an entire
approved combination was recovered.

| Case | Dense top 5, 5/5 → 10/10 | Lexical top 5, 5/5 → 10/10 | Fused top 5, 5/5 → 10/10 | Full union, 5/5 → 10/10 | Selected, 5/5 → 10/10 |
| --- | --- | --- | --- | --- | --- |
| C1 | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y |
| C2 | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y |
| C3 | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y |
| C4 | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y |
| C5 | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y | 1/1 Y → 1/1 Y |
| C6 | 2/3 N → 2/3 N | 1/3 N → 1/3 N | 1/3 N → 1/3 N | 2/3 N → 3/3 Y | 1/3 N → 1/3 N |

C6's sole approved combination, C6-S1, jointly requires E5, E6 and E7:

| Member | 5/5 dense | 5/5 lexical | 5/5 fused | 5/5 selected | 10/10 dense | 10/10 lexical | 10/10 fused | 10/10 selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E5 | absent | absent | absent | absent | 6 | 8 | 7 | absent |
| E6 | 5 | absent | 8 | absent | 5 | 9 | 6 | absent |
| E7 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 |

The 10/10 union recovered the complete nominated combination that the 5/5
union did not. The 10/10 selector retained only the first five fused candidates,
so E5 and E6 were present in the union but not selected. This distinction is
candidate recovery versus loss at the fixed selection limit, not a judgment
that the selected evidence is semantically sufficient or insufficient.

## Operations and cleanup

Each report records one client open and one retrieval initialization, six
hybrid calls, six dense calls, six lexical calls and six selections, all
attempted and completed. Both record zero provider calls, no primary failure
and no cleanup failures. Client closure and disposable-copy removal passed,
and each run's final original-index fingerprint matched its pre-copy
fingerprint. The operator acknowledged stopped writers; the acknowledgement
and fingerprint checks do not independently prove that no writer was active.

## Limitations and proposed follow-up

These results measure recovery of the seed's approved question-specific
supporting chunks and jointly required combinations. Every other chunk remains
unjudged; unjudged does not mean irrelevant. Missing nominated support does not
prove the corpus lacks alternative support. The seed has six questions across
five topics and is a small development set, not a representative benchmark.
The results do not establish semantic sufficiency, answer quality, citation
entailment or completeness, regulatory correctness, or production readiness;
they do not approve a production candidate depth or selection limit.

A subsequent analysis could inspect the **recorded fused prefixes at cutoffs
5, 6 and 7 across all six cases**, using the existing reports and approved seed
without new retrieval. This would be retrospective and exploratory: the
cutoffs were motivated by the observed C6 fused ranks. Those prefixes would not
be observed selector executions at limits 6 or 7, and the analysis should not
claim token costs without the necessary evidence text and an appropriate
measurement method. Any such analysis remains a separate reviewed checkpoint.
