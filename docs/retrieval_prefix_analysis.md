# Retrospective fused-prefix analysis

## Scope and inputs

This is a **retrospective, exploratory** analysis of recorded fused rankings.
Cutoffs 5, 6 and 7 were motivated by the previously observed C6 ranks; they
were not independently prespecified evaluation thresholds. No corpus, model or
Chroma data was loaded, and no retrieval, evidence selection or provider
operation was performed for this analysis.

Analysis HEAD: `906b2999dff62c96666675270c3404b15ea6a618`.

| Input | Path | Verified SHA-256 |
| --- | --- | --- |
| 5/5 report | [`data/evaluation/runs/mdr-retrieval-selected-v2-f007f21-attempt01.json`](../data/evaluation/runs/mdr-retrieval-selected-v2-f007f21-attempt01.json) | `d2606de7c1af2d530c95dc485bcf6fdbcbb1de836d6b8d91365b8d135cced61c` |
| 10/10 report | [`data/evaluation/runs/mdr-retrieval-selected-depth10-v2-bcc0cdb-attempt01.json`](../data/evaluation/runs/mdr-retrieval-selected-depth10-v2-bcc0cdb-attempt01.json) | `08d853838d3cc6fa493b84ac0645048b411c4b9da16cbf3ac5c8c428922f5bf0` |
| Approved seed | [`data/evaluation/retrieval_seed_v1.json`](../data/evaluation/retrieval_seed_v1.json) | `b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258` |

Both report hashes match the [candidate-depth comparison](retrieval_candidate_depth_comparison.md),
and both reports reference the verified seed hash. The reports have different
execution HEADs and runner hashes; their recorded configuration, identity,
package and original-index fingerprint comparisons are described in that
document. Those checks do not prove all runtime conditions were identical.

## Calculation method

For each case and report, take the first `min(cutoff, available fused count)`
recorded fused chunk IDs in their stored rank order. Map the case's supporting
judgments to chunk IDs using the approved seed. Count the approved IDs in the
prefix. A complete-combination flag is true when all jointly required members
of at least one approved combination occur in the prefix. This follows the
existing `assess_ranking` semantics: separate combinations are alternatives,
and every member within a combination is jointly required.

The calculation checked contiguous fused ranks and unique IDs. Its cutoff-five
counts and complete-combination flags matched the fused top-five assessment
stored in each report. No additional relevance judgments were made.

## Results

Each cell is `actual prefix length; approved found/total; complete combination`.
`Y` means at least one approved combination is complete; `N` means none is.
The denominator is the case's nominated supporting chunks, not all relevant
evidence in the corpus.

| Run | Case | Available fused candidates | Cutoff 5 | Cutoff 6 | Cutoff 7 |
| --- | --- | ---: | --- | --- | --- |
| 5/5 | C1 | 7 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 5/5 | C2 | 8 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 5/5 | C3 | 6 | 5; 1/1; Y | 6; 1/1; Y | 6; 1/1; Y |
| 5/5 | C4 | 8 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 5/5 | C5 | 6 | 5; 1/1; Y | 6; 1/1; Y | 6; 1/1; Y |
| 5/5 | C6 | 8 | 5; 1/3; N | 6; 1/3; N | 7; 1/3; N |
| 10/10 | C1 | 15 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 10/10 | C2 | 16 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 10/10 | C3 | 14 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 10/10 | C4 | 14 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 10/10 | C5 | 14 | 5; 1/1; Y | 6; 1/1; Y | 7; 1/1; Y |
| 10/10 | C6 | 13 | 5; 1/3; N | 6; 2/3; N | 7; 3/3; Y |

The 5/5 report has only six fused candidates for C3 and C5, so their
cutoff-seven prefixes contain six items rather than seven.

## C6 member inclusion

C6-S1 is the sole approved C6 combination and jointly requires E5, E6 and E7.

| Run | Cutoff | E5 | E6 | E7 | Complete C6-S1? |
| --- | ---: | --- | --- | --- | --- |
| 5/5 | 5 | No | No | Yes, rank 1 | No |
| 5/5 | 6 | No | No | Yes, rank 1 | No |
| 5/5 | 7 | No | No | Yes, rank 1 | No |
| 10/10 | 5 | No | No | Yes, rank 1 | No |
| 10/10 | 6 | No | Yes, rank 6 | Yes, rank 1 | No |
| 10/10 | 7 | Yes, rank 7 | Yes, rank 6 | Yes, rank 1 | Yes |

In the 5/5 full union, E6 is at fused rank 8 and E5 is absent; neither can
appear in its first seven positions. In the 10/10 union, E6 and E5 occur at
ranks 6 and 7 respectively.

## Interpretation limits

These are calculations over stored rankings, **not observed selector
executions at limits 6 or 7**. Both original runs actually used a selection
limit of five. All chunks without approved question-specific supporting
judgments remain unjudged, not irrelevant. The six questions cover five topics
and form a small development seed, not a representative benchmark.

This analysis makes no claim of semantic sufficiency, answer quality,
regulatory correctness, or production readiness. It recommends no production
candidate count, cutoff, selection limit, routing rule or sufficiency policy.
It makes no token-cost estimate: the necessary full evidence text and an
appropriate token measurement were not part of this analysis.
