# Retrieval seed baseline comparison

## Purpose and provenance

This note compares two completed retrieval experiments against the approved development seed. Both evaluated commit `815ab47db2f748cfb47707a3a686bdd0e19df04c` and seed `mdr-retrieval-seed-v1` (`sha256:b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258`). The frozen corpus has 18 sources and 938 chunks, with corpus identity `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab` and ChunkSet identity `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b`.

- [Baseline report](../data/evaluation/runs/mdr-retrieval-baseline-815ab47.json) — SHA-256 `414cd14b5e449625d426c6d1563ea2a569f850d70e9447fa028e029ed8c56373`
- [Depth-10 report](../data/evaluation/runs/mdr-retrieval-depth10-815ab47.json) — SHA-256 `b8c6fc9cc7cbfb00517c63a8054a2289fdaa9a65d2fb18ee3022d255d44f01b2`

| Run | Dense candidates | Lexical candidates | BM25 `k1` | BM25 `b` | RRF rank constant | Reporting cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 5 | 5 | 1.2 | 0.75 | 2.0 | 5 |
| Depth 10 | 10 | 10 | 1.2 | 0.75 | 2.0 | 5 |

Only the two branch candidate depths changed, from 5 to 10. The other retrieval parameters and the reporting cutoff stayed fixed. Both reports also record the same chunking configuration, dense collection, lexical index, and normalized `Alibaba-NLP/gte-modernbert-base` embedding configuration and revision.

## Results

Each cell below is `approved chunks found / approved chunks total; any approved support?; complete approved combination?`, calculated over that branch's first five results. “Any” means at least one nominated supporting chunk was found; “complete” means every jointly required member of an approved combination was found.

| Case | Baseline dense | Baseline lexical | Baseline fused | Depth-10 dense | Depth-10 lexical | Depth-10 fused |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes |
| C2 | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes |
| C3 | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes |
| C4 | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes |
| C5 | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes | 1/1; yes; yes |
| C6 | 2/3; yes; no | 1/3; yes; no | 1/3; yes; no | 2/3; yes; no | 1/3; yes; no | 1/3; yes; no |

Execution status is separate from evidence coverage: both reports completed, each made six hybrid calls, all six case records have status `ok`, and neither report records an execution error.

C6's sole approved combination, C6-S1, jointly requires E5 + E6 + E7. The following are ranks in the complete stored branch and fused rankings; `—` means the member was absent from that run's candidate list or fused union.

| Member | Baseline dense | Baseline lexical | Baseline fused | Depth-10 dense | Depth-10 lexical | Depth-10 fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E5 | — | — | — | 6 | 8 | 7 |
| E6 | 5 | — | 8 | 5 | 9 | 6 |
| E7 | 2 | 1 | 1 | 2 | 1 | 1 |

The baseline full fused union contains 2/3 approved members (E6 and E7), while its fused top five contains 1/3 (E7). The depth-10 full fused union contains all 3/3 members and therefore the complete approved combination, but its fused top five still contains only 1/3 (E7); E6 and E5 are fused ranks 6 and 7.

## Interpretation and limitations

Deeper branch retrieval captured the full approved C6 combination in the fused candidate union, but it did not put that combination in the fused top five. This evidence-coverage result does not judge retrieved chunks outside the approved set: unjudged chunks remain unjudged, and missing approved evidence does not prove that all retrieved evidence is irrelevant or that no alternative support exists.

The seed has six questions across five topics. C1 and C2 share evidence and are paired query formulations of the vigilance-reporting topic. This is a small development seed, not a representative benchmark. These experiments do not approve production parameters, a sufficiency threshold, or a new output cutoff. Successful execution likewise does not establish answer quality or evidence sufficiency.

## Next checkpoint

Before Gemini integration, design the evidence-selection step and the handling of incomplete evidence combinations. That design—including selection rules, sufficiency behavior, and all related parameters—remains unresolved.
