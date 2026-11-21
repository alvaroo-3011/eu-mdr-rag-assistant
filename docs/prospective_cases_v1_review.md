# Prospective cases v1: frozen development judgments and evidence review

The user approved the eight assistant-assisted development cases and judgments, including the P3/P4 decisions below. This is not independent regulatory validation or an independent benchmark: prior corpus knowledge and results influenced its design. The [frozen artifact](../data/evaluation/prospective_cases_v1.json) is authoritative for this development evaluation; the [draft](../data/evaluation/prospective_cases_v1.draft.json) is its working review copy. Accepted combinations are non-exhaustive and all other chunks are unjudged. Approval does not authorize an evaluation run.

Frozen artifact SHA-256: `b24050f555c31507a4f9521fbb8d2cf50a9235a08375f1fccbb5111dc2b6b758`. The hash covers the exact file bytes: UTF-8 without BOM, LF line endings, and one final LF. The hash is recorded here outside the JSON.

## Scope and method

Analysis HEAD: `f69912b211bbadda8b8ba96cd2f1c189f8f490aa`. The approved manifest and source checksums were verified; deterministic ingestion and the existing baseline chunking profile produced 18 sources and 938 chunks, corpus identity `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab` and chunk-set identity `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b`. Passage and structured-table context were inspected in normalized prepared sources. There was no embedding, retrieval, Chroma or provider operation. Chunk IDs are deterministic IDs from this chunk set, not retrieval results.

The cited consolidated MDR is `eu-regulation-2017-745@sha256:6c9193c32149080ef77edae4afdff2e885599d9851feec5cf05c1a26c78258fc`; legal locators, not invented page numbers, identify its passages. The Annex XVI Q&A is `annex-xvi-transition-provisions-qa@sha256:5bda9501826a7c9a968a8283e230af657e71b39bddeaaca30f5feb877804d581`; the EtO guidance is `mdcg-2024-13@sha256:c3a830a89489ef6e75fa375200a42d5aec9771866a2db160616a9ae0a9dbfdd5`. Guidance is non-binding. Exact excerpts and locations are in the draft; ellipses indicate omitted intervening text.

## Answerable cases

* **P1 — surveillance and PSUR frequency.** Article 83(1)-(2) and Article 86(1) support complementary system-function and update-frequency elements. The two cited chunks do not each contain the complete requested answer. Review equivalent guidance and ensure the question is distinct enough from the development seed.
* **P2 — importer identification and distributor sampling.** Article 13(3) gives the identification details, locations and no-obscuring qualification; Article 14(2) identifies the checks covered by representative sampling. The dual-applicable MDCG 2021-27 chunk `chunk:sha256:da229a705fe07e95a08d6787110535761003590fba1c14eb3a9b0a287ceee4a7` does not itself give the complete Article 13(3) details or no-obscuring qualification. Even paired with Article 13(3), it identifies the permitted sampling points and describes information generally but does not explicitly state the distributor check that the declaration has been drawn up. The pair is therefore not verified complete for this exact rubric. Other alternatives remain open.
* **P3 — Annex XVI Table 1 and Q10.** The question explicitly asks for Q4 Table 1 and Q10. The approved substantive complete combination is Table 1 chunk `fbd8c151` and Q10 chunk `a3519082`; Q4 introduction chunk `472ed78a` is supporting context, not a mandatory member. Table 1's header spans 22 June 2023 through 31 December 2029; its first condition cell contains both ongoing conditions, and the next row's three dates align with the corresponding condition cells below. Relationships were inspected in the structured table. Additional dated conditions apply; the two ongoing conditions do not constitute a complete product-eligibility assessment.
* **P4 — EtO manufacturing versus health-institution cartridges.** The approved substantive complete combination is the section 2.1 continuation `21cf16a6` and section 2.2 `0bf1e179`. The section 2.1 opening `8fc606e6` provides context but is not required for the substantive rubric. The non-binding guidance qualification is scored separately in answer review; disclaimer chunk `7341bbb7` supports that qualification outside the substantive retrieval denominator. This separation does not allow unsupported generated claims. Dual MDR/IVDR guidance remains eligible for MDR questions.

Each mandatory substantive element is separately mapped to exact excerpts, source/version, locator and chunk ID in the frozen artifact. Discontinuous quotations are separate; summaries are labelled as summaries. Accepted combinations are non-exhaustive; the identified P2 guidance alternative is not established as complete for the approved rubric. Other chunks remain unjudged. Structural support and these development judgments do not establish semantic correctness.


| Elements | Verified supporting chunks (SHA-256 prefixes; full IDs and quotations in draft) |
| --- | --- |
| P1.1–P1.2: system design, QMS and lifetime data/action | `80400e02` |
| P1.3–P1.4: IIb/III and IIa PSUR frequency | `9872b7f4` |
| P2.1–P2.2: importer details, location and no-obscuring | `c0e3ce70` |
| P2.3–P2.7: sampling permission, checks and point (c) qualification | `3592de24` |
| P3.1: Table 1 end date | `fbd8c151` |
| P3.2–P3.3: ongoing conditions | `fbd8c151` |
| P3.4–P3.5: cumulative and stricter rules | `a3519082` |
| P3.6: additional dated conditions and eligibility limit | `fbd8c151` |
| P4.1–P4.2: manufacturing use, scope and assessment | `21cf16a6` |
| P4.3: health-institution cartridges | `0bf1e179` |
| P4.4: non-binding guidance, separate answer review | `7341bbb7` |

## Unsupported-answer review

For P5 and P6, case-insensitive literal searches were run over all 938 normalized chunk texts from all 18 manifest sources, including extracted table text; counts below are chunk counts, not independent document counts. This is a corpus-wide text search, not a search of external sources or a proof of absence. It can miss synonyms, split terms, extraction omissions and implicit facts. Context was inspected around generic sterilisation/validation references in MDR Annex II and actor-registration context in MDCG 2021-1. The user approved the development refusal judgments; the searches do not establish exhaustive absence.

* **P5:** `example device alpha`, `example-device-alpha`, `example device`, and `unpublished` each returned 0 chunks. Broader searches found `alpha` in 5, `pass` in 4, `fail` in 15, `sterilis` in 37, `sterilisation validation` in 1, `sterilization validation` in 0, and `validation report` in 1. Generic requirements or unrelated pass/fail words do not establish the fictional unpublished test outcome. Accept an `insufficient_evidence` response with a nonblank limitation, empty answer blocks and null clarification question; it may ask for the report in the limitation. Invented pass/fail results, measurements, dates, laboratories or reports fail the rubric.
* **P6:** `belgium`, `belgian`, `belgique`, `belgien`, `2032`, and `registration fee` each returned 0 chunks. `actor registration` appeared in 7 chunks, `eudamed` in 102, and the broad substring `fee` in 22; these broad hits include unrelated words and are not a fee inventory. Accept `insufficient_evidence` with empty answer blocks, a nonblank limitation and null clarification question. Invented amounts, a zero-fee assertion, a schedule or a claim about whether a future fee will exist fail the rubric. The question specifies manufacturer **actor** registration; a missing amount does not establish future policy.

## Clarification and scope

P7 intentionally leaves the device, intended use, component and relevant contact conditions unspecified. The current response validator requires `needs_clarification` to have empty `answer_blocks` and one nonblank `clarification_question`. The optional `limitation` may be null or nonblank and can contain brief general framing; no device-specific material recommendation or unsupported regulatory assertion belongs there. Ask one useful question about the device, purpose, material or contact facts; do not require an exhaustive questionnaire. The user approved this development oracle; its reliability remains untested, and an `answered` response would require changing the oracle. There is no approved deterministic ambiguity policy.

P8 asks only about IVDR performance evaluation. Its approved development `out_of_scope` oracle concerns the request and MDR MVP boundary, not classification of an actual product. MDR Article 1(6)(a) is optional contextual regulatory evidence, not a required positive retrieval combination. This does not exclude sources marked applicable to both regimes from MDR questions, and it does not settle mixed-scope product routing.

## Remaining limits

The user-approved judgments are assistant-assisted development judgments, not independent regulatory validation. Alternative complete combinations have not been exhaustively reviewed, and unjudged chunks must not be treated as irrelevant. The P5/P6 searches do not prove absence; the P7 ambiguity oracle has not been tested; and P8 does not determine an actual product classification or mixed-scope policy.

The existing retrieval-seed assessment supports positive chunk/combination recovery, but does not itself evaluate refusal, clarification, scope or the separately scored P4 answer qualification; those require an appropriate reviewed evaluation method. No production threshold, routing policy, semantic-sufficiency or answer-quality claim follows from this artifact. No evaluation has been executed or authorized in this checkpoint.
