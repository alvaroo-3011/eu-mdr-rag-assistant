# Retrieval evaluation seed v1 — review record

## Status and scope

This document records the review basis for `data/evaluation/retrieval_seed_v1.json`. The user approved the six revised questions and their nominated evidence in chat on 2026-10-16 after reviewing the evidence packets and targeted corrections.

The seed contains six questions across five evidence topics and uses seven distinct supporting chunks. C1 and C2 are paired formulations of the same vigilance-reporting-deadlines topic: C1 contains the explicit Article 87 reference, while C2 is a natural paraphrase. They intentionally share one evidence record.

Approval establishes question-specific support against the frozen normalized JSON and HTML corpus:

- 18 sources and 938 reconstructed chunks;
- corpus identity `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab`;
- ChunkSet identity `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b`.

It does not independently verify extraction fidelity against the original guidance PDFs. The consolidated MDR source is frozen HTML and is traced with legal locators rather than PDF pages.

## Judgment vocabulary

- **Supporting:** the examined chunk supports at least one required answer element for the particular question. This is a question-specific judgment, not a universal property of the chunk.
- **Context only:** the examined chunk is useful for understanding the question or evidence but does not support a required answer element. No context-only judgments are included in seed v1.
- **Reviewed non-supporting:** the examined chunk was reviewed for the question and found not to support a required answer element. No reviewed-non-supporting judgments are included in seed v1.
- **Unjudged:** no judgment has been approved for that chunk against that question. Every corpus chunk outside the nominated supporting judgments remains unjudged.

A supporting chunk may be partial. A **complete support combination** is the approved set of evidence members that together covers the required answer elements. Members within one combination are jointly required; separate combinations would represent alternatives. C1–C5 each have one single-member combination. C6 has one three-member combination. No alternative combination is approved in v1.

## Cases and approved evidence

### C1 — explicit Article 87 reference

Question: “Under MDR Article 87(3)–(5), what are the maximum reporting deadlines for a serious incident generally, a serious public health threat, and death or an unanticipated serious deterioration in health?”

Approved evidence: `chunk:sha256:1efd49b589aee298bcb333ddbaf830a1c49ba2d845921df67112cfaa7c3762ac`.

The evidence preserves the immediate-reporting and causal-relationship qualifications together with the 15-, 2-, and 10-day maxima.

### C2 — natural paraphrase of C1

Question: “How quickly must a medical-device manufacturer report a serious incident, and how do the deadlines change for a serious public health threat or for death or an unexpected serious deterioration in health?”

Approved evidence: the same shared Article 87 chunk as C1. C2 tests a natural formulation without the legal citation; it does not create a second evidence topic.

### C3 — SSCP and its MDR provision

Question: “According to MDCG 2026-4, what does SSCP stand for, and which MDR provision does the paper associate with its upload during certificate-information registration in EUDAMED?”

Approved evidence: `chunk:sha256:578c8b018fbfeadefdbedf0c8e72b1c85eb0deb4f0b6d765fdf397a6ae2b8f4a`.

The required elements are limited to the SSCP expansion, Article 32(1) MDR, and the paper's qualification that the policy is evolving. The source also discusses IVDR, but it is retained because it permits MDR; SSP and the IVDR pairing are not answer requirements.

### C4 — Article 16 guidance interpretation

Question: “According to MDCG 2021-26, what does ‘necessary in order to market’ mean for Article 16(2) relabelling or repackaging, and what examples of such conditions does the guidance give?”

Approved evidence: `chunk:sha256:209728e547695cd97ef6f0ce87d4ead7203e9b8bcca8193b6eda897c87517736`.

The evidence requires a case-by-case assessment and identifies the examples as non-exhaustive. Its normalized hierarchy contains an inherited heading about legacy devices that does not describe the Question 3 passage. The question-and-answer text itself is coherent, and the limitation is retained in the evidence record.

### C5 — VGL structured table

Question: “According to the November 2024 EUDAMED gradual-roll-out Q&A, when do the VGL-module obligations become applicable, and which MDR provisions and subjects does its Q12 table list as concerned?”

Approved evidence: `chunk:sha256:f283e74476d9465ce3169bd67603e773b89b240b7f4bba28c833065813c3e49a`.

The original `ChunkTableSegment` is preserved as two exact headers and four exact rows. The applicability statement is a source-specific conditional trigger: six months after publication in the OJEU of the notice confirming VGL-module functionality. The seed does not infer whether that notice has since been published or assign a current calendar date. The Article 90 MDR statement is a separate note after the table, not a fifth table row.

### C6 — jointly required multi-chunk support

Question: “According to MDCG 2021-3, how are custom-made, adaptable, and patient-matched devices distinguished, and what MDR market pathway applies to adaptable and patient-matched devices?”

Approved combination:

1. `chunk:sha256:7580367708439c9ae365425fd80e78e30b8308780369a3065fc02daa38f27ee4` — CMD definition and mass-produced exclusions.
2. `chunk:sha256:8c3330ef3d135e5bcf7e273f1a7d585981b26cf0f383ba29db9ccb9659a9c862` — adaptable and patient-matched characteristics and qualifications.
3. `chunk:sha256:25433d42e8af9d7eaaf321539c530cf3f983b61a8a0ae206f9b135cae71359b9` — standard MDR pathway conclusion.

All three chunks are needed for this nominated evidence combination. This does not mean it is the only valid combination in the corpus. C6 retains the Article 16(1) qualification that a person adapting, adjusting, assembling, or shaping an adaptable device is not regarded as its manufacturer only while the stated compliance and intended-purpose conditions remain satisfied. Its patient-matched production element describes a process capable of being validated and reproduced; it does not claim that process validation has been completed. The third chunk ends with an irrelevant, incomplete heading for the next question; that trailing text is not approved evidence.

## Review method and limits

Evidence discovery started from frozen corpus content, not from retrieval results or rankings. Relevant passages and surrounding context were inspected in the reconstructed original `ChunkSet`, mapped back to normalized blocks, legal locators or physical PDF pages, and checked for MDR permission. The C5 cells were checked against the original structured table record rather than inferred from flattened chunk text.

Consequently:

- technical retrieval success is not a retrieval-quality judgment;
- a relevant chunk may supply only part of a complete answer;
- different chunks or combinations may provide valid alternative support;
- unexamined chunks are not treated as irrelevant;
- failed retrieval would not establish that the corpus lacks support;
- optional alternative candidates discussed during review remain unjudged and are not serialized as judgments.

Ambiguous questions, unsupported questions, and IVDR-focused scope cases are deferred. This development seed is not a comprehensive benchmark. It establishes no retrieval-quality result, metric, threshold, routing rule, tuning choice, or production retrieval parameter.

## Change control

The evidence is bound to the recorded corpus and ChunkSet identities. Any change to a question, required answer element, judgment, source version, chunk identity, excerpt, locator, table cell, or support combination requires renewed evidence review. Material changes should be saved under a new explicit format or artifact version rather than silently changing v1.
