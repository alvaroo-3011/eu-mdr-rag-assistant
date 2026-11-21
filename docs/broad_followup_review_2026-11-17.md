# Broad-answer follow-up review

## Incident submission and evidence

The single live report records an `answered` response, 7,838 / 16,384 input tokens and application request ID `gemini-request:sha256:11eb0e406671a2ce86bbc5c16d36042384ecdae7d6156e83fbe93720a0e48544`. The [offline reconstruction](../data/evaluation/incident_request_reconstruction.json) produced the **same** content-bound request ID under `mdr-grounding/v5`. It used the verified ChunkSet and a network-blocked, fingerprinted disposable index copy; the original index fingerprint was unchanged. This establishes a match to the locally prepared request identity, not what happened internally at the provider.

The matching request selected Article 87 as citation 1 and Article 88 as citation 4. The live answer covered serious-incident deadlines and field safety corrective-action reporting using citation 1, but did not discuss trend reporting or cite citation 4. The v5 instruction required coverage of central aspects but did not explicitly remind the model to consider distinct vigilance categories. That is a plausible contributing factor; the model's internal reason for omitting Article 88 is not observable. Citation validity does not establish completeness.

## Changes and offline evidence

The revised prompt (`mdr-grounding/v6`) asks the model to consider serious incidents, field safety corrective actions and trend reporting separately when relevant, with their supported conditions. It asks for ongoing post-market surveillance activities and the use of their outputs, and for supported recall-related communications and reporting as well as corrective action. These are coverage instructions, not fixed answers or replacement evidence. If a central aspect lacks support, the existing insufficient-evidence status remains available.

The reusable topic retriever (`broad-mdr-facets/v2`) recognizes post-market/surveillance language and broad after-market paraphrases, and separately recall/withdrawal language. It nominates at most three additional local BM25 searches and prioritizes validated source chunks within the existing configured five-chunk cap. It does not split or rewrite those chunks. The [four-question offline diagnostic](../data/evaluation/broad_followup_offline_diagnostic.json) selected:

| Topic | Selected regulation evidence | Prepared request bytes |
|---|---|---:|
| Problem reporting | Articles 87 and 88 | 29,548 |
| After-market activities | Articles 83, 87 and 88 | 30,118 |
| Recall | Articles 10(12), 87 and 89(8) | 31,547 |
| MDR/IVDR comparison | No topic facets; existing ordinary retrieval | 21,060 |

Sizes are UTF-8 bytes of the complete prepared request, not token counts. The unchanged complete-request token-count gate still runs before each generation. The offline diagnostic made zero provider calls and confirmed the index fingerprint was unchanged. Full chunk text, source identities and locators are available in the artifact. These selections are not proof that all possible facts, qualifications or scope issues are covered.

The scope instruction now states explicitly that an inseparable comparison requiring both MDR and IVDR requirements is `out_of_scope`, while a request with separately answerable MDR and IVDR parts should receive the established `needs_clarification` behavior. Existing IVDR retrieval exclusions and citation validation are unchanged. This is a prompt-level instruction rather than a semantic verifier, so live responses must still be checked.

## Outstanding provider 503 failures

The earlier 20-question regression had eight `generateContent` HTTP 503 errors. Their provider statuses/reasons were not recorded, and the succeeding single attempt does not establish the cause of the intermittent failures. The bounded diagnostics added in the preceding review remain available; no automatic retries were introduced. See [the 503 review](gemini_503_review_2026-11-17.md).

## Separate candidateCount compatibility item

Google's Gemini 3 guidance recommends omitting `candidate_count`, while this application currently sends `candidateCount: 1`. That compatibility item remains unresolved and was not changed as part of this content work. It is not established as the cause of the observed 503s.

## Live review required

The four selected passages and prompt instructions do not guarantee what a model will answer. A live review should check the reporting categories and qualifiers, substantive ongoing surveillance, supported recall steps and communications, scope behavior for the MDR/IVDR comparison, citations, and fresh token counts. The [four-question list](../data/evaluation/broad_followup_questions_4.json) supports that review; no provider call was made for this implementation.
