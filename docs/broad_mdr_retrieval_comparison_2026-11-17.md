# Broad MDR retrieval: offline comparison

The [baseline diagnostic](../data/evaluation/broad_mdr_baseline_diagnostic.json) and [revised diagnostic](../data/evaluation/broad_mdr_revised_diagnostic_final.json) used the same configured corpus, ChunkSet, retrieval counts and five-chunk selection cap. The revised artifact records the commit, prompt and retrieval-policy versions, implementation hashes, non-secret configuration, rankings, selected full texts and locator probes. It used a network-blocked process and a fingerprinted disposable index copy; the original index fingerprint matched before and after. Neither run proves what exact evidence the earlier live baseline supplied, because that report did not retain its request.

| Question | Baseline reconstruction | Revised reconstruction | Complete request UTF-8 bytes, baseline → revised |
|---|---|---|---:|
| MDR scope | Article 1(1) not retrieved | Article 1(1)–(2) chunk retrieved and selected | 19,749 → 24,129 |
| Classification | Article 51 not retrieved | Article 51(1) and an Annex VIII implementation-principles chunk retrieved and selected | 24,655 → 27,597 |
| Notified body | Article 52 not retrieved | Three Article 52 chunks selected, covering III/IIb, IIa/I, and the separate IIb implantable-device exception list | 36,550 → 25,976 |
| QMS | Article 10(9) selected with large neighboring evidence | The same complete Article 10(9) chunk, including points (a)–(m), is prioritized; full text remains intact | 54,611 → 36,734 |
| Incident reporting | Article 87 retrieved but omitted; Article 88 not retrieved | Article 87(1)–(8) and Article 88 chunks retrieved and selected | 30,044 → 28,405 |

These byte sizes are not token counts. The QMS baseline received a provider count of 19,191 tokens; the revised request has not been counted by the provider. A fresh count and the unchanged 16,384-token input gate apply to every generation. A smaller byte size does not prove it will pass.

## Strategy and limits

The original dense query and ordinary lexical query still run. Recognized topic cues nominate at most three additional local BM25 searches. Each facet hit must be returned within the configured lexical candidate depth and match an article/paragraph or annex locator in the authoritative regulation ChunkSet. The merged lexical branch has composite ranks; its raw scores originate from their respective searches and are not compared across searches. Existing fusion validates the complete candidate union against the ChunkSet. A coverage selector then reserves space for the retrieved facet hits, fills remaining places in fused order, and retains original fused candidates and complete chunks for citation mapping. It does not fabricate missing hits or provide answers.

The cues cover scope, classification, notified bodies, QMS, incident reporting, PRRC and post-market surveillance. They are reusable topic patterns, not matches to the five exact questions, but they are not a general semantic planner: other phrasings or topics can fall back to ordinary retrieval, a facet can miss at the configured search depth, and a nominated provision may not be sufficient for a particular question. Explicit article/annex requests and IVDR/IVD, in-vitro or FDA requests are not expanded by this policy. Existing clarification and scope instructions remain in force.

The existing `retrieval.max_chunks` setting remains configurable and stays at five: the five diagnostics require at most three reserved chunks, leaving two places for ordinary fused evidence. Lowering the cap or lexical candidate depth can omit facets; increasing it can worsen the token budget. Five chunks do not prove complete topic coverage, and no automatic semantic verifier has been added. Context is reduced by omitting source block-offset records from the model projection; trusted local chunks, passage text, tables, annotations and legal locators remain intact. No passage is truncated, so duplication or large chunks can still cause a budget rejection.

Seven differently worded local BM25 checks found all nominated facets for scope, classification, notified bodies, QMS, incidents, PRRC and post-market surveillance. Those checks and the diagnostic do not establish answer correctness or completeness. Review each live answer, conditions, exceptions and citation support before the full regression run.
