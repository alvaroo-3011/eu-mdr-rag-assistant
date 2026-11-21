# EU MDR Assistant: 20-question run

Run stopped after question 1 because token counting failed with a `PermissionError` in the restricted server environment. No HTTP status was recorded; this is not evidence of provider quota or rate limiting. Restarting the server with different network permissions would end the browser session. Questions 2–20 were not submitted. No response or timing is inferred for them.

- Server: `http://127.0.0.1:18502`, process command line verified as `streamlit run main.py`; diagnostics enabled.
- Commit: `d3b7eb25f98ca808a9a904f47bed3d6430294095`. Worktree included unrelated staged changes and an uncommitted UI regression test; application files were at the commit.
- Current local configuration: `models/gemini-3.5-flash-lite`, input limit 16,384 tokens, output limit 1,024 tokens, temperature 1.0, thinking level MINIMAL; dense and lexical candidates 10 each, maximum 5 chunks. Transport timeout 30 seconds, retries 0, TLS verification enabled. Free-tier account status was not independently verified.
- Elapsed time is click to first observed final result; observation was periodic, so 12.9 seconds is an upper-bound observation rather than an exact rendering timestamp. Typing was excluded. The 120-second timeout was not reached.

| # | Question | Elapsed seconds | Actual outcome | Displayed response | Citations | Error stage | Exception class | HTTP status | Input tokens / limit |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | Under MDR Article 83(1)-(2), what must a manufacturer's post-market surveillance system do, and how often must class IIa versus class IIb and III manufacturers update their PSUR under Article 86(1)? | ≤12.9 | Error | The provider request or validated response failed; no partial answer is shown. | None | countTokens | PermissionError | not recorded | not recorded / 16,384 |
| 2 | What qualifications and responsibilities does MDR Article 15 specify for a person responsible for regulatory compliance? | — | Not run | — | — | — | — | — | — |
| 3 | For an imported MDR device, what importer identification details and locations are specified in Article 13(3), and which distributor checks may use representative sampling under Article 14(2)? | — | Not run | — | — | — | — | — | — |
| 4 | What does SSCP stand for, and which MDR provision does MDCG 2026-4 cite for uploading it to EUDAMED? | — | Not run | — | — | — | — | — | — |
| 5 | According to Q4’s Table 1 in the Annex XVI transition Q&A, what transition end date and two ongoing conditions are listed? What does Q10 say about dual-purpose devices when the applicable requirements differ? | — | Not run | — | — | — | — | — | — |
| 6 | Did the unpublished sterilisation validation test for the fictional Example Device Alpha pass or fail? | — | Not run | — | — | — | — | — | — |
| 7 | What material must my device be made from to comply with the MDR? | — | Not run | — | — | — | — | — | — |
| 8 | What MDR labelling requirements apply to my device? | — | Not run | — | — | — | — | — | — |
| 9 | Under the IVDR, what performance evaluation requirements apply to my in vitro diagnostic test? | — | Not run | — | — | — | — | — | — |
| 10 | What does the MDR require for a manufacturer’s post-market surveillance system, and what performance evaluation does the IVDR require for an IVD? | — | Not run | — | — | — | — | — | — |
| 11 | Under MDR Article 85, what must a class I manufacturer include in its post-market surveillance report, and when should it update the report? | — | Not run | — | — | — | — | — | — |
| 12 | What does MDR Article 10(9) require the manufacturer’s quality management system to cover? | — | Not run | — | — | — | — | — | — |
| 13 | What are the three reporting deadlines in MDR Article 87 for serious incidents, and what circumstances correspond to each deadline? | — | Not run | — | — | — | — | — | — |
| 14 | Under MDR Article 13, what should an importer do if it has reason to believe a device already placed on the market is nonconforming? | — | Not run | — | — | — | — | — | — |
| 15 | What information does MDR Article 19 require in an EU declaration of conformity, and which annex lists its minimum contents? | — | Not run | — | — | — | — | — | — |
| 16 | According to MDCG 2024-13, how does the treatment of ethylene oxide used during sterile-device manufacture differ from that of cartridges used by a health institution to sterilise reusable devices? | — | Not run | — | — | — | — | — | — |
| 17 | What class is my device under the MDR? | — | Not run | — | — | — | — | — | — |
| 18 | What exact fee in euros will Belgium’s competent authority charge in 2032 for registering a manufacturer as an MDR actor in EUDAMED? | — | Not run | — | — | — | — | — | — |
| 19 | What steps does the MDR specify for reporting a statistically significant increase in the frequency or severity of incidents under Article 88? | — | Not run | — | — | — | — | — | — |
| 20 | What are the US FDA requirements for a 510(k) submission for a reusable surgical instrument? | — | Not run | — | — | — | — | — | — |

Outcomes: 1 error, 19 not run. Total observed submission time: ≤12.9 seconds. Median and slowest completed request: ≤12.9 seconds (one observation). Error breakdown: 1 at `countTokens`, `PermissionError`, HTTP status not recorded. No app token-budget rejection or provider quota/rate-limit response was recorded.
