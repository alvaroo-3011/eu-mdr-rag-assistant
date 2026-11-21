# Local MVP acceptance observations — 14 November 2026

These are limited, user-reported Streamlit observations, not an independent
regulatory review or a complete MVP acceptance assessment. P1, P5, P7 and P8
passed across separate live runs; they were not rerun for this note.

## Final implementation and local startup

- Implementation commit: `967cfd50d43c0a5ee6f4edcdd9087f13264aa439`.
- The earlier startup check preceded that commit. A fresh PowerShell session
  subsequently launched the committed implementation using the command below.
  The server bound to `127.0.0.1:8502` and its health endpoint returned `ok`;
  port 8501 was already in use.
- Streamlit's native initial-render test passed and confirmed the question
  field and submit button without making a provider call. Browser automation
  was unavailable, so this is not a visual browser check of the new instance.
- The four authorized operational configuration fields were inspected for this
  checkpoint: model `models/gemini-3.5-flash-lite`, 10 dense candidates,
  10 lexical candidates and a maximum of 5 selected chunks. All match the
  requested profile. This is a current configuration observation, not a
  retained record of every earlier live request. Provider billing state was
  not independently inspected.

```powershell
Set-Location '<project-root>'
.\.venv\Scripts\python.exe -m streamlit run main.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
```

## User-reported live outcomes

| Case | Recorded live outcome |
| --- | --- |
| P1 | User reported a pass. The response and its request metadata were not retained for independent review here. |
| P5 | Initially displayed the generic provider/validated-response error. On the later attempt, the user reported a pass with a wording caveat: the fictional test's pass/fail outcome remained undetermined, but “the supplied evidence corpus contains no information” could be read as a claim about the whole corpus. |
| P7 | An initial v4-era submission displayed `insufficient_evidence`. After adding the local device-clarification rule and restarting Streamlit, the user reported the retest passed. The successful response and request metadata were not retained here. |
| P8 | Initially displayed the generic provider/validated-response error. On the later attempt, the user reported a pass: the app explained that the IVDR performance-evaluation question was outside MDR scope and gave no IVDR requirements. |

The source available for the successful P7 restart identifies the configured
prompt as `mdr-grounding/v4`. Process/source timing supports that version being
loaded, but no per-request prompt-version log was retained. For the exact P7
question, the new local policy matches an explicitly MDR-scoped, undescribed
device-material question and returns one fixed clarification before retrieval
or a provider request; consequently the v4 prompt is not sent on that local
path. The successful browser result is user-reported, not a retained execution
trace. The rule is intentionally narrow; other wording and IVDR/mixed-scope
questions continue through the existing path.

The earlier generic errors do not identify whether counting, generation, or
response validation failed. These observations do not establish broad routing
accuracy, evidence sufficiency, citation entailment, free-tier billing status,
or production readiness.

The separate retrieval-evaluation script retains an open cleanup limitation:
its Windows ACL preflight checks the temporary parent and immediate parent,
but does not establish that higher path components cannot be replaced by
another principal. Use a temporary path with trusted ancestors until resolved.
