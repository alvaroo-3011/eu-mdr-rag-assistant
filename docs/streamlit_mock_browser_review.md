# Streamlit mock browser review

## Review scope

The browser review occurred on 30 October 2026 using a guarded synthetic
Streamlit preview. The preview rendered the production `main.py` interface but
patched the stable configuration, credential, and execution boundaries. Its
answers were validated synthetic `GeminiTransportRun` fixtures projected
through the real presentation boundary. It did not use the real configuration,
credentials, corpus, Chroma index, embedding model, retrieval, token counting,
or Gemini.

Browser automation was blocked because the available automation capability
could not establish the browser URL with sufficient confidence. The user then
reviewed the preview manually. The observations below are therefore
user-reported manual-review evidence, not independently reproduced browser
automation results.

## Repository facts independently inspected

- Repository HEAD at this documentation checkpoint:
  `b10faa1e9d37a7a87c61823e980c5fe63d92a3b7` (`Pin Streamlit and validate
  native interface behavior`).
- No applicable `AGENTS.md` was present.
- The preview wrapper remained outside the repository at
  `<temp-dir>\mdr_streamlit_mock_preview_9b3227c21f8a4936a6d18c04046fcc76.py`.
- Its reviewed SHA-256 was
  `f2cea4d53190a8f08c2fe23d51402ae5b1e84c7170c3bcda0e2bb4d02c853d16`.
- No tests or application backends were run during this documentation
  checkpoint.

## User-reported browser observations

Screenshots shared in ChatGPT showed:

- readable wrapping of the intentionally long synthetic answer;
- expanded authoritative citation metadata and hierarchy information; and
- model-supplied HTML-like text displayed literally as text rather than being
  interpreted as HTML.

The user also confirmed that pressing Tab produced a visible keyboard-focus
indicator.

A screenshot showed a mock execution count of `14`. This establishes only the
displayed value in that screenshot. The exact before-and-after counter
transitions for an ordinary rerun and a subsequent explicit submission were
not explicitly confirmed.

The user reported that a rerun left the screen unchanged. This is recorded as
a manual observation, but it does not independently establish the execution
counter transition.

## Checks not established by this review

No other preview scenario is marked as passed without direct evidence. In
particular, this review does not establish:

- empty-submission behavior;
- insufficient-evidence, clarification, or out-of-scope rendering;
- sanitized-failure behavior or recovery after failure;
- an exact zero-increment ordinary rerun followed by an exact one-increment
  explicit submission;
- complete keyboard navigation order or accessibility behavior beyond the
  reported visible Tab focus;
- layout behavior across a defined set of desktop and narrow viewport sizes;
  or
- live application startup and resource cleanup with the real local corpus and
  services.

## Cleanup

Cleanup was completed on 31 October 2026. The previously recorded PID `3848`
was stale. Port `58079` was still owned by Python PID `14232`. A privileged,
read-only command-line inspection verified that PID `14232` was running the
exact temporary wrapper above with `127.0.0.1:58079`, headless mode, usage
telemetry disabled, and file watching disabled. Only that verified process was
stopped. PID `14232` exited and port `58079` no longer accepted connections.

## Limitations

This was a synthetic presentation review. It does not establish regulatory or
legal correctness, semantic support for citations, evidence sufficiency,
retrieval quality, model quality, production configuration suitability, or
live end-to-end behavior. It does not replace the existing deterministic tests,
future browser accessibility/visual QA, retrieval evaluation, or an explicitly
authorized live validation.
