# EU MDR RAG assistant

A closed-corpus question-answering assistant for the EU Medical Device
Regulation (Regulation (EU) 2017/745) and 17 MDCG / European Commission
guidance documents. It combines hybrid retrieval (dense Chroma + BM25) with
Google Gemini generation and returns answers with validated citations to the
source chunks.

**Status and disclaimer**

- Personal, non-commercial learning prototype.
- Informational use only. It is **not legal or regulatory advice**. Answers may
  be incomplete or wrong, and every citation must be verified by a human
  against the original source.
- Not affiliated with, endorsed by, or developed for any employer or
  organisation.
- Built only from public sources. No confidential or proprietary data is
  included.
- Intended to run **locally on 127.0.0.1 only**. Do not expose the app to the
  internet or a shared network.

**Quick setup** (Python 3.12, from the repository root)

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python.exe
cp config/local_app_config.example.json config/local_app_config.json  # then review it
export GEMINI_API_KEY="<your-key>"   # environment variable only; PowerShell: $env:GEMINI_API_KEY = "<your-key>"
.venv/bin/python -m streamlit run main.py --server.address 127.0.0.1 --server.port 8501
```

Never put the API key in a file or commit it. A local Chroma index under
`data/indexes/` is also required (it is not committed); see the technical
notes below.

## Sources and attribution

The corpus in `data/rag_ready/` is derived from public EU documents,
reformatted and chunked for retrieval:

- Regulation (EU) 2017/745 on medical devices, consolidated text from
  EUR-Lex: <https://eur-lex.europa.eu/eli/reg/2017/745/oj>
- MDCG endorsed guidance documents and related European Commission
  guidance: <https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en>

Source: European Union / European Commission, (c) European Union. Reuse is
permitted under Commission Decision 2011/833/EU on the reuse of Commission
documents (<https://eur-lex.europa.eu/eli/dec/2011/833/oj>), with
acknowledgement of the source. The versions here are modified (converted and
chunked) and are not official. Only the legislation published in the Official
Journal of the European Union is deemed authentic.

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The corpus
documents in `data/rag_ready/` keep their original EU reuse terms and are
**not** covered by the code licence.

# Technical notes

## EU MDR local RAG interface

This repository contains an English-language, closed-corpus EU MDR assistant.
The local interface submits one explicit question through the existing hybrid
retrieval, evidence selection, Gemini request-budget, generation, response
validation, and trusted-citation pipeline.

## Streamlit dependency

`streamlit==1.66.1` is explicitly pinned in `requirements.txt` and installed in
the current Python 3.12 `.venv`. A new environment must install that exact pin
and the rest of the repository's pinned requirements before launch. No other
web framework is supported.

## Configuration

The sole operational configuration mechanism is
`config/local_app_config.json`. Copy
`config/local_app_config.example.json` to that ignored path and review every
value. The example is explicitly a **C3 diagnostic profile that is unapproved
for production**. In particular, the Gemini model's current availability,
capabilities, and free-tier eligibility must be checked before a future live
run. Retrieval, selection, sufficiency, routing, and production settings remain
evaluation decisions rather than approved defaults.

The JSON file requires explicit project/manifest/index paths, Chroma namespace
and migration hash, frozen chunking and embedding profiles, BM25 and retrieval
values, generation settings, application token budgets, and transport limits.
Secret-bearing fields are rejected.

Set the API key only at the future execution boundary, for example in the
current PowerShell process:

```powershell
$env:GEMINI_API_KEY = "<key>"
```

Do not put the key in JSON or commit it. This project remains free-tier-only
with **EUR 0 authorized paid spending**: do not activate billing and do not use
a paid fallback or another model automatically. Previous C3 count/generation
authorizations are consumed. When a submission reaches generation, it uses a
fresh complete-request count and the application budget gate. The narrow local
device-clarification and empty-evidence paths can return without a provider
request.

## Launch

From a fresh PowerShell session in the checkout, create the Python 3.12
environment and install the pinned dependencies if they are not already
present. Confirm `python --version` reports Python 3.12; on this checkout the
Windows `py -3.12` launcher does not find the installed interpreter:

```powershell
# Run from the root of your local checkout
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `config/local_app_config.json` from the example only if it does not
already exist:

```powershell
if (-not (Test-Path -LiteralPath 'config/local_app_config.json')) {
    Copy-Item -LiteralPath 'config/local_app_config.example.json' -Destination 'config/local_app_config.json'
}
```

Review the paths and generation/budget settings. For the
P1-profile acceptance run set `retrieval.dense_candidate_count` and
`retrieval.lexical_candidate_count` to `10`, and `retrieval.max_chunks` to
`5`. Keep the key out of the JSON file. Set `GEMINI_API_KEY` in this PowerShell
session, or inherit it from an existing trusted environment; the process needs
it when a question is submitted.

Start the server on the local loopback interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
```

Open <http://127.0.0.1:8501>. Keep the PowerShell session open while using the
app; press Ctrl+C there to stop it. If port 8501 is already occupied, use an
unused loopback port instead (for example, replace `8501` with `8502` in the
command and URL).

Rendering and ordinary Streamlit reruns do not call the provider. One nonblank
form submission constructs submission-scoped resources, invokes the application
pipeline once, and closes the Gemini and Chroma resources on success or
failure. The most recent validated display projection is retained only in the
Streamlit session; no conversation or answer file is written automatically.
The local device-clarification rule can return before retrieval or a Gemini
operation, although submission-scoped resources are initialized first. Other
nonempty-evidence questions follow the existing count, budget, and generation
path. Restart the server after source changes because the launch command
disables Streamlit's file watcher.

Chroma is opened against explicitly existing storage with migration validation,
telemetry disabled, and no collection creation or rebuild. Its ordinary startup
is not genuinely read-only and may create SQLite journals, WAL/shared-memory
files, or other writes in the configured persistence directory.

## Native interface validation

Run the focused Streamlit framework test with:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/integration/test_streamlit_interface.py
```

This uses Streamlit's native `AppTest` facility to execute `main.py`, interact
with its form and session state, and inspect rendered framework elements. The
application execution boundary is replaced with deterministic synthetic
responses, while the real presentation projection remains in use. It neither
starts a live Streamlit server nor performs corpus, Chroma, model, credential,
retrieval, token-count, generation, or network operations.

Passing AppTest validates framework-level Python behavior; it is not browser
visual QA and does not establish layout, styling, accessibility, or behavior in
a real browser. It is also not a live end-to-end validation of local resources
or Gemini.

## Safety and validation limits

The interface accepts no uploads and performs no PDF extraction or browsing.
It warns against confidential, personal, patient, client-identifying, or other
sensitive input and presents outputs as informational rather than legal advice.

Provider output is displayed only after the existing structural response and
citation validators accept it. Citations are rendered from trusted chunk/source
metadata, not model-authored bibliographic text. Structural validity and known
citation identity do not prove semantic support, evidence sufficiency, legal
correctness, retrieval quality, or generation quality. Full evaluation,
deterministic routing, sufficiency policy, production operational values, and
deployment readiness remain unresolved.

No live startup, retrieval, embedding, token count, generation, credential
inspection, or network call was performed in the UI implementation checkpoint.

The separate retrieval-evaluation script's temporary-directory cleanup still
has an open security-review limitation: its Windows ACL preflight checks the
temporary parent and its immediate parent, but does not establish that higher
path components cannot be replaced by another principal. Use only a temporary
path whose ancestors are trusted until that gap is resolved. The local MVP
acceptance history, including earlier intermittent provider errors, is recorded
in [the acceptance note](docs/local_mvp_acceptance_2026-11-14.md).
