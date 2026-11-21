# Project contract

## Approved MVP

- Build an English-language, closed-corpus assistant for concrete EU MDR questions.
- Use the MDR and the 17 selected MDCG guidances in the approved corpus.
- Run initially as a single-user local Windows/CPU Streamlit application.
- Expand to a small tester group only after acceptance gates pass.
- Use Gemini for grounded generation and local retrieval.
- Give every substantive regulatory claim numbered citations.
- Refuse unsupported answers and, where useful, suggest relevant out-of-corpus catalogue documents without pretending they were searched.

## Collaboration protocol

- Work collaboratively and one step at a time.
- Clarify the immediate objective, propose the next step, obtain approval at material decision gates, implement, let the user test or confirm, and then continue.
- Explain unfamiliar concepts in plain language while preserving technical accuracy.
- Prefer concrete commands and short checkpoints.
- Do not silently change approved choices.
- Record unresolved decisions explicitly instead of filling them with defaults.

## Product boundaries

- Treat incoming JSON as already extracted and validated.
- Do not accept or extract raw PDFs in Streamlit.
- Do not provide live web search in the MVP.
- Do not answer IVDR-focused questions in the MDR MVP.
- Do not use Gemini's uncited background knowledge as regulatory evidence.
- Do not add technical-document uploads, multilingual support, public deployment, user accounts, persistent monitoring, or automatic conversation persistence to the MVP.
- Keep memory within the session. Save material only through an explicit user action such as `Save as test case`.
- Show a warning against confidential, personal, or client-identifying input and an informational-use disclaimer.
- Do not present outputs as a replacement for professional or legal advice.

## Change control

Treat later explicit user decisions and approved project artifacts as amendments. When a proposed change affects scope, source authority, privacy, traceability, evaluation gates, or architecture, explain the impact and request approval before implementation.
