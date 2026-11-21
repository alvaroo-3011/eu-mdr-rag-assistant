# Retrieval startup smoke review

## Review scope

This document records the completed bounded offline retrieval-startup smoke
check. It deliberately separates:

- repository and script facts independently inspected during this documentation
  checkpoint;
- the command and terminal output reported by the user; and
- guarantees that follow from the reviewed script's control flow if that
  reviewed script produced the reported output.

The smoke check was not rerun while preparing this document. No test, corpus or
model load, Chroma open, retrieval, credential inspection, or network operation
was performed during documentation.

## Independently inspected facts

- Repository HEAD was
  `7162b0f9ec7463d059861ecdaf21ab496c0f5f1d` (`7162b0f Enforce
  local-only embedding model loading`).
- The nine pre-existing staged entries were present and were not changed by
  this documentation checkpoint.
- The reviewed temporary script was
  `<temp-dir>\mdr_startup_smoke_7162b0f.py`.
- Its independently recomputed SHA-256 was
  `9213aa67fbd188d6558ae7b16eafdc31e7fe3b9e04a1365a409ba6332cdae62a`.
- `requirements.txt` pins Chroma `1.5.9`.
- The frozen retrieval seed records 18 sources, 938 chunks, corpus identity
  `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab`,
  and ChunkSet identity
  `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b`.
- The existing retrieval run artifact records dense collection
  `mdr-dense-623682d019d735ec03ccc2b4`.

## Supplied command and user-reported output

The user supplied the following command for execution. The actual command line
as executed was not pasted or independently observed, so this review does not
represent the supplied line as confirmed shell history.

```powershell
if ((Get-FileHash -LiteralPath "<temp-dir>\mdr_startup_smoke_7162b0f.py" -Algorithm SHA256).Hash -ne "9213aa67fbd188d6558ae7b16eafdc31e7fe3b9e04a1365a409ba6332cdae62a") { throw "Script hash mismatch" }; & .\.venv\Scripts\python.exe -B "<temp-dir>\mdr_startup_smoke_7162b0f.py" --project-root "<project-root>" --persistence-directory "<project-root>\data\indexes\chroma" --acknowledge-all-writers-stopped
```

The user reported the following terminal output verbatim:

```text
offline startup smoke check: PASS
sources: 18
chunks: 938
corpus_identity: corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab
chunk_set_identity: chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b
collection: mdr-dense-623682d019d735ec03ccc2b4
tenant/database: default_tenant/default_database
chromadb_version: 1.5.9
migration_hash_algorithm: md5
embedding_backend_loaded: false
```

The numeric process exit code was not separately supplied and is not inferred
here.

## Recorded startup result

The user-reported output matches the repository's expected frozen startup
checkpoint for:

| Item | Expected or reported value |
| --- | --- |
| Sources | `18` |
| Chunks | `938` |
| Corpus identity | `corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab` |
| ChunkSet identity | `chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b` |
| Dense collection | `mdr-dense-623682d019d735ec03ccc2b4` |
| Chroma version | `1.5.9` |
| Tenant/database | `default_tenant/default_database` |
| Observed migration hash algorithm | `md5` |
| Embedding backend loaded | `false` |

The migration hash, tenant/database, Chroma version, and unloaded-backend state
in this result are user-reported execution observations. The Chroma version,
frozen identities, counts, and collection name also agree with independently
inspected repository artifacts.

## Guarantees from the reviewed script's control flow

For the reviewed script with the SHA-256 recorded above:

- Execution requires the explicit `--acknowledge-all-writers-stopped` operator
  acknowledgement. The flag records the precondition; it does not itself prove
  that no writer exists.
- The original persistence directory is fingerprinted before copying. The
  script creates a unique temporary directory, copies the complete persistence
  directory, compares the copy with the original fingerprint, and passes only
  the disposable copy to `open_existing_chroma_client`.
- Tenant, database, and migration metadata are inspected on the disposable
  copy. The temporary SQLite inspection connection is explicitly closed before
  Chroma opens that copy, including when inspection fails.
- The caller-owned Chroma client is closed in a `finally` block.
- The `TemporaryDirectory` context must exit successfully, including deletion
  of the disposable copy, before execution reaches the final original-index
  fingerprint comparison.
- The outer `finally` compares the final original-index fingerprint with the
  initial fingerprint.
- All success output, including `offline startup smoke check: PASS`, occurs only
  after client closure, temporary-copy cleanup, and successful final
  original-index fingerprint comparison.
- The script checks that the transformer's lazy backend remains `None` after
  startup.
- It invokes no retrieval query, embedding inference, provider token-count
  operation, or generation. Startup does perform the intended dense collection
  record-count validation and ordinary in-memory length checks; these are not
  generation token counting.
- Ordinary Chroma or SQLite startup writes are confined to the disposable copy.

Accordingly, if the reviewed script produced the user-reported PASS output, its
control flow establishes that startup used only the disposable complete index
copy, the client-close and cleanup paths completed, the final original
fingerprint matched, and the embedding backend remained unloaded.

## Limits of this smoke check

This checkpoint does not establish:

- that the pinned transformer model is present and loadable from the local
  cache;
- full stored-record, document, metadata, or vector integrity beyond the
  existing collection identity and record-count validation;
- retrieval or ranking quality;
- generation, grounding, citation, or answer quality; or
- an independently observed process exit code or exact executed shell command.

It also does not independently prove the absence of a writer during copying;
that remains the acknowledged operational precondition.
