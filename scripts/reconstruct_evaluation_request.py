"""Reconstruct a stored selection's request without Chroma or provider calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.application.local_config import load_local_app_config
from src.generation.evidence_context import build_evidence_context
from src.generation.gemini_request import GEMINI_PROMPT_VERSION, prepare_gemini_request
from src.ingestion.chunk_set_builder import build_chunk_set
from src.ingestion.corpus_loader import load_corpus
from src.ingestion.manifest import load_manifest
from src.retrieval.dense_index import chunk_metadata
from src.retrieval.dense_results import DenseCandidate, deserialize_chunk_metadata
from src.retrieval.evidence_selection import select_evidence
from src.retrieval.fusion import fuse_candidates


def run(diagnostic_path: Path, live_path: Path, case_number: int, output: Path):
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if type(case_number) is not int or not 1 <= case_number <= len(diagnostic["cases"]):
        raise ValueError("Invalid case number")
    case = diagnostic["cases"][case_number - 1]
    live = live_path.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^## {case_number}\s*\n(.*?)(?=^## \d+\s*$|\Z)", live)
    if match is None:
        raise ValueError("Missing live case")
    body = match.group(1)
    question = re.search(r"(?m)^Question: (.+)$", body)
    saved = re.search(r"(?m)^Application request ID: (gemini-request:sha256:[0-9a-f]{64})$", body)
    if question is None or saved is None or question.group(1) != case["question"]:
        raise ValueError("Live question or request identity missing or mismatched")
    c = load_local_app_config(ROOT / "config/local_app_config.json")
    manifest = load_manifest(c.manifest_path, project_root=c.project_root, verify_files=True)
    chunks = build_chunk_set(manifest, load_corpus(manifest, project_root=c.project_root), c.chunking_config)
    if chunks.chunk_set_id != diagnostic["chunk_set_id"]:
        raise ValueError("ChunkSet identity mismatch")
    by_id = {chunk.chunk_id: chunk for chunk in chunks.chunks}
    chosen = [by_id[row["chunk_id"]] for row in case["selected"]]
    # The synthetic ranks reproduce selection order only; they do not claim to
    # reconstruct dense scores or the original retrieval provenance.
    dense = tuple(DenseCandidate(i, chunk.chunk_id, chunk.text,
                  deserialize_chunk_metadata(chunk_metadata(chunk)), float(i))
                  for i, chunk in enumerate(chosen, 1))
    fused = fuse_candidates(chunk_set=chunks, dense_candidates=dense, lexical_candidates=(),
                            rank_constant=c.retrieval_config.rank_constant)
    selection = select_evidence(chunk_set=chunks, fused_candidates=fused, max_chunks=len(chosen))
    context = build_evidence_context(selection=selection, chunk_set=chunks)
    request = prepare_gemini_request(question=case["question"], context=context,
                                     config=c.generation_config, chunk_set=chunks)
    result = {"case": case_number, "question": case["question"], "prompt_version": GEMINI_PROMPT_VERSION,
              "chunk_set_id": chunks.chunk_set_id, "saved_request_id": saved.group(1),
              "reconstructed_request_id": request.request_id, "request_id_matches": request.request_id == saved.group(1),
              "source_diagnostic_request_id": case.get("application_request_id"),
              "provider_calls": 0, "index_opened": False,
              "selected": [{"citation_id": str(i), "chunk_id": chunk.chunk_id,
                            "source_identity": chunk.source.source_identity,
                            "locators": [[f"{p.kind}: {p.value}" for p in path] for path in chunk.legal_locators]}
                           for i, chunk in enumerate(chosen, 1)]}
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return result["request_id_matches"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--live-report", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.diagnostic, args.live_report, args.case, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
