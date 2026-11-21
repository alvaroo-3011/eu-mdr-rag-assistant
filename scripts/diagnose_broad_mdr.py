"""Offline baseline diagnostic using a fingerprinted disposable Chroma copy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_retrieval_selected import (
    _block_network, _remove_disposable_directory, _require_plain_windows_path,
    _require_private_windows_temp_parent, fingerprint,
)
from src.application.chroma_startup import open_existing_chroma_client
from src.application.local_config import load_local_app_config
from src.application.retrieval_startup import initialize_retrieval
from src.generation.evidence_context import build_evidence_context
from src.generation.gemini_request import GEMINI_PROMPT_VERSION, prepare_gemini_request
from src.retrieval.evidence_selection import select_evidence
from src.retrieval.evidence_selection import select_coverage_evidence
from src.retrieval.broad_coverage import BROAD_RETRIEVAL_VERSION, BroadLexicalRetriever
from src.retrieval.fusion import fuse_candidates

QUESTIONS = ROOT / "data/evaluation/broad_mdr_questions_5.json"
# Diagnostic probes, not routing rules or an exhaustive relevance judgment.
PROBES = (("1", "2"), ("51",), ("52",), ("10",), ("87", "88"))


def describe(chunk):
    return {
        "chunk_id": chunk.chunk_id, "source_identity": chunk.source.source_identity,
        "locators": [[f"{part.kind}: {part.value}" for part in path] for path in chunk.legal_locators],
        "hierarchies": [[entry.title for entry in path] for path in chunk.hierarchies],
        "text": chunk.text, "characters": len(chunk.text),
        "utf8_bytes": len(chunk.text.encode("utf-8")),
    }


def run(*, output, temp_parent, writers_stopped, questions_path=QUESTIONS, probes=PROBES):
    if not writers_stopped:
        raise ValueError("Index writers must be stopped before taking a copy.")
    c = load_local_app_config(ROOT / "config/local_app_config.json")
    root = c.persistence_directory.resolve(strict=True)
    temp_parent = temp_parent.resolve(strict=True)
    _require_plain_windows_path(temp_parent)
    _require_private_windows_temp_parent(temp_parent)
    if temp_parent == root or temp_parent.is_relative_to(root) or output.resolve().is_relative_to(root):
        raise ValueError("Temporary and output paths must be outside the index.")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if type(questions) is not list or not questions or len(questions) != len(probes):
        raise ValueError("Unexpected diagnostic question count.")
    with output.open("x", encoding="utf-8") as out, _block_network():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        before = fingerprint(root)
        scratch = Path(tempfile.mkdtemp(prefix="mdr-selected-eval-", dir=temp_parent))
        client = None
        try:
            copy = scratch / "chroma"
            shutil.copytree(root, copy, symlinks=True)
            if fingerprint(copy) != before or fingerprint(root) != before:
                raise ValueError("Original index changed or copy differs.")
            client = open_existing_chroma_client(
                persistence_directory=copy, tenant=c.chroma_tenant, database=c.chroma_database,
                migration_hash_algorithm=c.migration_hash_algorithm,
            )
            chunks, dense, lexical = initialize_retrieval(
                project_root=c.project_root, manifest_path=c.manifest_path,
                chunking_config=c.chunking_config, embedding_config=c.embedding_config,
                bm25_config=c.bm25_config, dense_client=client,
            )
            by_id = {chunk.chunk_id: chunk for chunk in chunks.chunks}
            report = {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "prompt_version": GEMINI_PROMPT_VERSION,
                "broad_retrieval_version": BROAD_RETRIEVAL_VERSION,
                "implementation_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
                    "src/retrieval/broad_coverage.py", "src/retrieval/evidence_selection.py",
                    "src/application/question_answering.py", "src/generation/evidence_context.py",
                    "src/generation/gemini_request.py", "scripts/diagnose_broad_mdr.py")},
                "configuration": {"retrieval": asdict(c.retrieval_config), "generation": asdict(c.generation_config),
                                  "budget": asdict(c.budget), "bm25": asdict(c.bm25_config),
                                  "embedding": asdict(c.embedding_config), "chunking": asdict(c.chunking_config)},
                "corpus_identity": chunks.corpus_identity, "chunk_set_id": chunks.chunk_set_id,
                "chunk_count": len(chunks.chunks), "lexical_index_id": lexical.index_id,
                "provider_calls": 0, "baseline_request_note": "Baseline did not retain exact requests or selected evidence; current retrieval is a reconstruction, not proof of the original selection.",
                "cases": [],
            }
            for question, articles in zip(questions, probes):
                d = dense.retrieve(question, candidate_count=c.retrieval_config.dense_candidate_count)
                broad = BroadLexicalRetriever(lexical, chunks)
                l = broad.retrieve(question, candidate_count=c.retrieval_config.lexical_candidate_count)
                f = fuse_candidates(chunk_set=chunks, dense_candidates=d, lexical_candidates=l,
                                    rank_constant=c.retrieval_config.rank_constant)
                selected = (select_coverage_evidence(chunk_set=chunks, fused_candidates=f,
                            max_chunks=c.retrieval_config.max_chunks, priority_ids=broad.priority_ids)
                            if broad.priority_ids else
                            select_evidence(chunk_set=chunks, fused_candidates=f, max_chunks=c.retrieval_config.max_chunks))
                context = build_evidence_context(selection=selected, chunk_set=chunks)
                request = prepare_gemini_request(question=question, context=context,
                                                 config=c.generation_config, chunk_set=chunks)
                def ranking(rows):
                    return [{"rank": r.rank, **describe(by_id[r.chunk_id])} for r in rows]
                chosen = [item.chunk for item in selected.selected]
                overlaps = []
                for i, a in enumerate(chosen):
                    for b in chosen[i+1:]:
                        match = SequenceMatcher(None, a.text, b.text, autojunk=False).find_longest_match()
                        if match.size >= 120:
                            overlaps.append({"first": a.chunk_id, "second": b.chunk_id, "shared_characters": match.size,
                                             "shared_text": a.text[match.a:match.a+match.size]})
                fused_ids = {item.chunk_id for item in f}
                selected_ids = {item.chunk_id for item in chosen}
                probes = []
                for chunk in chunks.chunks:
                    if chunk.source.document_id != "eu-regulation-2017-745":
                        continue
                    if any(part.kind == "article" and part.value in articles for path in chunk.legal_locators for part in path):
                        probes.append({**describe(chunk), "position": "selected" if chunk.chunk_id in selected_ids else
                                       "retrieved_not_selected" if chunk.chunk_id in fused_ids else "not_retrieved"})
                report["cases"].append({"question": question, "dense": ranking(d), "lexical": ranking(l),
                                        "fused": ranking(f), "selected": [describe(x) for x in chosen],
                                        "facet_searches": broad.facet_results,
                                        "priority_ids": list(broad.priority_ids), "selection_strategy": selected.strategy,
                                        "application_request_id": request.request_id,
                                        "request_characters": len(request.request_json),
                                        "request_utf8_bytes": len(request.request_json.encode("utf-8")),
                                        "overlaps": overlaps, "probed_articles": list(articles), "corpus_probes": probes})
            if client is not None:
                client.close()
                client = None
            if fingerprint(root) != before:
                raise ValueError("Original index changed during diagnostic.")
            report["original_index_fingerprint_unchanged"] = True
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write("\n")
        finally:
            if client is not None:
                client.close()
            _remove_disposable_directory(scratch, temp_parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-parent", type=Path, required=True)
    parser.add_argument("--writers-stopped", action="store_true")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--probes", help="Semicolon-separated groups of article numbers, comma-separated within each question")
    args = parser.parse_args()
    probes = tuple(tuple(part.split(",")) for part in args.probes.split(";")) if args.probes else PROBES
    if any(not item.isdecimal() for group in probes for item in group):
        raise ValueError("Probe article numbers must be numeric.")
    run(output=args.output, temp_parent=args.temp_parent, writers_stopped=args.writers_stopped,
        questions_path=args.questions, probes=probes)


if __name__ == "__main__":
    main()
