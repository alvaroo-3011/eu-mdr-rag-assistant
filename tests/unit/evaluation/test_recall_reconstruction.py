"""Offline evidence and identity checks for the recorded recall attempt."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _local_json(relative):
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"Local diagnostic artifact is not available: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _has(chunk, *parts):
    return any(all(part in path for part in parts) for path in chunk["locators"])


def test_original_request_matches_and_contained_uncited_87_and_89():
    rebuilt = _local_json("data/evaluation/recall_request_reconstruction.json")
    original = _local_json("data/evaluation/broad_followup_offline_diagnostic.json")["cases"][2]
    live_path = ROOT / "docs/broad-followup-four-live.md"
    if not live_path.is_file():
        pytest.skip("Local live report is not available")
    report = live_path.read_text(encoding="utf-8")
    case = re.search(r"(?ms)^## 3\s*\n(.*?)(?=^## 4\s*$|\Z)", report).group(1)
    saved = re.search(r"(?m)^Application request ID: (\S+)$", case).group(1)
    assert saved == rebuilt["saved_request_id"] == rebuilt["reconstructed_request_id"]
    assert saved == original["application_request_id"] and rebuilt["request_id_matches"]
    assert _has(original["selected"][2], "article: 87", "paragraph: 8")
    assert _has(original["selected"][4], "article: 89", "paragraph: 8")
    assert "field safety corrective action" in original["selected"][2]["text"].casefold()
    assert "field safety notice" in original["selected"][4]["text"].casefold()
    assert "Citation [3]:" not in case and "Citation [5]:" not in case
    assert rebuilt["provider_calls"] == 0 and rebuilt["index_opened"] is False


def test_revised_recall_selection_preserves_original_passages_and_definition():
    original = _local_json("data/evaluation/broad_followup_offline_diagnostic.json")["cases"][2]
    revised = _local_json("data/evaluation/recall_revised_offline_diagnostic.json")
    selected = revised["cases"][0]["selected"]
    assert revised["provider_calls"] == 0 and revised["original_index_fingerprint_unchanged"]
    assert all(facet["match"] for facet in revised["cases"][0]["facet_searches"])
    for article, paragraph in (("10", "12"), ("87", "8"), ("89", "8")):
        assert any(_has(chunk, f"article: {article}", f"paragraph: {paragraph}") for chunk in selected)
    definition = next(chunk for chunk in selected if _has(chunk, "article: 2", "point: 68"))
    assert _has(definition, "article: 2", "point: 69")
    assert "prevent or reduce the risk of a serious incident" in definition["text"]
    for article in ("87", "89"):
        old = next(chunk for chunk in original["selected"] if _has(chunk, f"article: {article}"))
        new = next(chunk for chunk in selected if _has(chunk, f"article: {article}"))
        assert new["chunk_id"] == old["chunk_id"] and new["text"] == old["text"]
    assert revised["cases"][0]["application_request_id"] != original["application_request_id"]
