"""Verify the recorded offline reconstruction, without using a provider."""
import json
import re
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_single_live_request_identity_and_selected_evidence():
    diagnostic_path = ROOT / "data/evaluation/incident_request_reconstruction.json"
    live_path = ROOT / "docs/device-incident-single-live.md"
    if not diagnostic_path.is_file() or not live_path.is_file():
        pytest.skip("Local reconstruction and live report are not available")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    live = live_path.read_text(encoding="utf-8")
    match = re.search(r"Application request ID: (gemini-request:sha256:[0-9a-f]{64})", live)
    assert match is not None
    case = diagnostic["cases"][4]
    assert case["application_request_id"] == match.group(1)
    assert diagnostic["prompt_version"] == "mdr-grounding/v5"
    assert diagnostic["provider_calls"] == 0
    assert diagnostic["original_index_fingerprint_unchanged"] is True
    assert any("article: 87" in path for path in case["selected"][0]["locators"])
    assert any("article: 88" in path for path in case["selected"][3]["locators"])
