"""Offline tests for the single local JSON configuration boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.application.local_config import (
    LocalConfigError,
    load_local_app_config,
)
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
from src.retrieval.embeddings import GTE_MODERNBERT_BASE


def _raw_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "profile_label": "Synthetic offline test profile",
        "paths": {
            "project_root": "..",
            "manifest_path": "data/corpus_manifest.json",
            "persistence_directory": "data/indexes/chroma",
        },
        "chroma": {
            "tenant": "default_tenant",
            "database": "default_database",
            "migration_hash_algorithm": "md5",
        },
        "profiles": {
            "chunking": "baseline_frozen_18_source_v1",
            "embedding": "gte_modernbert_base_pinned_cpu",
        },
        "bm25": {"k1": 1.2, "b": 0.75},
        "retrieval": {
            "dense_candidate_count": 3,
            "lexical_candidate_count": 2,
            "rank_constant": 2.0,
            "max_chunks": 2,
        },
        "generation": {
            "model": "models/gemini-synthetic",
            "max_output_tokens": 256,
            "temperature": 0.25,
            "thinking_budget": 32,
            "thinking_level": None,
        },
        "budget": {"input_token_limit": 123, "output_token_limit": 256},
        "transport": {
            "api_version": "v1beta",
            "timeout_seconds": 1.25,
            "max_response_bytes": 20000,
            "verify_tls": True,
            "trust_env": False,
            "follow_redirects": False,
            "retries": 0,
        },
    }


class LocalConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_dir = self.root / "config"
        self.config_dir.mkdir()
        data = self.root / "data"
        data.mkdir()
        (data / "corpus_manifest.json").write_text("{}", encoding="utf-8")
        index = data / "indexes" / "chroma"
        index.mkdir(parents=True)
        (index / "chroma.sqlite3").write_bytes(b"synthetic-not-sqlite")
        self.path = self.config_dir / "local_app_config.json"

    def _write(self, value: object) -> None:
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_complete_explicit_config_reuses_existing_validated_profiles(self):
        self._write(_raw_config())
        config = load_local_app_config(self.path)

        self.assertEqual(config.project_root, self.root.resolve())
        self.assertEqual(config.manifest_path, (self.root / "data/corpus_manifest.json").resolve())
        self.assertEqual(
            config.persistence_directory,
            (self.root / "data/indexes/chroma").resolve(),
        )
        self.assertIs(config.chunking_config, BASELINE_CHUNKING_CONFIG)
        self.assertIs(config.embedding_config, GTE_MODERNBERT_BASE)
        self.assertEqual((config.bm25_config.k1, config.bm25_config.b), (1.2, 0.75))
        self.assertEqual(config.retrieval_config.dense_candidate_count, 3)
        self.assertEqual(config.generation_config.model, "models/gemini-synthetic")
        self.assertEqual(config.budget.input_token_limit, 123)
        self.assertTrue(config.http_backend_config.verify_tls)
        self.assertFalse(config.http_backend_config.trust_env)
        self.assertFalse(config.http_backend_config.follow_redirects)
        self.assertEqual(config.http_backend_config.retries, 0)

    def test_missing_extra_duplicate_and_secret_fields_fail_closed(self):
        base = _raw_config()
        cases = []
        missing = deepcopy(base)
        del missing["retrieval"]
        cases.append(json.dumps(missing))
        extra = deepcopy(base)
        extra["other"] = 1
        cases.append(json.dumps(extra))
        cases.append('{"schema_version":"1.0","schema_version":"1.0"}')
        secret = deepcopy(base)
        secret["generation"]["api_key"] = "SHOULD_NEVER_APPEAR"
        cases.append(json.dumps(secret))

        for text in cases:
            with self.subTest(text=text[:40]):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(LocalConfigError) as caught:
                    load_local_app_config(self.path)
                self.assertNotIn("SHOULD_NEVER_APPEAR", str(caught.exception))

    def test_invalid_operational_values_fail_before_any_resource_boundary(self):
        changes = (
            ("retrieval", "dense_candidate_count", 0),
            ("generation", "model", "not-a-model-resource"),
            ("budget", "output_token_limit", 255),
            ("transport", "verify_tls", False),
            ("transport", "trust_env", True),
            ("transport", "follow_redirects", True),
            ("transport", "retries", 1),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                raw = _raw_config()
                raw[section][key] = value
                self._write(raw)
                with self.assertRaises(LocalConfigError):
                    load_local_app_config(self.path)

    def test_missing_manifest_index_or_database_fails_during_config_loading(self):
        self._write(_raw_config())
        targets = (
            self.root / "data/corpus_manifest.json",
            self.root / "data/indexes/chroma/chroma.sqlite3",
            self.root / "data/indexes/chroma",
        )
        for target in targets:
            with self.subTest(target=target.name):
                if target.is_dir():
                    for child in target.iterdir():
                        child.unlink()
                    target.rmdir()
                else:
                    target.unlink()
                with self.assertRaises(LocalConfigError):
                    load_local_app_config(self.path)
                if target.suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"{}" if target.suffix == ".json" else b"synthetic")
                else:
                    target.mkdir(parents=True)
                    (target / "chroma.sqlite3").write_bytes(b"synthetic")


if __name__ == "__main__":
    unittest.main()
