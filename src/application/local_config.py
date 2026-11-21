"""Single-file explicit configuration for the local MDR application.

The JSON configuration contains operational values only. Credentials are
rejected and must be acquired separately at the submitted-question boundary.
Loading validates the complete configuration and existing paths before any
Chroma, retrieval, embedding, or provider resource is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Literal

from src.generation.gemini_request import GeminiGenerationConfig
from src.generation.gemini_transport import GeminiTransportConfig
from src.generation.request_budget import RequestBudget, validate_generation_budget
from src.ingestion.chunk_models import ChunkingConfig
from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
from src.retrieval.embeddings import EmbeddingModelConfig, GTE_MODERNBERT_BASE
from src.retrieval.evidence_selection import EvidenceSelectionConfig
from src.retrieval.hybrid_query import HybridRetrievalConfig
from src.retrieval.lexical_query import BM25Config


class LocalConfigError(ValueError):
    """The local application configuration is missing, unsafe, or invalid."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalConfig:
    """Explicit application-level retrieval and mechanical selection settings."""

    dense_candidate_count: int
    lexical_candidate_count: int
    rank_constant: float
    max_chunks: int

    def __post_init__(self) -> None:
        try:
            hybrid = HybridRetrievalConfig(
                dense_candidate_count=self.dense_candidate_count,
                lexical_candidate_count=self.lexical_candidate_count,
                rank_constant=self.rank_constant,
            )
            selection = EvidenceSelectionConfig(max_chunks=self.max_chunks)
        except ValueError as exc:
            raise LocalConfigError("Invalid retrieval configuration.") from exc
        object.__setattr__(self, "rank_constant", hybrid.rank_constant)
        object.__setattr__(self, "max_chunks", selection.max_chunks)


@dataclass(frozen=True, slots=True, kw_only=True)
class HTTPBackendConfig:
    """Fixed-safe HTTPX backend policy, supplied explicitly in configuration."""

    verify_tls: bool
    trust_env: bool
    follow_redirects: bool
    retries: int

    def __post_init__(self) -> None:
        if self.verify_tls is not True:
            raise LocalConfigError("verify_tls must be true.")
        if self.trust_env is not False:
            raise LocalConfigError("trust_env must be false.")
        if self.follow_redirects is not False:
            raise LocalConfigError("follow_redirects must be false.")
        if type(self.retries) is not int or self.retries != 0:
            raise LocalConfigError("retries must be the integer zero.")


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalAppConfig:
    """Validated explicit settings for one local application submission."""

    profile_label: str
    project_root: Path
    manifest_path: Path
    persistence_directory: Path
    chroma_tenant: str
    chroma_database: str
    migration_hash_algorithm: Literal["md5", "sha256"]
    chunking_profile_name: str
    chunking_config: ChunkingConfig
    embedding_profile_name: str
    embedding_config: EmbeddingModelConfig
    bm25_config: BM25Config
    retrieval_config: RetrievalConfig
    generation_config: GeminiGenerationConfig
    budget: RequestBudget
    transport_config: GeminiTransportConfig
    http_backend_config: HTTPBackendConfig

    def __post_init__(self) -> None:
        for name in (
            "profile_label", "chroma_tenant", "chroma_database",
            "chunking_profile_name", "embedding_profile_name",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise LocalConfigError(f"{name} must be a nonblank string.")
        for name in ("project_root", "manifest_path", "persistence_directory"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise LocalConfigError(f"{name} must be an absolute pathlib.Path.")
        if self.migration_hash_algorithm not in ("md5", "sha256"):
            raise LocalConfigError("migration_hash_algorithm must be md5 or sha256.")
        expected = (
            (self.chunking_config, ChunkingConfig, "chunking_config"),
            (self.embedding_config, EmbeddingModelConfig, "embedding_config"),
            (self.bm25_config, BM25Config, "bm25_config"),
            (self.retrieval_config, RetrievalConfig, "retrieval_config"),
            (self.generation_config, GeminiGenerationConfig, "generation_config"),
            (self.budget, RequestBudget, "budget"),
            (self.transport_config, GeminiTransportConfig, "transport_config"),
            (self.http_backend_config, HTTPBackendConfig, "http_backend_config"),
        )
        for value, kind, name in expected:
            if type(value) is not kind:
                raise LocalConfigError(f"{name} must be {kind.__name__}.")
            replace(value)
        try:
            validate_generation_budget(
                generation_config=self.generation_config,
                budget=self.budget,
            )
        except ValueError as exc:
            raise LocalConfigError("Invalid generation/application budget relationship.") from exc


_TOP_LEVEL_KEYS = {
    "schema_version", "profile_label", "paths", "chroma", "profiles", "bm25",
    "retrieval", "generation", "budget", "transport",
}
_SECRET_KEYS = {
    "api_key", "apikey", "gemini_api_key", "authorization", "credential",
    "credentials", "password", "secret", "client_secret",
}


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LocalConfigError("Configuration contains a duplicate JSON key.")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise LocalConfigError("Configuration contains a nonstandard JSON number.")


def _reject_secrets(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            normalized = key.strip().lower().replace("-", "_")
            if normalized in _SECRET_KEYS:
                raise LocalConfigError(
                    "Secret-bearing fields are forbidden in local application configuration."
                )
            _reject_secrets(item)
    elif type(value) is list:
        for item in value:
            _reject_secrets(item)


def _mapping(value: object, *, name: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise LocalConfigError(f"{name} must contain exactly the documented fields.")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise LocalConfigError(f"{name} must be a nonblank string.")
    return value


def _resolve_path(value: object, *, name: str, base: Path) -> Path:
    text = _text(value, name=name)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def validate_local_app_paths(config: LocalAppConfig) -> None:
    """Fail before resource construction when required local storage is absent."""

    if type(config) is not LocalAppConfig:
        raise LocalConfigError("config must be LocalAppConfig.")
    replace(config)
    if not config.project_root.is_dir():
        raise LocalConfigError("Configured project_root is not an existing directory.")
    if not config.manifest_path.is_file():
        raise LocalConfigError("Configured manifest_path is not an existing file.")
    try:
        config.manifest_path.relative_to(config.project_root)
    except ValueError:
        raise LocalConfigError("manifest_path must remain inside project_root.") from None
    if not config.persistence_directory.is_dir():
        raise LocalConfigError(
            "Configured Chroma persistence_directory is not an existing directory."
        )
    database_path = config.persistence_directory / "chroma.sqlite3"
    if not database_path.is_file() or database_path.stat().st_size < 1:
        raise LocalConfigError(
            "Configured Chroma storage requires a non-empty chroma.sqlite3 file."
        )


def load_local_app_config(config_path: str | Path) -> LocalAppConfig:
    """Load the one supported JSON mechanism and validate all explicit values."""

    try:
        path = Path(config_path).resolve(strict=True)
        if not path.is_file():
            raise LocalConfigError("Local application configuration is not a file.")
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_constant=_constant,
        )
        _reject_secrets(raw)
        root = _mapping(raw, name="configuration", keys=_TOP_LEVEL_KEYS)
        if root["schema_version"] != "1.0":
            raise LocalConfigError("Unsupported local configuration schema_version.")

        paths = _mapping(
            root["paths"], name="paths",
            keys={"project_root", "manifest_path", "persistence_directory"},
        )
        project_root = _resolve_path(
            paths["project_root"], name="paths.project_root", base=path.parent,
        )
        manifest_path = _resolve_path(
            paths["manifest_path"], name="paths.manifest_path", base=project_root,
        )
        persistence_directory = _resolve_path(
            paths["persistence_directory"],
            name="paths.persistence_directory", base=project_root,
        )

        chroma = _mapping(
            root["chroma"], name="chroma",
            keys={"tenant", "database", "migration_hash_algorithm"},
        )
        profiles = _mapping(
            root["profiles"], name="profiles",
            keys={"chunking", "embedding"},
        )
        chunking_name = _text(profiles["chunking"], name="profiles.chunking")
        embedding_name = _text(profiles["embedding"], name="profiles.embedding")
        if chunking_name != "baseline_frozen_18_source_v1":
            raise LocalConfigError("Unknown chunking profile.")
        if embedding_name != "gte_modernbert_base_pinned_cpu":
            raise LocalConfigError("Unknown embedding profile.")

        bm25 = _mapping(root["bm25"], name="bm25", keys={"k1", "b"})
        retrieval = _mapping(
            root["retrieval"], name="retrieval",
            keys={
                "dense_candidate_count", "lexical_candidate_count",
                "rank_constant", "max_chunks",
            },
        )
        generation = _mapping(
            root["generation"], name="generation",
            keys={
                "model", "max_output_tokens", "temperature",
                "thinking_budget", "thinking_level",
            },
        )
        budget = _mapping(
            root["budget"], name="budget",
            keys={"input_token_limit", "output_token_limit"},
        )
        transport = _mapping(
            root["transport"], name="transport",
            keys={
                "api_version", "timeout_seconds", "max_response_bytes",
                "verify_tls", "trust_env", "follow_redirects", "retries",
            },
        )

        config = LocalAppConfig(
            profile_label=_text(root["profile_label"], name="profile_label"),
            project_root=project_root,
            manifest_path=manifest_path,
            persistence_directory=persistence_directory,
            chroma_tenant=_text(chroma["tenant"], name="chroma.tenant"),
            chroma_database=_text(chroma["database"], name="chroma.database"),
            migration_hash_algorithm=chroma["migration_hash_algorithm"],
            chunking_profile_name=chunking_name,
            chunking_config=BASELINE_CHUNKING_CONFIG,
            embedding_profile_name=embedding_name,
            embedding_config=GTE_MODERNBERT_BASE,
            bm25_config=BM25Config(k1=bm25["k1"], b=bm25["b"]),
            retrieval_config=RetrievalConfig(**retrieval),
            generation_config=GeminiGenerationConfig(**generation),
            budget=RequestBudget(**budget),
            transport_config=GeminiTransportConfig(
                api_version=transport["api_version"],
                timeout_seconds=transport["timeout_seconds"],
                max_response_bytes=transport["max_response_bytes"],
            ),
            http_backend_config=HTTPBackendConfig(
                verify_tls=transport["verify_tls"],
                trust_env=transport["trust_env"],
                follow_redirects=transport["follow_redirects"],
                retries=transport["retries"],
            ),
        )
        validate_local_app_paths(config)
        return config
    except LocalConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalConfigError("Unable to read valid local application configuration.") from exc
    except Exception as exc:
        raise LocalConfigError("Invalid local application configuration.") from exc


__all__ = [
    "LocalConfigError", "RetrievalConfig", "HTTPBackendConfig", "LocalAppConfig",
    "validate_local_app_paths", "load_local_app_config",
]
