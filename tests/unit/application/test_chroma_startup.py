"""Offline tests for fail-closed caller-owned Chroma startup."""

from __future__ import annotations

from inspect import Parameter, Signature, signature
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, call

import pytest

from src.application.chroma_startup import open_existing_chroma_client


_EXPECTED_SETTINGS = {
    "_env_file": None,
    "chroma_api_impl": "chromadb.api.rust.RustBindingsAPI",
    "allow_reset": False,
    "anonymized_telemetry": False,
    "chroma_product_telemetry_impl": (
        "chromadb.telemetry.product.posthog.Posthog"
    ),
    "chroma_otel_collection_endpoint": "",
    "chroma_otel_collection_headers": {},
    "chroma_otel_granularity": None,
    "migrations": "validate",
}


class _CallerOwnedClient:
    """Record every operation the startup helper must not perform."""

    def __init__(self) -> None:
        self.operations: list[str] = []

    def close(self) -> None:
        self.operations.append("close")

    def get_collection(self, **values: object) -> None:
        del values
        self.operations.append("get_collection")

    def create_collection(self, **values: object) -> None:
        del values
        self.operations.append("create_collection")

    def get_or_create_collection(self, **values: object) -> None:
        del values
        self.operations.append("get_or_create_collection")

    def reset(self) -> None:
        self.operations.append("reset")


def _existing_storage(tmp_path: Path) -> Path:
    storage = tmp_path / "existing-chroma"
    storage.mkdir()
    (storage / "chroma.sqlite3").write_bytes(b"synthetic-not-sqlite")
    return storage


def _block_chroma_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)
    monkeypatch.setitem(sys.modules, "chromadb.config", None)


def _install_fake_chroma(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings_factory: Mock | None = None,
    client_factory: Mock | None = None,
) -> tuple[Mock, Mock]:
    settings = settings_factory if settings_factory is not None else Mock()
    client = client_factory if client_factory is not None else Mock()
    chromadb_module = ModuleType("chromadb")
    chromadb_module.__path__ = []  # type: ignore[attr-defined]
    config_module = ModuleType("chromadb.config")
    chromadb_module.PersistentClient = client  # type: ignore[attr-defined]
    chromadb_module.config = config_module  # type: ignore[attr-defined]
    config_module.Settings = settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", chromadb_module)
    monkeypatch.setitem(sys.modules, "chromadb.config", config_module)
    return settings, client


def test_contract_is_keyword_only_without_defaults() -> None:
    function_signature = signature(open_existing_chroma_client)
    assert tuple(function_signature.parameters) == (
        "persistence_directory",
        "tenant",
        "database",
        "migration_hash_algorithm",
    )
    assert all(
        parameter.kind is Parameter.KEYWORD_ONLY
        and parameter.default is Signature.empty
        for parameter in function_signature.parameters.values()
    )


@pytest.mark.parametrize(
    "invalid_hash",
    (None, True, 1, "", "MD5", "sha-256"),
)
def test_invalid_migration_hash_stops_before_path_or_chroma(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_hash: object,
) -> None:
    _block_chroma_imports(monkeypatch)

    with pytest.raises(ValueError, match="migration_hash_algorithm"):
        open_existing_chroma_client(
            persistence_directory=tmp_path / "also-missing",
            tenant="tenant",
            database="database",
            migration_hash_algorithm=invalid_hash,  # type: ignore[arg-type]
        )


def test_missing_persistence_path_stops_before_chroma_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _block_chroma_imports(monkeypatch)

    with pytest.raises(FileNotFoundError):
        open_existing_chroma_client(
            persistence_directory=tmp_path / "missing",
            tenant="tenant",
            database="database",
            migration_hash_algorithm="md5",
        )


def test_non_directory_persistence_path_stops_before_chroma_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-a-directory"
    path.write_bytes(b"file")
    _block_chroma_imports(monkeypatch)

    with pytest.raises(NotADirectoryError):
        open_existing_chroma_client(
            persistence_directory=path,
            tenant="tenant",
            database="database",
            migration_hash_algorithm="md5",
        )


@pytest.mark.parametrize("database_state", ("missing", "directory", "empty"))
def test_invalid_database_file_stops_before_chroma_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    database_state: str,
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    database_path = storage / "chroma.sqlite3"
    if database_state == "directory":
        database_path.mkdir()
    elif database_state == "empty":
        database_path.touch()
    _block_chroma_imports(monkeypatch)

    expected_error = FileNotFoundError if database_state == "missing" else ValueError
    with pytest.raises(expected_error):
        open_existing_chroma_client(
            persistence_directory=storage,
            tenant="tenant",
            database="database",
            migration_hash_algorithm="sha256",
        )


@pytest.mark.parametrize("migration_hash", ("md5", "sha256"))
def test_success_uses_exact_settings_and_preserves_caller_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    migration_hash: str,
) -> None:
    storage = _existing_storage(tmp_path)
    settings_object = object()
    returned_client = _CallerOwnedClient()
    settings_factory = Mock(return_value=settings_object)
    client_factory = Mock(return_value=returned_client)
    _install_fake_chroma(
        monkeypatch,
        settings_factory=settings_factory,
        client_factory=client_factory,
    )
    tenant = "explicit-tenant"
    database = "explicit-database"

    result = open_existing_chroma_client(
        persistence_directory=storage,
        tenant=tenant,
        database=database,
        migration_hash_algorithm=migration_hash,  # type: ignore[arg-type]
    )

    assert result is returned_client
    settings_factory.assert_called_once_with(
        **_EXPECTED_SETTINGS,
        migrations_hash_algorithm=migration_hash,
    )
    client_factory.assert_called_once_with(
        path=storage.resolve(strict=True),
        settings=settings_object,
        tenant=tenant,
        database=database,
    )
    assert client_factory.call_args.kwargs["tenant"] is tenant
    assert client_factory.call_args.kwargs["database"] is database
    assert returned_client.operations == []


def test_settings_failure_propagates_without_client_construction_or_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = _existing_storage(tmp_path)
    failure = RuntimeError("synthetic settings failure")
    settings_factory = Mock(side_effect=failure)
    client_factory = Mock()
    _install_fake_chroma(
        monkeypatch,
        settings_factory=settings_factory,
        client_factory=client_factory,
    )

    with pytest.raises(RuntimeError) as caught:
        open_existing_chroma_client(
            persistence_directory=storage,
            tenant="tenant",
            database="database",
            migration_hash_algorithm="md5",
        )

    assert caught.value is failure
    assert settings_factory.call_count == 1
    client_factory.assert_not_called()


def test_client_failure_propagates_once_without_retry_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = _existing_storage(tmp_path)
    settings_object = object()
    failure = RuntimeError("synthetic client failure")
    settings_factory = Mock(return_value=settings_object)
    client_factory = Mock(side_effect=failure)
    _install_fake_chroma(
        monkeypatch,
        settings_factory=settings_factory,
        client_factory=client_factory,
    )

    with pytest.raises(RuntimeError) as caught:
        open_existing_chroma_client(
            persistence_directory=storage,
            tenant="tenant",
            database="database",
            migration_hash_algorithm="sha256",
        )

    assert caught.value is failure
    settings_factory.assert_called_once()
    assert client_factory.call_args_list == [call(
        path=storage.resolve(strict=True),
        settings=settings_object,
        tenant="tenant",
        database="database",
    )]
