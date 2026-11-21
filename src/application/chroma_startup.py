"""Open caller-owned Chroma only after fail-closed storage preflight.

This module performs no collection operation and imports Chroma only after the
supplied existing-storage path passes validation. The returned client remains
fully caller-owned, including closure.

The preflight proves only that a non-empty ``chroma.sqlite3`` regular file is
present. It does not validate database contents, eliminate the check/open race,
restrict later caller writes, or guarantee that SQLite startup creates no
journal, WAL, shared-memory, or other sidecar writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from chromadb.api.client import Client


def open_existing_chroma_client(
    *,
    persistence_directory: str | Path,
    tenant: str,
    database: str,
    migration_hash_algorithm: Literal["md5", "sha256"],
) -> Client:
    """Validate existing storage and construct one migration-validating client.

    Chroma Settings and client-construction exceptions propagate unchanged.
    No retry, fallback, repair, migration application, reset, or lifecycle
    management is performed here.
    """

    if (
        type(migration_hash_algorithm) is not str
        or migration_hash_algorithm not in ("md5", "sha256")
    ):
        raise ValueError(
            "migration_hash_algorithm must be exactly 'md5' or 'sha256'."
        )

    resolved_directory = Path(persistence_directory).resolve(strict=True)
    if not resolved_directory.is_dir():
        raise NotADirectoryError(
            f"Chroma persistence path is not a directory: {resolved_directory}"
        )

    database_path = resolved_directory / "chroma.sqlite3"
    if not database_path.exists():
        raise FileNotFoundError(
            f"Existing Chroma database is required: {database_path}"
        )
    if not database_path.is_file():
        raise ValueError(
            f"Chroma database path must be a regular file: {database_path}"
        )
    if database_path.stat().st_size < 1:
        raise ValueError(
            f"Chroma database file must be non-empty: {database_path}"
        )

    # Delayed imports keep rejected preflight paths independent of Chroma and
    # its compiled bindings.
    from chromadb import PersistentClient
    from chromadb.config import Settings

    settings = Settings(
        _env_file=None,
        chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
        allow_reset=False,
        anonymized_telemetry=False,
        chroma_product_telemetry_impl=(
            "chromadb.telemetry.product.posthog.Posthog"
        ),
        chroma_otel_collection_endpoint="",
        chroma_otel_collection_headers={},
        chroma_otel_granularity=None,
        migrations="validate",
        migrations_hash_algorithm=migration_hash_algorithm,
    )
    return cast(
        "Client",
        PersistentClient(
            path=resolved_directory,
            settings=settings,
            tenant=tenant,
            database=database,
        ),
    )


__all__ = ["open_existing_chroma_client"]
