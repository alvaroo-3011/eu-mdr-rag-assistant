"""Provider-free, stop-on-failure evaluation of reviewed retrieval cases.

This is an experiment runner, not a production configuration or authorization
to run it. It opens only a disposable copy of a caller-specified existing index.
Its output is reserved exclusively before resource access. Incomplete evidence
recovery is an evaluation result, not an operational failure.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_seed import (  # noqa: E402
    assess_ranking, parse_seed, ranking_records, validate_seed_against_chunks,
)
from src.evaluation.prospective_cases import (  # noqa: E402
    parse_prospective, project_retrieval_report, validate_prospective_against_chunks,
)


SEED_HASH = "b545a7f9d4aeba5f728a985296b95015592328a3331e14c24928b713989b1258"
PROSPECTIVE_HASH = "b24050f555c31507a4f9521fbb8d2cf50a9235a08375f1fccbb5111dc2b6b758"
SOURCE_COUNT = 18
CHUNK_COUNT = 938
CORPUS_ID = "corpus:sha256:7121229f6c645f3ffdee7a208ea9886b85171fcc90db2fd7148ab028d98a2eab"
CHUNK_SET_ID = "chunk-set:sha256:48c56cb87716c5da233522ecacfb712f6ebbc8dea19b66102f9916eebe8a035b"
CHUNKING_ID = "chunking-config:sha256:84eab6ff2f1bbd9cfb55f20cd63ac21d0cf29125d7e521fa7c9d7fa94e3ff634"
COLLECTION = "mdr-dense-623682d019d735ec03ccc2b4"
LEXICAL_ID = "lexical-index:sha256:81c6503252c5879bc1aae552ffe41f025212509d8b118fb1d817b951b1427204"
EXPECTED_EMBEDDING = {
    "model_id": "Alibaba-NLP/gte-modernbert-base",
    "revision": "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
    "dimension": 768, "max_sequence_length": 8192,
    "device": "cpu", "normalize_embeddings": True,
}
PARAMETERS = {
    "bm25": {"k1": 1.2, "b": 0.75}, "rank_constant": 2.0,
    "assessment_cutoff": 5, "max_selected_chunks": 5,
    "tenant": "default_tenant", "database": "default_database",
    "migration_hash_algorithm": "md5",
}


class ExperimentError(RuntimeError):
    """A bounded experiment precondition or report operation failed."""


@dataclass(frozen=True)
class _WindowsAccessGrant:
    sid: str
    mask: int
    flags: int


@dataclass(frozen=True)
class _WindowsDirectorySecurity:
    owner_sid: str
    dacl_present: bool
    grants: tuple[_WindowsAccessGrant, ...]


_WINDOWS_SYSTEM_SID = "S-1-5-18"
_WINDOWS_ADMINISTRATORS_SID = "S-1-5-32-544"
_WINDOWS_CREATOR_OWNER_SID = "S-1-3-0"
_WINDOWS_OWNER_RIGHTS_SID = "S-1-3-4"
_WINDOWS_INHERIT_ONLY_ACE = 0x08
_WINDOWS_CHILD_INHERITANCE = 0x03
_WINDOWS_TEMP_MUTATION = (
    0x00000002  # FILE_ADD_FILE / FILE_WRITE_DATA
    | 0x00000004  # FILE_ADD_SUBDIRECTORY / FILE_APPEND_DATA
    | 0x00000040  # FILE_DELETE_CHILD
    | 0x00010000  # DELETE
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x10000000  # GENERIC_ALL
    | 0x40000000  # GENERIC_WRITE
)
_WINDOWS_ENTRY_PARENT_MUTATION = (
    0x00000040  # FILE_DELETE_CHILD
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x10000000  # GENERIC_ALL
)


def _windows_api():
    if os.name != "nt":
        raise ExperimentError("A private Windows temporary parent is required.")
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.GetAclInformation.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int,
    ]
    advapi32.GetAclInformation.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return advapi32, kernel32


def _sid_string(advapi32, kernel32, sid) -> str:
    text = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return text.value
    finally:
        kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))


def _windows_current_user_sid() -> str:
    advapi32, kernel32 = _windows_api()
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008,
                                     ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = wintypes.DWORD()
        if advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(size)):
            raise ExperimentError("Unexpected Windows token information response.")
        if ctypes.get_last_error() != 122 or size.value == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(token, 1, buffer, size,
                                             ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p)).contents.value
        return _sid_string(advapi32, kernel32, sid)
    finally:
        kernel32.CloseHandle(token)


def _windows_directory_security(path: Path) -> _WindowsDirectorySecurity:
    advapi32, kernel32 = _windows_api()
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    error = advapi32.GetNamedSecurityInfoW(
        str(path), 1, 0x00000001 | 0x00000004,
        ctypes.byref(owner), None, ctypes.byref(dacl), None,
        ctypes.byref(descriptor),
    )
    if error:
        raise ctypes.WinError(error)

    class _AclSizeInformation(ctypes.Structure):
        _fields_ = [("ace_count", wintypes.DWORD),
                    ("bytes_in_use", wintypes.DWORD),
                    ("bytes_free", wintypes.DWORD)]

    class _AceHeader(ctypes.Structure):
        _fields_ = [("ace_type", ctypes.c_ubyte),
                    ("ace_flags", ctypes.c_ubyte),
                    ("ace_size", ctypes.c_ushort)]

    try:
        dacl_present = wintypes.BOOL()
        dacl_defaulted = wintypes.BOOL()
        actual_dacl = ctypes.c_void_p()
        if not advapi32.GetSecurityDescriptorDacl(
                descriptor, ctypes.byref(dacl_present), ctypes.byref(actual_dacl),
                ctypes.byref(dacl_defaulted)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not dacl_present.value or not actual_dacl.value:
            return _WindowsDirectorySecurity(
                owner_sid=_sid_string(advapi32, kernel32, owner),
                dacl_present=False, grants=(),
            )

        info = _AclSizeInformation()
        if not advapi32.GetAclInformation(actual_dacl, ctypes.byref(info),
                                          ctypes.sizeof(info), 2):
            raise ctypes.WinError(ctypes.get_last_error())
        grants = []
        allow_types = {0x00, 0x05, 0x09, 0x0B}
        object_allow_types = {0x05, 0x0B}
        deny_types = {0x01, 0x06, 0x0A, 0x0C}
        for index in range(info.ace_count):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(actual_dacl, index, ctypes.byref(ace)):
                raise ctypes.WinError(ctypes.get_last_error())
            header = ctypes.cast(ace, ctypes.POINTER(_AceHeader)).contents
            if header.ace_type in deny_types:
                continue
            if header.ace_type not in allow_types:
                raise ExperimentError(
                    "Temporary parent uses an unsupported Windows access rule."
                )
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            sid_offset = 8
            if header.ace_type in object_allow_types:
                object_flags = ctypes.c_uint32.from_address(ace.value + 8).value
                sid_offset = 12
                if object_flags & 0x01:
                    sid_offset += 16
                if object_flags & 0x02:
                    sid_offset += 16
            if header.ace_size < sid_offset + 8:
                raise ExperimentError(
                    "Temporary parent has a malformed Windows access rule."
                )
            sid = ctypes.c_void_p(ace.value + sid_offset)
            grants.append(_WindowsAccessGrant(
                sid=_sid_string(advapi32, kernel32, sid), mask=mask,
                flags=header.ace_flags,
            ))
        return _WindowsDirectorySecurity(
            owner_sid=_sid_string(advapi32, kernel32, owner),
            dacl_present=True, grants=tuple(grants),
        )
    finally:
        kernel32.LocalFree(descriptor)


def _windows_security_is_private(security: _WindowsDirectorySecurity,
                                 current_user_sid: str, *, entry_parent: bool) -> bool:
    trusted = {current_user_sid, _WINDOWS_SYSTEM_SID, _WINDOWS_ADMINISTRATORS_SID,
               _WINDOWS_CREATOR_OWNER_SID, _WINDOWS_OWNER_RIGHTS_SID}
    if not security.dacl_present or security.owner_sid not in trusted:
        return False
    for grant in security.grants:
        if grant.sid in trusted:
            continue
        applies_here = not grant.flags & _WINDOWS_INHERIT_ONLY_ACE
        if entry_parent:
            if applies_here and grant.mask & _WINDOWS_ENTRY_PARENT_MUTATION:
                return False
        elif ((applies_here or grant.flags & _WINDOWS_CHILD_INHERITANCE)
              and grant.mask & _WINDOWS_TEMP_MUTATION):
            return False
    return True


def _require_plain_windows_path(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink() or current.is_junction():
            raise ExperimentError(
                "Temporary parent path must not contain a link or junction."
            )


def _require_private_windows_temp_parent(path: Path) -> None:
    try:
        current_user_sid = _windows_current_user_sid()
        security = _windows_directory_security(path)
        parent_security = (_windows_directory_security(path.parent)
                           if path.parent != path else security)
    except (OSError, ValueError) as exc:
        raise ExperimentError(
            "Unable to establish private Windows permissions for temporary parent."
        ) from exc
    if (not _windows_security_is_private(
            security, current_user_sid, entry_parent=False)
            or not _windows_security_is_private(
                parent_security, current_user_sid, entry_parent=True)):
        raise ExperimentError(
            "Temporary parent Windows permissions allow an untrusted principal "
            "to replace its contents."
        )


def _remove_disposable_directory(scratch: Path, temp: Path) -> None:
    if (scratch.parent != temp or not scratch.name.startswith("mdr-selected-eval-")
            or scratch.is_symlink() or scratch.is_junction()):
        raise ExperimentError("Refusing unsafe temporary directory cleanup.")
    shutil.rmtree(scratch)
    if scratch.exists():
        raise ExperimentError("Disposable directory remains after cleanup.")


def _validate_candidate_count(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExperimentError("Candidate count must be a positive nonboolean integer.")
    return value


def _parse_candidate_count(value: str) -> int:
    try:
        return _validate_candidate_count(int(value))
    except (ValueError, ExperimentError) as exc:
        raise argparse.ArgumentTypeError("Candidate count must be a positive integer.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(directory: Path) -> dict[str, dict]:
    """Hash a complete tree without following links or special files."""
    if directory.is_symlink() or directory.is_junction():
        raise ExperimentError("Index root must not be a symbolic link or junction.")
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ExperimentError("Index path must be a directory.")

    def fail_enumeration(exc: OSError) -> None:
        raise ExperimentError("Unable to enumerate the complete index tree.") from exc

    result = {}
    for base, dirs, files in os.walk(root, followlinks=False, onerror=fail_enumeration):
        base_path = Path(base)
        for name in sorted(dirs + files):
            path = base_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or path.is_junction():
                raise ExperimentError("Index tree contains a symbolic link or junction.")
            if path.is_dir():
                result[relative + "/"] = {"type": "directory"}
            elif path.is_file():
                result[relative] = {"type": "file", "size": path.stat().st_size,
                                    "sha256": _sha256(path)}
            else:
                raise ExperimentError("Index tree contains a nonregular entry.")
    if "chroma.sqlite3" not in result or result["chroma.sqlite3"].get("size", 0) < 1:
        raise ExperimentError("Nonempty existing chroma.sqlite3 is required.")
    return result


def _provenance(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout.rstrip("\r\n")
    paths = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short", "--untracked-files=all"),
        "code_sha256": {str(p.relative_to(root)).replace("\\", "/"): _sha256(p)
                        for p in paths},
        "package_versions": {name: version(name) for name in
                             ("chromadb", "numpy", "torch", "transformers", "huggingface-hub")},
    }


@contextmanager
def _block_network():
    def forbidden(*_args, **_kwargs):
        raise ExperimentError("Network access is forbidden during this experiment.")
    with (patch.object(socket.socket, "connect", forbidden),
          patch.object(socket.socket, "connect_ex", forbidden),
          patch.object(socket, "create_connection", forbidden)):
        yield


class _ObservedRetriever:
    def __init__(self, retriever, operations: dict, branch: str):
        self._retriever = retriever
        self._operations = operations
        self._branch = branch

    @property
    def chunk_set_id(self):
        return self._retriever.chunk_set_id

    def retrieve(self, query, *, candidate_count):
        counts = self._operations[self._branch]
        counts["attempted"] += 1
        value = self._retriever.retrieve(query, candidate_count=candidate_count)
        counts["completed"] += 1
        return value


def _failure(stage: str, exc: Exception) -> dict:
    # Avoid serializing exception text, paths from nested backends, or secrets.
    return {"stage": stage, "error_type": type(exc).__name__,
            "message": "Operation failed; inspect the stage and local diagnostics."}


def _default_dependencies():
    from src.application.chroma_startup import open_existing_chroma_client
    from src.application.retrieval_startup import initialize_retrieval
    from src.ingestion.chunking_profiles import BASELINE_CHUNKING_CONFIG
    from src.retrieval.dense_index import dense_collection_name
    from src.retrieval.embeddings import GTE_MODERNBERT_BASE
    from src.retrieval.evidence_selection import select_evidence
    from src.retrieval.hybrid_query import retrieve_hybrid
    from src.retrieval.lexical_query import BM25Config
    return (open_existing_chroma_client, initialize_retrieval, BASELINE_CHUNKING_CONFIG,
            GTE_MODERNBERT_BASE, dense_collection_name, select_evidence,
            retrieve_hybrid, BM25Config)


@_block_network()
def run_experiment(*, project_root: Path, seed_path: Path, original_index: Path,
                   temp_parent: Path, output_path: Path, acknowledge_writers_stopped: bool,
                   dense_candidate_count: int, lexical_candidate_count: int,
                   mode: str = "seed",
                   dependencies=None, provenance=_provenance) -> dict:
    """Run the specified reviewed experiment; caller must authorize real access.

    The result file is exclusively reserved before reading the seed or index.
    Reservations remain on write/interruption failure for manual inspection.
    This function contains no provider or credential path.
    """
    dense_candidate_count = _validate_candidate_count(dense_candidate_count)
    lexical_candidate_count = _validate_candidate_count(lexical_candidate_count)
    if mode not in ("seed", "prospective"):
        raise ExperimentError("Unknown evaluation mode.")
    if acknowledge_writers_stopped is not True:
        raise ExperimentError("Explicit acknowledgement that all index writers are stopped is required.")
    root = Path(project_root).resolve(strict=True)
    index_input = Path(original_index)
    if index_input.is_symlink() or index_input.is_junction():
        raise ExperimentError("Index root must not be a symbolic link or junction.")
    index = index_input.resolve(strict=False)
    temp_input = Path(temp_parent)
    _require_plain_windows_path(temp_input)
    temp = temp_input.resolve(strict=True)
    output = Path(output_path).resolve(strict=False)
    if not root.is_dir() or not temp.is_dir() or not output.parent.is_dir():
        raise ExperimentError("Project, temporary parent and output parent must exist.")
    if temp == index or temp.is_relative_to(index) or output.is_relative_to(index):
        raise ExperimentError("Temporary and output paths must be outside the original index.")
    _require_private_windows_temp_parent(temp)
    try:
        handle = output.open("x", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ExperimentError("Unable to reserve the report path; no resource was opened.") from exc

    counts = {name: {"attempted": 0, "completed": 0} for name in
              ("client_open", "initialization", "hybrid", "dense", "lexical", "selection")}
    report = {
        "format_version": "retrieval-evidence-run/v2", "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "finished_at_utc": None,
        "provenance": None, "seed": None, "configuration": None,
        "expected_identities": {"source_count": SOURCE_COUNT, "chunk_count": CHUNK_COUNT,
            "corpus_identity": CORPUS_ID, "chunk_set_identity": CHUNK_SET_ID,
            "chunking_config_id": CHUNKING_ID, "dense_collection": COLLECTION,
            "lexical_index_id": LEXICAL_ID},
        "observed_identities": None, "operations": counts, "provider_calls": 0,
        "cases": [{"case_id": f"C{i}", "topic_id": None, "question": None,
                   "status": "not_run", "rankings": None, "assessment": None,
                   "approved_evidence_positions": None, "failure": None}
                  for i in range(1, 7)],
        "primary_failure": None, "cleanup_failures": [],
        "cleanup": {"client_close": "not_started", "disposable_copy_removal": "not_started",
                    "original_fingerprint_before": None, "original_fingerprint_after": None,
                    "original_fingerprint_match": None},
        "interpretation": {"judgment_policy": "other chunks unjudged",
            "semantic_sufficiency_established": False, "answer_quality_established": False,
            "production_readiness_established": False},
    }
    if mode == "prospective":
        report["format_version"] = "prospective-retrieval-run/v1"
        report.pop("seed")
        report["artifact"] = None
        report["cases"] = [{"case_id": f"P{i}", "question": None,
            "status": "not_run" if i <= 4 else "not_executed", "rankings": None,
            "assessment": None if i <= 4 else {"applicability": "not_applicable"},
            "answer_behavior": "not_evaluated", "semantic_review": "not_evaluated",
            "failure": None} for i in range(1, 9)]
    stage = "provenance"
    client = None
    scratch = None
    before = None
    try:
        with _block_network():
            report["provenance"] = provenance(root)
            stage = "seed" if mode == "seed" else "artifact"
            seed_bytes = Path(seed_path).read_bytes()
            seed_hash = hashlib.sha256(seed_bytes).hexdigest()
            if mode == "seed":
                if seed_hash != SEED_HASH:
                    raise ExperimentError("Seed SHA-256 does not match the approved seed.")
                seed = parse_seed(seed_bytes)
                cases = seed["cases"]
                if [case["case_id"] for case in cases] != [f"C{i}" for i in range(1, 7)]:
                    raise ExperimentError("The approved six cases must be in order.")
                if seed["artifact_id"] != "mdr-retrieval-seed-v1":
                    raise ExperimentError("Unexpected seed artifact identity.")
                manifest_path = seed["frozen_corpus"]["manifest_path"]
                if manifest_path != "data/corpus_manifest.json":
                    raise ExperimentError("Unexpected manifest path in approved seed.")
                report["seed"] = {"path": str(Path(seed_path).resolve()), "sha256": seed_hash,
                                  "artifact_id": seed["artifact_id"]}
                report["cases"] = [{"case_id": c["case_id"], "topic_id": c["topic_id"],
                                    "question": c["question"], "status": "not_run",
                                    "rankings": None, "assessment": None,
                                    "approved_evidence_positions": None, "failure": None}
                                   for c in cases]
            else:
                if seed_hash != PROSPECTIVE_HASH:
                    raise ExperimentError("Prospective artifact SHA-256 does not match approval.")
                seed = parse_prospective(seed_bytes)
                by_id = {c["id"]: c for c in seed["cases"]}
                cases = [by_id[f"P{i}"] for i in range(1, 9)]
                manifest_path = "data/corpus_manifest.json"
                report["artifact"] = {"path": str(Path(seed_path).resolve()),
                                      "sha256": seed_hash, "artifact_id": seed["artifact"]}
                for case, record in zip(cases, report["cases"]):
                    record["question"] = case["question"]
            if report["provenance"]["package_versions"]["chromadb"] != "1.5.9":
                raise ExperimentError("Chroma version does not match the reviewed version.")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            (open_client, initialize, chunk_config, embedding_config, collection_name,
             select, hybrid, bm25_type) = dependencies or _default_dependencies()
            if asdict(embedding_config) != EXPECTED_EMBEDDING:
                raise ExperimentError("Embedding profile does not match the reviewed identity.")
            report["configuration"] = {**PARAMETERS,
                "dense_candidate_count": dense_candidate_count,
                "lexical_candidate_count": lexical_candidate_count, "project_root": str(root),
                "manifest_path": str(root / manifest_path),
                "original_index_path": str(index), "temp_parent": str(temp),
                "output_path": str(output),
                "chunking": asdict(chunk_config), "embedding": asdict(embedding_config),
                "writers_stopped_acknowledged": True}
            manifest = (root / manifest_path).resolve()
            if not manifest.is_relative_to(root):
                raise ExperimentError("Manifest must be inside the project root.")
            stage = "original_fingerprint_before"
            before = fingerprint(index)
            report["cleanup"]["original_fingerprint_before"] = before
            stage = "copy"
            scratch = Path(tempfile.mkdtemp(prefix="mdr-selected-eval-", dir=temp))
            copy = scratch / "chroma"
            report["configuration"]["disposable_index_path"] = str(copy)
            shutil.copytree(index, copy, symlinks=True)
            if fingerprint(copy) != before or fingerprint(index) != before:
                raise ExperimentError("Index changed or copy fingerprint does not match.")
            stage = "client_open"
            counts["client_open"]["attempted"] += 1
            client = open_client(persistence_directory=copy, tenant=PARAMETERS["tenant"],
                                 database=PARAMETERS["database"],
                                 migration_hash_algorithm=PARAMETERS["migration_hash_algorithm"])
            counts["client_open"]["completed"] += 1
            stage = "initialization"
            counts["initialization"]["attempted"] += 1
            chunks, dense, lexical = initialize(
                project_root=root, manifest_path=manifest, chunking_config=chunk_config,
                embedding_config=embedding_config,
                bm25_config=bm25_type(k1=1.2, b=0.75), dense_client=client,
            )
            counts["initialization"]["completed"] += 1
            stage = "identity_validation"
            if mode == "seed":
                validate_seed_against_chunks(seed, chunks)
            else:
                validate_prospective_against_chunks(seed, chunks)
            actual = {"source_count": len(chunks.manifest.sources), "chunk_count": len(chunks.chunks),
                "corpus_identity": chunks.corpus_identity, "chunk_set_identity": chunks.chunk_set_id,
                "chunking_config_id": chunks.chunking_config_id,
                "dense_collection": collection_name(chunks, embedding_config),
                "lexical_index_id": lexical.index_id}
            report["observed_identities"] = actual
            if actual != report["expected_identities"]:
                raise ExperimentError("Observed identities or counts do not match the experiment.")
            observed_dense = _ObservedRetriever(dense, counts, "dense")
            observed_lexical = _ObservedRetriever(lexical, counts, "lexical")
            for case, record in zip(cases, report["cases"]):
                if mode == "prospective" and record["status"] == "not_executed":
                    continue
                try:
                    stage = "retrieval"
                    counts["hybrid"]["attempted"] += 1
                    fused = hybrid(case["question"], chunk_set=chunks,
                        dense_retriever=observed_dense, lexical_retriever=observed_lexical,
                        dense_candidate_count=dense_candidate_count,
                        lexical_candidate_count=lexical_candidate_count, rank_constant=2.0)
                    counts["hybrid"]["completed"] += 1
                    stage = "selection"
                    counts["selection"]["attempted"] += 1
                    selection = select(chunk_set=chunks, fused_candidates=fused, max_chunks=5)
                    counts["selection"]["completed"] += 1
                    stage = "assessment"
                    rankings = ranking_records(fused)
                    rankings["selected"] = [{"rank": i, "chunk_id": item.chunk.chunk_id,
                        "fused_rank": item.candidate.rank}
                        for i, item in enumerate(selection.selected, 1)]
                    if mode == "prospective":
                        projected = project_retrieval_report(artifact=seed, case_id=case["id"],
                            rankings={name: [row["chunk_id"] for row in rankings[name]]
                                      for name in ("dense", "lexical", "fused", "selected")}, cutoff=5)
                        record.update(status="ok", rankings=rankings, assessment=projected["retrieval"])
                    else:
                        assessment = {name: assess_ranking(case, seed["evidence"],
                            [row["chunk_id"] for row in rankings[name]], cutoff=5)
                            for name in ("dense", "lexical", "fused", "selected")}
                        assessment["fused_union"] = assess_ranking(case, seed["evidence"],
                            [row["chunk_id"] for row in rankings["fused"]], cutoff=max(1, len(fused)))
                        evidence_ids = {row["evidence_id"]: row["chunk_id"] for row in seed["evidence"]}
                        positions = []
                        for judgment in case["supporting_judgments"]:
                            evidence_id = judgment["evidence_id"]
                            chunk_id = evidence_ids[evidence_id]
                            ranks = {name: next((row["rank"] for row in rankings[name]
                                                 if row["chunk_id"] == chunk_id), None)
                                     for name in ("dense", "lexical", "fused", "selected")}
                            positions.append({"evidence_id": evidence_id, "chunk_id": chunk_id,
                                              "ranks": ranks})
                        record.update(status="ok", rankings=rankings, assessment=assessment,
                                      approved_evidence_positions=positions)
                except Exception as exc:
                    record.update(status="failed", failure=_failure(stage, exc))
                    raise
                except BaseException as exc:
                    record.update(status="failed", failure=_failure(stage, exc))
                    raise
            report["status"] = "completed"
    except Exception as exc:
        report["primary_failure"] = _failure(stage, exc)
    except BaseException as exc:
        report["primary_failure"] = _failure(stage, exc)
        raise
    finally:
        if client is not None:
            try:
                client.close()
                report["cleanup"]["client_close"] = "passed"
            except Exception as exc:
                report["cleanup"]["client_close"] = "failed"
                report["cleanup_failures"].append(_failure("client_close", exc))
        if scratch is not None:
            try:
                _remove_disposable_directory(scratch, temp)
                report["cleanup"]["disposable_copy_removal"] = "passed"
            except Exception as exc:
                report["cleanup"]["disposable_copy_removal"] = "failed"
                report["cleanup_failures"].append(_failure("disposable_copy_removal", exc))
        if before is not None:
            try:
                after = fingerprint(index)
                report["cleanup"]["original_fingerprint_after"] = after
                report["cleanup"]["original_fingerprint_match"] = before == after
                if before != after:
                    raise ExperimentError("Original index fingerprint changed.")
            except Exception as exc:
                report["cleanup_failures"].append(_failure("original_fingerprint_after", exc))
        if report["cleanup_failures"]:
            report["status"] = "cleanup_failed"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            with handle:
                handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
        except Exception as exc:
            raise ExperimentError("Report write failed; the reserved file may be incomplete.") from exc
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "seed-path", "original-index", "temp-parent", "output-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dense-candidates", dest="dense_candidate_count",
                        type=_parse_candidate_count, required=True)
    parser.add_argument("--lexical-candidates", dest="lexical_candidate_count",
                        type=_parse_candidate_count, required=True)
    parser.add_argument("--acknowledge-writers-stopped", action="store_true", required=True)
    parser.add_argument("--mode", choices=("seed", "prospective"), default="seed")
    args = parser.parse_args(argv)
    try:
        result = run_experiment(**vars(args))
    except Exception as exc:
        print(f"Experiment reservation/report failure: {type(exc).__name__}.", file=sys.stderr)
        return 1
    print(f"Experiment status: {result['status']}; report: {args.output_path}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
