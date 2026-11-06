from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import AppConfig, SourceDefinition, validate_source_url
from .errors import ConfigurationError, SourceError

CHUNK_SIZE = 64 * 1024
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_hash(payload: dict[str, object]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "manifest_hash"}
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def read_source_lock(config: AppConfig, verify_files: bool = True) -> dict[str, object]:
    path = config.source_lock_path
    if not path.is_file():
        raise SourceError(f"Source lock not found: {path}. Import the required CDC files first.")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"Cannot read source lock {path}: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("manifest_hash") != canonical_manifest_hash(lock):
        raise SourceError(f"Source lock manifest hash is invalid: {path}")
    if lock.get("release") != config.release:
        raise SourceError(
            f"Source lock release {lock.get('release')!r} does not match configured "
            f"release {config.release!r}."
        )
    entries = lock.get("sources")
    if not isinstance(entries, dict):
        raise SourceError("Source lock has no valid sources mapping.")
    if verify_files:
        for source_id, entry in entries.items():
            if not isinstance(entry, dict):
                raise SourceError(f"Invalid source-lock entry for {source_id}.")
            local_path = config.data_dir / str(entry.get("local_filename", ""))
            if not local_path.is_file():
                raise SourceError(f"Locked source file is missing: {local_path}")
            actual_hash = sha256_file(local_path)
            if actual_hash != entry.get("sha256"):
                raise SourceError(f"Locked source file hash changed: {local_path}")
    return lock


def source_file_path(config: AppConfig, source_id: str) -> Path:
    try:
        definition = config.sources[source_id]
    except KeyError as exc:
        raise SourceError(
            f"Unknown source id {source_id!r}. Configured sources: "
            f"{', '.join(sorted(config.sources))}."
        ) from exc
    return config.data_dir / "sources" / source_id / definition.filename


def import_source(config: AppConfig, source_id: str, input_path: Path) -> Path:
    definition = _definition(config, source_id)
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise SourceError(f"Manual source file does not exist: {input_path}")
    if input_path.name == ".env":
        raise SourceError("Refusing to import a file named .env.")
    if input_path.stat().st_size > config.max_download_bytes:
        raise SourceError(
            f"Source file exceeds the configured {config.max_download_bytes}-byte limit."
        )
    destination = source_file_path(config, source_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(destination)
    try:
        with input_path.open("rb") as source, temp_path.open("wb") as target:
            _copy_bounded(source, target, config.max_download_bytes)
            target.flush()
            os.fsync(target.fileno())
        validate_source_file(temp_path, definition, config)
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    _register_source(config, definition, destination, input_path.name)
    return destination


def download_source(config: AppConfig, source_id: str) -> Path:
    definition = _definition(config, source_id)
    validate_source_url(definition.url, config.allowed_hosts)
    destination = source_file_path(config, source_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(config.download_retries):
        temp_path = _temporary_path(destination)
        try:
            _download_once(config, definition.url, temp_path)
            validate_source_file(temp_path, definition, config)
            os.replace(temp_path, destination)
            _register_source(config, definition, destination, definition.filename)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError, SourceError) as exc:
            temp_path.unlink(missing_ok=True)
            if not _is_retryable(exc) or attempt + 1 >= config.download_retries:
                raise SourceError(f"Could not download {source_id} from its configured URL: {exc}") from exc
            last_error = exc
            retry_after = _retry_after(exc)
            delay = min(retry_after if retry_after is not None else (2**attempt), 8.0)
            time.sleep(delay + random.uniform(0.0, min(delay * 0.1, 0.5)))
    raise SourceError(f"Download failed after bounded retries: {last_error}")


def validate_source_file(path: Path, definition: SourceDefinition, config: AppConfig) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(4096)
    except OSError as exc:
        raise SourceError(f"Cannot read source file {path}: {exc}") from exc
    if not header:
        raise SourceError(f"Source file is empty: {path}")
    lowered_header = header.lstrip().lower()
    if lowered_header.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise SourceError(f"Source response looks like an error page, not {definition.file_format}.")
    if definition.file_format == "pdf":
        if b"%PDF-" not in header[:1024]:
            raise SourceError(f"Expected a PDF signature in {path}.")
        return
    if not header.startswith(ZIP_SIGNATURES):
        raise SourceError(f"Expected a ZIP signature in {path}.")
    _inspect_zip(path, config)


def inspect_zip(path: Path, config: AppConfig) -> list[dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > config.max_archive_members:
            raise SourceError(
                f"ZIP has {len(infos)} members; configured limit is {config.max_archive_members}."
            )
        total_size = 0
        safe_infos = []
        seen: set[str] = set()
        for info in infos:
            _validate_zip_member(info, seen)
            total_size += info.file_size
            if info.file_size > config.max_archive_member_bytes:
                raise SourceError(f"ZIP member exceeds size limit: {info.filename}")
            compressed = max(info.compress_size, 1)
            if info.file_size / compressed > config.max_compression_ratio:
                raise SourceError(f"ZIP member exceeds compression-ratio limit: {info.filename}")
            safe_infos.append(info)
        if total_size > config.max_archive_uncompressed_bytes:
            raise SourceError(
                "ZIP total uncompressed size exceeds configured limit "
                f"{config.max_archive_uncompressed_bytes}."
            )
        bad_member = archive.testzip()
        if bad_member:
            raise SourceError(f"ZIP CRC check failed for member {bad_member!r}.")
        return [
            {
                "filename": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
            }
            for info in safe_infos
            if not info.is_dir()
        ]


def _inspect_zip(path: Path, config: AppConfig) -> None:
    try:
        inspect_zip(path, config)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise SourceError(f"Invalid or unsafe ZIP archive {path}: {exc}") from exc


def _validate_zip_member(info: zipfile.ZipInfo, seen: set[str]) -> None:
    name = info.filename
    member_path = PurePosixPath(name.replace("\\", "/"))
    if (
        not name
        or name.startswith(("/", "\\"))
        or ":" in name.split("/", 1)[0]
        or ".." in member_path.parts
        or member_path.is_absolute()
    ):
        raise SourceError(f"Unsafe ZIP member path: {name!r}")
    if name in seen:
        raise SourceError(f"Duplicate ZIP member path: {name!r}")
    seen.add(name)
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise SourceError(f"ZIP member is a symbolic link: {name!r}")


def _definition(config: AppConfig, source_id: str) -> SourceDefinition:
    try:
        return config.sources[source_id]
    except KeyError as exc:
        raise SourceError(
            f"Unknown source id {source_id!r}. Configured sources: "
            f"{', '.join(sorted(config.sources))}."
        ) from exc


def _download_once(config: AppConfig, url: str, temp_path: Path) -> None:
    opener = build_opener(_AllowedRedirectHandler(config.allowed_hosts))
    request = Request(url, headers={"User-Agent": "icd-kev-dataset-builder/0.1"})
    with opener.open(request, timeout=config.request_timeout_seconds) as response:
        final_url = response.geturl()
        validate_source_url(final_url, config.allowed_hosts)
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise SourceError("Download returned an invalid Content-Length header.") from exc
            if declared_size > config.max_download_bytes:
                raise SourceError("Download Content-Length exceeds the configured size limit.")
        with temp_path.open("wb") as output:
            _copy_bounded(response, output, config.max_download_bytes)
            output.flush()
            os.fsync(output.fileno())


def _copy_bounded(source: BinaryIO, target: BinaryIO, byte_limit: int) -> None:
    total = 0
    while True:
        chunk = source.read(CHUNK_SIZE)
        if not chunk:
            return
        total += len(chunk)
        if total > byte_limit:
            raise SourceError(f"Source exceeds the configured {byte_limit}-byte limit.")
        target.write(chunk)


def _register_source(
    config: AppConfig,
    definition: SourceDefinition,
    path: Path,
    original_filename: str,
) -> None:
    zip_members: list[dict[str, object]] = []
    if definition.file_format == "zip":
        zip_members = inspect_zip(path, config)
    relative_name = path.relative_to(config.data_dir).as_posix()
    try:
        lock = read_source_lock(config, verify_files=False)
    except SourceError:
        if config.source_lock_path.exists():
            raise
        lock = {
            "schema_version": 1,
            "release": config.release,
            "release_year": config.release_year,
            "coding_system": config.coding_system,
            "sources": {},
        }
    entries = lock.get("sources")
    if not isinstance(entries, dict):
        raise SourceError("Existing source lock has an invalid sources mapping.")
    for existing_id, entry in entries.items():
        if existing_id == definition.source_id:
            continue
        if not isinstance(entry, dict):
            raise SourceError(f"Invalid source-lock entry for {existing_id}.")
        existing_path = config.data_dir / str(entry.get("local_filename", ""))
        if not existing_path.is_file() or sha256_file(existing_path) != entry.get("sha256"):
            raise SourceError(f"Previously locked source is missing or changed: {existing_path}")
    release_markers = sorted(
        {
            str(config.release_year)
            for item in zip_members
            if str(config.release_year) in str(item["filename"])
        }
    )
    source_info: dict[str, object] = {
        "source_id": definition.source_id,
        "role": definition.role,
        "source_url": definition.url,
        "original_filename": original_filename,
        "configured_filename": definition.filename,
        "local_filename": relative_name,
        "sha256": sha256_file(path),
        "imported_at": datetime.now(UTC).isoformat(),
        "format": definition.file_format,
        "required": definition.required,
        "verified_release_metadata": {
            "configured_release": config.release,
            "configured_release_year": config.release_year,
            "verified_member_name_markers": release_markers,
            "verified": bool(release_markers),
        },
        "unresolved_release_uncertainty": (
            []
            if release_markers
            else [
                "The configured release year was not confirmed in archive member names; "
                "verify source contents before treating this source as release-verified."
            ]
        ),
    }
    if definition.file_format == "zip":
        source_info["archive_members"] = zip_members
        source_info["addendum_handling"] = (
            "Archived and checksummed as reference only; never applied over the selected "
            "main release during knowledge-base construction."
            if definition.source_id == "addenda"
            else None
        )
    entries[definition.source_id] = source_info
    lock["sources"] = entries
    lock["manifest_hash"] = canonical_manifest_hash(lock)
    atomic_write_json(config.source_lock_path, lock)


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    return Path(name)


def _retry_after(exc: Exception) -> float | None:
    if isinstance(exc, HTTPError):
        value = exc.headers.get("Retry-After")
        if value:
            try:
                return max(float(value), 0.0)
            except ValueError:
                return None
    return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or 500 <= exc.code <= 599
    return isinstance(exc, (URLError, TimeoutError, OSError))


class _AllowedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, request, response, code, message, headers, new_url):
        try:
            validate_source_url(new_url, self.allowed_hosts)
        except ConfigurationError as exc:
            raise SourceError(f"Refusing redirect to an untrusted URL: {new_url}") from exc
        return super().redirect_request(request, response, code, message, headers, new_url)
