from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import ConfigurationError


@dataclass(frozen=True)
class SourceDefinition:
    source_id: str
    role: str
    url: str
    filename: str
    file_format: str
    required: bool


@dataclass(frozen=True)
class AppConfig:
    root: Path
    release: str
    release_year: int
    coding_system: str
    data_dir: Path
    allowed_hosts: frozenset[str]
    request_timeout_seconds: int
    download_retries: int
    max_download_bytes: int
    max_archive_members: int
    max_archive_uncompressed_bytes: int
    max_archive_member_bytes: int
    max_compression_ratio: int
    sources: dict[str, SourceDefinition]

    @property
    def source_lock_path(self) -> Path:
        return self.root / "sources.lock.json"

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"


def default_config_path(root: Path) -> Path:
    return root / "config" / "default.toml"


def load_config(root: Path, config_path: Path | None = None) -> AppConfig:
    root = root.resolve()
    path = (config_path or default_config_path(root)).resolve()
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"Invalid TOML configuration {path}: {exc}") from exc

    project = raw.get("project")
    security = raw.get("security")
    if not isinstance(project, dict) or not isinstance(security, dict):
        raise ConfigurationError("Configuration must define [project] and [security] tables.")

    data_dir_value = project.get("data_dir")
    if not isinstance(data_dir_value, str) or not data_dir_value:
        raise ConfigurationError("project.data_dir must be a non-empty path.")
    data_dir = Path(data_dir_value)
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    host_values = security.get("allowed_hosts")
    if not isinstance(host_values, list) or not host_values:
        raise ConfigurationError("security.allowed_hosts must be a non-empty list.")
    allowed_hosts = frozenset(str(host).lower() for host in host_values)

    source_rows = raw.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise ConfigurationError("Configuration must define at least one [[sources]] entry.")
    sources: dict[str, SourceDefinition] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            raise ConfigurationError("Each [[sources]] entry must be a table.")
        try:
            definition = SourceDefinition(
                source_id=str(row["id"]),
                role=str(row["role"]),
                url=str(row["url"]),
                filename=str(row["filename"]),
                file_format=str(row["format"]),
                required=bool(row["required"]),
            )
        except KeyError as exc:
            raise ConfigurationError(f"Source definition is missing {exc.args[0]!r}.") from exc
        if definition.source_id in sources:
            raise ConfigurationError(f"Duplicate source id: {definition.source_id}")
        validate_source_url(definition.url, allowed_hosts)
        if definition.file_format not in {"zip", "pdf"}:
            raise ConfigurationError(
                f"Unsupported format {definition.file_format!r} for {definition.source_id}."
            )
        if Path(definition.filename).name != definition.filename:
            raise ConfigurationError(
                f"Source filename must be a basename: {definition.filename!r}."
            )
        sources[definition.source_id] = definition

    integer_keys = (
        "request_timeout_seconds",
        "download_retries",
        "max_download_bytes",
        "max_archive_members",
        "max_archive_uncompressed_bytes",
        "max_archive_member_bytes",
        "max_compression_ratio",
    )
    limits: dict[str, int] = {}
    for key in integer_keys:
        value = security.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"security.{key} must be a positive integer.")
        limits[key] = value
    if limits["download_retries"] > 10:
        raise ConfigurationError("security.download_retries cannot exceed 10.")

    release_year = project.get("release_year")
    release = project.get("release")
    coding_system = project.get("coding_system")
    if not isinstance(release_year, int) or not isinstance(release, str):
        raise ConfigurationError("project.release and project.release_year are required.")
    if not isinstance(coding_system, str) or not coding_system:
        raise ConfigurationError("project.coding_system must be a non-empty string.")

    return AppConfig(
        root=root,
        release=release,
        release_year=release_year,
        coding_system=coding_system,
        data_dir=data_dir.resolve(),
        allowed_hosts=allowed_hosts,
        request_timeout_seconds=limits["request_timeout_seconds"],
        download_retries=limits["download_retries"],
        max_download_bytes=limits["max_download_bytes"],
        max_archive_members=limits["max_archive_members"],
        max_archive_uncompressed_bytes=limits["max_archive_uncompressed_bytes"],
        max_archive_member_bytes=limits["max_archive_member_bytes"],
        max_compression_ratio=limits["max_compression_ratio"],
        sources=sources,
    )


def validate_source_url(url: str, allowed_hosts: frozenset[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise ConfigurationError(f"Source URL must use HTTPS and include a host: {url!r}.")
    if parsed.hostname.lower() not in allowed_hosts:
        raise ConfigurationError(
            f"Source host {parsed.hostname!r} is not in the configured allowlist."
        )
    if parsed.username or parsed.password:
        raise ConfigurationError("Credentials are not allowed in source URLs.")

