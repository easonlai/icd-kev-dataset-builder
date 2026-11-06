#!/usr/bin/env python3
"""Archive generated data from the previous run before starting a fresh run."""

from __future__ import annotations

import argparse
import fcntl
import shutil
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TextIO


def configured_data_dir(root: Path) -> Path:
    config_path = root / "config" / "default.toml"
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        configured = config["project"]["data_dir"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Cannot read project data directory from {config_path}: {exc}") from exc
    if not isinstance(configured, str) or not configured:
        raise RuntimeError("project.data_dir must be a non-empty path.")
    data_dir = Path(configured)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    data_dir = data_dir.resolve()
    if not data_dir.is_relative_to(root):
        raise RuntimeError(f"Refusing to clean data outside the project directory: {data_dir}")
    return data_dir


def _validate_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing to move a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Expected a directory, found another file type: {path}")


def _unique_archive_path(data_dir: Path, timestamp: str) -> Path:
    archive_root = data_dir / "archive"
    _validate_directory(archive_root)
    candidate = archive_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = archive_root / f"{timestamp}-{suffix:02d}"
        suffix += 1
    return candidate


def _state_payload(state_dir: Path) -> list[Path]:
    if not state_dir.exists():
        return []
    _validate_directory(state_dir)
    return [path for path in state_dir.iterdir() if path.name != "plan.lock"]


def archive_run(data_dir: Path, timestamp: str | None = None) -> Path | None:
    """Move state payload and output snapshots to a unique archive; preserve sources/reference."""
    data_dir = data_dir.resolve()
    state_dir = data_dir / "state"
    output_dir = data_dir / "output"
    _validate_directory(state_dir)
    _validate_directory(output_dir)
    payload = _state_payload(state_dir)
    if not payload and not output_dir.exists():
        return None

    lock_handle: TextIO | None = None
    try:
        if state_dir.exists():
            lock_handle = (state_dir / "plan.lock").open("a+")
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    "A generate/review command is active; wait for it to finish."
                ) from exc

        # Recheck after obtaining the run lock.
        payload = _state_payload(state_dir)
        _validate_directory(output_dir)
        if not payload and not output_dir.exists():
            return None

        stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not stamp or any(char not in allowed for char in stamp):
            raise RuntimeError(
                "Archive timestamp may contain only letters, digits, hyphens, and underscores."
            )
        archive_dir = _unique_archive_path(data_dir, stamp)
        archive_dir.mkdir(parents=True)

        if payload:
            archived_state = archive_dir / "state"
            archived_state.mkdir()
            for path in payload:
                shutil.move(str(path), str(archived_state / path.name))
        if output_dir.exists():
            shutil.move(str(output_dir), str(archive_dir / "output"))
        return archive_dir
    finally:
        if lock_handle is not None:
            lock_handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive data/state and data/output from the previous run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without changing files.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Proceed without the interactive CLEAN confirmation.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    try:
        if not (root / "pyproject.toml").is_file():
            raise RuntimeError(f"Not a project root: {root}")
        data_dir = configured_data_dir(root)
        state_dir = data_dir / "state"
        output_dir = data_dir / "output"
        _validate_directory(state_dir)
        _validate_directory(output_dir)
        targets = _state_payload(state_dir)
        if output_dir.exists():
            targets.append(output_dir)
        if not targets:
            print("No previous run state or output was found.")
            return 0

        print(
            "The following run data will be archived; "
            "source and reference data will not be changed:"
        )
        for target in targets:
            print(f"  {target}")
        if args.dry_run:
            print("Dry run only; no files were moved.")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    "Confirmation needs a terminal. Review with --dry-run, "
                    "then pass --yes to proceed."
                )
            if input("Type CLEAN to archive these run files: ") != "CLEAN":
                print("Cancelled; no files were moved.")
                return 1

        archive_dir = archive_run(data_dir)
        if archive_dir is None:
            print("No previous run state or output was found.")
        else:
            print(f"Archived previous run data to {archive_dir}")
            print("Start a fresh plan with: .venv/bin/python -m icd_kev_dataset_builder.cli plan")
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"clean-run error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
