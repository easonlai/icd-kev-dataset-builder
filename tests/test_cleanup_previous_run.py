import fcntl
from pathlib import Path

import pytest

from scripts.cleanup_previous_run import archive_run, configured_data_dir


def test_archive_run_moves_only_state_and_output(tmp_path: Path):
    data_dir = tmp_path / "data"
    state = data_dir / "state"
    output = data_dir / "output"
    sources = data_dir / "sources"
    reference = data_dir / "reference"
    for directory in (state, output, sources, reference):
        directory.mkdir(parents=True)
    (state / "plan.sqlite").write_text("state")
    (state / "plan.lock").write_text("")
    (output / "train.jsonl").write_text("{}\n")
    (sources / "source.zip").write_text("source")
    (reference / "codebook.json").write_text("{}")

    archive = archive_run(data_dir, "trial")

    assert archive == data_dir / "archive" / "trial"
    assert (archive / "state" / "plan.sqlite").read_text() == "state"
    assert (archive / "output" / "train.jsonl").read_text() == "{}\n"
    assert not (state / "plan.sqlite").exists()
    assert (state / "plan.lock").is_file()
    assert (sources / "source.zip").is_file()
    assert (reference / "codebook.json").is_file()


def test_archive_run_uses_unique_archive_directory(tmp_path: Path):
    data_dir = tmp_path / "data"
    (data_dir / "output").mkdir(parents=True)
    (data_dir / "output" / "x.jsonl").write_text("{}\n")
    (data_dir / "archive" / "trial").mkdir(parents=True)

    archive = archive_run(data_dir, "trial")

    assert archive == data_dir / "archive" / "trial-01"
    assert (archive / "output" / "x.jsonl").is_file()


def test_archive_run_refuses_symlinked_output(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (data_dir / "output").symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        archive_run(data_dir, "trial")


def test_archive_run_refuses_active_pipeline_lock(tmp_path: Path):
    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)
    (state / "plan.sqlite").write_text("state")

    with (state / "plan.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active"):
            archive_run(data_dir, "trial")

    assert (state / "plan.sqlite").is_file()


def test_configured_data_dir_must_stay_inside_project(tmp_path: Path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config" / "default.toml").write_text('[project]\ndata_dir = "../outside"\n')

    with pytest.raises(RuntimeError, match="outside"):
        configured_data_dir(root)
