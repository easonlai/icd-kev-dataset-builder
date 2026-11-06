from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .errors import PlanningError
from .planner import PlannedSlot, check_plan_integrity, plan_hash

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE families (
    family_id TEXT PRIMARY KEY,
    split TEXT NOT NULL CHECK (split IN ('train','validation','test'))
);
CREATE TABLE slots (
    slot_id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES families(family_id),
    split TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('MAPPED','OUT_OF_SCOPE','NEEDS_REVIEW')),
    target_code TEXT,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned'
        CHECK (status IN ('planned','in_progress','accepted','exhausted'))
);
CREATE TABLE attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id TEXT NOT NULL REFERENCES slots(slot_id),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    outcome TEXT NOT NULL DEFAULT 'uncertain',
    detail_json TEXT
);
CREATE TABLE accepted_records (
    slot_id TEXT PRIMARY KEY REFERENCES slots(slot_id),
    record_json TEXT NOT NULL,
    accepted_at TEXT NOT NULL
);
"""


# Created idempotently so plan state made by an earlier milestone is upgraded in place.
EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id TEXT NOT NULL REFERENCES slots(slot_id),
    attempt_id INTEGER NOT NULL REFERENCES attempts(attempt_id),
    record_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_review'
        CHECK (status IN ('pending_review','accepted','rejected')),
    reasons_json TEXT NOT NULL DEFAULT '[]',
    review_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS duplicate_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
    other_slot_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    score REAL NOT NULL,
    decision TEXT NOT NULL
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def connect_state(path: Path) -> sqlite3.Connection:
    """Open existing plan state for read/write work; the caller closes it."""
    if not path.is_file():
        raise PlanningError(f"No plan state at {path}; run `icd-data plan` first.")
    connection = _connect(path)
    with connection:
        connection.executescript(EXTRA_SCHEMA)
    return connection


def initialise_state(
    path: Path,
    slots: list[PlannedSlot],
    *,
    manifest_hash: str,
    release: str,
    seed: int,
    family_size: int,
) -> str:
    """Create the plan once, or verify an existing one. Returns 'created' or 'resumed'."""
    check_plan_integrity(slots)
    expected = {
        "source_manifest_hash": manifest_hash,
        "release": release,
        "seed": str(seed),
        "family_size": str(family_size),
        "plan_hash": plan_hash(slots),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.is_file()
    with closing(_connect(path)) as connection:
        if existing:
            stored = dict(connection.execute("SELECT key, value FROM meta"))
            mismatched = [k for k, v in expected.items() if stored.get(k) != v]
            if mismatched:
                raise PlanningError(
                    "Existing plan state does not match the current inputs ("
                    + ", ".join(mismatched)
                    + f"). Refusing to mix releases or change splits; use a new state file: {path}"
                )
            return "resumed"
        try:
            with connection:
                connection.executescript("BEGIN;" + SCHEMA + EXTRA_SCHEMA)
                connection.executemany("INSERT INTO meta VALUES (?, ?)", expected.items())
                families = sorted({(s.family_id, s.split) for s in slots})
                connection.executemany("INSERT INTO families VALUES (?, ?)", families)
                connection.executemany(
                    "INSERT INTO slots (slot_id, family_id, split, outcome, target_code, spec_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            s.slot_id,
                            s.family_id,
                            s.split,
                            s.outcome,
                            s.target_code,
                            json.dumps(s.spec(), sort_keys=True),
                        )
                        for s in slots
                    ],
                )
        except Exception:
            connection.close()
            path.unlink(missing_ok=True)
            raise
    return "created"


def read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlanningError(f"No plan state at {path}; run `icd-data plan` first.")
    with closing(_connect(path)) as connection:
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        by_split: dict[str, dict[str, int]] = {}
        for split, status, count in connection.execute(
            "SELECT split, status, COUNT(*) FROM slots GROUP BY split, status"
        ):
            by_split.setdefault(split, {})[status] = count
        outcomes = {
            outcome: count
            for outcome, count in connection.execute(
                "SELECT outcome, COUNT(*) FROM slots GROUP BY outcome"
            )
        }
        split_leaks = connection.execute(
            "SELECT COUNT(*) FROM slots s JOIN families f USING (family_id) WHERE s.split != f.split"
        ).fetchone()[0]
        accepted = connection.execute("SELECT COUNT(*) FROM accepted_records").fetchone()[0]
        attempts = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    return {
        "meta": meta,
        "slots_by_split": by_split,
        "slots_by_outcome": outcomes,
        "family_split_violations": split_leaks,
        "accepted_records": accepted,
        "attempts": attempts,
    }
