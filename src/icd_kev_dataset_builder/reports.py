from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import AppConfig
from .dedupe import DuplicateIndex
from .errors import ReviewError
from .sources import atomic_write_bytes, atomic_write_json
from .state import connect_state

SPLITS = ("train", "validation", "test")


def output_dir(config: AppConfig) -> Path:
    return config.data_dir / "output"


def quota_report(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT split, outcome, COALESCE(target_code, ''), status, COUNT(*) FROM slots "
        "GROUP BY split, outcome, target_code, status"
    ).fetchall()
    cells: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for split, outcome, code, status, count in rows:
        cells[(split, outcome, code)][status] += count
    by_split: dict[str, dict[str, int]] = {s: {"planned_quota": 0, "accepted": 0, "shortfall": 0} for s in SPLITS}
    shortfalls = []
    for (split, outcome, code), counts in sorted(cells.items()):
        total = sum(counts.values())
        accepted = counts["accepted"]
        by_split[split]["planned_quota"] += total
        by_split[split]["accepted"] += accepted
        by_split[split]["shortfall"] += total - accepted
        if accepted < total:
            shortfalls.append(
                {"split": split, "outcome": outcome, "target_code": code or None,
                 "quota": total, "accepted": accepted, "exhausted": counts["exhausted"],
                 "unfinished": total - accepted - counts["exhausted"]}
            )
    totals = {k: sum(v[k] for v in by_split.values()) for k in ("planned_quota", "accepted", "shortfall")}
    return {"by_split": by_split, "totals": totals, "shortfalls": shortfalls,
            "complete": totals["shortfall"] == 0}


def integrity_report(connection: sqlite3.Connection, manifest_hash: str) -> dict[str, Any]:
    records = [json.loads(r[0]) for r in connection.execute("SELECT record_json FROM accepted_records")]
    family_splits: dict[str, set[str]] = defaultdict(set)
    index = DuplicateIndex()
    problems: list[str] = []
    for record in records:
        family_splits[record["scenario_family_id"]].add(record["split"])
        if record["source_manifest_hash"] != manifest_hash:
            problems.append(f"{record['record_id']}: manifest hash mismatch")
        if record["provenance"]["human_review"] != "not_run":
            problems.append(f"{record['record_id']}: unexpected human_review value")
    for family, splits in family_splits.items():
        if len(splits) > 1:
            problems.append(f"family {family} spans splits {sorted(splits)}")
    cross_split = 0
    for record in sorted(records, key=lambda r: r["record_id"]):
        for dup in index.find(record["record_id"], record["medical_description"]):
            if dup.other_split != record["split"]:
                cross_split += 1
                problems.append(f"{record['record_id']}: {dup.kind} duplicate across splits")
        index.add(record["record_id"], record["split"], record["medical_description"])
    return {
        "records_checked": len(records),
        "family_split_violations": sum(1 for s in family_splits.values() if len(s) > 1),
        "cross_split_duplicates": cross_split,
        "problems": problems,
        "passed": not problems,
        "semantic_duplicate_check": "NOT_RUN (lexical exact and word 3-gram near-duplicate only)",
    }


def run_audit(state_path: Path, manifest_hash: str) -> dict[str, Any]:
    connection = connect_state(state_path)
    try:
        return {
            "quota": quota_report(connection),
            "integrity": integrity_report(connection, manifest_hash),
            "candidate_status": dict(
                connection.execute("SELECT status, COUNT(*) FROM candidates GROUP BY status").fetchall()
            ),
            "attempt_outcomes": dict(
                connection.execute("SELECT outcome, COUNT(*) FROM attempts GROUP BY outcome").fetchall()
            ),
        }
    finally:
        connection.close()


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records).encode("utf-8")


def _unsupported_branches(config: AppConfig) -> Any:
    path = config.reference_dir / "kb_audit.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["unsupported_branches"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Cannot read unsupported-branch report from {path}: {exc}") from exc


def finalize(
    config: AppConfig,
    state_path: Path,
    manifest_hash: str,
    *,
    sample_size: int = 25,
    review_deployments: dict[str, str] | None = None,
) -> dict[str, Any]:
    audit = run_audit(state_path, manifest_hash)
    if not audit["integrity"]["passed"]:
        raise ReviewError("Integrity audit failed; refusing to finalize: " + "; ".join(audit["integrity"]["problems"][:5]))
    connection = connect_state(state_path)
    try:
        records = [json.loads(r[0]) for r in connection.execute(
            "SELECT record_json FROM accepted_records ORDER BY slot_id")]
        dup_rows = connection.execute(
            "SELECT other_slot_id, kind, score, decision FROM duplicate_decisions").fetchall()
        rejected = connection.execute(
            "SELECT reasons_json FROM candidates WHERE status = 'rejected'").fetchall()
        generation_deployments = sorted({r["provenance"]["generation_deployment"] for r in records})
        review_deps = sorted({r["provenance"].get("review_deployment") or "" for r in records} - {""})
    finally:
        connection.close()
    out = output_dir(config)
    counts = {}
    for split in SPLITS:
        subset = sorted((r for r in records if r["split"] == split), key=lambda r: r["record_id"])
        atomic_write_bytes(out / f"{split}.master.jsonl", _jsonl(subset))
        counts[split] = len(subset)

    reasons: Counter[str] = Counter()
    for (blob,) in rejected:
        for reason in json.loads(blob):
            reasons[reason.split(":")[0]] += 1
    atomic_write_json(out / "quota_report.json", audit["quota"])
    atomic_write_json(out / "audit_report.json", audit)
    atomic_write_json(out / "duplicate_report.json", {
        "decisions": [{"other_slot_id": a, "kind": b, "score": c, "decision": d} for a, b, c, d in dup_rows],
        "rejection_reasons": dict(reasons),
    })

    # Stratified human-review sample: round-robin over (split, outcome) in hash order.
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[(record["split"], record["outcome"])].append(record)
    for members in strata.values():
        members.sort(key=lambda r: hashlib.sha256(r["record_id"].encode()).hexdigest())
    sample: list[dict[str, Any]] = []
    while len(sample) < min(sample_size, len(records)):
        progressed = False
        for key in sorted(strata):
            if strata[key] and len(sample) < sample_size:
                sample.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    atomic_write_bytes(out / "human_review_sample.jsonl", _jsonl(sample))

    complete = audit["quota"]["complete"]
    card = {
        "title": "Synthetic ICD-10-CM outpatient notes (pilot scope)",
        "synthetic_only": True,
        "contains_real_patient_data": False,
        "coding_system": config.coding_system,
        "release": config.release,
        "source_manifest_hash": manifest_hash,
        "status": "COMPLETE_QUOTA" if complete else "PARTIAL",
        "record_counts": counts,
        "quota": audit["quota"]["totals"],
        "generation_deployments": generation_deployments,
        "review_deployments": review_deps,
        "review_note": "Model review is a blind re-code; the target-conditioned consistency review was NOT_RUN. "
        "The reviewer may share the generating deployment, so independence is not established.",
        "human_review": "NOT_RUN for every record; a stratified sample file is provided for manual review.",
        "deployment_identity": "The provider-side model identity behind a deployment name is not independently verified.",
        "unsupported_branches": _unsupported_branches(config),
        "scope": "Short English synthetic outpatient notes about one focused diagnosis; "
        "not full-encounter, inpatient, or reimbursement coding.",
        "prompt_versions": {"generation": "generate.v1", "review": "review.v1"},
        "limitations": [
            "Narratives are model-generated and carry model bias; no clinical validation was performed.",
            "Coding labels are derived from the official release tabular source, not from the narratives.",
            "Near-duplicate detection is lexical; semantic duplicate detection was NOT_RUN.",
        ],
        "audit_passed": audit["integrity"]["passed"],
    }
    atomic_write_json(out / "dataset_card.json", card)
    atomic_write_json(out / "config_snapshot.json", {
        "coding_system": config.coding_system, "release": config.release,
        "source_ids": sorted(config.sources), "secrets": "redacted: no environment values are recorded",
    })
    return {"counts": counts, "status": card["status"], "output_dir": str(out), "audit": audit}
