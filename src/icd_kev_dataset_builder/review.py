from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dedupe import DuplicateIndex
from .errors import LLMConfigurationError, LLMRefusalError, LLMRequestError, ReviewError
from .generation import MAX_CONSECUTIVE_FAILED_BATCHES, acquire_run_lock, parse_model_json, utc_now
from .llm import LLMClient, LLMSettings
from .reference import ReferenceBank
from .schemas import MasterRecord
from .state import connect_state

REVIEW_PROMPT_VERSION = "review.v1"


@dataclass
class ReviewSummary:
    reviewed: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    stopped_reason: str = "completed"

    def reject(self, reasons: list[str]) -> None:
        self.rejected += 1
        for reason in reasons:
            key = reason.split(":")[0]
            self.reasons[key] = self.reasons.get(key, 0) + 1


def reference_bank_text(bank: ReferenceBank, codes: list[str]) -> str:
    return "\n".join(f"{code}: {bank.codebook[code]}" for code in codes)


def build_review_prompt(note: str, bank_text: str) -> str:
    return json.dumps({"reference_bank": bank_text, "note": note}, ensure_ascii=False, indent=1)


def evaluate_agreement(record: MasterRecord, review: dict[str, Any], bank: ReferenceBank) -> list[str]:
    """Compare the blind re-code with the stored label. Returns disagreement reasons."""
    outcome = str(review.get("outcome", "")).upper()
    code = review.get("code")
    code = str(code).strip().upper() if code else None
    if outcome not in {"MAPPED", "OUT_OF_SCOPE", "NEEDS_REVIEW"}:
        return ["review_malformed: unknown outcome"]
    expected = record.outcome.value
    if outcome != expected:
        return [f"review_disagreement: reviewer said {outcome}, stored {expected}"]
    if expected == "MAPPED":
        if code != record.target_code:
            return ["review_disagreement: reviewer chose a different code"]
    elif expected == "OUT_OF_SCOPE":
        if not code or code not in bank.codebook:
            return ["review_disagreement: reviewer code is not a valid complete code"]
        pilot = bank.context["pilot"]
        if code in pilot:
            return ["review_disagreement: reviewer chose a pilot code"]
        # Same-category comparison is a heuristic for agreement, not a hierarchy inference.
        if code != record.outside_code and code[:3] != (record.outside_code or "")[:3]:
            return ["review_disagreement: reviewer chose an unrelated outside code"]
    return []


def local_checks(record: MasterRecord) -> list[str]:
    reasons = []
    if len(record.medical_description.split()) < 8:
        reasons.append("local_check: note is too short")
    if not record.source_references:
        reasons.append("local_check: no source references")
    if record.outcome.value == "MAPPED" and f"codebook:{record.target_code}" not in record.source_references:
        reasons.append("local_check: target code is not referenced to the codebook")
    return reasons


def load_accepted_index(connection: sqlite3.Connection) -> DuplicateIndex:
    index = DuplicateIndex()
    for slot_id, record_json in connection.execute("SELECT slot_id, record_json FROM accepted_records"):
        record = json.loads(record_json)
        index.add(slot_id, record["split"], record["medical_description"])
    return index


def run_review(
    *,
    state_path: Path,
    bank: ReferenceBank,
    client: LLMClient,
    settings: LLMSettings,
    system_prompt: str,
    max_requests: int = 50,
    max_attempts_per_slot: int = 3,
) -> ReviewSummary:
    summary = ReviewSummary()
    connection = connect_state(state_path)
    lock = acquire_run_lock(state_path)
    bank_codes = sorted(bank.context["pilot"])
    bank_text = reference_bank_text(bank, bank_codes)
    try:
        index = load_accepted_index(connection)
        pending = connection.execute(
            "SELECT candidate_id, slot_id, record_json FROM candidates "
            "WHERE status = 'pending_review' ORDER BY candidate_id"
        ).fetchall()
        consecutive_errors = 0
        for candidate_id, slot_id, record_json in pending:
            if summary.reviewed + summary.errors >= max_requests:
                summary.stopped_reason = "max_requests reached"
                break
            record = MasterRecord.model_validate_json(record_json)
            reasons = local_checks(record)
            slot_split = record.split.value
            duplicates = index.find(slot_id, record.medical_description)
            for dup in duplicates:
                decision = "rejected"
                if dup.other_split != slot_split:
                    reasons.append(f"cross_split_leakage: {dup.kind} match with {dup.other_slot_id}")
                else:
                    reasons.append(f"duplicate: {dup.kind} match with {dup.other_slot_id}")
                with connection:
                    connection.execute(
                        "INSERT INTO duplicate_decisions (candidate_id, other_slot_id, kind, score, decision) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (candidate_id, dup.other_slot_id, dup.kind, dup.score, decision),
                    )
            review: dict[str, Any] | None = None
            if not reasons:
                try:
                    result = client.complete(
                        settings.review_deployment,
                        system_prompt,
                        build_review_prompt(record.medical_description, bank_text),
                    )
                except LLMConfigurationError:
                    raise
                except LLMRefusalError:
                    # A refusal is permanent for this note; leaving it pending would re-bill every run.
                    summary.reviewed += 1
                    reasons.append("review_refused: reviewer refused or content filter blocked the note")
                    _reject(connection, candidate_id, slot_id, reasons, None, max_attempts_per_slot)
                    summary.reject(reasons)
                    continue
                except LLMRequestError:
                    summary.errors += 1
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_FAILED_BATCHES:
                        summary.stopped_reason = "stopped after repeated review errors"
                        break
                    continue
                consecutive_errors = 0
                summary.reviewed += 1
                review = {
                    "request_id": result.request_id,
                    "deployment": result.deployment,
                    "finish_reason": result.finish_reason,
                    "prompt_version": REVIEW_PROMPT_VERSION,
                }
                if result.finish_reason != "stop":
                    reasons.append(f"review_malformed: finish_reason {result.finish_reason}")
                else:
                    try:
                        parsed = parse_model_json(result.text)
                        review["result"] = parsed
                        reasons += evaluate_agreement(record, parsed, bank)
                    except ValueError as exc:
                        reasons.append(f"review_malformed: {exc}")
            if reasons:
                _reject(connection, candidate_id, slot_id, reasons, review, max_attempts_per_slot)
                summary.reject(reasons)
                continue
            payload = record.model_dump(mode="json")
            payload["provenance"].update(
                local_review="passed",
                model_review="passed",
                review_deployment=review["deployment"] if review else None,
            )
            final = MasterRecord.model_validate(payload)
            with connection:
                connection.execute(
                    "INSERT INTO accepted_records (slot_id, record_json, accepted_at) VALUES (?, ?, ?)",
                    (slot_id, final.model_dump_json(), utc_now()),
                )
                connection.execute(
                    "UPDATE candidates SET status = 'accepted', review_json = ? WHERE candidate_id = ?",
                    (json.dumps(review), candidate_id),
                )
                connection.execute("UPDATE slots SET status = 'accepted' WHERE slot_id = ?", (slot_id,))
            index.add(slot_id, slot_split, final.medical_description)
            summary.accepted += 1
    finally:
        connection.close()
        lock.close()
    return summary


def _reject(
    connection: sqlite3.Connection,
    candidate_id: int,
    slot_id: str,
    reasons: list[str],
    review: dict[str, Any] | None,
    max_attempts: int,
) -> None:
    attempts = connection.execute("SELECT COUNT(*) FROM attempts WHERE slot_id = ?", (slot_id,)).fetchone()[0]
    with connection:
        connection.execute(
            "UPDATE candidates SET status = 'rejected', reasons_json = ?, review_json = ? WHERE candidate_id = ?",
            (json.dumps(reasons), json.dumps(review) if review else None, candidate_id),
        )
        connection.execute(
            "UPDATE slots SET status = ? WHERE slot_id = ?",
            ("exhausted" if attempts >= max_attempts else "planned", slot_id),
        )


__all__ = ["ReviewError", "run_review", "ReviewSummary"]
