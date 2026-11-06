from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .errors import GenerationError, LLMConfigurationError, LLMRequestError
from .llm import LLMClient, LLMSettings
from .reference import ContextPacket, ReferenceBank, leaked_codes
from .schemas import LineageNode, MasterRecord
from .state import connect_state

PROMPT_VERSION = "generate.v1"
DEFAULT_LIMIT = 5
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_ATTEMPTS = 3
MAX_CONSECUTIVE_FAILED_BATCHES = 3
MAX_SENTENCES = 8
MAX_SENTENCE_CHARS = 400
_PHI_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:MRN|SSN)\b", re.IGNORECASE),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_prompt(root: Path, name: str) -> str:
    path = root / "prompts" / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GenerationError(f"Cannot read prompt file {path}: {exc}") from exc


@dataclass(frozen=True)
class GenerationOptions:
    limit: int | None = DEFAULT_LIMIT
    batch_size: int = DEFAULT_BATCH_SIZE
    max_requests: int = 20
    max_total_tokens: int | None = None
    max_attempts_per_slot: int = DEFAULT_MAX_ATTEMPTS
    dry_run: bool = False
    outcome: str | None = None


@dataclass
class GenerationSummary:
    selected_slots: int = 0
    requests: int = 0
    candidates: int = 0
    failed_slots: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stopped_reason: str = "completed"
    dry_run: bool = False
    failures: dict[str, int] = field(default_factory=dict)

    def note_failure(self, reason: str) -> None:
        self.failed_slots += 1
        self.failures[reason] = self.failures.get(reason, 0) + 1


@dataclass(frozen=True)
class BatchItem:
    slot_id: str
    spec: dict[str, Any]
    packet: ContextPacket


def parse_model_json(text: str) -> dict[str, Any]:
    """Strict JSON parse with a plain-text fallback that strips a Markdown code fence."""
    candidate = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def build_user_prompt(items: list[BatchItem]) -> tuple[str, dict[str, str]]:
    """Return the request body and the item-key to slot mapping. Slot IDs are never sent."""
    from .scenarios import LENGTH_TEXT, STYLE_TEXT, TRAIT_TEXT

    references: dict[str, Any] = {}
    reference_by_family: dict[str, str] = {}
    groups: dict[str, str] = {}
    requested = []
    key_to_slot: dict[str, str] = {}
    for number, item in enumerate(items, start=1):
        key = f"item_{number}"
        key_to_slot[key] = item.slot_id
        family = item.spec["family_id"]
        if family not in reference_by_family:
            reference_by_family[family] = f"reference_{len(reference_by_family) + 1}"
            references[reference_by_family[family]] = item.packet.prompt_dict()
        groups.setdefault(family, f"group_{len(groups) + 1}")
        entry: dict[str, Any] = {
            "item": key,
            "scenario_group": groups[family],
            "reference": reference_by_family[family],
            "note_style": STYLE_TEXT[item.spec["note_style"]],
            "length": LENGTH_TEXT[item.spec["length"]],
            "requirement": TRAIT_TEXT[item.spec["trait"]],
        }
        if item.packet.scenario_hint:
            entry["contradiction"] = item.packet.scenario_hint
        requested.append(entry)
    body = json.dumps(
        {"requested_items": requested, "references": references}, indent=1, ensure_ascii=False
    )
    return body, key_to_slot


def _free_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("evidence_summary",):
        parts.append(str(item.get(key, "")))
    for key in ("missing_information", "documentation_conflicts", "additional_coding_notes"):
        parts.extend(str(v) for v in item.get(key, []) or [])
    return " ".join(parts)


def build_record(
    *,
    slot_id: str,
    spec: dict[str, Any],
    packet: ContextPacket,
    bank: ReferenceBank,
    item: dict[str, Any],
    deployment: str,
    attempt_id: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> MasterRecord:
    """Validate one model item and assemble a master record. Raises ValueError with reasons."""
    sentences = item.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("sentences must be a non-empty list")
    if len(sentences) > MAX_SENTENCES or any(
        not isinstance(s, str) or not s.strip() or len(s) > MAX_SENTENCE_CHARS for s in sentences
    ):
        raise ValueError("sentences must be 1-8 non-empty strings within the length limit")
    cleaned = [s.strip() for s in sentences]
    text = " ".join(cleaned)
    spans, cursor = [], 0
    for number, sentence in enumerate(cleaned, start=1):
        spans.append({"sentence_id": f"s{number}", "start": cursor, "end": cursor + len(sentence)})
        cursor += len(sentence) + 1

    focus_indexes = item.get("focus_sentence_indexes") or []
    if not focus_indexes or any(
        not isinstance(i, int) or not 0 <= i < len(cleaned) for i in focus_indexes
    ):
        raise ValueError("focus_sentence_indexes must reference existing sentences")
    evidence = []
    for entry in item.get("supporting_evidence", []) or []:
        index = entry.get("sentence_index") if isinstance(entry, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(cleaned):
            raise ValueError("supporting_evidence cites a missing sentence")
        evidence.append({"sentence_id": f"s{index + 1}", "quote": str(entry.get("quote", ""))})

    narrative_text = text + " " + _free_text(item)
    # Any pilot code in the narrative is treated as leakage, not only the slot's own answer.
    leaks = leaked_codes(narrative_text, set(packet.forbidden_codes) | set(bank.context["pilot"]))
    if leaks:
        raise ValueError(f"answer-code leakage in generated text: {leaks}")
    if any(p.search(narrative_text) for p in _PHI_PATTERNS):
        raise ValueError("generated text contains an identifier-like pattern")
    if sum(1 for ch in narrative_text if ord(ch) > 127) > 0.05 * len(narrative_text):
        raise ValueError("generated text does not look like English")

    outcome = spec["outcome"]
    lineage: list[LineageNode] = []
    references = list(packet.source_refs)
    target_code = spec["target_code"]
    outside_code = None
    if outcome == "MAPPED":
        steps = bank.context["pilot"][target_code]["lineage"]
        lineage = [LineageNode(**{k: s[k] for k in ("code", "description", "kind")}) for s in steps]
        references += [f"tabular:{s['node_id']}" for s in steps] + [f"codebook:{target_code}"]
    elif outcome == "OUT_OF_SCOPE":
        outside_code = packet.outside_code
        references.append(f"codebook:{outside_code}")
    try:
        return MasterRecord(
            record_id="rec-" + hashlib.sha256(slot_id.encode()).hexdigest()[:12],
            split=spec["split"],
            scenario_family_id=spec["family_id"],
            synthetic=True,
            coding_system=bank.config.coding_system,
            coding_release=bank.config.release,
            source_manifest_hash=bank.manifest_hash,
            encounter_setting="outpatient",
            medical_description=text,
            sentences=spans,
            focus_sentence_ids=[f"s{i + 1}" for i in sorted(set(focus_indexes))],
            outcome=outcome,
            target_code=target_code if outcome == "MAPPED" else None,
            outside_code=outside_code,
            lineage=lineage,
            supporting_evidence=evidence,
            missing_information=[str(v) for v in item.get("missing_information", []) or []],
            documentation_conflicts=[str(v) for v in item.get("documentation_conflicts", []) or []],
            additional_coding_notes=[str(v) for v in item.get("additional_coding_notes", []) or []],
            source_references=references,
            evidence_summary=str(item.get("evidence_summary", "")).strip(),
            provenance={
                "generation_deployment": deployment,
                "prompt_version": PROMPT_VERSION,
                "context_packet_hash": packet.digest,
                "attempt_ids": [str(attempt_id)],
                "generated_at": utc_now(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
    except ValidationError as exc:
        reasons = "; ".join(sorted({err["msg"] for err in exc.errors()}))
        raise ValueError(f"schema validation failed: {reasons}") from exc


def acquire_run_lock(state_path: Path) -> Any:
    """Exclusive advisory lock so two runs cannot abandon each other's open attempts."""
    import fcntl

    handle = open(state_path.with_suffix(".lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise GenerationError("Another generate/review run is using this state; wait for it to finish.") from exc
    return handle


def recover_interrupted(connection: sqlite3.Connection, max_attempts: int) -> int:
    """Mark attempts left open by a crash as uncertain and release their slots."""
    with connection:
        cursor = connection.execute(
            "UPDATE attempts SET outcome = 'uncertain_abandoned', finished_at = ? "
            "WHERE finished_at IS NULL",
            (utc_now(),),
        )
        connection.execute(
            "UPDATE slots SET status = CASE WHEN (SELECT COUNT(*) FROM attempts a "
            "WHERE a.slot_id = slots.slot_id) >= ? THEN 'exhausted' ELSE 'planned' END "
            "WHERE status = 'in_progress' AND NOT EXISTS (SELECT 1 FROM candidates c "
            "WHERE c.slot_id = slots.slot_id AND c.status = 'pending_review')",
            (max_attempts,),
        )
    return cursor.rowcount


def _select_slots(connection: sqlite3.Connection, options: GenerationOptions) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT s.slot_id, s.spec_json, (SELECT COUNT(*) FROM attempts a WHERE a.slot_id = s.slot_id) "
        "FROM slots s WHERE s.status = 'planned'" + (" AND s.outcome = ?" if options.outcome else ""),
        (options.outcome,) if options.outcome else (),
    ).fetchall()
    eligible = [
        (attempts, hashlib.sha256(slot_id.encode()).hexdigest(), json.loads(spec))
        for slot_id, spec, attempts in rows
        if attempts < options.max_attempts_per_slot
    ]
    eligible.sort(key=lambda row: (row[0], row[1]))
    chosen = [spec for _, _, spec in eligible]
    return chosen if options.limit is None else chosen[: options.limit]


def _batches(specs: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Group slots by split only, keeping family members adjacent, so small limits still fill batches."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(spec["split"], []).append(spec)
    batches = []
    for split in sorted(grouped):
        members = sorted(grouped[split], key=lambda s: (s["family_id"], s["slot_id"]))
        for start in range(0, len(members), size):
            batches.append(members[start : start + size])
    return batches


def _release_slot(connection: sqlite3.Connection, slot_id: str, max_attempts: int) -> None:
    attempts = connection.execute(
        "SELECT COUNT(*) FROM attempts WHERE slot_id = ?", (slot_id,)
    ).fetchone()[0]
    connection.execute(
        "UPDATE slots SET status = ? WHERE slot_id = ?",
        ("exhausted" if attempts >= max_attempts else "planned", slot_id),
    )


def run_generation(
    *,
    state_path: Path,
    bank: ReferenceBank,
    client: LLMClient | None,
    settings: LLMSettings | None,
    system_prompt: str,
    options: GenerationOptions,
) -> GenerationSummary:
    """Generate candidate records for a bounded number of planned slots."""
    if options.batch_size < 1:
        raise GenerationError("batch_size must be at least 1.")
    if not options.dry_run and (client is None or settings is None):
        raise GenerationError("A configured model client is required unless --dry-run is used.")
    summary = GenerationSummary(dry_run=options.dry_run)
    connection = connect_state(state_path)
    lock = None if options.dry_run else acquire_run_lock(state_path)
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        if meta.get("source_manifest_hash") != bank.manifest_hash or meta.get("release") != bank.config.release:
            raise GenerationError("Plan state is bound to a different release/manifest; refusing to mix.")
        if not options.dry_run:
            recover_interrupted(connection, options.max_attempts_per_slot)
        specs = _select_slots(connection, options)
        summary.selected_slots = len(specs)
        packet_cache: dict[str, ContextPacket] = {}
        consecutive_failures = 0
        for batch in _batches(specs, options.batch_size):
            if summary.requests >= options.max_requests and not options.dry_run:
                summary.stopped_reason = "max_requests reached"
                break
            used = summary.prompt_tokens + summary.completion_tokens
            if options.max_total_tokens is not None and used >= options.max_total_tokens:
                summary.stopped_reason = "max_total_tokens reached"
                break
            items = []
            for spec in batch:
                packet = packet_cache.get(spec["family_id"])
                if packet is None:
                    packet = packet_cache[spec["family_id"]] = bank.packet(spec)
                items.append(BatchItem(spec["slot_id"], spec, packet))
            body, key_to_slot = build_user_prompt(items)
            forbidden = {c for item in items for c in item.packet.forbidden_codes}
            leaks = leaked_codes(body + system_prompt, forbidden)
            if leaks:
                raise GenerationError(f"Answer-code leakage in a generation request: {leaks}")
            if options.dry_run:
                summary.requests += 1
                continue
            outcome = _run_batch(
                connection, client, settings, system_prompt, body, key_to_slot, items, bank, options, summary
            )
            consecutive_failures = 0 if outcome else consecutive_failures + 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILED_BATCHES:
                summary.stopped_reason = "stopped after repeated failed batches"
                break
    finally:
        connection.close()
        if lock is not None:
            lock.close()
    return summary


def _run_batch(
    connection: sqlite3.Connection,
    client: LLMClient | None,
    settings: LLMSettings | None,
    system_prompt: str,
    body: str,
    key_to_slot: dict[str, str],
    items: list[BatchItem],
    bank: ReferenceBank,
    options: GenerationOptions,
    summary: GenerationSummary,
) -> bool:
    assert client is not None and settings is not None
    attempt_ids: dict[str, int] = {}
    with connection:
        for item in items:
            cursor = connection.execute(
                "INSERT INTO attempts (slot_id, started_at, outcome) VALUES (?, ?, 'uncertain')",
                (item.slot_id, utc_now()),
            )
            attempt_ids[item.slot_id] = int(cursor.lastrowid)
            connection.execute("UPDATE slots SET status = 'in_progress' WHERE slot_id = ?", (item.slot_id,))

    def fail_all(reason: str, detail: dict[str, Any]) -> bool:
        with connection:
            for item in items:
                _finish_attempt(connection, attempt_ids[item.slot_id], reason, detail)
                _release_slot(connection, item.slot_id, options.max_attempts_per_slot)
        for _ in items:
            summary.note_failure(reason)
        return False

    summary.requests += 1
    try:
        result = client.complete(settings.generation_deployment, system_prompt, body)
    except LLMConfigurationError:
        fail_all("configuration_error", {})
        raise
    except LLMRequestError as exc:
        return fail_all("request_error", {"error": type(exc).__name__})
    summary.prompt_tokens += result.prompt_tokens
    summary.completion_tokens += result.completion_tokens
    detail = {
        "request_id": result.request_id,
        "deployment": result.deployment,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "request_attempts": result.attempts,
        "batch_size": len(items),
        "prompt_version": PROMPT_VERSION,
    }
    if result.finish_reason != "stop":
        return fail_all(f"finish_reason_{result.finish_reason}", detail)
    try:
        parsed = parse_model_json(result.text)
        by_key = {str(e.get("item")): e for e in parsed.get("items", []) if isinstance(e, dict)}
    except (ValueError, AttributeError) as exc:
        return fail_all("malformed_json", {**detail, "error": str(exc)})

    any_success = False
    share_prompt = result.prompt_tokens // len(items)
    share_completion = result.completion_tokens // len(items)
    for key, slot_id in key_to_slot.items():
        item = next(i for i in items if i.slot_id == slot_id)
        attempt_id = attempt_ids[slot_id]
        entry = by_key.get(key)
        try:
            if entry is None:
                raise ValueError("item missing from the model response")
            record = build_record(
                slot_id=slot_id,
                spec=item.spec,
                packet=item.packet,
                bank=bank,
                item=entry,
                deployment=result.deployment,
                attempt_id=attempt_id,
                prompt_tokens=share_prompt,
                completion_tokens=share_completion,
            )
        except ValueError as exc:
            with connection:
                _finish_attempt(connection, attempt_id, "invalid", {**detail, "error": str(exc)})
                _release_slot(connection, slot_id, options.max_attempts_per_slot)
            summary.note_failure("invalid")
            continue
        with connection:
            connection.execute(
                "INSERT INTO candidates (slot_id, attempt_id, record_json, created_at) VALUES (?, ?, ?, ?)",
                (slot_id, attempt_id, record.model_dump_json(), utc_now()),
            )
            _finish_attempt(connection, attempt_id, "generated", detail)
        summary.candidates += 1
        any_success = True
    return any_success


def _finish_attempt(connection: sqlite3.Connection, attempt_id: int, outcome: str, detail: dict[str, Any]) -> None:
    connection.execute(
        "UPDATE attempts SET finished_at = ?, outcome = ?, detail_json = ? WHERE attempt_id = ?",
        (utc_now(), outcome, json.dumps(detail, sort_keys=True), attempt_id),
    )
