"""Schema adapter from the builder's labelled tasks to Kev's native labelled JSONL.

Target format (kev skills/kev-finetune/references/data-format.md, kev@2074cc4): one System One request per
line, ``{"state", "questions": {qid: {"type", "instructions", "criteria", "label"}}}``.
Routing tasks become ``choice`` questions; evidence tasks become one ``noul`` question per sentence.
Only the described variant is used because Kev binds option descriptions, not opaque IDs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import ReviewError

MAX_OPTIONS = 255
STATE_CHARS_LIMIT = 1400
BRANCH_TOKEN_LIMIT = 1024
REQUEST_TOKEN_LIMIT = 2048
CHARS_PER_TOKEN = 3.5  # conservative estimate; the real tokenizer check runs in Kev (kev_modal.py::validate)

SPECIAL_DESCRIPTIONS = {
    "STOP": "The current position is already the complete, most specific code the note supports.",
    "NO_MATCH": "None of the listed options is supported by the note; the current branch is wrong.",
    "OUT_OF_SCOPE": "The documented diagnosis lies outside the supported scope.",
    "NEEDS_REVIEW": "The documentation is contradictory or unresolved, so a human must review it.",
}
EVIDENCE_INSTRUCTION = (
    "Is the quoted sentence part of the documentation that supports the diagnosis that is the focus of this note? "
    "Sentence: \"{text}\""
)


def _option_text(option: dict[str, Any]) -> str:
    if option["id"] in SPECIAL_DESCRIPTIONS:
        return SPECIAL_DESCRIPTIONS[option["id"]]
    code = option.get("code") or ""
    return f"{code}: {option['description']}" if code else option["description"]


def _parent_text(parent: dict[str, Any]) -> str:
    code = parent.get("code") or ""
    return f"{parent['description']} ({code})" if code else parent["description"]


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def to_native(task: dict[str, Any]) -> dict[str, Any]:
    """One labelled builder task -> one Kev record. Raises on anything Kev would reject."""
    data = task["input"]
    state = data["description"]
    if not state.strip():
        raise ReviewError(f"{task['task_id']}: empty description")
    if task["task_type"] in ("routing", "routing_no_match"):
        options = data["options"]
        if not 1 <= len(options) <= MAX_OPTIONS:
            raise ReviewError(f"{task['task_id']}: {len(options)} options; Kev allows 1..{MAX_OPTIONS}")
        criteria = {o["id"]: _option_text(o) for o in options}
        if len(criteria) != len(options) or task["label"] not in criteria:
            raise ReviewError(f"{task['task_id']}: duplicate option IDs or a label outside the options")
        instructions = f"{data['question']} Current position in the ICD-10-CM hierarchy: {_parent_text(data['current_parent'])}."
        # The one-hot target trains exactly like the hard label, but Kev's augment() and none_pair() then only
        # permute options. Without it they delete the gold option and relabel the question "none of the above",
        # which would corrupt OUT_OF_SCOPE / NEEDS_REVIEW / NO_MATCH, since this data has its own special options.
        questions = {"route": {"type": "choice", "instructions": instructions, "criteria": criteria,
                               "label": task["label"], "target": {task["label"]: 1.0}}}
    elif task["task_type"] == "evidence_selection":
        focus = set(task["label"]["focus_sentence_ids"])
        questions = {
            f"sentence_{n}": {
                "type": "noul",
                "instructions": EVIDENCE_INSTRUCTION.format(text=s["text"]),
                "label": s["sentence_id"] in focus,
            }
            for n, s in enumerate(data["sentences"], 1)
        }
    else:
        raise ReviewError(f"{task['task_id']}: unsupported task type {task['task_type']!r}")
    return {
        "state": state,
        "questions": questions,
        "_meta": {"task_id": task["task_id"], "case_id": task["case_id"], "task_type": task["task_type"], "split": task["split"]},
    }


def request_problems(record: dict[str, Any]) -> list[str]:
    """Estimated limit checks; the exact tokenizer check happens in Kev."""
    problems = []
    if len(record["state"]) > STATE_CHARS_LIMIT:
        problems.append(f"state has {len(record['state'])} characters (limit about {STATE_CHARS_LIMIT})")
    total = estimate_tokens(record["state"])
    for qid, q in record["questions"].items():
        branch = estimate_tokens(json.dumps({k: q.get(k) for k in ("instructions", "criteria")}, ensure_ascii=False))
        total += branch
        if branch > BRANCH_TOKEN_LIMIT:
            problems.append(f"question {qid!r} is about {branch} tokens (limit {BRANCH_TOKEN_LIMIT})")
    if total > REQUEST_TOKEN_LIMIT:
        problems.append(f"request is about {total} tokens (limit {REQUEST_TOKEN_LIMIT})")
    return problems


def partition(split: str, case_id: str) -> str:
    """Kev wants train / calibration / development. Our validation split is halved by case; test stays test."""
    if split == "train":
        return "train"
    if split == "test":
        return "test"
    if split == "validation":
        return "calibration" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0 else "development"
    raise ReviewError(f"Unknown split {split!r}")


def convert(tasks: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Group converted records by Kev partition; oversize records are reported, never silently shortened."""
    parts: dict[str, list[dict[str, Any]]] = {"train": [], "calibration": [], "development": [], "test": []}
    problems: list[str] = []
    for task in tasks:
        record = to_native(task)
        issues = request_problems(record)
        if issues:
            problems.append(f"{task['task_id']}: " + "; ".join(issues))
            continue
        parts[partition(task["split"], task["case_id"])].append(record)
    return parts, problems
