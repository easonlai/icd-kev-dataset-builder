from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .config import AppConfig
from .errors import ReviewError
from .kev_native import convert
from .reference import load_pilot_context
from .reports import SPLITS, _jsonl, output_dir
from .sources import atomic_write_bytes, atomic_write_json, read_source_lock

SPECIALS = ("NO_MATCH", "OUT_OF_SCOPE", "NEEDS_REVIEW")
STOP = "STOP"
QUESTION = "Which option best matches the clinical documentation at this point in the code hierarchy?"


def _rank(task_id: str, key: str) -> str:
    return hashlib.sha256(f"{task_id}|{key}".encode()).hexdigest()


def _option(node: dict[str, Any]) -> dict[str, str]:
    return {"id": node["node_id"], "code": node["code"] or "", "description": node["description"]}


def _special(name: str) -> dict[str, str]:
    return {"id": name, "code": "", "description": name.replace("_", " ").lower()}


def _select_options(
    task_id: str, correct: str, candidates: list[dict[str, Any]], specials: tuple[str, ...], count: int
) -> list[dict[str, str]]:
    """Always keep the correct option; fill the rest with deterministically ordered distractors."""
    room = max(count - len(specials), 1)
    pool = [c for c in candidates if c["node_id"] != correct]
    pool.sort(key=lambda c: _rank(task_id, c["node_id"]))
    chosen = [c for c in candidates if c["node_id"] == correct] + pool[: room - (correct in {c["node_id"] for c in candidates})]
    options = [_option(c) for c in chosen] + [_special(s) for s in specials]
    options.sort(key=lambda o: _rank(task_id, o["id"]))
    return options


def _compact(options: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"id": o["id"], "code": o["code"]} for o in options]


def _task(
    record: dict[str, Any], kind: str, index: int, parent: dict[str, Any] | None,
    options: list[dict[str, str]], label: str,
) -> dict[str, Any]:
    task_id = f"{record['record_id']}:{kind}:{index}"
    return {
        "task_id": task_id,
        "case_id": record["record_id"],
        "split": record["split"],
        "task_type": kind,
        "input": {
            "description": record["medical_description"],
            "current_parent": _option(parent) if parent else {"id": "ROOT", "code": "", "description": "ICD-10-CM root"},
            "question": QUESTION,
            "options": options,
        },
        "label": label,
    }


def routing_tasks(
    record: dict[str, Any], ctx: dict[str, Any], count: int, max_no_match: int = 2
) -> list[dict[str, Any]]:
    """Every routing task offers the same special options so option sets never reveal the label.

    STOP is offered only where the current parent is itself a complete, reportable code.
    """
    tasks: list[dict[str, Any]] = []
    roots = ctx["roots"]
    if record["outcome"] != "MAPPED":
        task_id = f"{record['record_id']}:routing:0"
        # All chapters are distractors, so a pilot-only option list cannot hint at the outcome.
        tasks.append(_task(record, "routing", 0, None,
                           _select_options(task_id, "", roots, SPECIALS, count), record["outcome"]))
        return tasks
    entry = ctx["pilot"].get(record["target_code"])
    if entry is None:
        raise ReviewError(f"No pilot context for {record['target_code']}; run `kb pilot`.")
    lineage = entry["lineage"]
    levels = [roots] + entry["options_by_level"]
    parents: list[dict[str, Any] | None] = [None] + lineage
    for step, node in enumerate(lineage):
        task_id = f"{record['record_id']}:routing:{step}"
        tasks.append(_task(record, "routing", step, parents[step],
                           _select_options(task_id, node["node_id"], levels[step], SPECIALS, count),
                           node["node_id"]))
    stop_options = (STOP,) + SPECIALS
    step = len(lineage)
    task_id = f"{record['record_id']}:routing:{step}"
    tasks.append(_task(record, "routing", step, lineage[-1],
                       _select_options(task_id, "", [], stop_options, count), STOP))

    # Wrong-branch negatives: a wrong parent that has children, or a sibling leaf code that is
    # a valid stop point but is not supported by the note. Both are labelled NO_MATCH.
    negatives: list[tuple[str, dict[str, Any], list[dict[str, Any]], tuple[str, ...]]] = []
    for number, wrong in enumerate(entry["wrong_parents"]):
        negatives.append((f"wrong_parent:{number}", wrong["node"], wrong["children"], SPECIALS))
    if len(lineage) >= 2:
        target_id = lineage[-1]["node_id"]
        for sibling in entry["options_by_level"][-2]:
            if sibling["node_id"] != target_id and sibling["is_reportable"] and not sibling["has_children"]:
                negatives.append((f"wrong_leaf:{sibling['node_id']}", sibling, [], stop_options))
    negatives.sort(key=lambda n: _rank(record["record_id"], n[0]))
    for number, (key, parent, children, specials) in enumerate(negatives[:max_no_match]):
        task_id = f"{record['record_id']}:routing_no_match:{number}"
        tasks.append(_task(record, "routing_no_match", number, parent,
                           _select_options(task_id, "", children, specials, count), "NO_MATCH"))
    return tasks


def evidence_task(record: dict[str, Any]) -> dict[str, Any] | None:
    if record["outcome"] != "MAPPED":
        return None
    return {
        "task_id": f"{record['record_id']}:evidence:0",
        "case_id": record["record_id"],
        "split": record["split"],
        "task_type": "evidence_selection",
        "input": {
            "description": record["medical_description"],
            "sentences": [
                {"sentence_id": s["sentence_id"],
                 "text": record["medical_description"][s["start"]:s["end"]]}
                for s in record["sentences"]
            ],
            "question": "Which sentences support the diagnosis that is the focus of this note?",
        },
        "label": {"focus_sentence_ids": record["focus_sentence_ids"],
                  "supporting_quotes": record["supporting_evidence"]},
    }


def export_kev(config: AppConfig, *, option_count: int = 24, max_no_match: int = 2) -> dict[str, Any]:
    ctx = load_pilot_context(config)
    lock = read_source_lock(config)
    if ctx["source_manifest_hash"] != lock["manifest_hash"]:
        raise ReviewError("Pilot context is stale for the current source lock; rerun `kb pilot`.")
    out = output_dir(config)
    kev = out / "kev"
    manifest: dict[str, Any] = {"task_counts": {}, "option_count": option_count, "max_no_match_per_case": max_no_match}
    all_described: list[dict[str, Any]] = []
    for split in SPLITS:
        path = out / f"{split}.master.jsonl"
        if not path.is_file():
            raise ReviewError(f"Missing {path}; run `icd-data finalize` first.")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        labelled: list[dict[str, Any]] = []
        for record in records:
            labelled += routing_tasks(record, ctx, option_count, max_no_match)
            ev = evidence_task(record)
            if ev:
                labelled.append(ev)
        variants = {
            "described": labelled,
            "compact": [
                {**t, "input": {**t["input"], "options": _compact(t["input"]["options"])}}
                if "options" in t["input"] else t
                for t in labelled
            ],
        }
        for variant, tasks in variants.items():
            inference = [{k: v for k, v in t.items() if k != "label"} for t in tasks]
            atomic_write_bytes(kev / f"{split}.{variant}.labelled.jsonl", _jsonl(tasks))
            atomic_write_bytes(kev / f"{split}.{variant}.inference.jsonl", _jsonl(inference))
        manifest["task_counts"][split] = dict(Counter(t["task_type"] for t in labelled))
        all_described += labelled
    parts, problems = convert(all_described)
    for name, rows in parts.items():
        atomic_write_bytes(kev / "native" / f"{name}.jsonl", _jsonl(rows))
    manifest["native"] = {
        "format": "Kev labelled JSONL (kev skills/kev-finetune/references/data-format.md); described variant only",
        "records": {name: len(rows) for name, rows in parts.items()},
        "skipped_over_limit": problems,
    }
    manifest.update({
        "release": config.release,
        "source_manifest_hash": lock["manifest_hash"],
        "compatibility": "Generic files: best-effort shapes. native/: Kev labelled JSONL checked offline with Kev's split_data.py only; no training run, no tokenizer check.",
        "notes": "Inference files omit labels. Gold lineage and evidence never appear in task inputs.",
    })
    atomic_write_json(kev / "manifest.json", manifest)
    return manifest
