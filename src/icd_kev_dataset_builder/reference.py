from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .errors import KnowledgeBaseError
from .scenarios import NEEDS_REVIEW_SCENARIOS, OUTSIDE_CODES_BY_TRAIT
from .sources import atomic_write_json, read_source_lock

CODE_PATTERN = re.compile(r"\b[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or the to with without not nec nos".split()
)
_USEFUL_NOTE_KINDS = ("excludes1", "excludes2", "includes", "codefirst", "codealso", "useadditionalcode")
_OUTPATIENT_RULE_QUERY = (
    "outpatient services uncertain diagnosis probable suspected questionable rule out working "
    "diagnosis do not code symptoms signs"
)
NOTE_CHAR_LIMIT = 240
CHUNK_CHAR_LIMIT = 420


def pilot_context_path(config: AppConfig) -> Path:
    return config.reference_dir / "pilot_context.json"


def sanitize_reference_text(text: str) -> str:
    """Replace code-like tokens so reference excerpts cannot leak an answer code."""
    return re.sub(r"\s+", " ", CODE_PATTERN.sub("[code]", text)).strip()


def leaked_codes(text: str, codes: Iterable[str]) -> list[str]:
    """Codes found in text, with or without the decimal point (handles codes such as I10)."""
    found = []
    for code in sorted({c for c in codes if c}):
        compact = code.replace(".", "")
        stem, _, tail = code.partition(".")
        pattern = re.escape(stem) + (r"\.?" + re.escape(tail) if tail else "")
        if re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text, flags=re.IGNORECASE) or (
            re.search(rf"(?<![A-Za-z0-9]){re.escape(compact)}(?![A-Za-z0-9])", text, re.IGNORECASE)
        ):
            found.append(code)
    return found


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """Transparent lexical retrieval. It does not guarantee every applicable rule is found."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings[token].append((doc_id, tf))
        self.size = len(texts)
        self.avg_length = (sum(self.lengths) / self.size) if self.size else 0.0

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = math.log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc_id, tf in posting:
                if allowed is not None and doc_id not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / (self.avg_length or 1))
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


def build_pilot_context(config: AppConfig, pilot_codes: tuple[str, ...]) -> dict[str, Any]:
    """Slice the verified hierarchy into the branches the pilot needs. Offline."""
    lock = read_source_lock(config)
    reference = config.reference_dir
    audit = _load_json(reference / "kb_audit.json")
    if audit.get("source_manifest_hash") != lock["manifest_hash"]:
        raise KnowledgeBaseError("Knowledge base is stale for the current source lock; rebuild it.")
    hierarchy = _load_json(reference / "hierarchy.json")
    if hierarchy.get("source_manifest_hash") != lock["manifest_hash"]:
        raise KnowledgeBaseError("Hierarchy is not bound to the current source manifest.")
    nodes = {node["node_id"]: node for node in hierarchy["nodes"]}
    lineage_ids = {
        step["node_id"] for code in pilot_codes for step in audit["pilot_target_paths"][code]
    }

    def option(node_id: str) -> dict[str, Any]:
        node = nodes[node_id]
        return {
            "node_id": node_id,
            "code": node["code"],
            "description": node["description"],
            "kind": node["kind"],
            "is_reportable": bool(node.get("is_reportable")),
            "has_children": bool(node["children"]),
            "contains_pilot": node_id in lineage_ids,
        }

    pilot: dict[str, Any] = {}
    for code in pilot_codes:
        lineage = audit["pilot_target_paths"][code]
        target = nodes[lineage[-1]["node_id"]]
        notes: list[dict[str, str]] = []
        seen: set[str] = set()
        for note in list(target["local_coding_notes"]) + list(target["inherited_coding_notes"]):
            text = note["text"][:NOTE_CHAR_LIMIT]
            if note["kind"] in _USEFUL_NOTE_KINDS and text not in seen:
                seen.add(text)
                notes.append({"kind": note["kind"], "text": text})
        options_by_level = [
            [option(child) for child in nodes[step["node_id"]]["children"]] for step in lineage
        ]
        parent = nodes[lineage[-2]["node_id"]] if len(lineage) > 1 else None
        distinctions = (
            [nodes[c]["description"] for c in parent["children"] if c != target["node_id"]][:10]
            if parent
            else []
        )
        wrong_parents = []
        for depth in range(2, len(lineage)):
            siblings = [
                c
                for c in nodes[lineage[depth - 1]["node_id"]]["children"]
                if c != lineage[depth]["node_id"] and nodes[c]["children"]
            ]
            for sibling_id in sorted(siblings)[:2]:
                wrong_parents.append(
                    {
                        "depth": depth,
                        "node": option(sibling_id),
                        "children": [option(c) for c in nodes[sibling_id]["children"]][:12],
                    }
                )
        pilot[code] = {
            "description": target["description"],
            "lineage": lineage,
            "coding_notes": notes[:8],
            "distinctions": distinctions,
            "options_by_level": options_by_level,
            "wrong_parents": wrong_parents,
        }
    document = {
        "schema_version": 1,
        "release": config.release,
        "source_manifest_hash": lock["manifest_hash"],
        "roots": [option(root) for root in hierarchy["roots"]],
        "pilot": pilot,
    }
    atomic_write_json(pilot_context_path(config), document)
    return document


def load_pilot_context(config: AppConfig) -> dict[str, Any]:
    path = pilot_context_path(config)
    if not path.is_file():
        raise KnowledgeBaseError(f"Pilot context missing: run `icd-data kb pilot` first ({path}).")
    document = _load_json(path)
    lock = read_source_lock(config)
    if document.get("source_manifest_hash") != lock["manifest_hash"]:
        raise KnowledgeBaseError("Pilot context is stale for the current source lock; rebuild it.")
    return document


@dataclass(frozen=True)
class ContextPacket:
    concept: str
    scenario_hint: str | None
    distinctions: list[str]
    coding_notes: list[str]
    index_entries: list[str]
    guideline_excerpts: list[str]
    general_rules: list[str]
    source_refs: list[str]
    forbidden_codes: tuple[str, ...] = field(default=(), repr=False)
    outside_code: str | None = None

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "focused_concept": self.concept,
            "scenario_hint": self.scenario_hint,
            "related_distinctions": self.distinctions,
            "coding_note_summaries": self.coding_notes,
            "alphabetic_index_entries": self.index_entries,
            "guideline_excerpts": self.guideline_excerpts,
            "general_outpatient_rules": self.general_rules,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.prompt_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReferenceBank:
    """Manifest-bound access to the codebook, pilot context, and BM25 reference retrieval."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.lock = read_source_lock(config)
        self.context = load_pilot_context(config)
        codebook = _load_json(config.reference_dir / "codebook.json")
        if codebook.get("source_manifest_hash") != self.lock["manifest_hash"]:
            raise KnowledgeBaseError("Codebook is not bound to the current source manifest.")
        self.codebook: dict[str, str] = {c["code"]: c["description"] for c in codebook["codes"]}
        self._chunks: list[dict[str, Any]] | None = None
        self._index: BM25Index | None = None
        self._general: list[tuple[str, str]] | None = None

    @property
    def manifest_hash(self) -> str:
        return str(self.lock["manifest_hash"])

    def verified_description(self, code: str) -> str:
        try:
            return self.codebook[code]
        except KeyError as exc:
            raise KnowledgeBaseError(
                f"{code} is not a complete code in the {self.config.release} code descriptions."
            ) from exc

    def outside_code_for(self, family_id: str, trait: str) -> str:
        choices = OUTSIDE_CODES_BY_TRAIT[trait]
        rng = random.Random(int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:8], "big"))
        code = choices[rng.randrange(len(choices))]
        self.verified_description(code)
        return code

    def needs_review_scenario(self, family_id: str) -> tuple[str, str]:
        rng = random.Random(int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:8], "big"))
        return NEEDS_REVIEW_SCENARIOS[rng.randrange(len(NEEDS_REVIEW_SCENARIOS))]

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        path = self.config.reference_dir / "reference_chunks.jsonl"
        chunks = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                chunk = json.loads(line)
                if chunk.get("source_manifest_hash") != self.manifest_hash:
                    raise KnowledgeBaseError("A reference chunk is not bound to the current manifest.")
                chunks.append(chunk)
        self._chunks = chunks
        self._index = BM25Index([c["text"] for c in chunks])

    def retrieve(self, query: str, source: str, k: int) -> list[dict[str, Any]]:
        self._ensure_index()
        assert self._index is not None and self._chunks is not None
        allowed = {i for i, c in enumerate(self._chunks) if c["source"] == source}
        return [self._chunks[i] for i, _ in self._index.search(query, k, allowed)]

    def general_rules(self) -> list[tuple[str, str]]:
        if self._general is None:
            self._general = [
                (c["chunk_id"], sanitize_reference_text(c["text"])[:CHUNK_CHAR_LIMIT])
                for c in self.retrieve(_OUTPATIENT_RULE_QUERY, "official_guidelines", 2)
            ]
        return self._general

    def packet(self, spec: dict[str, Any]) -> ContextPacket:
        """Build a code-free context packet for a planned slot; codes stay in forbidden_codes."""
        outcome, code = spec["outcome"], spec["target_code"]
        hint: str | None = None
        outside: str | None = None
        notes: list[str] = []
        distinctions: list[str] = []
        forbidden: list[str] = []
        if outcome == "MAPPED":
            entry = self.context["pilot"][code]
            concept = entry["description"]
            distinctions = [sanitize_reference_text(d) for d in entry["distinctions"]]
            notes = [sanitize_reference_text(n["text"]) for n in entry["coding_notes"]]
            forbidden = [code] + [s["code"] for s in entry["lineage"] if s["code"]]
        elif outcome == "OUT_OF_SCOPE":
            outside = self.outside_code_for(spec["family_id"], spec["trait"])
            concept = self.verified_description(outside)
            forbidden = [outside]
        else:
            concept, hint = self.needs_review_scenario(spec["family_id"])
        query = f"{concept} {hint or ''}"
        index_hits = self.retrieve(query, "alphabetic_index", 3)
        guideline_hits = self.retrieve(query, "official_guidelines", 2)
        general = self.general_rules()
        packet = ContextPacket(
            concept=sanitize_reference_text(concept),
            scenario_hint=hint,
            distinctions=distinctions,
            coding_notes=notes,
            index_entries=[sanitize_reference_text(c["text"])[:CHUNK_CHAR_LIMIT] for c in index_hits],
            guideline_excerpts=[sanitize_reference_text(c["text"])[:CHUNK_CHAR_LIMIT] for c in guideline_hits],
            general_rules=[text for _, text in general],
            source_refs=[c["chunk_id"] for c in index_hits + guideline_hits] + [cid for cid, _ in general],
            forbidden_codes=tuple(forbidden),
            outside_code=outside,
        )
        leaks = leaked_codes(json.dumps(packet.prompt_dict()), forbidden)
        if leaks:
            raise KnowledgeBaseError(f"Answer-code leakage in the context packet: {leaks}")
        return packet


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"Cannot read reference artifact {path}: {exc}") from exc
