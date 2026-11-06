from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

NEAR_DUPLICATE_THRESHOLD = 0.8


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", text.lower())).strip()


def _shingles(text: str, n: int = 3) -> frozenset[str]:
    words = normalise(text).split()
    if len(words) < n:
        return frozenset({" ".join(words)})
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass(frozen=True)
class Duplicate:
    other_slot_id: str
    kind: str  # exact | near
    score: float
    other_split: str


class DuplicateIndex:
    """Exact-hash and word-3-gram Jaccard near-duplicate detection over accepted notes."""

    def __init__(self, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> None:
        self.threshold = threshold
        self._hashes: dict[str, str] = {}
        self._shingles: dict[str, frozenset[str]] = {}
        self._splits: dict[str, str] = {}
        self._inverted: dict[str, set[str]] = defaultdict(set)

    def add(self, slot_id: str, split: str, text: str) -> None:
        self._hashes[hashlib.sha256(normalise(text).encode()).hexdigest()] = slot_id
        shingles = _shingles(text)
        self._shingles[slot_id] = shingles
        self._splits[slot_id] = split
        for shingle in shingles:
            self._inverted[shingle].add(slot_id)

    def find(self, slot_id: str, text: str) -> list[Duplicate]:
        found: list[Duplicate] = []
        exact = self._hashes.get(hashlib.sha256(normalise(text).encode()).hexdigest())
        if exact is not None and exact != slot_id:
            found.append(Duplicate(exact, "exact", 1.0, self._splits[exact]))
        shingles = _shingles(text)
        candidates = {o for s in shingles for o in self._inverted.get(s, ())} - {slot_id}
        for other in sorted(candidates):
            score = jaccard(shingles, self._shingles[other])
            if score >= self.threshold and not any(d.other_slot_id == other for d in found):
                found.append(Duplicate(other, "near", round(score, 4), self._splits[other]))
        return found
