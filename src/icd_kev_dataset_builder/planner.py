from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

from .errors import PlanningError
from .kb import PILOT_TARGET_CODES
from .schemas import Outcome, Split

# (mapped cases per target code, OUT_OF_SCOPE, NEEDS_REVIEW) per split.
QUOTAS: dict[Split, tuple[int, int, int]] = {
    Split.TRAIN: (50, 250, 250),
    Split.VALIDATION: (8, 50, 50),
    Split.TEST: (16, 100, 100),
}
NOTE_STYLES = (
    "soap_brief",
    "free_text_paragraph",
    "bulleted_problem_list",
    "telephone_followup",
    "abbreviation_heavy",
)
MAPPED_TRAITS = (
    "plain",
    "negation_of_related_condition",
    "history_vs_current",
    "irrelevant_distractor",
    "sparse_detail",
    "detailed_exam",
)
OUT_OF_SCOPE_TRAITS = ("nonpilot_diagnosis", "nonpilot_branch_of_pilot_condition")
NEEDS_REVIEW_TRAITS = ("documentation_conflict", "unresolved_detail_required_for_code")
LENGTHS = ("short", "medium")
DEFAULT_FAMILY_SIZE = 2


@dataclass(frozen=True)
class PlannedSlot:
    slot_id: str
    split: str
    family_id: str
    outcome: str
    target_code: str | None
    note_style: str
    trait: str
    length: str
    paraphrase_index: int

    def spec(self) -> dict[str, object]:
        return asdict(self)


def expected_totals() -> dict[str, int]:
    totals = {
        split.value: mapped * len(PILOT_TARGET_CODES) + oos + review
        for split, (mapped, oos, review) in QUOTAS.items()
    }
    totals["all"] = sum(totals.values())
    return totals


def _rng(seed: int, family_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{family_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_plan(seed: int, family_size: int = DEFAULT_FAMILY_SIZE) -> list[PlannedSlot]:
    """Plan every master-record slot. Pure and reproducible for a given seed."""
    if family_size < 1:
        raise PlanningError("family_size must be at least 1.")
    slots: list[PlannedSlot] = []
    for split, (mapped, oos, review) in QUOTAS.items():
        groups: list[tuple[str, str, str | None, int, tuple[str, ...]]] = []
        for code in PILOT_TARGET_CODES:
            groups.append((Outcome.MAPPED.value, code.replace(".", ""), code, mapped, MAPPED_TRAITS))
        groups.append((Outcome.OUT_OF_SCOPE.value, "OOS", None, oos, OUT_OF_SCOPE_TRAITS))
        groups.append((Outcome.NEEDS_REVIEW.value, "NR", None, review, NEEDS_REVIEW_TRAITS))

        style_cursor = 0
        for outcome, label, code, count, traits in groups:
            for family_no, first in enumerate(range(0, count, family_size)):
                family_id = f"{split.value}-{label}-{family_no:03d}"
                rng = _rng(seed, family_id)
                trait = traits[rng.randrange(len(traits))]
                length = LENGTHS[rng.randrange(len(LENGTHS))]
                # Round-robin across the split so every style appears in every split.
                style = NOTE_STYLES[style_cursor % len(NOTE_STYLES)]
                style_cursor += 1
                for offset in range(min(family_size, count - first)):
                    slots.append(
                        PlannedSlot(
                            slot_id=f"{family_id}-{offset}",
                            split=split.value,
                            family_id=family_id,
                            outcome=outcome,
                            target_code=code,
                            note_style=style,
                            trait=trait,
                            length=length,
                            paraphrase_index=offset,
                        )
                    )
    return slots


def plan_hash(slots: list[PlannedSlot]) -> str:
    payload = json.dumps([slot.spec() for slot in slots], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_plan_integrity(slots: list[PlannedSlot]) -> None:
    """Fail loudly if quotas, uniqueness, or family/split isolation are violated."""
    ids = [slot.slot_id for slot in slots]
    if len(set(ids)) != len(ids):
        raise PlanningError("Plan contains duplicate slot IDs.")
    family_splits: dict[str, set[str]] = {}
    for slot in slots:
        family_splits.setdefault(slot.family_id, set()).add(slot.split)
    leaking = sorted(f for f, splits in family_splits.items() if len(splits) > 1)
    if leaking:
        raise PlanningError(f"Scenario families span more than one split: {leaking[:5]}")

    for split, (mapped, oos, review) in QUOTAS.items():
        in_split = [slot for slot in slots if slot.split == split.value]
        for code in PILOT_TARGET_CODES:
            got = sum(1 for s in in_split if s.target_code == code)
            if got != mapped:
                raise PlanningError(f"{split.value}/{code}: planned {got}, quota {mapped}.")
        for outcome, quota in ((Outcome.OUT_OF_SCOPE, oos), (Outcome.NEEDS_REVIEW, review)):
            got = sum(1 for s in in_split if s.outcome == outcome.value)
            if got != quota:
                raise PlanningError(f"{split.value}/{outcome.value}: planned {got}, quota {quota}.")
        styles = {s.note_style for s in in_split}
        if styles != set(NOTE_STYLES):
            raise PlanningError(f"{split.value} does not use every note style: {sorted(styles)}")
