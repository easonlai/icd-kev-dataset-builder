from __future__ import annotations

import sqlite3
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from icd_kev_dataset_builder.errors import PlanningError
from icd_kev_dataset_builder.kb import PILOT_TARGET_CODES
from icd_kev_dataset_builder.planner import (
    NOTE_STYLES,
    build_plan,
    check_plan_integrity,
    expected_totals,
)
from icd_kev_dataset_builder.schemas import MasterRecord
from icd_kev_dataset_builder.state import initialise_state, read_status

TEXT = "Follow-up for diabetes. Glucose remains high."


def record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "record_id": "r1",
        "split": "train",
        "scenario_family_id": "train-E1165-000",
        "synthetic": True,
        "coding_system": "ICD-10-CM",
        "coding_release": "FIXTURE",
        "source_manifest_hash": "h",
        "encounter_setting": "outpatient",
        "medical_description": TEXT,
        "sentences": [
            {"sentence_id": "s1", "start": 0, "end": 23},
            {"sentence_id": "s2", "start": 24, "end": 45},
        ],
        "focus_sentence_ids": ["s1"],
        "outcome": "MAPPED",
        "target_code": "E11.65",
        "lineage": [{"code": "E11.65", "description": "x", "kind": "diag"}],
        "supporting_evidence": [{"sentence_id": "s2", "quote": "Glucose remains high"}],
        "evidence_summary": "Fixture summary.",
        "provenance": {
            "generation_deployment": "d",
            "prompt_version": "v1",
            "context_packet_hash": "c",
            "attempt_ids": ["a1"],
            "generated_at": "2026-01-01T00:00:00+00:00",
        },
    }
    base.update(overrides)
    return base


class PlannerTests(unittest.TestCase):
    def test_quota_totals_match_the_agreed_4500(self) -> None:
        self.assertEqual(
            expected_totals(), {"train": 3000, "validation": 500, "test": 1000, "all": 4500}
        )
        self.assertEqual(len(build_plan(seed=1)), 4500)

    def test_plan_is_reproducible_and_seed_sensitive(self) -> None:
        self.assertEqual(build_plan(7), build_plan(7))
        self.assertNotEqual(build_plan(7), build_plan(8))

    def test_integrity_passes_and_every_split_uses_every_style(self) -> None:
        slots = build_plan(3)
        check_plan_integrity(slots)
        for split in ("train", "validation", "test"):
            self.assertEqual({s.note_style for s in slots if s.split == split}, set(NOTE_STYLES))

    def test_family_spanning_splits_is_rejected(self) -> None:
        slots = build_plan(3)
        slots[0] = replace(slots[0], split="test")
        with self.assertRaisesRegex(PlanningError, "span more than one split"):
            check_plan_integrity(slots)

    def test_missing_slot_breaks_quota(self) -> None:
        with self.assertRaisesRegex(PlanningError, "quota"):
            check_plan_integrity(build_plan(3)[:-1])

    def test_each_code_has_exact_quota_per_split(self) -> None:
        slots = build_plan(3)
        counts = {
            (s.split, s.target_code) for s in slots if s.target_code == PILOT_TARGET_CODES[0]
        }
        self.assertEqual(counts, {("train", "E10.9"), ("validation", "E10.9"), ("test", "E10.9")})


class StateTests(unittest.TestCase):
    def kwargs(self, **overrides: object) -> dict[str, object]:
        base = {"manifest_hash": "m1", "release": "FY2027", "seed": 5, "family_size": 2}
        base.update(overrides)
        return base

    def test_create_then_resume_does_not_duplicate_or_change_splits(self) -> None:
        slots = build_plan(5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.sqlite"
            self.assertEqual(initialise_state(path, slots, **self.kwargs()), "created")
            before = read_status(path)
            self.assertEqual(initialise_state(path, slots, **self.kwargs()), "resumed")
            after = read_status(path)
        self.assertEqual(before, after)
        self.assertEqual(sum(sum(v.values()) for v in after["slots_by_split"].values()), 4500)
        self.assertEqual(after["family_split_violations"], 0)

    def test_resume_refuses_a_different_release_manifest_or_seed(self) -> None:
        slots = build_plan(5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.sqlite"
            initialise_state(path, slots, **self.kwargs())
            for override in ({"manifest_hash": "m2"}, {"release": "FY2028"}):
                with self.assertRaisesRegex(PlanningError, "does not match"):
                    initialise_state(path, slots, **self.kwargs(**override))
            with self.assertRaisesRegex(PlanningError, "does not match"):
                initialise_state(path, build_plan(6), **self.kwargs(seed=6))

    def test_invalid_plan_leaves_no_state_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.sqlite"
            with self.assertRaises(PlanningError):
                initialise_state(path, build_plan(5)[:-1], **self.kwargs())
            self.assertFalse(path.exists())

    def test_database_blocks_duplicate_acceptance(self) -> None:
        slots = build_plan(5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.sqlite"
            initialise_state(path, slots, **self.kwargs())
            connection = sqlite3.connect(path)
            insert = "INSERT INTO accepted_records VALUES (?, '{}', 't')"
            connection.execute(insert, (slots[0].slot_id,))
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(insert, (slots[0].slot_id,))
            connection.close()

    def test_missing_state_is_an_explicit_error(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PlanningError, "run `icd-data plan`"):
                read_status(Path(tmp) / "none.sqlite")


class MasterRecordTests(unittest.TestCase):
    def test_valid_mapped_record(self) -> None:
        self.assertEqual(MasterRecord(**record()).target_code, "E11.65")

    def test_rejects_non_synthetic_and_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            MasterRecord(**record(synthetic=False))
        with self.assertRaises(ValidationError):
            MasterRecord(**record(confidence=0.9))

    def test_rejects_evidence_that_is_not_an_exact_span(self) -> None:
        bad = [{"sentence_id": "s2", "quote": "Glucose is high"}]
        with self.assertRaisesRegex(ValidationError, "exact span"):
            MasterRecord(**record(supporting_evidence=bad))

    def test_mapped_requires_pilot_code_and_lineage_ending_at_target(self) -> None:
        with self.assertRaisesRegex(ValidationError, "pilot list"):
            MasterRecord(**record(target_code="Z00.0"))
        wrong = [{"code": "E11", "description": "x", "kind": "diag"}]
        with self.assertRaisesRegex(ValidationError, "end at the target"):
            MasterRecord(**record(lineage=wrong))

    def test_out_of_scope_needs_nonpilot_outside_code(self) -> None:
        base = {"outcome": "OUT_OF_SCOPE", "target_code": None, "lineage": []}
        with self.assertRaisesRegex(ValidationError, "non-pilot"):
            MasterRecord(**record(**base, outside_code="E11.65"))
        self.assertEqual(
            MasterRecord(**record(**base, outside_code="E66.9")).outside_code, "E66.9"
        )

    def test_needs_review_must_explain_and_carry_no_code(self) -> None:
        base = {"outcome": "NEEDS_REVIEW", "target_code": None, "lineage": [], "supporting_evidence": []}
        with self.assertRaisesRegex(ValidationError, "unresolved"):
            MasterRecord(**record(**base))
        ok = MasterRecord(**record(**base, documentation_conflicts=["Type 1 vs type 2."]))
        self.assertEqual(ok.provenance.human_review.value, "not_run")

    def test_sentence_span_errors(self) -> None:
        overlap = [
            {"sentence_id": "s1", "start": 0, "end": 23},
            {"sentence_id": "s2", "start": 10, "end": 45},
        ]
        with self.assertRaisesRegex(ValidationError, "overlaps"):
            MasterRecord(**record(sentences=overlap))
        with self.assertRaisesRegex(ValidationError, "not defined"):
            MasterRecord(**record(focus_sentence_ids=["s9"]))


if __name__ == "__main__":
    unittest.main()
