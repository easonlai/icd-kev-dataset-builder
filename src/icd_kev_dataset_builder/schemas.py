from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .kb import PILOT_TARGET_CODES


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Outcome(StrEnum):
    MAPPED = "MAPPED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sentence(_Strict):
    sentence_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class EvidenceQuote(_Strict):
    sentence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class LineageNode(_Strict):
    code: str
    description: str
    kind: str


class Provenance(_Strict):
    generation_deployment: str = Field(min_length=1)
    review_deployment: str | None = None
    prompt_version: str = Field(min_length=1)
    context_packet_hash: str = Field(min_length=1)
    attempt_ids: list[str] = Field(min_length=1)
    generated_at: str
    local_review: ReviewStatus = ReviewStatus.PENDING
    model_review: ReviewStatus = ReviewStatus.NOT_RUN
    # Models may never mark a record human-reviewed; only a person can, outside this pipeline.
    human_review: ReviewStatus = ReviewStatus.NOT_RUN
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)


class MasterRecord(_Strict):
    record_id: str = Field(min_length=1)
    split: Split
    scenario_family_id: str = Field(min_length=1)
    synthetic: Literal[True]
    coding_system: str
    coding_release: str
    source_manifest_hash: str = Field(min_length=1)
    encounter_setting: Literal["outpatient"]
    medical_description: str = Field(min_length=1)
    sentences: list[Sentence] = Field(min_length=1)
    focus_sentence_ids: list[str] = Field(min_length=1)

    outcome: Outcome
    target_code: str | None = None
    outside_code: str | None = None
    lineage: list[LineageNode] = Field(default_factory=list)
    supporting_evidence: list[EvidenceQuote] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    documentation_conflicts: list[str] = Field(default_factory=list)
    additional_coding_notes: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    evidence_summary: str = Field(min_length=1)

    provenance: Provenance

    @model_validator(mode="after")
    def _check_consistency(self) -> MasterRecord:
        text = self.medical_description
        ids = [sentence.sentence_id for sentence in self.sentences]
        if len(set(ids)) != len(ids):
            raise ValueError("sentence IDs must be unique")
        previous_end = 0
        for sentence in self.sentences:
            if sentence.end > len(text) or sentence.start >= sentence.end:
                raise ValueError(f"sentence {sentence.sentence_id} has an invalid span")
            if sentence.start < previous_end:
                raise ValueError(f"sentence {sentence.sentence_id} overlaps an earlier sentence")
            previous_end = sentence.end
        unknown_focus = set(self.focus_sentence_ids) - set(ids)
        if unknown_focus:
            raise ValueError(f"focus sentence IDs are not defined: {sorted(unknown_focus)}")

        spans = {s.sentence_id: text[s.start : s.end] for s in self.sentences}
        for item in self.supporting_evidence:
            if item.sentence_id not in spans:
                raise ValueError(f"evidence cites unknown sentence {item.sentence_id}")
            if item.quote not in spans[item.sentence_id]:
                raise ValueError(
                    f"evidence quote is not an exact span of sentence {item.sentence_id}"
                )

        if self.outcome is Outcome.MAPPED:
            if self.target_code not in PILOT_TARGET_CODES:
                raise ValueError("MAPPED records need a target_code from the pilot list")
            if self.outside_code is not None:
                raise ValueError("MAPPED records must not carry an outside_code")
            if not self.lineage or self.lineage[-1].code != self.target_code:
                raise ValueError("MAPPED lineage must come from the tabular source and end at the target")
            if not self.supporting_evidence:
                raise ValueError("MAPPED records need supporting evidence")
        elif self.outcome is Outcome.OUT_OF_SCOPE:
            if self.target_code is not None:
                raise ValueError("OUT_OF_SCOPE records must not carry a pilot target_code")
            if not self.outside_code or self.outside_code in PILOT_TARGET_CODES:
                raise ValueError("OUT_OF_SCOPE records need a verified non-pilot outside_code")
        else:
            if self.target_code is not None or self.outside_code is not None:
                raise ValueError("NEEDS_REVIEW records must not carry a code")
            if not (self.missing_information or self.documentation_conflicts):
                raise ValueError("NEEDS_REVIEW records must state what is unresolved")
        return self
