"""Offline tests for generation, review, dedupe, reports, and Kev export.

All model calls use a scripted fake client; nothing here is live verification.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from icd_kev_dataset_builder.dedupe import DuplicateIndex
from icd_kev_dataset_builder.generation import (
    GenerationOptions,
    parse_model_json,
    recover_interrupted,
    run_generation,
)
from icd_kev_dataset_builder.kb import PILOT_TARGET_CODES
from icd_kev_dataset_builder.kev import routing_tasks
from icd_kev_dataset_builder.llm import LLMResult
from icd_kev_dataset_builder.planner import build_plan
from icd_kev_dataset_builder.reference import ContextPacket
from icd_kev_dataset_builder.reports import integrity_report, quota_report
from icd_kev_dataset_builder.review import evaluate_agreement, run_review
from icd_kev_dataset_builder.schemas import MasterRecord
from icd_kev_dataset_builder.state import connect_state, initialise_state
from icd_kev_dataset_builder.errors import GenerationError

MANIFEST = "m" * 64
OUTSIDE = "Z00.00"


def fake_lineage(code: str) -> list[dict[str, str]]:
    return [
        {"code": "", "description": "Chapter", "kind": "chapter", "node_id": "n1"},
        {"code": code, "description": f"Target {code}", "kind": "code", "node_id": "n2"},
    ]


class FakeBank:
    def __init__(self) -> None:
        self.config = SimpleNamespace(coding_system="ICD-10-CM", release="FY2027")
        self.manifest_hash = MANIFEST
        self.codebook = {c: f"Description {c}" for c in PILOT_TARGET_CODES} | {
            OUTSIDE: "Outside", "Z00.01": "Outside two"
        }
        self.context = {"pilot": {c: {"lineage": fake_lineage(c)} for c in PILOT_TARGET_CODES}}

    def packet(self, spec: dict[str, Any]) -> ContextPacket:
        return ContextPacket(
            concept="a concept",
            scenario_hint="a conflict" if spec["outcome"] == "NEEDS_REVIEW" else None,
            distinctions=[], coding_notes=[], index_entries=[], guideline_excerpts=[],
            general_rules=[], source_refs=["chunk:1"],
            forbidden_codes=(spec["target_code"],) if spec["target_code"] else (),
            outside_code=OUTSIDE if spec["outcome"] == "OUT_OF_SCOPE" else None,
        )


def result(text: str, finish: str = "stop") -> LLMResult:
    return LLMResult(text=text, finish_reason=finish, request_id="req", deployment="dep",
                     prompt_tokens=10, completion_tokens=20, attempts=1, started_at=0.0,
                     finished_at=1.0, parameters_sent=())


_NOTE_COUNTER = iter(range(10**6))
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]


def good_items(user: str) -> dict[str, Any]:
    requested = json.loads(user)["requested_items"]
    items = []
    for entry in requested:
        n = next(_NOTE_COUNTER)
        w = [_WORDS[(n // 10**i) % 10] for i in range(3)]
        sentences = [
            f"Patient {w[0]} reports a persistent cough lasting {n + 2} weeks with {w[1]} onset.",
            f"Examination today shows mild wheeze near the {w[2]} region on {w[1]} review {n}.",
        ]
        items.append({
            "item": entry["item"], "sentences": sentences, "focus_sentence_indexes": [0],
            "supporting_evidence": [{"sentence_index": 0, "quote": "persistent cough"}],
            "missing_information": ["duration detail"],
            "documentation_conflicts": ["a conflict"] if "contradiction" in entry else [],
            "additional_coding_notes": [], "evidence_summary": "Cough is documented.",
        })
    return {"items": items}


class ScriptedClient:
    def __init__(self, handler: Callable[[str, str, str], LLMResult]) -> None:
        self.handler = handler
        self.calls = 0

    def complete(self, deployment: str, system: str, user: str) -> LLMResult:
        self.calls += 1
        return self.handler(deployment, system, user)


@pytest.fixture()
def state(tmp_path):
    path = tmp_path / "plan.sqlite"
    initialise_state(path, build_plan(7, 2), manifest_hash=MANIFEST, release="FY2027", seed=7, family_size=2)
    return path


SETTINGS = SimpleNamespace(generation_deployment="gen", review_deployment="rev")


def generate(state, handler, **kw):
    options = GenerationOptions(limit=kw.pop("limit", 3), batch_size=kw.pop("batch_size", 3), **kw)
    client = ScriptedClient(handler)
    summary = run_generation(state_path=state, bank=FakeBank(), client=client, settings=SETTINGS,
                             system_prompt="system", options=options)
    return summary, client


def ok_handler(_d, _s, user):
    return result(json.dumps(good_items(user)))


def rows(state, sql):
    with sqlite3.connect(state) as connection:
        return connection.execute(sql).fetchall()


def test_parse_model_json_accepts_fence_and_rejects_bad():
    assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_model_json("not json")
    with pytest.raises(ValueError):
        parse_model_json("[1]")


def test_generation_creates_pending_candidates_and_never_sends_slot_ids(state):
    seen = []

    def handler(d, s, user):
        seen.append(user)
        return ok_handler(d, s, user)

    summary, _ = generate(state, handler, limit=4, batch_size=4)
    assert summary.candidates == summary.selected_slots == 4
    assert not any("slot-" in u or "family" in u for u in seen)
    statuses = rows(state, "SELECT status FROM candidates")
    assert {s[0] for s in statuses} == {"pending_review"}
    record = MasterRecord.model_validate_json(rows(state, "SELECT record_json FROM candidates")[0][0])
    assert record.synthetic and record.provenance.human_review.value == "not_run"


def test_limit_and_request_budget_are_enforced(state):
    summary, client = generate(state, ok_handler, limit=6, batch_size=1, max_requests=2)
    assert client.calls == 2
    assert summary.stopped_reason == "max_requests reached"


def test_dry_run_makes_no_calls(state):
    client = ScriptedClient(ok_handler)
    summary = run_generation(state_path=state, bank=FakeBank(), client=None, settings=None,
                             system_prompt="s", options=GenerationOptions(limit=3, dry_run=True))
    assert summary.requests > 0 and client.calls == 0
    assert rows(state, "SELECT COUNT(*) FROM attempts") == [(0,)]


@pytest.mark.parametrize(
    "handler,reason",
    [
        (lambda d, s, u: result("garbage"), "malformed_json"),
        (lambda d, s, u: result("{}", "length"), "finish_reason_length"),
        (lambda d, s, u: result('{"items": []}'), "invalid"),
    ],
)
def test_bad_model_output_is_recorded_and_slot_released(state, handler, reason):
    summary, _ = generate(state, handler, limit=2, batch_size=2)
    assert summary.candidates == 0 and summary.failures.get(reason) == 2
    assert rows(state, "SELECT COUNT(*) FROM candidates") == [(0,)]
    assert {r[0] for r in rows(state, "SELECT status FROM slots WHERE status != 'planned'")} <= {"exhausted"}


def test_answer_code_leak_in_output_is_rejected(state):
    def leaky(_d, _s, user):
        payload = good_items(user)
        for item in payload["items"]:
            item["sentences"][0] += " Diagnosis " + PILOT_TARGET_CODES[0] + "."
        return result(json.dumps(payload))

    summary, _ = generate(state, leaky, limit=6, batch_size=6)
    assert summary.candidates == 0 and summary.failures.get("invalid", 0) >= 3
    assert summary.stopped_reason == "stopped after repeated failed batches"


def test_partial_batch_keeps_valid_items(state):
    def partial(_d, _s, user):
        payload = good_items(user)
        payload["items"] = payload["items"][:1]
        return result(json.dumps(payload))

    summary, _ = generate(state, partial, limit=3, batch_size=3)
    assert summary.candidates >= 1
    assert summary.failed_slots == summary.selected_slots - summary.candidates


def test_interrupted_attempts_are_recovered(state):
    generate(state, lambda d, s, u: result("garbage"), limit=1, batch_size=1)
    connection = connect_state(state)
    with connection:
        connection.execute("UPDATE slots SET status = 'in_progress' WHERE slot_id = (SELECT slot_id FROM attempts LIMIT 1)")
        connection.execute("UPDATE attempts SET finished_at = NULL")
    assert recover_interrupted(connection, 3) == 1
    assert connection.execute("SELECT outcome FROM attempts").fetchone()[0] == "uncertain_abandoned"
    assert connection.execute("SELECT COUNT(*) FROM slots WHERE status = 'in_progress'").fetchone()[0] == 0
    connection.close()


def test_state_bound_to_other_manifest_is_refused(state):
    bank = FakeBank()
    bank.manifest_hash = "x" * 64
    with pytest.raises(GenerationError):
        run_generation(state_path=state, bank=bank, client=ScriptedClient(ok_handler), settings=SETTINGS,
                       system_prompt="s", options=GenerationOptions(limit=1))


def reviewer_for(state, wrong: set[str] | None = None):
    """Build a reviewer that answers with each candidate's stored label unless told otherwise."""
    answers = {}
    for (blob,) in rows(state, "SELECT record_json FROM candidates"):
        rec = json.loads(blob)
        outcome, code = rec["outcome"], rec["target_code"] or rec["outside_code"]
        if wrong and rec["record_id"] in wrong:
            outcome, code = ("NEEDS_REVIEW", None) if outcome != "NEEDS_REVIEW" else ("MAPPED", PILOT_TARGET_CODES[0])
        answers[rec["medical_description"]] = {"outcome": outcome, "code": code,
                                               "supporting_quotes": [], "rationale": "x"}

    def handler(_d, _s, user):
        return result(json.dumps(answers[json.loads(user)["note"]]))

    return handler


def test_review_accepts_agreeing_candidates_and_rejects_disagreement(state):
    generate(state, ok_handler, limit=4, batch_size=4)
    first = json.loads(rows(state, "SELECT record_json FROM candidates ORDER BY candidate_id")[0][0])
    summary = run_review(state_path=state, bank=FakeBank(), client=ScriptedClient(reviewer_for(state, {first["record_id"]})),
                         settings=SETTINGS, system_prompt="r")
    assert summary.accepted == 3 and summary.rejected == 1
    assert rows(state, "SELECT COUNT(*) FROM accepted_records") == [(3,)]
    assert rows(state, "SELECT status FROM slots WHERE slot_id IN (SELECT slot_id FROM candidates WHERE status='rejected')") == [("planned",)]
    accepted = MasterRecord.model_validate_json(rows(state, "SELECT record_json FROM accepted_records")[0][0])
    assert accepted.provenance.model_review.value == "passed"
    assert accepted.provenance.human_review.value == "not_run"
    assert accepted.provenance.review_deployment == "dep"


def test_review_malformed_output_rejects(state):
    generate(state, ok_handler, limit=2, batch_size=2)
    summary = run_review(state_path=state, bank=FakeBank(), client=ScriptedClient(lambda d, s, u: result("nope")),
                         settings=SETTINGS, system_prompt="r")
    assert summary.accepted == 0 and summary.reasons.get("review_malformed") == 2


def test_evaluate_agreement_rules():
    bank = FakeBank()
    code = PILOT_TARGET_CODES[0]
    mapped = SimpleNamespace(outcome=SimpleNamespace(value="MAPPED"), target_code=code, outside_code=None)
    assert evaluate_agreement(mapped, {"outcome": "MAPPED", "code": code}, bank) == []
    assert evaluate_agreement(mapped, {"outcome": "MAPPED", "code": PILOT_TARGET_CODES[1]}, bank)
    oos = SimpleNamespace(outcome=SimpleNamespace(value="OUT_OF_SCOPE"), target_code=None, outside_code=OUTSIDE)
    assert evaluate_agreement(oos, {"outcome": "OUT_OF_SCOPE", "code": "Z00.01"}, bank) == []
    assert evaluate_agreement(oos, {"outcome": "OUT_OF_SCOPE", "code": code}, bank)
    assert evaluate_agreement(oos, {"outcome": "OUT_OF_SCOPE", "code": "Q99.9"}, bank)
    assert evaluate_agreement(oos, {"outcome": "BOGUS"}, bank)


def test_duplicate_index_exact_near_and_distinct():
    index = DuplicateIndex()
    text = "Patient reports a persistent dry cough lasting two weeks with mild wheeze noted today."
    index.add("a", "train", text)
    assert index.find("b", text.upper())[0].kind == "exact"
    near = text.replace("today", "now")
    assert index.find("b", near)[0].kind == "near"
    assert index.find("b", "Completely different note about a sprained ankle after a fall yesterday.") == []


def test_cross_split_duplicate_rejected_at_review(state):
    generate(state, ok_handler, limit=1, batch_size=1)
    record = json.loads(rows(state, "SELECT record_json FROM candidates")[0][0])
    other = rows(state, f"SELECT slot_id FROM slots WHERE split != '{record['split']}' LIMIT 1")[0][0]
    clone = {**record, "record_id": "rec-clone", "split": "test" if record["split"] != "test" else "train"}
    connection = connect_state(state)
    with connection:
        connection.execute("INSERT INTO accepted_records VALUES (?, ?, 't')", (other, json.dumps(clone)))
    connection.close()
    summary = run_review(state_path=state, bank=FakeBank(), client=ScriptedClient(reviewer_for(state)),
                         settings=SETTINGS, system_prompt="r")
    assert summary.accepted == 0 and summary.reasons.get("cross_split_leakage") == 1


def test_quota_and_integrity_reports_flag_shortfall(state):
    generate(state, ok_handler, limit=3, batch_size=3)
    run_review(state_path=state, bank=FakeBank(), client=ScriptedClient(reviewer_for(state)),
               settings=SETTINGS, system_prompt="r")
    connection = connect_state(state)
    quota = quota_report(connection)
    integrity = integrity_report(connection, MANIFEST)
    connection.close()
    assert quota["totals"]["planned_quota"] == 4500 and quota["totals"]["accepted"] == 3
    assert quota["complete"] is False and quota["shortfalls"]
    assert integrity["passed"] and integrity["records_checked"] == 3
    assert integrity_report(connect_state(state), "other")["passed"] is False


def test_kev_routing_labels_no_gold_leak_and_no_option_shortcut():
    entry_lineage = fake_lineage(PILOT_TARGET_CODES[0])
    option = lambda nid, code, leaf=False: {"node_id": nid, "code": code, "description": nid, "kind": "code",
                                           "is_reportable": leaf, "has_children": not leaf, "contains_pilot": True}
    ctx = {
        "roots": [option("n1", ""), option("r2", ""), option("r3", "")],
        "pilot": {PILOT_TARGET_CODES[0]: {
            "lineage": entry_lineage,
            "options_by_level": [[option("n2", PILOT_TARGET_CODES[0], True), option("s1", "X1", True)], []],
            "wrong_parents": [{"node": option("w1", "W"), "children": [option("w1a", "W.1", True)]}],
        }},
    }
    record = {"record_id": "rec-1", "split": "train", "outcome": "MAPPED", "target_code": PILOT_TARGET_CODES[0],
              "medical_description": "Some note."}
    tasks = routing_tasks(record, ctx, 6, max_no_match=5)
    assert [t["label"] for t in tasks[:3]] == ["n1", "n2", "STOP"]
    assert sorted(t["label"] for t in tasks[3:]) == ["NO_MATCH", "NO_MATCH"]  # wrong parent + wrong leaf
    assert len(routing_tasks(record, ctx, 6, max_no_match=1)) == 4
    for task in tasks:
        ids = {o["id"] for o in task["input"]["options"]}
        assert task["label"] in ids and {"NO_MATCH", "OUT_OF_SCOPE", "NEEDS_REVIEW"} <= ids
        assert "lineage" not in json.dumps(task["input"])
    # STOP is offered only when the parent is a complete code, and then with identical option sets
    stop_sets = {tuple(sorted(o["id"] for o in t["input"]["options"])) for t in tasks if "STOP" in {o["id"] for o in t["input"]["options"]}}
    assert len(stop_sets) == 1
    oos = routing_tasks({**record, "outcome": "OUT_OF_SCOPE", "target_code": None}, ctx, 6)
    assert oos[0]["label"] == "OUT_OF_SCOPE"
    assert {o["id"] for o in oos[0]["input"]["options"]} & {"r2", "r3", "n1"}
    assert routing_tasks(record, ctx, 6) == routing_tasks(record, ctx, 6)


# ---- LLM adapter, leakage helper, review refusal, finalize, CLI guard (all mocked/offline) ----

import httpx2
import openai

from icd_kev_dataset_builder.cli import main
from icd_kev_dataset_builder.errors import LLMConfigurationError, LLMRefusalError, LLMRequestError
from icd_kev_dataset_builder.llm import LLMClient, LLMSettings
from icd_kev_dataset_builder.reference import leaked_codes


def _settings(**kw):
    base = dict(endpoint="https://x.openai.azure.com", api_key="k", api_version=None,
                generation_deployment="gen", review_deployment="gen", mode="v1",
                max_output_tokens=100, temperature=0.5)
    return LLMSettings(**{**base, **kw})


def _api_error(cls, status, message="boom", headers=None):
    request = httpx2.Request("POST", "https://x.openai.azure.com/")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls(message, response=response, body=None)


def _ok_response(content="{}", refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
                           _request_id="rid")


class FakeSDK:
    def __init__(self, outcomes):
        self.outcomes, self.kwargs = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_llm_retries_rate_limit_honouring_retry_after():
    sleeps = []
    sdk = FakeSDK([_api_error(openai.RateLimitError, 429, headers={"retry-after": "2"}), _ok_response()])
    result = LLMClient(_settings(), sdk, sleep=sleeps.append).complete("d", "s", "u")
    assert result.attempts == 2 and result.request_id == "rid" and result.prompt_tokens == 3
    assert 2.0 <= sleeps[0] <= 3.0


def test_llm_gives_up_after_bounded_retries():
    errors = [_api_error(openai.InternalServerError, 500) for _ in range(10)]
    client = LLMClient(_settings(max_retries=2), FakeSDK(errors), sleep=lambda _s: None)
    with pytest.raises(LLMRequestError):
        client.complete("d", "s", "u")


def test_llm_disables_unsupported_optional_parameter_and_retries():
    sdk = FakeSDK([_api_error(openai.BadRequestError, 400, "Unsupported parameter: 'temperature'"), _ok_response()])
    client = LLMClient(_settings(), sdk, sleep=lambda _s: None)
    result = client.complete("d", "s", "u")
    assert "temperature" in sdk.kwargs[0] and "temperature" not in sdk.kwargs[1]
    assert "temperature" in client.disabled_parameters and "temperature" not in result.parameters_sent


def test_llm_auth_error_is_configuration_error_and_not_retried():
    sdk = FakeSDK([_api_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMConfigurationError):
        LLMClient(_settings(), sdk, sleep=lambda _s: None).complete("d", "s", "u")
    assert len(sdk.kwargs) == 1


def test_llm_refusal_and_content_filter_are_refusals():
    with pytest.raises(LLMRefusalError):
        LLMClient(_settings(), FakeSDK([_ok_response(refusal="no")])).complete("d", "s", "u")
    filtered = _api_error(openai.BadRequestError, 400, "content_filter triggered")
    with pytest.raises(LLMRefusalError):
        LLMClient(_settings(), FakeSDK([filtered])).complete("d", "s", "u")


def test_settings_reject_untrusted_endpoint_and_missing_values():
    env = {"AZURE_OPENAI_ENDPOINT": "https://evil.example.com", "AZURE_OPENAI_API_KEY": "k",
           "AZURE_OPENAI_GENERATION_DEPLOYMENT": "d"}
    with pytest.raises(LLMConfigurationError):
        LLMSettings.from_environment(env)
    with pytest.raises(LLMConfigurationError):
        LLMSettings.from_environment({})
    assert "secret-value" not in repr(_settings(api_key="secret-value"))


def test_leaked_codes_handles_decimal_and_nondecimal_forms():
    assert leaked_codes("Dx I10 noted", ["I10"]) == ["I10"]
    assert leaked_codes("code E11.65 here", ["E11.65"]) == ["E11.65"]
    assert leaked_codes("code E1165 here", ["E11.65"]) == ["E11.65"]
    assert leaked_codes("hemoglobin I100 units", ["I10"]) == []
    assert leaked_codes("nothing", ["E11.65", ""]) == []


def test_review_refusal_rejects_instead_of_staying_pending(state):
    generate(state, ok_handler, limit=2, batch_size=2)

    def refuse(_d, _s, _u):
        raise LLMRefusalError("blocked")

    summary = run_review(state_path=state, bank=FakeBank(), client=ScriptedClient(refuse),
                         settings=SETTINGS, system_prompt="r")
    assert summary.rejected == 2 and summary.reasons.get("review_refused") == 2
    assert rows(state, "SELECT COUNT(*) FROM candidates WHERE status = 'pending_review'") == [(0,)]


def test_finalize_writes_partial_dataset_with_flags(state, tmp_path):
    from icd_kev_dataset_builder.reports import finalize

    generate(state, ok_handler, limit=4, batch_size=4)
    run_review(state_path=state, bank=FakeBank(), client=ScriptedClient(reviewer_for(state)),
               settings=SETTINGS, system_prompt="r")
    reference = tmp_path / "data" / "reference"
    reference.mkdir(parents=True)
    (reference / "kb_audit.json").write_text(json.dumps({"unsupported_branches": {"count": 0}}))
    config = SimpleNamespace(data_dir=tmp_path / "data", reference_dir=reference,
                             coding_system="ICD-10-CM", release="FY2027", sources={"a": 1})
    result = finalize(config, state, MANIFEST, sample_size=2)
    out = tmp_path / "data" / "output"
    assert result["status"] == "PARTIAL"
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["synthetic_only"] is True and card["status"] == "PARTIAL"
    total = sum(len((out / f"{s}.master.jsonl").read_text().splitlines()) for s in ("train", "validation", "test"))
    assert total == 4 and len((out / "human_review_sample.jsonl").read_text().splitlines()) == 2
    for line in (out / "train.master.jsonl").read_text().splitlines():
        MasterRecord.model_validate_json(line)


def test_cli_full_run_requires_explicit_request_budget(capsys):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert main(["--root", str(root), "generate", "--full-run"]) == 2
    assert "--max-requests" in capsys.readouterr().err


def test_any_pilot_code_in_output_is_leakage(state):
    other = PILOT_TARGET_CODES[-1]

    def leaky(_d, _s, user):
        payload = good_items(user)
        for item in payload["items"]:
            item["sentences"][0] += f" Also see {other}."
        return result(json.dumps(payload))

    summary, _ = generate(state, leaky, limit=2, batch_size=2)
    assert summary.candidates == 0 and summary.failures.get("invalid") == 2


def test_batches_fill_across_targets_within_a_split_only(state):
    from icd_kev_dataset_builder.generation import _batches

    specs = [json.loads(r[0]) for r in rows(state, "SELECT spec_json FROM slots LIMIT 400")]
    batches = _batches(specs, 5)
    assert all(len({s["split"] for s in b}) == 1 and len(b) <= 5 for b in batches)
    assert sum(len(b) for b in batches) == len(specs)


def test_concurrent_run_is_refused_by_lock(state):
    from icd_kev_dataset_builder.generation import acquire_run_lock

    handle = acquire_run_lock(state)
    try:
        with pytest.raises(GenerationError):
            generate(state, ok_handler, limit=1, batch_size=1)
    finally:
        handle.close()
    summary, _ = generate(state, ok_handler, limit=1, batch_size=1)
    assert summary.candidates == 1


# ---- Kev native adapter (offline; format check only, not a training run) ----
def test_kev_native_adapter_maps_routing_and_evidence_and_rejects_bad_input():
    from icd_kev_dataset_builder.kev_native import convert, to_native

    routing = {
        "case_id": "c1", "task_id": "c1:routing:0", "task_type": "routing", "split": "validation", "label": "n1",
        "input": {"description": "Some note.", "question": "Which?", "current_parent": {"id": "ROOT", "code": "", "description": "ICD-10-CM root"},
                  "options": [{"id": "n1", "code": "I10", "description": "Essential hypertension"},
                              {"id": "NO_MATCH", "code": "", "description": "no match"}]},
    }
    native = to_native(routing)
    q = native["questions"]["route"]
    assert q["type"] == "choice" and q["label"] == "n1" and q["label"] in q["criteria"]
    assert q["target"] == {"n1": 1.0}
    assert q["criteria"]["n1"] == "I10: Essential hypertension" and "wrong" in q["criteria"]["NO_MATCH"]
    evidence = {
        "case_id": "c1", "task_id": "c1:evidence:0", "task_type": "evidence_selection", "split": "train",
        "label": {"focus_sentence_ids": ["s2"]},
        "input": {"description": "A. B.", "question": "?", "sentences": [{"sentence_id": "s1", "text": "A."}, {"sentence_id": "s2", "text": "B."}]},
    }
    ev = to_native(evidence)["questions"]
    assert [ev["sentence_1"]["label"], ev["sentence_2"]["label"]] == [False, True]
    parts, problems = convert([routing, evidence])
    assert not problems and len(parts["train"]) == 1 and len(parts["calibration"]) + len(parts["development"]) == 1
    too_long = {**evidence, "input": {**evidence["input"], "description": "x" * 2000}}
    assert convert([too_long])[1]
    with pytest.raises(Exception):
        to_native({**routing, "label": "missing"})
