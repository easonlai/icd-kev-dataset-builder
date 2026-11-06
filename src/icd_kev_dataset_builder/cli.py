from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import AppConfig, load_config
from .errors import DatasetBuilderError, KnowledgeBaseError
from .kb import build_knowledge_base
from .planner import DEFAULT_FAMILY_SIZE, build_plan, expected_totals
from .state import initialise_state, read_status
from .sources import (
    download_source,
    import_source,
    read_source_lock,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icd-data",
        description="Build a version-locked ICD-10-CM reference knowledge base.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root.")
    parser.add_argument("--config", type=Path, help="TOML configuration path.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Check configuration and local source readiness.")

    sources = commands.add_parser("sources", help="Download or import official source files.")
    source_commands = sources.add_subparsers(dest="sources_command", required=True)
    download = source_commands.add_parser("download", help="Download configured CDC sources.")
    download.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Configured source id; repeat to select multiple. Defaults to required sources.",
    )
    download.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional configured sources when no --source-id is supplied.",
    )
    manual_import = source_commands.add_parser(
        "import", help="Import a manually downloaded source file without modifying it."
    )
    manual_import.add_argument("--source-id", required=True)
    manual_import.add_argument("--path", type=Path, required=True)

    kb = commands.add_parser("kb", help="Build and audit the reference knowledge base.")
    kb_commands = kb.add_subparsers(dest="kb_command", required=True)
    kb_commands.add_parser("build", help="Parse the locked official sources and validate pilot codes.")
    kb_commands.add_parser("pilot", help="Build the offline pilot context used by generation and export.")
    kb_commands.add_parser("audit", help="Check built artifacts against the current source lock.")

    plan = commands.add_parser("plan", help="Create or verify the offline generation plan (no API calls).")
    plan.add_argument("--seed", type=int, default=20270, help="Planning seed.")
    plan.add_argument("--family-size", type=int, default=DEFAULT_FAMILY_SIZE)
    commands.add_parser("status", help="Report durable plan state (no API calls).")

    gen = commands.add_parser("generate", help="Generate candidate notes for planned slots (paid model calls).")
    gen.add_argument("--limit", type=int, default=5, help="Maximum slots to attempt (default 5).")
    gen.add_argument("--batch-size", type=int, default=5)
    gen.add_argument("--max-requests", type=int, default=None, help="Request budget (default 20).")
    gen.add_argument("--max-tokens", type=int, default=None, help="Stop after this many total tokens.")
    gen.add_argument("--max-attempts", type=int, default=3, help="Attempts per slot before shortfall.")
    gen.add_argument("--outcome", choices=["MAPPED", "OUT_OF_SCOPE", "NEEDS_REVIEW"], help="Only attempt slots with this outcome (smoke tests).")
    gen.add_argument("--dry-run", action="store_true", help="Build prompts and checks without calling the model.")
    gen.add_argument("--probe", action="store_true", help="Send one tiny request to check the connection.")
    gen.add_argument("--full-run", action="store_true", help="Required to attempt every remaining slot.")

    rev = commands.add_parser("review", help="Review and accept pending candidates (paid model calls).")
    rev.add_argument("--max-requests", type=int, default=50)
    rev.add_argument("--max-attempts", type=int, default=3)

    commands.add_parser("audit", help="Audit accepted dataset quotas and split integrity (offline).")
    fin = commands.add_parser("finalize", help="Write master JSONL files and reports (offline).")
    fin.add_argument("--sample-size", type=int, default=25)
    export = commands.add_parser("export-kev", help="Export Kev routing/evidence tasks (offline).")
    export.add_argument("--option-count", type=int, default=24)
    export.add_argument("--max-no-match", type=int, default=2, help="NO_MATCH negatives per mapped case.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.root, args.config)
        if args.command == "doctor":
            return _doctor(config)
        if args.command == "sources":
            return _sources(config, args)
        if args.command == "kb" and args.kb_command == "build":
            audit = build_knowledge_base(config)
            print(
                f"Built {audit['catalogue_code_count']} codes and validated "
                f"{audit['pilot_targets_validated']} pilot target paths for {audit['release']}."
            )
            print(f"Reference artifacts: {config.reference_dir}")
            return 0
        if args.command == "kb" and args.kb_command == "pilot":
            from .kb import PILOT_TARGET_CODES
            from .reference import build_pilot_context

            doc = build_pilot_context(config, PILOT_TARGET_CODES)
            print(f"Pilot context written for {len(doc['pilot'])} codes.")
            return 0
        if args.command == "kb" and args.kb_command == "audit":
            return _audit(config)
        if args.command == "plan":
            return _plan(config, args.seed, args.family_size)
        if args.command == "status":
            return _status(config)
        if args.command == "generate":
            return _generate(config, args)
        if args.command == "review":
            return _review(config, args)
        if args.command == "audit":
            return _dataset_audit(config)
        if args.command == "finalize":
            return _finalize(config, args)
        if args.command == "export-kev":
            return _export_kev(config, args)
        parser.error("Unsupported command.")
    except DatasetBuilderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _doctor(config: AppConfig) -> int:
    print(f"icd-data {__version__}")
    print(f"Workspace: {config.root}")
    print(f"Configured release: {config.release} ({config.coding_system})")
    print("Source status:")
    try:
        lock = read_source_lock(config)
    except DatasetBuilderError as exc:
        print(f"  source lock unavailable: {exc}")
        entries = {}
    else:
        entries = lock["sources"]
    for source_id, definition in config.sources.items():
        entry = entries.get(source_id)
        if entry:
            print(f"  {source_id}: imported, sha256={entry['sha256']}")
        else:
            qualifier = "required" if definition.required else "optional"
            print(f"  {source_id}: missing ({qualifier})")
    if find_spec("pypdf") is None:
        print("PDF extraction: unavailable (install the optional .[pdf] dependency)")
    else:
        print("PDF extraction: available")
    return 0


def _sources(config: AppConfig, args: argparse.Namespace) -> int:
    if args.sources_command == "import":
        saved_path = import_source(config, args.source_id, args.path)
        print(f"Imported {args.source_id} to {saved_path}")
        lock = read_source_lock(config)
        print(f"Source manifest hash: {lock['manifest_hash']}")
        return 0

    if args.source_ids:
        source_ids = args.source_ids
    else:
        source_ids = [
            source_id
            for source_id, definition in config.sources.items()
            if definition.required or args.include_optional
        ]
    for source_id in source_ids:
        saved_path = download_source(config, source_id)
        print(f"Downloaded {source_id} to {saved_path}")
    lock = read_source_lock(config)
    print(f"Source manifest hash: {lock['manifest_hash']}")
    return 0


def _state_path(config: AppConfig) -> Path:
    return config.data_dir / "state" / "plan.sqlite"


def _plan(config: AppConfig, seed: int, family_size: int) -> int:
    lock = read_source_lock(config)
    audit_path = config.reference_dir / "kb_audit.json"
    if not audit_path.is_file():
        raise KnowledgeBaseError("Run `icd-data kb build` before planning.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("source_manifest_hash") != lock["manifest_hash"]:
        raise KnowledgeBaseError("Knowledge base is stale for the current source lock; rebuild it.")
    slots = build_plan(seed, family_size)
    action = initialise_state(
        _state_path(config),
        slots,
        manifest_hash=str(lock["manifest_hash"]),
        release=config.release,
        seed=seed,
        family_size=family_size,
    )
    print(f"Plan {action}: {len(slots)} slots (expected quota total {expected_totals()['all']}).")
    print(f"State: {_state_path(config)}")
    return 0


def _status(config: AppConfig) -> int:
    report = read_status(_state_path(config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["family_split_violations"] == 0 else 2


def _audit(config: AppConfig) -> int:
    lock = read_source_lock(config)
    reference_dir = config.reference_dir
    required_artifacts = (
        reference_dir / "codebook.json",
        reference_dir / "hierarchy.json",
        reference_dir / "reference_chunks.jsonl",
        reference_dir / "kb_audit.json",
    )
    missing = [path for path in required_artifacts if not path.is_file()]
    if missing:
        raise KnowledgeBaseError(
            "Reference artifacts are incomplete; run `icd-data kb build`. Missing: "
            + ", ".join(str(path) for path in missing)
        )
    reports = [
        json.loads((reference_dir / "codebook.json").read_text(encoding="utf-8")),
        json.loads((reference_dir / "hierarchy.json").read_text(encoding="utf-8")),
        json.loads((reference_dir / "kb_audit.json").read_text(encoding="utf-8")),
    ]
    stale = [
        label
        for label, document in zip(("codebook", "hierarchy", "audit"), reports, strict=True)
        if document.get("source_manifest_hash") != lock["manifest_hash"]
    ]
    if stale:
        raise KnowledgeBaseError(
            "Artifacts do not match the current source manifest: "
            + ", ".join(stale)
            + ". Rebuild the knowledge base."
        )
    chunk_lines = (reference_dir / "reference_chunks.jsonl").read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(chunk_lines, start=1):
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"Malformed reference chunk at line {line_number}: {exc}"
            ) from exc
        if chunk.get("source_manifest_hash") != lock["manifest_hash"]:
            raise KnowledgeBaseError(
                f"Reference chunk at line {line_number} is not bound to the current source manifest."
            )
    print(
        f"Audit passed: {len(reports[0]['codes'])} codes, "
        f"{reports[2]['pilot_targets_validated']} pilot targets, "
        f"{len(chunk_lines)} reference chunks."
    )
    print(f"Source manifest: {lock['manifest_hash']}")
    return 0


def _model_client(config: AppConfig):
    from .llm import LLMClient, LLMSettings, load_dotenv

    load_dotenv(config.root / ".env")
    settings = LLMSettings.from_environment()
    return settings, LLMClient(settings)


def _generate(config: AppConfig, args: argparse.Namespace) -> int:
    from .errors import GenerationError
    from .generation import GenerationOptions, load_prompt, run_generation
    from .reference import ReferenceBank

    if args.limit < 1:
        raise GenerationError("--limit must be at least 1.")
    if args.full_run and not args.dry_run and args.max_requests is None:
        raise GenerationError("--full-run needs an explicit --max-requests budget.")
    options = GenerationOptions(
        limit=None if args.full_run else args.limit,
        batch_size=args.batch_size,
        max_requests=20 if args.max_requests is None else args.max_requests,
        max_total_tokens=args.max_tokens,
        max_attempts_per_slot=args.max_attempts,
        dry_run=args.dry_run,
        outcome=args.outcome,
    )
    bank = ReferenceBank(config)
    settings = client = None
    if not args.dry_run:
        settings, client = _model_client(config)
        if args.probe:
            print(json.dumps(client.probe(), indent=2, default=str))
    summary = run_generation(
        state_path=_state_path(config), bank=bank, client=client, settings=settings,
        system_prompt=load_prompt(config.root, "generate.v1"), options=options,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _review(config: AppConfig, args: argparse.Namespace) -> int:
    from .generation import load_prompt
    from .reference import ReferenceBank
    from .review import run_review

    settings, client = _model_client(config)
    summary = run_review(
        state_path=_state_path(config), bank=ReferenceBank(config), client=client, settings=settings,
        system_prompt=load_prompt(config.root, "review.v1"),
        max_requests=args.max_requests, max_attempts_per_slot=args.max_attempts,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _dataset_audit(config: AppConfig) -> int:
    from .reports import run_audit

    report = run_audit(_state_path(config), str(read_source_lock(config)["manifest_hash"]))
    summary = {
        "quota_totals": report["quota"]["totals"],
        "quota_by_split": report["quota"]["by_split"],
        "quota_complete": report["quota"]["complete"],
        "shortfall_cells": len(report["quota"]["shortfalls"]),
        "candidate_status": report["candidate_status"],
        "attempt_outcomes": report["attempt_outcomes"],
        "integrity": report["integrity"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Per-cell shortfalls are written by `finalize` to data/output/quota_report.json.")
    return 0 if report["integrity"]["passed"] else 2


def _finalize(config: AppConfig, args: argparse.Namespace) -> int:
    from .reports import finalize

    result = finalize(config, _state_path(config), str(read_source_lock(config)["manifest_hash"]),
                      sample_size=args.sample_size)
    print(f"Finalized ({result['status']}): {result['counts']} -> {result['output_dir']}")
    return 0


def _export_kev(config: AppConfig, args: argparse.Namespace) -> int:
    from .kev import export_kev

    manifest = export_kev(config, option_count=args.option_count, max_no_match=args.max_no_match)
    print(json.dumps(manifest["task_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
