from __future__ import annotations

import csv
import importlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .config import AppConfig
from .errors import KnowledgeBaseError, SourceError
from .sources import atomic_write_bytes, atomic_write_json, inspect_zip, read_source_lock

PILOT_TARGET_CODES = (
    "E10.9",
    "E10.65",
    "E11.9",
    "E11.65",
    "E11.21",
    "E11.22",
    "E11.29",
    "E11.40",
    "E11.42",
    "E11.649",
    "J45.20",
    "J45.21",
    "J45.30",
    "J45.31",
    "J45.40",
    "J45.41",
    "J44.9",
    "J44.1",
    "J44.0",
    "J18.9",
    "I10",
    "I25.10",
    "I25.119",
    "I48.0",
    "I48.11",
    "I48.19",
    "I48.21",
    "I50.22",
    "I50.32",
    "I50.42",
    "N18.1",
    "N18.2",
    "N18.30",
    "N18.31",
    "N18.32",
    "N18.4",
    "N18.5",
    "N18.6",
    "N18.9",
    "N39.0",
    "K21.9",
    "K21.00",
    "K21.01",
    "K29.50",
    "K29.51",
    "K29.70",
    "K29.71",
    "K25.9",
    "K26.9",
    "K59.00",
)

_CODE_TOKEN = re.compile(r"^[A-Z][0-9A-Z][A-Z0-9](?:\.?[A-Z0-9]{1,4})?$")
_LEADING_CODE = re.compile(r"^\s*([A-Z][0-9A-Z][A-Z0-9](?:\.?[A-Z0-9]{1,4})?)\b")
_NOTE_TAGS = {
    "codealso",
    "codefirst",
    "excludes1",
    "excludes2",
    "includes",
    "inclusionterm",
    "note",
    "useadditionalcode",
    "useadditionalcodedesc",
    "sevenchrdef",
    "sevenchrnote",
    "sequencing",
    "includesnote",
}
_HIERARCHY_TAGS = {"chapter", "section", "diag"}


def canonical_code(value: str) -> str | None:
    token = value.strip().upper()
    if not _CODE_TOKEN.fullmatch(token):
        return None
    normalized = token.replace(".", "")
    return f"{normalized[:3]}.{normalized[3:]}" if len(normalized) > 3 else normalized


def build_knowledge_base(config: AppConfig) -> dict[str, Any]:
    lock = read_source_lock(config)
    entries = lock["sources"]
    required_ids = ("table_and_index", "code_descriptions", "guidelines", "addenda")
    missing = [
        source_id
        for source_id in required_ids
        if source_id not in entries or not config.sources.get(source_id)
    ]
    if missing:
        raise KnowledgeBaseError(
            "Knowledge-base construction requires imported sources: " + ", ".join(missing)
        )

    table_path = _locked_path(config, entries, "table_and_index")
    descriptions_path = _locked_path(config, entries, "code_descriptions")
    guidelines_path = _locked_path(config, entries, "guidelines")
    addenda_path = _locked_path(config, entries, "addenda")

    codebook, description_sources = parse_code_descriptions(descriptions_path, config)
    tabular_xml, tabular_member = _find_tabular_xml(table_path, config)
    index_xml, index_member = _find_index_xml(table_path, config, excluded={tabular_member})
    hierarchy = parse_tabular_hierarchy(
        tabular_xml,
        tabular_member,
        {record["code"] for record in codebook},
        config.release,
        str(lock["manifest_hash"]),
    )
    index_chunks = parse_index_chunks(index_xml, index_member, str(lock["manifest_hash"]))
    guideline_chunks, guideline_pages = parse_guidelines(
        guidelines_path, str(lock["manifest_hash"])
    )
    addenda_members = [item["filename"] for item in inspect_zip(addenda_path, config)]

    code_lookup = {record["code"]: record for record in codebook}
    path_by_code = _reportable_paths(hierarchy)
    target_paths: dict[str, list[dict[str, str]]] = {}
    invalid_targets: list[str] = []
    for code in PILOT_TARGET_CODES:
        if code not in code_lookup:
            invalid_targets.append(f"{code}: absent from the official code-description list")
            continue
        lineage = path_by_code.get(code)
        if not lineage:
            invalid_targets.append(
                f"{code}: complete code exists but no official tabular hierarchy path was found"
            )
            continue
        target_paths[code] = lineage
    if invalid_targets:
        raise KnowledgeBaseError(
            "Pilot target validation failed for the configured release "
            f"{config.release}:\n- " + "\n- ".join(invalid_targets)
        )

    unplaced_codes = sorted(set(code_lookup) - set(path_by_code))
    audit: dict[str, Any] = {
        "schema_version": 1,
        "coding_system": config.coding_system,
        "release": config.release,
        "release_year": config.release_year,
        "source_manifest_hash": lock["manifest_hash"],
        "source_verification_uncertainty": {
            source_id: entry.get("unresolved_release_uncertainty", [])
            for source_id, entry in entries.items()
            if entry.get("unresolved_release_uncertainty")
        },
        "catalogue_code_count": len(codebook),
        "hierarchy_node_count": len(hierarchy["nodes"]),
        "reportable_code_path_count": len(path_by_code),
        "unplaced_code_count": len(unplaced_codes),
        "unplaced_code_examples": unplaced_codes[:100],
        "pilot_target_count": len(PILOT_TARGET_CODES),
        "pilot_targets_validated": len(target_paths),
        "pilot_target_paths": target_paths,
        "description_sources": description_sources,
        "tabular_source_member": tabular_member,
        "index_source_member": index_member,
        "guideline_page_count": guideline_pages,
        "index_chunk_count": len(index_chunks),
        "guideline_chunk_count": len(guideline_chunks),
        "addenda_members": addenda_members,
        "addenda_handling": (
            "The addenda archive is retained and checksummed as reference only. "
            "No addendum is applied over the selected main-release tabular data."
        ),
        "unsupported_branches": {
            "count": len(unplaced_codes),
            "code_examples": unplaced_codes[:100],
            "note": (
                "Codes in the description catalogue without a tabular path are not "
                "assigned invented lineage. Pilot targets are blocked unless every "
                "target has a verified complete-code path."
            ),
        },
    }

    codebook_document = {
        "schema_version": 1,
        "coding_system": config.coding_system,
        "release": config.release,
        "release_year": config.release_year,
        "source_manifest_hash": lock["manifest_hash"],
        "codes": codebook,
    }
    all_chunks = index_chunks + guideline_chunks
    _write_jsonl(config.reference_dir / "reference_chunks.jsonl", all_chunks)
    atomic_write_json(config.reference_dir / "codebook.json", codebook_document)
    atomic_write_json(config.reference_dir / "hierarchy.json", hierarchy)
    atomic_write_json(config.reference_dir / "kb_audit.json", audit)
    return audit


def parse_code_descriptions(path: Path, config: AppConfig) -> tuple[list[dict[str, str]], list[str]]:
    candidates: dict[str, set[str]] = defaultdict(set)
    used_members: list[str] = []
    with zipfile.ZipFile(path) as archive:
        members = inspect_zip(path, config)
        text_infos = [
            archive.getinfo(str(item["filename"]))
            for item in members
            if Path(str(item["filename"])).suffix.lower() in {".txt", ".csv", ".xml"}
        ]
        preferred = [
            info
            for info in text_infos
            if "code" in Path(info.filename).name.lower()
            and "order" not in Path(info.filename).name.lower()
            and "addenda" not in Path(info.filename).name.lower()
        ]
        selected = preferred or text_infos
        for info in selected:
            raw = archive.read(info)
            if Path(info.filename).suffix.lower() == ".xml":
                found = _parse_description_xml(raw, info.filename)
            else:
                found = _parse_description_text(raw, info.filename)
            if found:
                used_members.append(info.filename)
                for code, description in found.items():
                    candidates[code].add(description)
    if not candidates:
        raise KnowledgeBaseError(
            f"No ICD-10-CM code descriptions could be parsed from {path}. "
            "Inspect archive members and provide the official code-description archive."
        )
    records: list[dict[str, str]] = []
    for code, descriptions in sorted(candidates.items()):
        preferred_description = sorted(descriptions, key=lambda value: (-len(value), value))[0]
        records.append(
            {
                "code": code,
                "description": preferred_description,
                "description_variants": " | ".join(sorted(descriptions)),
            }
        )
    return records, sorted(used_members)


def parse_tabular_hierarchy(
    xml_bytes: bytes,
    source_member: str,
    reportable_codes: set[str],
    release: str,
    manifest_hash: str,
) -> dict[str, Any]:
    root = _safe_xml_root(xml_bytes, source_member)
    nodes: list[dict[str, Any]] = []
    nodes_by_id: dict[str, dict[str, Any]] = {}
    roots: list[str] = []

    def walk(element: ElementTree.Element, parent_id: str | None) -> None:
        tag = _local_name(element.tag)
        current_parent = parent_id
        if tag in _HIERARCHY_TAGS:
            name_element = next(
                (child for child in list(element) if _local_name(child.tag) == "name"),
                None,
            )
            desc_element = next(
                (
                    child
                    for child in list(element)
                    if _local_name(child.tag) in {"desc", "description"}
                ),
                None,
            )
            name_text = _element_text(name_element)
            description = _element_text(desc_element) or name_text or tag
            code = _extract_leading_code(name_text)
            node_id = f"node-{len(nodes) + 1:07d}"
            node: dict[str, Any] = {
                "node_id": node_id,
                "kind": tag,
                "code": code,
                "name": name_text,
                "description": description,
                "parent_id": parent_id,
                "children": [],
                "is_reportable": code in reportable_codes if code else False,
                "source": {"member": source_member},
                "local_coding_notes": _collect_local_notes(element),
                "inherited_coding_notes": [],
                "seventh_character_extensions": _seventh_characters(element),
            }
            nodes.append(node)
            nodes_by_id[node_id] = node
            if parent_id is None:
                roots.append(node_id)
            else:
                parent = nodes_by_id[parent_id]
                parent["children"].append(node_id)
            current_parent = node_id
        for child in list(element):
            walk(child, current_parent)

    walk(root, None)
    if not nodes or not any(node["kind"] == "diag" for node in nodes):
        raise KnowledgeBaseError(
            f"Tabular XML member {source_member!r} contains no recognizable chapter, "
            "section, and diagnosis hierarchy."
        )

    _expand_seventh_characters(nodes, nodes_by_id, reportable_codes, source_member)
    by_id = nodes_by_id
    for node in nodes:
        parent_id = node["parent_id"]
        inherited: list[dict[str, str]] = []
        while parent_id is not None:
            parent = by_id[parent_id]
            inherited[0:0] = parent["local_coding_notes"]
            parent_id = parent["parent_id"]
        node["inherited_coding_notes"] = inherited
    return {
        "schema_version": 1,
        "coding_system": "ICD-10-CM",
        "release": release,
        "source_manifest_hash": manifest_hash,
        "source_member": source_member,
        "roots": roots,
        "nodes": nodes,
    }


def parse_index_chunks(xml_bytes: bytes, source_member: str, manifest_hash: str) -> list[dict[str, Any]]:
    root = _safe_xml_root(xml_bytes, source_member)
    chunks: list[dict[str, Any]] = []
    candidates = [
        element
        for element in root.iter()
        if _local_name(element.tag).lower() in {"mainterm", "entry", "term"}
    ]
    if not candidates:
        candidates = [root]
    for position, element in enumerate(candidates, start=1):
        text = _element_text(element)
        if not text:
            continue
        chunks.extend(
            _split_reference_text(
                text,
                source="alphabetic_index",
                source_member=source_member,
                locator=f"{source_member}#entry-{position}",
                manifest_hash=manifest_hash,
            )
        )
    if not chunks:
        raise KnowledgeBaseError(f"Alphabetic Index XML has no readable entries: {source_member}")
    return chunks


def parse_guidelines(path: Path, manifest_hash: str) -> tuple[list[dict[str, Any]], int]:
    try:
        pdf_module = importlib.import_module("pypdf")
    except ImportError as exc:
        raise KnowledgeBaseError(
            "PDF text extraction requires the optional pypdf dependency. "
            "Install the project PDF extra with `python -m pip install -e '.[pdf]'`, "
            "then rebuild the knowledge base."
        ) from exc
    try:
        reader = pdf_module.PdfReader(str(path), strict=True)
    except Exception as exc:
        raise KnowledgeBaseError(f"Cannot parse official guidelines PDF {path}: {exc}") from exc
    if reader.is_encrypted:
        raise KnowledgeBaseError(f"Official guidelines PDF is encrypted: {path}")
    chunks: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        chunks.extend(
            _split_reference_text(
                text,
                source="official_guidelines",
                source_member=path.name,
                locator=f"{path.name}#page-{page_number}",
                manifest_hash=manifest_hash,
                page=page_number,
            )
        )
    if not chunks:
        raise KnowledgeBaseError(f"Official guidelines PDF contains no extractable text: {path}")
    return chunks, len(reader.pages)


def _find_tabular_xml(path: Path, config: AppConfig) -> tuple[bytes, str]:
    matches = []
    for name, raw in _xml_members(path, config):
        if "tabular" in name.lower() or _root_name(raw, name).lower().endswith("tabular"):
            matches.append((raw, name))
    if len(matches) != 1:
        raise KnowledgeBaseError(
            "Expected exactly one tabular XML member in the Table and Index archive; "
            f"found {len(matches)} candidates: {[name for _, name in matches]}."
        )
    return matches[0]


def _find_index_xml(
    path: Path, config: AppConfig, excluded: set[str]
) -> tuple[bytes, str]:
    matches = []
    for name, raw in _xml_members(path, config):
        if name in excluded:
            continue
        root = _safe_xml_root(raw, name)
        title = _element_text(next((c for c in root if _local_name(c.tag) == "title"), None))
        if "index to diseases" in title.lower():
            matches.append((raw, name))
    if len(matches) != 1:
        raise KnowledgeBaseError(
            "Expected exactly one Alphabetic Index XML member in the Table and Index archive; "
            f"found {len(matches)} candidates: {[name for _, name in matches]}."
        )
    return matches[0]


def _xml_members(path: Path, config: AppConfig) -> list[tuple[str, bytes]]:
    found: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(path) as archive:
        for item in inspect_zip(path, config):
            name = str(item["filename"])
            if Path(name).suffix.lower() == ".xml":
                found.append((name, archive.read(name)))
    return found


def _locked_path(config: AppConfig, entries: object, source_id: str) -> Path:
    if not isinstance(entries, dict) or source_id not in entries:
        raise SourceError(f"Required source {source_id!r} is not present in the source lock.")
    entry = entries[source_id]
    if not isinstance(entry, dict):
        raise SourceError(f"Invalid source-lock entry for {source_id!r}.")
    return config.data_dir / str(entry["local_filename"])


def _parse_description_xml(raw: bytes, source_name: str) -> dict[str, str]:
    root = _safe_xml_root(raw, source_name)
    found: dict[str, str] = {}
    for element in root.iter():
        if _local_name(element.tag).lower() not in {"diag", "code"}:
            continue
        name_text = ""
        description = ""
        for child in list(element):
            child_tag = _local_name(child.tag).lower()
            if child_tag in {"name", "code"} and not name_text:
                name_text = _element_text(child)
            elif child_tag in {"desc", "description"}:
                description = _element_text(child)
        code = _extract_leading_code(name_text) or canonical_code(name_text)
        if code and description:
            found[code] = description
    return found


def _parse_description_text(raw: bytes, source_name: str) -> dict[str, str]:
    text = _decode_text(raw, source_name)
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        fields: list[str] | None = None
        if "|" in stripped:
            fields = [field.strip() for field in stripped.split("|")]
        elif "\t" in stripped:
            fields = [field.strip() for field in stripped.split("\t")]
        elif "," in stripped:
            try:
                fields = [field.strip() for field in next(csv.reader([stripped]))]
            except csv.Error:
                fields = None
        if fields and fields:
            code = canonical_code(fields[0])
            if code:
                description_fields = [field for field in fields[1:] if field]
                if description_fields:
                    found[code] = max(description_fields, key=len)
                    continue
        match = _LEADING_CODE.match(stripped)
        if match:
            code = canonical_code(match.group(1))
            description = stripped[match.end() :].strip(" \t|,")
            if code and description:
                found[code] = description
    return found


def _safe_xml_root(raw: bytes, source_name: str) -> ElementTree.Element:
    if len(raw) > 100 * 1024 * 1024:
        raise KnowledgeBaseError(f"XML member exceeds the 100 MiB parser limit: {source_name}")
    declarations = raw.upper()
    if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
        raise KnowledgeBaseError(f"XML external declarations are forbidden: {source_name}")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise KnowledgeBaseError(f"Malformed XML member {source_name!r}: {exc}") from exc


def _collect_local_notes(element: ElementTree.Element) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []

    def visit(current: ElementTree.Element) -> None:
        tag = _local_name(current.tag).lower()
        if current is not element and tag in _HIERARCHY_TAGS:
            return
        text = _element_text(current)
        if tag in _NOTE_TAGS and text:
            notes.append({"kind": tag, "text": text})
        for child in list(current):
            visit(child)

    visit(element)
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for note in notes:
        key = (note["kind"], note["text"])
        if key not in seen:
            seen.add(key)
            unique.append(note)
    return unique


def _seventh_characters(element: ElementTree.Element) -> dict[str, str]:
    extensions: dict[str, str] = {}
    for child in list(element):
        if _local_name(child.tag) != "sevenChrDef":
            continue
        for extension in list(child):
            character = extension.get("char")
            if _local_name(extension.tag) == "extension" and character:
                extensions[character] = _element_text(extension)
    return extensions


def _expand_seventh_characters(
    nodes: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    reportable_codes: set[str],
    source_member: str,
) -> None:
    """Add terminal codes defined by an ancestor's official sevenChrDef.

    A candidate is added only when the official code list contains it; placeholder
    X characters pad shorter bases up to the sixth character.
    """
    existing = {node["code"] for node in nodes if node["code"]}
    for leaf in [node for node in nodes if node["kind"] == "diag" and not node["children"]]:
        holder: dict[str, Any] | None = leaf
        while holder is not None and not holder["seventh_character_extensions"]:
            parent_id = holder["parent_id"]
            holder = nodes_by_id[parent_id] if parent_id else None
        if holder is None or not leaf["code"]:
            continue
        base = leaf["code"].replace(".", "")
        if len(base) > 6:
            continue
        padded = base.ljust(6, "X")
        for character, meaning in holder["seventh_character_extensions"].items():
            code = canonical_code(padded + character)
            if code is None or code in existing or code not in reportable_codes:
                continue
            node_id = f"node-{len(nodes) + 1:07d}"
            node = {
                "node_id": node_id,
                "kind": "diag",
                "code": code,
                "name": code,
                "description": f"{leaf['description']}, {meaning}",
                "parent_id": leaf["node_id"],
                "children": [],
                "is_reportable": True,
                "source": {
                    "member": source_member,
                    "derivation": "seventh_character_expansion",
                    "definition_node_id": holder["node_id"],
                },
                "local_coding_notes": [],
                "inherited_coding_notes": [],
                "seventh_character_extensions": {},
            }
            nodes.append(node)
            nodes_by_id[node_id] = node
            leaf["children"].append(node_id)
            existing.add(code)


def _reportable_paths(hierarchy: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    by_id = {node["node_id"]: node for node in hierarchy["nodes"]}
    paths: dict[str, list[dict[str, str]]] = {}
    for node in hierarchy["nodes"]:
        if not node["is_reportable"] or not node["code"]:
            continue
        lineage: list[dict[str, str]] = []
        current: dict[str, Any] | None = node
        while current is not None:
            lineage.insert(
                0,
                {
                    "node_id": current["node_id"],
                    "kind": current["kind"],
                    "code": current["code"] or "",
                    "description": current["description"],
                },
            )
            parent_id = current["parent_id"]
            current = by_id.get(parent_id) if parent_id is not None else None
        existing = paths.get(node["code"])
        if existing and existing != lineage:
            raise KnowledgeBaseError(
                f"Code {node['code']} has multiple distinct tabular hierarchy paths."
            )
        paths[node["code"]] = lineage
    return paths


def _extract_leading_code(value: str) -> str | None:
    token = value.strip()
    if not _CODE_TOKEN.fullmatch(token):
        return None
    return canonical_code(token)


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _root_name(raw: bytes, source_name: str) -> str:
    try:
        return _local_name(_safe_xml_root(raw, source_name).tag)
    except KnowledgeBaseError:
        return ""


def _decode_text(raw: bytes, source_name: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise KnowledgeBaseError(f"Cannot decode text member {source_name!r}.")


def _split_reference_text(
    text: str,
    source: str,
    source_member: str,
    locator: str,
    manifest_hash: str,
    page: int | None = None,
    max_chars: int = 1800,
) -> list[dict[str, Any]]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []
    chunks: list[dict[str, Any]] = []
    remaining = normalized
    part = 1
    while remaining:
        if len(remaining) <= max_chars:
            chunk, remaining = remaining, ""
        else:
            boundary = remaining.rfind("\n", 0, max_chars + 1)
            if boundary < max_chars // 2:
                boundary = remaining.rfind(" ", 0, max_chars + 1)
            if boundary < max_chars // 2:
                boundary = max_chars
            chunk, remaining = remaining[:boundary].strip(), remaining[boundary:].strip()
        item: dict[str, Any] = {
            "chunk_id": f"{source}:{locator}:part-{part}",
            "source": source,
            "source_member": source_member,
            "locator": locator if part == 1 else f"{locator}#part-{part}",
            "text": chunk,
            "source_manifest_hash": manifest_hash,
        }
        if page is not None:
            item["page"] = page
        chunks.append(item)
        part += 1
    return chunks


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    atomic_write_bytes(path, data)
