from __future__ import annotations

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from icd_kev_dataset_builder.config import load_config
from icd_kev_dataset_builder.errors import KnowledgeBaseError
from icd_kev_dataset_builder.kb import (
    PILOT_TARGET_CODES,
    _parse_description_text,
    _reportable_paths,
    canonical_code,
    parse_code_descriptions,
    parse_index_chunks,
    parse_tabular_hierarchy,
)

ROOT = Path(__file__).resolve().parents[1]

TABULAR_FIXTURE = b"""\
<ICD10CM.tabular>
  <chapter><name>4</name><desc>Endocrine, nutritional and metabolic diseases</desc>
    <section><name>E08-E13</name><desc>Diabetes mellitus</desc>
      <diag><name>E11</name><desc>Type 2 diabetes mellitus</desc>
        <note>Use additional code for any insulin use.</note>
        <diag><name>E11.65</name><desc>Type 2 diabetes mellitus with hyperglycemia</desc></diag>
      </diag>
    </section>
  </chapter>
</ICD10CM.tabular>
"""

INDEX_FIXTURE = b"""\
<index><mainTerm><title>Diabetes</title><see>see condition</see></mainTerm>
<mainTerm><title>Fever</title></mainTerm></index>
"""


class KnowledgeBaseParsingTests(unittest.TestCase):
    def test_pilot_targets_are_exactly_fifty(self) -> None:
        self.assertEqual(len(PILOT_TARGET_CODES), 50)
        self.assertEqual(len(set(PILOT_TARGET_CODES)), 50)

    def test_code_normalization_handles_dotted_and_undotted_complete_codes(self) -> None:
        self.assertEqual(canonical_code("e1165"), "E11.65")
        self.assertEqual(canonical_code("E11.65"), "E11.65")
        self.assertEqual(canonical_code("E11."), None)

    def test_description_parser_supports_pipe_and_fixed_width_rows(self) -> None:
        parsed = _parse_description_text(
            b"E11.65 | Type 2 diabetes mellitus with hyperglycemia\n"
            b"A000    Cholera due to Vibrio cholerae\n",
            "structural fixture",
        )

        self.assertEqual(
            parsed["E11.65"], "Type 2 diabetes mellitus with hyperglycemia"
        )
        self.assertEqual(parsed["A00.0"], "Cholera due to Vibrio cholerae")

    def test_archive_code_parser_inspects_members_without_fixed_internal_names(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "code-descriptions.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "unexpected-name.txt",
                    "E11.65 | Type 2 diabetes with hyperglycemia\n",
                )
            records, member_names = parse_code_descriptions(path, load_config(ROOT))

        self.assertEqual(records[0]["code"], "E11.65")
        self.assertEqual(member_names, ["unexpected-name.txt"])

    def test_hierarchy_uses_xml_edges_and_retains_inherited_notes(self) -> None:
        hierarchy = parse_tabular_hierarchy(
            TABULAR_FIXTURE,
            "tabular-fixture.xml",
            {"E11.65"},
            "FIXTURE",
            "fixture-hash",
        )
        paths = _reportable_paths(hierarchy)

        self.assertEqual([item["code"] for item in paths["E11.65"]], ["", "", "E11", "E11.65"])
        target = next(node for node in hierarchy["nodes"] if node["code"] == "E11.65")
        self.assertIn(
            {"kind": "note", "text": "Use additional code for any insulin use."},
            target["inherited_coding_notes"],
        )

    def test_seventh_character_codes_are_expanded_from_official_definition(self) -> None:
        fixture = (
            b"<tabular><diag><name>S52.5</name><desc>Radius</desc>"
            b"<diag><name>S52.52</name><desc>Torus</desc>"
            b"<sevenChrDef><extension char=\"A\">initial</extension>"
            b"<extension char=\"D\">subsequent</extension></sevenChrDef>"
            b"<diag><name>S52.521</name><desc>Torus right</desc></diag></diag></diag></tabular>"
        )
        hierarchy = parse_tabular_hierarchy(
            fixture, "fx.xml", {"S52.521A"}, "FIXTURE", "h"
        )
        paths = _reportable_paths(hierarchy)

        self.assertIn("S52.521A", paths)
        self.assertNotIn("S52.521D", paths)

    def test_alphanumeric_second_character_codes_are_valid(self) -> None:
        self.assertEqual(canonical_code("QA00101"), "QA0.0101")
        self.assertEqual(canonical_code("C441021"), "C44.1021")

    def test_missing_hierarchy_edge_is_not_reconstructed_from_prefix(self) -> None:
        hierarchy = parse_tabular_hierarchy(
            b"<tabular><diag><name>E11</name><desc>Parent</desc></diag></tabular>",
            "minimal-fixture.xml",
            {"E11.65"},
            "FIXTURE",
            "fixture-hash",
        )

        self.assertNotIn("E11.65", _reportable_paths(hierarchy))

    def test_child_notes_are_not_misattributed_as_parent_local_notes(self) -> None:
        hierarchy = parse_tabular_hierarchy(
            b"<tabular><diag><name>E11</name><desc>Parent</desc><wrapper>"
            b"<diag><name>E11.65</name><desc>Child</desc>"
            b"<note>Child-only note</note></diag></wrapper></diag></tabular>",
            "nested-fixture.xml",
            {"E11.65"},
            "FIXTURE",
            "fixture-hash",
        )
        parent = next(node for node in hierarchy["nodes"] if node["code"] == "E11")
        child = next(node for node in hierarchy["nodes"] if node["code"] == "E11.65")

        self.assertEqual(parent["local_coding_notes"], [])
        self.assertEqual(
            child["local_coding_notes"],
            [{"kind": "note", "text": "Child-only note"}],
        )

    def test_xml_external_declarations_are_rejected(self) -> None:
        with self.assertRaisesRegex(KnowledgeBaseError, "external declarations"):
            parse_tabular_hierarchy(
                b'<!DOCTYPE x [<!ENTITY ext SYSTEM "file:///etc/passwd">]>'
                b"<tabular>&ext;</tabular>",
                "unsafe-fixture.xml",
                set(),
                "FIXTURE",
                "fixture-hash",
            )

    def test_index_entries_have_source_locators_and_manifest_binding(self) -> None:
        chunks = parse_index_chunks(INDEX_FIXTURE, "index-fixture.xml", "manifest-hash")

        self.assertTrue(chunks)
        self.assertIn("#entry-", chunks[0]["locator"])
        self.assertEqual(chunks[0]["source_manifest_hash"], "manifest-hash")


if __name__ == "__main__":
    unittest.main()
