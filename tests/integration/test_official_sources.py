from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from icd_kev_dataset_builder.config import load_config
from icd_kev_dataset_builder.kb import build_knowledge_base
from icd_kev_dataset_builder.sources import import_source

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY_VALUE = os.environ.get("ICD_OFFICIAL_SOURCE_DIR")
SOURCE_DIRECTORY = Path(SOURCE_DIRECTORY_VALUE).expanduser() if SOURCE_DIRECTORY_VALUE else None
REQUIRED_FILENAMES = {
    "table_and_index": "icd10cm-table-and-index-2027.zip",
    "code_descriptions": "icd10cm-code-descriptions-2027.zip",
    "guidelines": "ICD-10-CM-October-1-2026-FY27-Guidelines.pdf",
    "addenda": "icd10cm-addenda-2027.zip",
}


@unittest.skipUnless(
    SOURCE_DIRECTORY is not None and SOURCE_DIRECTORY.is_dir(),
    "NOT_RUN: set ICD_OFFICIAL_SOURCE_DIR to a directory of actual official FY2027 files",
)
class OfficialSourceIntegrationTests(unittest.TestCase):
    def test_real_sources_build_all_pilot_paths(self) -> None:
        assert SOURCE_DIRECTORY is not None
        missing = [
            str(SOURCE_DIRECTORY / filename)
            for filename in REQUIRED_FILENAMES.values()
            if not (SOURCE_DIRECTORY / filename).is_file()
        ]
        self.assertFalse(missing, f"Missing required CDC files: {missing}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / "config", root / "config")
            config = replace(
                load_config(root),
                root=root,
                data_dir=root / "data",
            )
            for source_id, filename in REQUIRED_FILENAMES.items():
                import_source(config, source_id, SOURCE_DIRECTORY / filename)
            audit = build_knowledge_base(config)

        self.assertEqual(audit["pilot_targets_validated"], 50)
        self.assertEqual(len(audit["pilot_target_paths"]), 50)
        self.assertTrue(audit["pilot_target_paths"]["E11.65"])


if __name__ == "__main__":
    unittest.main()
