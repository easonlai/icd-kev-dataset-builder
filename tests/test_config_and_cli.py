from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from icd_kev_dataset_builder.cli import main
from icd_kev_dataset_builder.config import load_config
from icd_kev_dataset_builder.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationAndCliTests(unittest.TestCase):
    def test_default_configuration_uses_the_exact_fy2027_sources(self) -> None:
        config = load_config(ROOT)

        self.assertEqual(config.release, "FY2027")
        self.assertEqual(config.release_year, 2027)
        self.assertEqual(
            config.sources["table_and_index"].url,
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/"
            "ICD10CM/2027/icd10cm-table-and-index-2027.zip",
        )
        self.assertEqual(len([s for s in config.sources.values() if s.required]), 4)

    def test_configuration_rejects_non_https_or_unallowlisted_hosts(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.toml"
            source = (ROOT / "config" / "default.toml").read_text(encoding="utf-8")
            source = source.replace(
                'allowed_hosts = ["ftp.cdc.gov", "www.cdc.gov"]',
                'allowed_hosts = ["example.com"]',
            )
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "not in the configured allowlist"):
                load_config(ROOT, config_path=path)

    def test_doctor_is_offline_and_reports_missing_sources(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            (root / "config" / "default.toml").write_text(
                (ROOT / "config" / "default.toml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            status = main(["--root", str(root), "doctor"])

        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
