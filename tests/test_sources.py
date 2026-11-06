from __future__ import annotations

import stat
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from icd_kev_dataset_builder.config import load_config
from icd_kev_dataset_builder.errors import SourceError
from icd_kev_dataset_builder.sources import (
    canonical_manifest_hash,
    import_source,
    inspect_zip,
    read_source_lock,
    validate_source_file,
)

ROOT = Path(__file__).resolve().parents[1]


def create_zip(path: Path, members: dict[str, bytes], compression: int = zipfile.ZIP_STORED) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


class SourceSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = load_config(ROOT)
        self.config = replace(config, root=self.root, data_dir=self.root / "data")

    def test_manual_import_preserves_bytes_and_records_manifest_hash(self) -> None:
        source = self.root / "manual.zip"
        create_zip(source, {"FY2027/ICD10CM_Tabular.xml": b"<sample/>"})
        original_bytes = source.read_bytes()

        destination = import_source(self.config, "table_and_index", source)
        lock = read_source_lock(self.config)

        self.assertEqual(destination.read_bytes(), original_bytes)
        self.assertEqual(lock["manifest_hash"], canonical_manifest_hash(lock))
        entry = lock["sources"]["table_and_index"]
        self.assertEqual(entry["original_filename"], "manual.zip")
        self.assertEqual(entry["verified_release_metadata"]["verified"], True)
        self.assertEqual(len(entry["sha256"]), 64)

    def test_reimport_updates_only_that_sources_lock_entry(self) -> None:
        first = self.root / "first.zip"
        second = self.root / "second.zip"
        create_zip(first, {"FY2027/first.txt": b"first"})
        create_zip(second, {"FY2027/second.txt": b"second"})
        import_source(self.config, "table_and_index", first)
        import_source(self.config, "code_descriptions", second)

        replacement = self.root / "replacement.zip"
        create_zip(replacement, {"FY2027/replacement.txt": b"replacement"})
        import_source(self.config, "table_and_index", replacement)
        lock = read_source_lock(self.config)

        self.assertEqual(set(lock["sources"]), {"table_and_index", "code_descriptions"})
        self.assertEqual(
            lock["sources"]["table_and_index"]["original_filename"], "replacement.zip"
        )

    def test_tampered_locked_file_is_rejected(self) -> None:
        source = self.root / "manual.zip"
        create_zip(source, {"FY2027/codes.txt": b"E119 | Example"})
        destination = import_source(self.config, "code_descriptions", source)
        destination.write_bytes(destination.read_bytes() + b"changed")

        with self.assertRaisesRegex(SourceError, "hash changed"):
            read_source_lock(self.config)

    def test_error_page_cannot_be_imported_as_zip(self) -> None:
        source = self.root / "download.zip"
        source.write_text("<html>access denied</html>", encoding="utf-8")

        with self.assertRaisesRegex(SourceError, "error page|ZIP signature"):
            import_source(self.config, "table_and_index", source)
        self.assertFalse(self.config.source_lock_path.exists())

    def test_zip_traversal_is_rejected(self) -> None:
        source = self.root / "unsafe.zip"
        create_zip(source, {"../../escape.txt": b"not allowed"})

        with self.assertRaisesRegex(SourceError, "Unsafe ZIP member"):
            validate_source_file(source, self.config.sources["table_and_index"], self.config)

    def test_zip_symbolic_link_is_rejected(self) -> None:
        source = self.root / "symlink.zip"
        with zipfile.ZipFile(source, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")

        with self.assertRaisesRegex(SourceError, "symbolic link"):
            inspect_zip(source, self.config)

    def test_zip_member_expansion_is_bounded(self) -> None:
        source = self.root / "compressed.zip"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("large.txt", b"A" * 100_000)
        strict_config = replace(self.config, max_compression_ratio=2)

        with self.assertRaisesRegex(SourceError, "compression-ratio"):
            inspect_zip(source, strict_config)

    def test_manifest_hash_detects_changed_metadata(self) -> None:
        manifest = {"release": "FY2027", "sources": {}}
        before = canonical_manifest_hash(manifest)
        manifest["release"] = "FY2026"

        self.assertNotEqual(before, canonical_manifest_hash(manifest))

    def test_zip_archive_with_valid_structural_fixture_passes(self) -> None:
        source = self.root / "fixture.zip"
        create_zip(source, {"fixture/codes.txt": b"Fixture only"})

        members = inspect_zip(source, self.config)

        self.assertEqual(members[0]["filename"], "fixture/codes.txt")
        self.assertEqual(members[0]["size"], len(b"Fixture only"))


if __name__ == "__main__":
    unittest.main()
