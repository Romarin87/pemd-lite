import os
import tempfile
import unittest
from pathlib import Path

from pemd_lite.charges import (
    LigParGenBackend,
    _snapshot_generated_files,
)


class LigParGenOutputCollectionTests(unittest.TestCase):
    def test_collect_outputs_ignores_stale_files_from_previous_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ligdir = root / "ligpargen_poly_10"
            ligdir.mkdir()
            stale_itp = ligdir / "poly_10.gmx.itp"
            stale_gro = ligdir / "poly_10.gmx.gro"
            stale_itp.write_text("old itp", encoding="utf-8")
            stale_gro.write_text("old gro", encoding="utf-8")

            cwd = Path.cwd()
            os.chdir(root)
            try:
                snapshot = _snapshot_generated_files([ligdir, root, Path("/tmp")], ["poly_10", "PAA"])
                backend = LigParGenBackend()
                found_itp, found_gro, found_csv = backend._collect_outputs(
                    ligdir=ligdir,
                    name="poly_10",
                    resname="PAA",
                    gmx_itp=stale_itp,
                    gmx_gro=stale_gro,
                    csv_path=ligdir / "poly_10.csv",
                    snapshot=snapshot,
                )
            finally:
                os.chdir(cwd)

            self.assertIsNone(found_itp)
            self.assertIsNone(found_gro)
            self.assertIsNone(found_csv)

    def test_collect_outputs_can_find_new_files_even_if_old_primary_stem_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ligdir = root / "ligpargen_poly_10"
            ligdir.mkdir()
            (ligdir / "poly_10.gmx.itp").write_text("old itp", encoding="utf-8")
            (ligdir / "poly_10.gmx.gro").write_text("old gro", encoding="utf-8")

            cwd = Path.cwd()
            os.chdir(root)
            try:
                snapshot = _snapshot_generated_files([ligdir, root, Path("/tmp")], ["poly_10", "PAA"])
                (ligdir / "PAA.gmx.itp").write_text("new itp", encoding="utf-8")
                (ligdir / "PAA.gmx.gro").write_text("new gro", encoding="utf-8")
                backend = LigParGenBackend()
                found_itp, found_gro, found_csv = backend._collect_outputs(
                    ligdir=ligdir,
                    name="poly_10",
                    resname="PAA",
                    gmx_itp=ligdir / "captured.gmx.itp",
                    gmx_gro=ligdir / "captured.gmx.gro",
                    csv_path=ligdir / "captured.csv",
                    snapshot=snapshot,
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(found_itp, ligdir / "captured.gmx.itp")
            self.assertEqual(found_gro, ligdir / "captured.gmx.gro")
            self.assertIsNone(found_csv)


if __name__ == "__main__":
    unittest.main()
