import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pemd_lite.pack import PackBuilder
from pemd_lite.project import PolymerSpec, Project, RunSpec


class _FakePackmol:
    def __init__(self, *args, **kwargs):
        self.generated = False
        self.ran = False

    def generate_input_file(self):
        self.generated = True

    def run_local(self):
        self.ran = True


class PackBuilderTests(unittest.TestCase):
    def _project(self, root: Path) -> Project:
        return Project(
            root=root,
            config_path=root / "md.json",
            polymer=PolymerSpec(
                name="poly_10",
                resname="MOL",
                repeating_unit="[*]CC[*]",
                left_cap="C[*]",
                right_cap="[*]C",
                length_short=4,
                length_long=20,
                numbers=20,
                charge=0.0,
                scale=1.0,
            ),
            run=RunSpec(),
            raw_config={},
        )

    def test_pack_can_stage_non_relaxed_polymer_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = self._project(root)
            source_pdb = root / "poly_10_build_N20.pdb"
            source_pdb.write_text("ATOM\n", encoding="utf-8")

            fake_runner = _FakePackmol()
            with patch("pemd_lite.pack.PEMDPackmol", return_value=fake_runner):
                result = PackBuilder(project).run(add_length_a=120.0, polymer_pdb=source_pdb)

            staged = project.artifacts.md_dir / "poly_10.pdb"
            self.assertTrue(staged.exists())
            self.assertEqual(staged.read_text(encoding="utf-8"), "ATOM\n")
            self.assertEqual(result.pack_pdb, project.artifacts.pack_pdb)
            self.assertTrue(fake_runner.generated)
            self.assertTrue(fake_runner.ran)


if __name__ == "__main__":
    unittest.main()
