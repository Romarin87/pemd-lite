import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types


if "MDAnalysis" not in sys.modules:
    sys.modules["MDAnalysis"] = types.SimpleNamespace(Universe=object, Writer=object)
if "openbabel" not in sys.modules:
    sys.modules["openbabel"] = types.SimpleNamespace(openbabel=types.SimpleNamespace())
if "parmed" not in sys.modules:
    sys.modules["parmed"] = types.SimpleNamespace(load_file=lambda *args, **kwargs: object())
if "foyer" not in sys.modules:
    sys.modules["foyer"] = types.SimpleNamespace(Forcefield=object)

from pemd_lite.pipeline import Pipeline
from pemd_lite.project import PolymerSpec, Project, RunSpec
import pemd_lite.forcefield as forcefield_mod
import pemd_lite.polymer as polymer_mod
import pemd_lite.relax as relax_mod


class PipelineFlowTests(unittest.TestCase):
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

    def test_relax_uses_parameterized_structure_from_forcefield(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = self._project(Path(tmpdir))
            forcefield = SimpleNamespace(polymer_gro=project.root / "MD_dir" / "poly_10_forcefield_N20.gro")
            polymer = SimpleNamespace(
                short_mol=object(),
                long_mol=object(),
                short_pdb=project.root / "poly_10_build_N4.pdb",
                long_pdb=project.root / "poly_10_build_N20.pdb",
            )

            with patch.object(polymer_mod.PolymerBuilder, "build_required", return_value=polymer), \
                patch.object(forcefield_mod.ForcefieldGenerator, "generate_polymer", return_value=forcefield), \
                patch.object(forcefield_mod.ForcefieldGenerator, "generate_small_molecule_forcefields"), \
                patch.object(relax_mod.RelaxRunner, "run", return_value=SimpleNamespace(relaxed_pdb=project.root / "MD_dir" / "poly_10_relax_N20.pdb")) as run_relax:
                Pipeline(project).run_until("relax_chain")

            run_relax.assert_called_once_with(forcefield.polymer_gro)


if __name__ == "__main__":
    unittest.main()
