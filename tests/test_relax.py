import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pemd_lite.project import PolymerSpec, Project, RunSpec
from pemd_lite.relax import BoxEstimator, RelaxOptions, RelaxRunner


class _FakeCommand:
    def __init__(self, calls, label):
        self._calls = calls
        self._label = label

    def run_local(self):
        self._calls.append(self._label)
        return self


class _FakeGromacs:
    def __init__(self):
        self.calls = []

    def commands_pdbtogro(self, *args, **kwargs):
        return _FakeCommand(self.calls, "pdb_to_gro")

    def gen_em_mdp_file(self, **kwargs):
        self.calls.append("gen_em_mdp")

    def commands_em(self, *args, **kwargs):
        return _FakeCommand(self.calls, "em")

    def gen_npt_mdp_file(self, **kwargs):
        self.calls.append("gen_npt_mdp")

    def commands_npt(self, *args, **kwargs):
        return _FakeCommand(self.calls, "npt")

    def gen_nvt_mdp_file(self, **kwargs):
        self.calls.append("gen_nvt_mdp")

    def commands_nvt(self, *args, **kwargs):
        return _FakeCommand(self.calls, "nvt")

    def commands_grotopdb(self, *args, **kwargs):
        return _FakeCommand(self.calls, "gro_to_pdb")


class RelaxRunnerTests(unittest.TestCase):
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
            run=RunSpec(relax_temperature=500, gpu=False),
            raw_config={},
        )

    def test_default_relax_order_remains_em_npt_nvt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = self._project(Path(tmpdir))
            fake_gmx = _FakeGromacs()
            runner = RelaxRunner(project)

            with patch.object(RelaxRunner, "_prepare_gmx", return_value=fake_gmx):
                result = runner.run(project.artifacts.forcefield_gro)

            self.assertEqual(result.completed_stages, ["pdb_to_gro", "em", "npt", "nvt"])
            self.assertEqual(
                fake_gmx.calls,
                [
                    "pdb_to_gro",
                    "gen_em_mdp",
                    "em",
                    "gen_npt_mdp",
                    "npt",
                    "gen_nvt_mdp",
                    "nvt",
                    "gro_to_pdb",
                ],
            )

    def test_relax_order_can_be_swapped_from_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = self._project(Path(tmpdir))
            fake_gmx = _FakeGromacs()
            runner = RelaxRunner(project)

            options = RelaxOptions(
                run_em=True,
                run_npt=True,
                run_nvt=True,
                stage_order=["em", "nvt", "npt"],
            )
            with patch.object(RelaxRunner, "_prepare_gmx", return_value=fake_gmx):
                result = runner.run(project.artifacts.forcefield_gro, options=options)

            self.assertEqual(result.completed_stages, ["pdb_to_gro", "em", "nvt", "npt"])
            self.assertEqual(
                fake_gmx.calls,
                [
                    "pdb_to_gro",
                    "gen_em_mdp",
                    "em",
                    "gen_nvt_mdp",
                    "nvt",
                    "gen_npt_mdp",
                    "npt",
                    "gro_to_pdb",
                ],
            )

    def test_nvt_stage_can_request_velocity_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = self._project(Path(tmpdir))
            fake_gmx = _FakeGromacs()
            recorded = {}
            runner = RelaxRunner(project)

            def record_nvt_kwargs(**kwargs):
                recorded.update(kwargs)
                fake_gmx.calls.append("gen_nvt_mdp")

            fake_gmx.gen_nvt_mdp_file = record_nvt_kwargs
            options = RelaxOptions(
                run_em=False,
                run_npt=False,
                run_nvt=True,
                nvt_gen_vel=True,
            )
            with patch.object(RelaxRunner, "_prepare_gmx", return_value=fake_gmx):
                result = runner.run(project.artifacts.forcefield_gro, options=options)

            self.assertEqual(result.completed_stages, ["pdb_to_gro", "nvt"])
            self.assertTrue(recorded["gen_vel"])

    def test_nvt_stage_can_override_tau_t(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = self._project(Path(tmpdir))
            fake_gmx = _FakeGromacs()
            recorded = {}
            runner = RelaxRunner(project)

            def record_nvt_kwargs(**kwargs):
                recorded.update(kwargs)
                fake_gmx.calls.append("gen_nvt_mdp")

            fake_gmx.gen_nvt_mdp_file = record_nvt_kwargs
            options = RelaxOptions(
                run_em=False,
                run_npt=False,
                run_nvt=True,
                nvt_tau_t_ps=0.1,
            )
            with patch.object(RelaxRunner, "_prepare_gmx", return_value=fake_gmx):
                runner.run(project.artifacts.forcefield_gro, options=options)

            self.assertEqual(recorded["tau_t_ps"], 0.1)

    def test_invalid_relax_stage_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = self._project(Path(tmpdir))
            runner = RelaxRunner(project)

            with self.assertRaises(ValueError):
                runner._effective_stage_order(
                    RelaxOptions(stage_order=["em", "anneal", "npt"])
                )


class BoxEstimatorTests(unittest.TestCase):
    def test_coords_from_pdb_falls_back_to_plain_pdb_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "test.pdb"
            pdb_path.write_text(
                "\n".join(
                    [
                        "TITLE     test",
                        "ATOM      1  C   UNL1    1       1.000   2.000   3.000  1.00  0.00",
                        "ATOM      2  O   UNL1    1       4.000   5.000   6.000  1.00  0.00",
                        "ENDMDL",
                        "",
                    ]
                )
            )

            with patch("pemd_lite.relax.Chem.MolFromPDBFile", return_value=None):
                coords = BoxEstimator._coords_from_pdb(pdb_path)

            self.assertEqual(coords.shape, (2, 3))
            self.assertTrue(
                np.allclose(coords, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            )


if __name__ == "__main__":
    unittest.main()
