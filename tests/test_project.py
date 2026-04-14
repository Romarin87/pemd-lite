import json
import tempfile
import unittest
from pathlib import Path

from pemd_lite import load


class ProjectLoadTests(unittest.TestCase):
    def test_load_project_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "polymer": {
                    "name": "poly_1",
                    "resname": "MOL",
                    "repeating_unit": "[*]CCO[*]",
                    "left_cap": "C[*]",
                    "right_cap": "[*]C",
                    "length": [4, 12],
                    "numbers": 20,
                    "charge": 0,
                    "scale": 1.0,
                },
                "Li_cation": {
                    "name": "Li",
                    "resname": "LI",
                    "charge": 1,
                    "scale": 0.8,
                    "numbers": 4,
                },
            }
            config_path = Path(tmpdir) / "md.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            project = load(config_path)

            self.assertEqual(project.polymer.name, "poly_1")
            self.assertEqual(project.polymer.length_short, 4)
            self.assertEqual(project.polymer.length_long, 12)
            self.assertEqual(len(project.small_molecules), 1)
            self.assertEqual(project.small_molecules[0].name, "Li")


if __name__ == "__main__":
    unittest.main()
