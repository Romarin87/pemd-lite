import json
import tempfile
import unittest
from pathlib import Path

from pemd_lite.table import generate_projects_from_table


class TableGenerationTests(unittest.TestCase):
    def test_generate_projects_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "input.csv"
            csv_path.write_text(
                "Name,SMILES,DP,length_short,Num_chain,Num_salt,left_cap,right_cap\n"
                "poly_2,[*]CC[*],10,4,8,2,C[*],[*]C\n",
                encoding="utf-8",
            )

            template_json = tmp_path / "template.json"
            template_json.write_text(
                json.dumps(
                    {
                        "polymer": {
                            "name": "",
                            "repeating_unit": "",
                            "left_cap": "",
                            "right_cap": "",
                            "length": [4, 4],
                            "numbers": 1,
                            "resname": "MOL",
                            "charge": 0,
                            "scale": 1.0,
                        },
                        "Li_cation": {"numbers": 0},
                        "salt_anion": {"numbers": 0},
                    }
                ),
                encoding="utf-8",
            )
            template_py = tmp_path / "template.py"
            template_py.write_text(
                "RUN_BUILD_POLYMER = True\n"
                "RUN_POLYMER_FORCEFIELD = True\n"
                "RUN_SMALL_MOLECULE_FF = True\n"
                "RUN_RELAX_CHAIN = True\n"
                "RUN_PACK_CELL = True\n"
                "RUN_BOX_MD = True\n",
                encoding="utf-8",
            )

            generated = generate_projects_from_table(
                xlsx_path=csv_path,
                out_base=tmp_path / "out",
                template_json=template_json,
                template_py=template_py,
                stage="pack_cell",
            )

            self.assertEqual([project.name for project in generated], ["poly_2"])
            rendered = json.loads((generated[0].root / "md.json").read_text(encoding="utf-8"))
            self.assertEqual(rendered["polymer"]["length"], [4, 10])
            self.assertEqual(rendered["Li_cation"]["numbers"], 2)
            self.assertEqual(rendered["salt_anion"]["numbers"], 2)


if __name__ == "__main__":
    unittest.main()
