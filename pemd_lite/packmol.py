from __future__ import annotations

import os
import random
import subprocess
import logging
from collections import defaultdict
from pathlib import Path
from shutil import which

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

def calc_mol_weight(pdb_file: str | Path) -> float:
    pdb_file = str(pdb_file)
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol)
            return float(Descriptors.MolWt(mol))
        raise ValueError(f"RDKit failed to parse PDB file: {pdb_file}")
    except Exception:
        atom_counts: dict[str, int] = defaultdict(int)
        with open(pdb_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    element = line[76:78].strip()
                    if not element:
                        atom_name = line[12:16].strip()
                        element = "".join(ch for ch in atom_name if ch.isalpha()).upper()[:2]
                    atom_counts[element] += 1
        atomic_weights = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "P": 30.974,
            "S": 32.06,
            "CL": 35.45,
            "BR": 79.904,
            "I": 126.904,
            "FE": 55.845,
            "ZN": 65.38,
        }
        total = 0.0
        for atom, count in atom_counts.items():
            weight = atomic_weights.get(atom.upper())
            if weight is None:
                raise ValueError(f"Unknown atom type '{atom}' in {pdb_file}")
            total += weight * count
        return total


def calculate_box_size(numbers, pdb_files, density):
    total_mass = 0.0
    for num, file in zip(numbers, pdb_files):
        total_mass += calc_mol_weight(file) * num / 6.022e23
    total_volume = total_mass / density
    return (total_volume * 1e24) ** (1 / 3)


class PEMDPackmol:
    def __init__(self, work_dir, molecule_list, density, add_length, packinp_name="pack.inp", packpdb_name="pack_cell.pdb"):
        self.work_dir = work_dir
        self.molecule_list = (
            [{"name": name, "number": number} for name, number in molecule_list.items()]
            if isinstance(molecule_list, dict)
            else molecule_list
        )
        self.density = density
        self.add_length = add_length
        self.packinp_name = packinp_name
        self.packpdb_name = packpdb_name
        self.compounds = [molecule["name"] for molecule in self.molecule_list]
        self.numbers = [molecule["number"] for molecule in self.molecule_list]

    def generate_input_file(self):
        os.makedirs(self.work_dir, exist_ok=True)
        packinp_path = os.path.join(self.work_dir, self.packinp_name)
        pdb_filenames = [f"{compound}.pdb" for compound in self.compounds]
        pdb_filepaths = [os.path.join(self.work_dir, f"{compound}.pdb") for compound in self.compounds]
        box_length = calculate_box_size(self.numbers, pdb_filepaths, self.density) + self.add_length
        text = "tolerance 2.0\n"
        text += "add_box_sides 1.2\n"
        text += f"output {self.packpdb_name}\n"
        text += "filetype pdb\n\n"
        text += f"seed {random.randint(1, 100000)}\n\n"
        for number, filename in zip(self.numbers, pdb_filenames):
            text += f"structure {filename}\n"
            text += f"  number {number}\n"
            text += f"  inside box 0.0 0.0 0.0 {box_length:.2f} {box_length:.2f} {box_length:.2f}\n"
            text += "end structure\n\n"
        with open(packinp_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        logger.info("Packmol input generated: %s box_length=%.2f", packinp_path, box_length)
        return packinp_path

    def run_local(self):
        current_path = os.getcwd()
        if not which("packmol"):
            raise RuntimeError("packmol executable is required in PATH.")
        try:
            os.chdir(self.work_dir)
            logger.info("Executing packmol in %s using %s", self.work_dir, self.packinp_name)
            result = subprocess.run(
                f"packmol < {self.packinp_name}",
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            Path("pack.out").write_text(result.stdout, encoding="utf-8")
            logger.info("Packmol completed successfully. stdout saved to %s", Path(self.work_dir) / "pack.out")
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in [172, 173]:
                logger.error("Packmol failed rc=%s stdout=%s stderr=%s", exc.returncode, exc.stdout[-2000:], exc.stderr[-2000:])
                raise ValueError(
                    f"Packmol failed with error code {exc.returncode}.\nStdout:\n{exc.stdout}\nStderr:\n{exc.stderr}"
                ) from exc
        finally:
            os.chdir(current_path)
