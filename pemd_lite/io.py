from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import MDAnalysis as mda
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem


def rdkitmol2xyz(name: str, mol: Chem.Mol, out_dir: str = ".", conf_id: int = 0) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    xyz_path = Path(out_dir) / f"{name}.xyz"
    try:
        Chem.MolToXYZFile(mol, str(xyz_path), confId=conf_id)
    except Exception:
        mol_path = Path(out_dir) / f"{name}.mol"
        Chem.MolToMolFile(mol, str(mol_path), confId=conf_id)
        ob_conversion = ob.OBConversion()
        ob_conversion.SetInAndOutFormats("mol", "xyz")
        ob_mol = ob.OBMol()
        ob_conversion.ReadFile(ob_mol, str(mol_path))
        ob_conversion.WriteFile(ob_mol, str(xyz_path))
    return str(xyz_path)


def smile_toxyz(name: str, smiles: str, out_dir: str = ".") -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    AllChem.EmbedMolecule(mol)
    mol.SetProp("_Name", name)
    AllChem.UFFOptimizeMolecule(mol)
    return rdkitmol2xyz(name, mol, out_dir, conf_id=-1)


def convert_gro_to_pdb(input_gro: Union[str, Path], output_pdb: Union[str, Path]) -> None:
    universe = mda.Universe(str(input_gro))
    with mda.Writer(str(output_pdb)) as writer:
        writer.write(universe.atoms)


def convert_rdkit_mol_to_mol2(mol: Chem.Mol, output_mol2: Union[str, Path], *, conf_id: int = 0) -> Path:
    output_path = Path(output_mol2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mol_path = output_path.with_suffix(".mol")
    Chem.MolToMolFile(mol, str(mol_path), confId=conf_id)

    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("mol", "mol2")
    ob_mol = ob.OBMol()
    if not ob_conversion.ReadFile(ob_mol, str(mol_path)):
        raise ValueError(f"Open Babel failed to read MOL file: {mol_path}")
    if not ob_conversion.WriteFile(ob_mol, str(output_path)):
        raise ValueError(f"Open Babel failed to write MOL2 file: {output_path}")
    return output_path


def extract_from_top(
    top_file: Union[str, Path],
    out_itp_file: Union[str, Path],
    *,
    nonbonded: bool = False,
    bonded: bool = False,
) -> None:
    sections_to_extract: list[str] = []
    if nonbonded:
        sections_to_extract = ["[ atomtypes ]"]
    elif bonded:
        sections_to_extract = [
            "[ moleculetype ]",
            "[ atoms ]",
            "[ bonds ]",
            "[ pairs ]",
            "[ angles ]",
            "[angles]",
            "[ dihedrals ]",
        ]

    lines = Path(top_file).read_text(encoding="utf-8").splitlines(keepends=True)
    extracted: list[str] = []
    current_section: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if stripped in sections_to_extract:
            current_section = stripped
            extracted.append(line)
        elif current_section and stripped.startswith(";"):
            extracted.append(line)
        elif current_section and stripped:
            extracted.append(line)
        elif current_section and stripped == "":
            extracted.append("\n")
            current_section = None

    Path(out_itp_file).write_text("".join(extracted), encoding="utf-8")
