from __future__ import annotations

import random
from pathlib import Path

from rdkit import Chem

from . import polymer_core


def gen_copolymer_3D(
    smiles_A,
    smiles_B,
    *,
    name: str | None = None,
    mode: str | None = None,
    length: int | None = None,
    frac_A: float = 0.5,
    block_sizes: list[int] | None = None,
    sequence: list[str] | None = None,
    optimize_every_n_steps: int = 1,
    left_cap_smiles: str | None = None,
    right_cap_smiles: str | None = None,
):
    if sequence is None:
        if mode == "homopolymer":
            if length is None:
                raise ValueError("length is required for homopolymer mode")
            sequence = ["A"] * length
        elif mode == "random":
            if length is None:
                raise ValueError("length is required for random mode")
            sequence = ["A" if random.random() < frac_A else "B" for _ in range(length)]
        elif mode == "alternating":
            if length is None:
                raise ValueError("length is required for alternating mode")
            sequence = ["A" if i % 2 == 0 else "B" for i in range(length)]
        elif mode == "block":
            if not block_sizes:
                raise ValueError("block_sizes is required for block mode")
            sequence = []
            for i, blk in enumerate(block_sizes):
                mon = "A" if i % 2 == 0 else "B"
                sequence += [mon] * blk
        else:
            raise ValueError("mode must be provided when sequence is None")

    return polymer_core.gen_sequence_copolymer_3D(
        name,
        smiles_A,
        smiles_B,
        sequence,
        optimize_every_n_steps=optimize_every_n_steps,
        left_cap_smiles=left_cap_smiles,
        right_cap_smiles=right_cap_smiles,
    )


def mol_to_pdb(work_dir, mol, name, resname, pdb_filename):
    work_path = Path(work_dir)
    pdb_file = work_path / pdb_filename
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{atom.GetSymbol():>4}")
        info.SetResidueName(str(resname)[:3].upper())
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(False)
        atom.SetMonomerInfo(info)
    Chem.MolToPDBFile(mol, str(pdb_file), confId=0)
