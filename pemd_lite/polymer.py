from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple

from rdkit import Chem

from .build_core import gen_copolymer_3D, mol_to_pdb

from .errors import ProjectValidationError
from .project import Project

logger = logging.getLogger(__name__)

@dataclass
class PolymerBuildOptions:
    optimize_every_n_steps: int = 2
    resume: bool = True
    force_rebuild_short_chain: bool = False
    force_rebuild_long_chain: bool = False


@dataclass
class PolymerBuildResult:
    short_mol: Chem.Mol
    long_mol: Chem.Mol
    short_pdb: Path
    long_pdb: Path


class PolymerBuilder:
    def __init__(self, project: Project, options: Optional[PolymerBuildOptions] = None):
        self.project = project
        self.options = options or PolymerBuildOptions(
            optimize_every_n_steps=project.polymer.optimize_every_n_steps,
            force_rebuild_short_chain=project.run.force_rebuild_short_chain,
            force_rebuild_long_chain=project.run.force_rebuild_long_chain,
        )

    def _load_existing(self, path: Path) -> Optional[Chem.Mol]:
        if not path.exists():
            return None
        logger.info("Reusing existing polymer chain PDB: %s", path)
        return Chem.MolFromPDBFile(str(path), removeHs=False)

    def _build_chain(self, length: int) -> Tuple[Chem.Mol, Path]:
        poly = self.project.polymer
        logger.info("Building polymer chain: name=%s length=%s optimize_every_n_steps=%s", poly.name, length, self.options.optimize_every_n_steps)
        mol = gen_copolymer_3D(
            smiles_A=poly.repeating_unit,
            smiles_B=poly.repeating_unit,
            name=poly.name,
            mode="homopolymer",
            length=length,
            optimize_every_n_steps=self.options.optimize_every_n_steps,
            left_cap_smiles=poly.left_cap,
            right_cap_smiles=poly.right_cap,
        )
        path = self.project.artifacts.polymer_chain_pdb(length)
        mol_to_pdb(self.project.root, mol, poly.name, poly.resname, path.name)
        logger.info("Polymer chain written: %s", path)
        return mol, path

    def build_required(self) -> PolymerBuildResult:
        short_path = self.project.artifacts.polymer_chain_pdb(self.project.polymer.length_short)
        long_path = self.project.artifacts.polymer_chain_pdb(self.project.polymer.length_long)

        short_mol = None
        long_mol = None
        if self.options.resume and not self.options.force_rebuild_short_chain:
            short_mol = self._load_existing(short_path)
        if self.options.resume and not self.options.force_rebuild_long_chain:
            long_mol = self._load_existing(long_path)

        if self.options.force_rebuild_short_chain:
            logger.warning("Force rebuild enabled for short chain: %s", short_path)
        if self.options.force_rebuild_long_chain:
            logger.warning("Force rebuild enabled for long chain: %s", long_path)

        if short_mol is None:
            short_mol, short_path = self._build_chain(self.project.polymer.length_short)
        if long_mol is None:
            long_mol, long_path = self._build_chain(self.project.polymer.length_long)

        if short_mol is None or long_mol is None:
            raise ProjectValidationError("Failed to obtain short and long polymer structures")
        return PolymerBuildResult(short_mol=short_mol, long_mol=long_mol, short_pdb=short_path, long_pdb=long_path)
