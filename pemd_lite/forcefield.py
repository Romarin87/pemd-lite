from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple

import parmed as pmd
from foyer import Forcefield as FoyerForcefield
from rdkit import Chem

from . import io
from .charge_transfer import apply_chg_to_poly, gen_ff_from_data
from .xml import XMLGenerator

from .charges import ChargeBackend, ChargeResult, DatabaseBackend, LigParGenBackend
from .errors import AtomTypingError, MissingBondParameterError, TopologyMismatchError
from .project import Project

logger = logging.getLogger(__name__)

@dataclass
class ChargeTransferPolicy:
    min_short_length: int = 3
    end_repeating_units: int = 1


@dataclass
class ForcefieldResult:
    polymer_bonded_itp: Path
    polymer_nonbonded_itp: Path
    polymer_gro: Path
    polymer_pdb: Path
    charge_result: ChargeResult


class ForcefieldGenerator:
    def __init__(
        self,
        project: Project,
        charge_backend: Optional[ChargeBackend] = None,
        charge_policy: Optional[ChargeTransferPolicy] = None,
    ):
        self.project = project
        self.charge_backend = charge_backend or LigParGenBackend()
        self.charge_policy = charge_policy or ChargeTransferPolicy()

    def _generate_xml(self, ordered_pdb: Path, gmx_itp: Path) -> Path:
        mol = Chem.MolFromPDBFile(str(ordered_pdb), removeHs=False)
        if mol is None:
            raise TopologyMismatchError(f"Could not parse ordered PDB: {ordered_pdb}")
        xml_path = self.project.artifacts.polymer_xml
        logger.info("Generating polymer XML from ordered PDB and LigParGen ITP: pdb=%s itp=%s xml=%s", ordered_pdb, gmx_itp, xml_path)
        XMLGenerator(gmx_itp, mol, xml_path).run()
        return xml_path

    def _classify_parameterization_error(self, exc: Exception) -> Exception:
        message = str(exc)
        if "Found no types for atom" in message:
            return AtomTypingError(message)
        if "Parameters have not been assigned to all bonds" in message:
            return MissingBondParameterError(message)
        return TopologyMismatchError(message)

    def _parameterize_long_chain(self, xml_path: Path, long_mol: Chem.Mol, long_pdb: Path) -> Tuple[Path, Path, Path]:
        artifacts = self.project.artifacts
        artifacts.md_dir.mkdir(parents=True, exist_ok=True)
        try:
            mol2_path = artifacts.md_dir / f"{self.project.polymer.name}_forcefield_N{self.project.polymer.length_long}.mol2"
            io.convert_rdkit_mol_to_mol2(long_mol, mol2_path)
            logger.info("Parameterizing long chain with foyer: xml=%s mol2=%s pdb=%s", xml_path, mol2_path, long_pdb)
            untyped = pmd.load_file(str(mol2_path), structure=True)
            typed = FoyerForcefield(forcefield_files=str(xml_path)).apply(
                untyped, verbose=True, use_residue_map=False
            )
        except Exception as exc:  # noqa: BLE001
            raise self._classify_parameterization_error(exc) from exc

        top_path = artifacts.forcefield_top
        gro_path = artifacts.forcefield_gro
        pdb_path = artifacts.forcefield_pdb
        bonded_itp = artifacts.bonded_itp
        nonbonded_itp = artifacts.nonbonded_itp

        typed.save(str(top_path), overwrite=True)
        typed.save(str(gro_path), overwrite=True)
        io.extract_from_top(top_path, nonbonded_itp, nonbonded=True, bonded=False)
        io.extract_from_top(top_path, bonded_itp, nonbonded=False, bonded=True)
        io.convert_gro_to_pdb(gro_path, pdb_path)
        logger.info("Long-chain parameterization outputs ready: top=%s gro=%s bonded=%s nonbonded=%s pdb=%s", top_path, gro_path, bonded_itp, nonbonded_itp, pdb_path)
        return bonded_itp, nonbonded_itp, pdb_path

    def generate_polymer(
        self,
        *,
        short_mol: Chem.Mol,
        long_mol: Chem.Mol,
        short_pdb: Path,
        long_pdb: Path,
    ) -> ForcefieldResult:
        if self.project.polymer.length_short < self.charge_policy.min_short_length:
            raise TopologyMismatchError(
                f"length_short={self.project.polymer.length_short} is below the minimum supported "
                f"value {self.charge_policy.min_short_length} for left/mid/right charge transfer."
            )
        if self.charge_policy.end_repeating_units < 1:
            raise TopologyMismatchError("end_repeating_units must be >= 1")
        logger.info(
            "Generating polymer forcefield: short_len=%s long_len=%s min_short_length=%s end_repeating_units=%s",
            self.project.polymer.length_short,
            self.project.polymer.length_long,
            self.charge_policy.min_short_length,
            self.charge_policy.end_repeating_units,
        )

        charge_result = self.charge_backend.generate_polymer_charges(
            project=self.project,
            short_pdb=short_pdb,
            short_mol=short_mol,
            short_smiles=Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(short_mol))),
        )
        xml_path = self._generate_xml(charge_result.ordered_pdb, charge_result.gmx_itp)
        bonded_itp, nonbonded_itp, pdb_path = self._parameterize_long_chain(xml_path, long_mol, long_pdb)

        apply_chg_to_poly(
            str(self.project.root),
            short_mol,
            long_mol,
            bonded_itp.name,
            charge_result.charge_table.frame,
            self.project.polymer.repeating_unit,
            self.charge_policy.end_repeating_units,
            self.project.polymer.scale,
            self.project.polymer.charge,
            length_short=self.project.polymer.length_short,
            left_cap_smiles=self.project.polymer.left_cap,
            right_cap_smiles=self.project.polymer.right_cap,
        )
        logger.info("Polymer bonded/nonbonded files updated with transferred charges.")
        return ForcefieldResult(
            polymer_bonded_itp=bonded_itp,
            polymer_nonbonded_itp=nonbonded_itp,
            polymer_gro=self.project.artifacts.forcefield_gro,
            polymer_pdb=pdb_path,
            charge_result=charge_result,
        )

    def generate_small_molecule_forcefields(self) -> None:
        for molecule in self.project.molecule_specs():
            if molecule.numbers <= 0:
                continue
            if molecule.ff_source != "database":
                continue
            logger.info("Generating small-molecule forcefield from bundled data: name=%s count=%s", molecule.name, molecule.numbers)
            gen_ff_from_data(str(self.project.root), molecule.name, molecule.scale, molecule.charge)
