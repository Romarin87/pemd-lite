from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil

from .packmol import PEMDPackmol

from .project import Project
from .relax import BoxEstimate

logger = logging.getLogger(__name__)

@dataclass
class PackResult:
    pack_input: Path
    pack_pdb: Path


class PackBuilder:
    def __init__(self, project: Project):
        self.project = project

    def _stage_packmol_inputs(self, artifacts) -> None:
        polymer_src = artifacts.relaxed_pdb
        polymer_dst = artifacts.md_dir / f"{self.project.polymer.name}.pdb"
        if not polymer_src.exists():
            raise FileNotFoundError(f"Relaxed polymer PDB not found for packmol: {polymer_src}")
        if polymer_src.resolve() != polymer_dst.resolve():
            shutil.copyfile(polymer_src, polymer_dst)
            logger.info("Staged polymer packmol input: %s -> %s", polymer_src, polymer_dst)

    def run(self, *, add_length_a: float) -> PackResult:
        artifacts = self.project.artifacts
        artifacts.md_dir.mkdir(parents=True, exist_ok=True)
        self._stage_packmol_inputs(artifacts)
        molecules = {self.project.polymer.name: self.project.polymer.numbers}
        for spec in self.project.molecule_specs():
            if spec.numbers > 0:
                molecules[spec.name] = spec.numbers
        runner = PEMDPackmol(
            artifacts.md_dir,
            molecules,
            self.project.run.density,
            add_length_a,
            artifacts.pack_input.name,
            artifacts.pack_pdb.name,
        )
        logger.info("Running packmol: molecules=%s density=%s add_length_a=%s", molecules, self.project.run.density, add_length_a)
        runner.generate_input_file()
        runner.run_local()
        logger.info("Packmol completed: input=%s output=%s", artifacts.pack_input, artifacts.pack_pdb)
        return PackResult(pack_input=artifacts.pack_input, pack_pdb=artifacts.pack_pdb)
