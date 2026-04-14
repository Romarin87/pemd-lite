from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Literal

from .project import Project, load
from .polymer import PolymerBuildResult


StageName = Literal["build_polymer", "generate_forcefield", "relax_chain", "pack_cell", "run_md"]


@dataclass
class PipelineResult:
    polymer: PolymerBuildResult | None = None
    forcefield: object | None = None
    relax: object | None = None
    box: object | None = None
    pack: object | None = None
    md: object | None = None


class Pipeline:
    def __init__(self, project: Project):
        self.project = project

    def run_until(self, stage: StageName) -> PipelineResult:
        from .polymer import PolymerBuilder

        result = PipelineResult()
        result.polymer = PolymerBuilder(self.project).build_required()
        if stage == "build_polymer":
            return result

        from .charges import LigParGenBackend
        from .forcefield import ForcefieldGenerator

        ff = ForcefieldGenerator(self.project, charge_backend=LigParGenBackend())
        result.forcefield = ff.generate_polymer(
            short_mol=result.polymer.short_mol,
            long_mol=result.polymer.long_mol,
            short_pdb=result.polymer.short_pdb,
            long_pdb=result.polymer.long_pdb,
        )
        ff.generate_small_molecule_forcefields()
        if stage == "generate_forcefield":
            return result

        from .relax import BoxEstimator, RelaxRunner
        from .pack import PackBuilder

        result.relax = RelaxRunner(self.project).run(result.polymer.long_pdb)
        if stage == "relax_chain":
            return result

        estimator = BoxEstimator()
        result.box = estimator.estimate_from_pdb(
            result.relax.relaxed_pdb,
            scale=self.project.run.add_length_scale,
            min_add_a=self.project.run.add_length_min_a,
        )
        result.pack = PackBuilder(self.project).run(add_length_a=result.box.add_length_a)
        if stage == "pack_cell":
            return result

        from .md import GromacsRunner

        result.md = GromacsRunner(self.project).run_pack_md(result.pack.pack_pdb.name)
        return result


def run_until(project: Project | str, stage: StageName) -> PipelineResult:
    if not isinstance(project, Project):
        project = load(project)
    return Pipeline(project).run_until(stage)


def run_polymer_pack_md(project: Project | str) -> PipelineResult:
    return run_until(project, "run_md")
