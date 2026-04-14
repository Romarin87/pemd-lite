from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing_extensions import Literal

from .gromacs import PEMDGROMACS

from .project import Project

logger = logging.getLogger(__name__)

MDStageName = Literal["pdb_to_gro", "em", "nvt", "npt", "production"]


@dataclass
class MDStep:
    stage: MDStageName
    output: str
    input_gro: str | None = None
    input_pdb: str | None = None
    steps: int | None = None
    temperature: float | None = None
    pressure: float | None = None


@dataclass
class StepResult:
    stage: MDStageName
    input_file: str
    output_file: str


@dataclass
class GromacsResult:
    work_dir: Path
    completed_stages: list[str]
    step_results: list[StepResult] = field(default_factory=list)


class MDFlow:
    def __init__(self, runner: GromacsRunner, initial_pdb: str):
        self.runner = runner
        self.initial_pdb = initial_pdb
        self.steps: list[MDStep] = []

    def pdb_to_gro(self, input_pdb: str | None = None) -> MDFlow:
        self.steps.append(
            MDStep(
                stage="pdb_to_gro",
                input_pdb=input_pdb or self.initial_pdb,
                output="boxmd_conf.gro",
            )
        )
        return self

    def em(self, *, input_gro: str | None = None, output: str = "em") -> MDFlow:
        self.steps.append(MDStep(stage="em", input_gro=input_gro, output=output))
        return self

    def nvt(
        self,
        *,
        input_gro: str | None = None,
        output: str = "nvt",
        steps: int | None = None,
        temperature: float | None = None,
    ) -> MDFlow:
        self.steps.append(
            MDStep(
                stage="nvt",
                input_gro=input_gro,
                output=output,
                steps=steps,
                temperature=temperature,
            )
        )
        return self

    def npt(
        self,
        *,
        input_gro: str | None = None,
        output: str = "npt",
        steps: int | None = None,
        temperature: float | None = None,
        pressure: float | None = None,
    ) -> MDFlow:
        self.steps.append(
            MDStep(
                stage="npt",
                input_gro=input_gro,
                output=output,
                steps=steps,
                temperature=temperature,
                pressure=pressure,
            )
        )
        return self

    def production(
        self,
        *,
        input_gro: str | None = None,
        output: str = "production",
        steps: int | None = None,
        temperature: float | None = None,
    ) -> MDFlow:
        self.steps.append(
            MDStep(
                stage="production",
                input_gro=input_gro,
                output=output,
                steps=steps,
                temperature=temperature,
            )
        )
        return self

    def run(self) -> GromacsResult:
        return self.runner.run_flow(self)


class GromacsRunner:
    def __init__(self, project: Project):
        self.project = project

    def _molecules(self) -> list[dict[str, object]]:
        molecules = [
            {
                "name": self.project.polymer.name,
                "number": self.project.polymer.numbers,
                "resname": self.project.polymer.resname,
            }
        ]
        for spec in self.project.molecule_specs():
            if spec.numbers > 0:
                molecules.append({"name": spec.name, "number": spec.numbers, "resname": spec.resname})
        return molecules

    def prepare_topology(self) -> PEMDGROMACS:
        gmx = PEMDGROMACS(
            str(self.project.artifacts.md_dir),
            self._molecules(),
            self.project.run.production_temperature,
            self.project.run.gpu,
        )
        gmx.gen_top_file(top_filename=self.project.artifacts.topology.name)
        return gmx

    def build_flow(self, packmol_pdb: str | None = None) -> MDFlow:
        return MDFlow(self, initial_pdb=packmol_pdb or self.project.artifacts.pack_pdb.name)

    def build_default_flow(self, packmol_pdb: str | None = None) -> MDFlow:
        initial_pdb = packmol_pdb or self.project.artifacts.pack_pdb.name
        flow = self.build_flow(initial_pdb)
        flow.pdb_to_gro(initial_pdb)
        if self.project.run.md_enable_em:
            flow.em(output="boxmd_em")
        if self.project.run.md_enable_nvt:
            flow.nvt(
                output="boxmd_nvt",
                steps=self.project.run.md_nvt_steps,
                temperature=self.project.run.production_temperature,
            )
        if self.project.run.md_enable_npt:
            flow.npt(
                output="boxmd_npt",
                steps=self.project.run.md_npt_steps,
                temperature=self.project.run.production_temperature,
                pressure=self.project.run.production_pressure,
            )
        if self.project.run.md_enable_production:
            flow.production(
                output="boxmd_production",
                steps=self.project.run.md_production_steps,
                temperature=self.project.run.production_temperature,
            )
        return flow

    def _infer_input(self, step: MDStep, previous_output: str | None) -> str:
        if step.stage == "pdb_to_gro":
            return step.input_pdb or self.project.artifacts.pack_pdb.name
        if step.input_gro is not None:
            return step.input_gro
        if previous_output is None:
            raise ValueError(f"{step.stage} requires an input_gro when no previous step exists")
        return previous_output

    def run_flow(self, flow: MDFlow) -> GromacsResult:
        gmx = self.prepare_topology()
        completed: list[str] = []
        step_results: list[StepResult] = []
        previous_output: str | None = None
        logger.info("Starting box MD flow with %s steps", len(flow.steps))

        for step in flow.steps:
            input_file = self._infer_input(step, previous_output)
            logger.info(
                "Running MD step: stage=%s input=%s output=%s steps=%s temperature=%s pressure=%s",
                step.stage,
                input_file,
                step.output,
                step.steps,
                step.temperature,
                step.pressure,
            )

            if step.stage == "pdb_to_gro":
                gmx.commands_pdbtogro(input_file, output_gro=step.output).run_local()
                output_file = step.output

            elif step.stage == "em":
                mdp_name = f"{step.output}.mdp"
                gmx.gen_em_mdp_file(mdp_name)
                gmx.commands_em(input_gro=input_file, output_str=step.output).run_local()
                output_file = f"{step.output}.gro"

            elif step.stage == "nvt":
                mdp_name = f"{step.output}.mdp"
                gmx.gen_nvt_mdp_file(
                    nsteps_nvt=step.steps or self.project.run.md_nvt_steps,
                    filename=mdp_name,
                    temperature=step.temperature if step.temperature is not None else self.project.run.production_temperature,
                )
                gmx.commands_nvt(input_gro=input_file, output_str=step.output).run_local()
                output_file = f"{step.output}.gro"

            elif step.stage == "npt":
                mdp_name = f"{step.output}.mdp"
                gmx.gen_npt_mdp_file(
                    filename=mdp_name,
                    nsteps_npt=step.steps or self.project.run.md_npt_steps,
                    pression=step.pressure if step.pressure is not None else self.project.run.production_pressure,
                    temperature=step.temperature if step.temperature is not None else self.project.run.production_temperature,
                )
                gmx.commands_npt(input_gro=input_file, output_str=step.output).run_local()
                output_file = f"{step.output}.gro"

            elif step.stage == "production":
                mdp_name = f"{step.output}.mdp"
                gmx.gen_nvt_mdp_file(
                    nsteps_nvt=step.steps or self.project.run.md_production_steps,
                    filename=mdp_name,
                    temperature=step.temperature if step.temperature is not None else self.project.run.production_temperature,
                )
                gmx.commands_nvt_product(input_gro=input_file, output_str=step.output).run_local()
                output_file = f"{step.output}.gro"

            else:
                raise ValueError(f"Unsupported MD stage: {step.stage}")

            previous_output = output_file
            completed.append(step.stage)
            step_results.append(StepResult(stage=step.stage, input_file=input_file, output_file=output_file))
            logger.info("Completed MD step: stage=%s output=%s", step.stage, output_file)

        logger.info("Box MD flow completed: stages=%s", completed)
        return GromacsResult(
            work_dir=self.project.artifacts.md_dir,
            completed_stages=completed,
            step_results=step_results,
        )

    def run_pack_md(self, packmol_pdb: str | None = None) -> GromacsResult:
        return self.build_default_flow(packmol_pdb).run()
