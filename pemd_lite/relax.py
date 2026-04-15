from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem

from .gromacs import PEMDGROMACS

from .project import Project

logger = logging.getLogger(__name__)

@dataclass
class RelaxOptions:
    temperature: float = 1000.0
    pressure: float = 1.0
    box_mode: Optional[Union[float, str]] = "editconf"
    dt_ps: float = 0.001
    tau_t_ps: float = 1.0
    nvt_tau_t_ps: Optional[float] = None
    tau_p_ps: float = 1.0
    run_em: bool = True
    run_npt: bool = True
    run_nvt: bool = True
    npt_steps: int = 200000
    nvt_steps: int = 200000
    nvt_gen_vel: bool = False
    em_output: Optional[str] = None
    npt_output: Optional[str] = None
    nvt_output: Optional[str] = None
    stage_order: Optional[List[str]] = None


@dataclass
class RelaxResult:
    relaxed_pdb: Path
    completed_stages: List[str]


@dataclass
class BoxEstimate:
    add_length_a: float
    end_to_end_a: float
    max_span_a: float
    source_pdb: Path


class BoxEstimator:
    @staticmethod
    def _coords_from_pdb(pdb_path: Path) -> np.ndarray:
        coords = []
        with pdb_path.open() as handle:
            for line in handle:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords.append((x, y, z))
        if coords:
            return np.array(coords, dtype=float)

        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is None or mol.GetNumConformers() == 0:
            raise ValueError(f"Could not read coordinates from {pdb_path}")
        conf = mol.GetConformer()
        return np.array(conf.GetPositions(), dtype=float)

    def estimate_from_pdb(self, pdb_path: Path, *, scale: float = 1.0, min_add_a: float = 100.0) -> BoxEstimate:
        coords = self._coords_from_pdb(pdb_path)
        spans = coords.max(axis=0) - coords.min(axis=0)
        max_span = float(np.max(spans))
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        end_to_end = float(np.max(distances))
        add_length = max(end_to_end, max_span) * float(scale) + float(min_add_a)
        logger.info(
            "Estimated box from %s: end_to_end_a=%.3f max_span_a=%.3f add_length_a=%.3f scale=%s min_add_a=%s",
            pdb_path,
            end_to_end,
            max_span,
            add_length,
            scale,
            min_add_a,
        )
        return BoxEstimate(add_length_a=add_length, end_to_end_a=end_to_end, max_span_a=max_span, source_pdb=pdb_path)


class RelaxRunner:
    _STAGE_DEFAULT_ORDER = ("em", "npt", "nvt")

    def __init__(self, project: Project):
        self.project = project

    def _prepare_gmx(self) -> PEMDGROMACS:
        molecules = [
            {
                "name": self.project.polymer.name,
                "number": 1,
                "resname": self.project.polymer.resname,
            }
        ]
        gmx = PEMDGROMACS(
            str(self.project.artifacts.md_dir),
            molecules,
            self.project.run.relax_temperature,
            self.project.run.gpu,
        )
        gmx.gen_top_file(top_filename=self.project.artifacts.relax_topology.name)
        return gmx

    def _default_stage_output(self, stage: str) -> str:
        return "{name}_relax_{stage}".format(name=self.project.polymer.name, stage=stage)

    @staticmethod
    def _resolve_box_mode(opts: RelaxOptions) -> Tuple[Optional[float], bool, Optional[float]]:
        mode = opts.box_mode
        if isinstance(mode, (int, float)):
            return float(mode), False, None
        if isinstance(mode, str):
            text = mode.strip()
            if text == "":
                return None, True, 1.2
            if text == "editconf":
                return None, True, 1.2
            match = re.fullmatch(r"editconf\s+-d\s+([0-9]*\.?[0-9]+)", text)
            if match:
                return None, True, float(match.group(1))
            try:
                return float(text), False, None
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported relax box mode: {mode!r}. Use a float, 'editconf', or 'editconf -d x.x'."
                ) from exc
        return None, True, 1.2

    @classmethod
    def _effective_stage_order(cls, opts: RelaxOptions) -> List[str]:
        requested = list(opts.stage_order) if opts.stage_order is not None else list(cls._STAGE_DEFAULT_ORDER)
        allowed = set(cls._STAGE_DEFAULT_ORDER)
        seen = set()
        ordered: List[str] = []
        enabled = {
            "em": opts.run_em,
            "npt": opts.run_npt,
            "nvt": opts.run_nvt,
        }
        for stage in requested:
            if stage not in allowed:
                raise ValueError(
                    "Unsupported relax stage in stage_order: {!r}. Allowed values are: {}.".format(
                        stage, ", ".join(cls._STAGE_DEFAULT_ORDER)
                    )
                )
            if stage in seen:
                raise ValueError(f"Duplicate relax stage in stage_order: {stage!r}")
            seen.add(stage)
            if enabled[stage]:
                ordered.append(stage)
        missing_enabled = [stage for stage in cls._STAGE_DEFAULT_ORDER if enabled[stage] and stage not in seen]
        if missing_enabled:
            logger.warning(
                "Relax stages enabled but omitted from stage_order; they will be skipped: %s. stage_order=%s",
                missing_enabled,
                requested,
            )
        return ordered

    def run(self, pdb_file: Path, options: Optional[RelaxOptions] = None) -> RelaxResult:
        opts = options or RelaxOptions(temperature=self.project.run.relax_temperature)
        gmx = self._prepare_gmx()
        box_length_nm, center_molecule, box_distance_nm = self._resolve_box_mode(opts)
        stage_order = self._effective_stage_order(opts)
        logger.info(
            "Starting relax: pdb=%s temperature=%s pressure=%s dt_ps=%s tau_t_ps=%s tau_p_ps=%s box_mode=%s resolved_box_length=%s center=%s distance=%s run_em=%s run_npt=%s run_nvt=%s stage_order=%s",
            pdb_file,
            opts.temperature,
            opts.pressure,
            opts.dt_ps,
            opts.tau_t_ps,
            opts.tau_p_ps,
            opts.box_mode,
            box_length_nm,
            center_molecule,
            box_distance_nm,
            opts.run_em,
            opts.run_npt,
            opts.run_nvt,
            stage_order,
        )

        conf_name = f"{self.project.polymer.name}_relax_conf.gro"
        gmx.commands_pdbtogro(
            str(pdb_file),
            box_length=box_length_nm,
            center=center_molecule,
            distance=box_distance_nm if center_molecule else None,
            output_gro=conf_name,
        ).run_local()
        completed: List[str] = ["pdb_to_gro"]
        current_gro = conf_name

        for stage in stage_order:
            if stage == "em":
                em_output = opts.em_output or self._default_stage_output("em")
                gmx.gen_em_mdp_file(filename=f"{em_output}.mdp")
                gmx.commands_em(input_gro=current_gro, output_str=em_output).run_local()
                current_gro = f"{em_output}.gro"
            elif stage == "npt":
                npt_output = opts.npt_output or self._default_stage_output("npt")
                gmx.gen_npt_mdp_file(
                    filename=f"{npt_output}.mdp",
                    nsteps_npt=opts.npt_steps,
                    pression=opts.pressure,
                    temperature=opts.temperature,
                    dt_ps=opts.dt_ps,
                    tau_t_ps=opts.tau_t_ps,
                    tau_p_ps=opts.tau_p_ps,
                )
                gmx.commands_npt(input_gro=current_gro, output_str=npt_output).run_local()
                current_gro = f"{npt_output}.gro"
            else:
                nvt_output = opts.nvt_output or self._default_stage_output("nvt")
                gmx.gen_nvt_mdp_file(
                    filename=f"{nvt_output}.mdp",
                    nsteps_nvt=opts.nvt_steps,
                    temperature=opts.temperature,
                    dt_ps=opts.dt_ps,
                    tau_t_ps=opts.nvt_tau_t_ps if opts.nvt_tau_t_ps is not None else opts.tau_t_ps,
                    gen_vel=opts.nvt_gen_vel,
                )
                gmx.commands_nvt(input_gro=current_gro, output_str=nvt_output).run_local()
                current_gro = f"{nvt_output}.gro"
            completed.append(stage)
            logger.info("Relax stage completed: %s -> %s", stage, current_gro)

        gmx.commands_grotopdb(gro_filename=current_gro, pdb_filename=self.project.artifacts.relaxed_pdb.name).run_local()
        logger.info("Relax completed: relaxed_pdb=%s completed_stages=%s", self.project.artifacts.relaxed_pdb, completed)
        return RelaxResult(
            relaxed_pdb=self.project.artifacts.relaxed_pdb,
            completed_stages=completed,
        )
