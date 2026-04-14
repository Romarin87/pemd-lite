from __future__ import annotations

import os
import shlex
import subprocess
import logging
from shutil import which
from typing import Optional

logger = logging.getLogger(__name__)

class PEMDGROMACS:
    def __init__(self, work_dir, molecules, temperature, gpu):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.molecules = molecules
        self.temperature = temperature
        self.commands = []
        self.gpu = gpu

        self.compounds = [molecule["name"] for molecule in self.molecules]
        self.resnames = [molecule["resname"] for molecule in self.molecules]
        self.numbers = [molecule["number"] for molecule in self.molecules]

    @staticmethod
    def _resolve_ntomp(default_value=8) -> int:
        env_value = os.environ.get("PEMD_GMX_NTOMP", "").strip()
        if env_value:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
        return int(default_value)

    def _gmx(self) -> str:
        preferred = os.environ.get("PEMD_GMX_EXEC", "").strip()
        if preferred:
            if which(preferred):
                return preferred
            raise RuntimeError(f"Requested PEMD_GMX_EXEC='{preferred}' not found in PATH.")

        candidates = ["gmx_mpi", "gmx"] if self.gpu else ["gmx_mpi", "gmx"]
        for candidate in candidates:
            if which(candidate):
                logger.info("Using GROMACS executable: %s", candidate)
                return candidate
        raise RuntimeError("Neither 'gmx' nor 'gmx_mpi' found in PATH.")

    def _mdrun_cmd(self, output_str: str, *, em_stage: bool = False) -> str:
        gmx_exec = self._gmx()
        ntomp = self._resolve_ntomp(default_value=8)
        cmd = [gmx_exec, "mdrun", "-v", "-deffnm", f"{self.work_dir}/{output_str}"]

        if self.gpu:
            cmd.extend(["-ntomp", str(ntomp)])
            flags_env = "PEMD_GPU_MDRUN_FLAGS_EM" if em_stage else "PEMD_GPU_MDRUN_FLAGS"
            default_flags = "" if em_stage else "-nb gpu -bonded gpu -pme gpu"
            extra_flags = os.environ.get(flags_env, default_flags).strip()
            if extra_flags:
                cmd.extend(shlex.split(extra_flags))
        else:
            cmd.extend(["-ntomp", str(ntomp)])

        return " ".join(cmd)

    def _resolve_path(self, path_like):
        if path_like is None:
            return None
        text = str(path_like)
        if os.path.isabs(text):
            return text
        return os.path.join(self.work_dir, text)

    def gen_top_file(self, top_filename="topol.top"):
        self.top_filename = top_filename
        top_filepath = os.path.join(self.work_dir, top_filename)
        contents = "; gromcs generation top file\n; Created by PEMD-Lite\n\n"
        contents += "[ defaults ]\n;nbfunc  comb-rule   gen-pairs   fudgeLJ  fudgeQQ\n"
        contents += "1        3           yes         0.5      0.5\n\n"
        contents += ";LOAD atomtypes\n[ atomtypes ]\n"
        for compound in self.compounds:
            contents += f'#include "{compound}_nonbonded.itp"\n'
        contents += "\n"
        for compound in self.compounds:
            contents += f'#include "{compound}_bonded.itp"\n'
        contents += "\n[ system ]\n;name "
        for compound in self.compounds:
            contents += compound
        contents += "\n\n[ molecules ]\n"
        for compound, fallback_resname, number in zip(self.compounds, self.resnames, self.numbers):
            molecule_name = self._resolve_moleculetype_name(compound, fallback_resname)
            contents += f"{molecule_name} {number}\n"
        contents += "\n"
        with open(top_filepath, "w", encoding="utf-8") as handle:
            handle.write(contents)

    def _resolve_moleculetype_name(self, compound, fallback_resname):
        bonded_itp = os.path.join(self.work_dir, f"{compound}_bonded.itp")
        if not os.path.exists(bonded_itp):
            return fallback_resname

        try:
            with open(bonded_itp, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return fallback_resname

        in_moleculetype = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped == "[ moleculetype ]":
                in_moleculetype = True
                continue
            if in_moleculetype:
                return stripped.split()[0]
        return fallback_resname

    def gen_em_mdp_file(self, filename="em.mdp"):
        filepath = os.path.join(self.work_dir, filename)
        contents = "; em.mdp - PEMD-Lite\n\n"
        contents += "integrator      = steep\nnsteps          = 50000\nemtol           = 1000.0\nemstep          = 0.01\n\n"
        contents += "; Parameters describing how to find the neighbors of each atom and how to calculate the interactions\n"
        contents += "nstlist         = 1\ncutoff-scheme   = Verlet\nns_type         = grid\nrlist           = 1.0\n"
        contents += "coulombtype     = PME\nrcoulomb        = 1.0\nrvdw            = 1.0\npbc             = xyz\n"
        contents += "; output control is on\nenergygrps      = System\n"
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(contents)

    def gen_nvt_mdp_file(
        self,
        nsteps_nvt=200000,
        filename="nvt.mdp",
        temperature: Optional[float] = None,
        dt_ps: float = 0.001,
        tau_t_ps: float = 1.0,
    ):
        ref_t = self.temperature if temperature is None else temperature
        filepath = os.path.join(self.work_dir, filename)
        contents = "; nvt.mdp - PEMD-Lite\n\n"
        contents += f"; RUN CONTROL PARAMETERS\nintegrator            = md\ndt                    = {dt_ps:.6f} \n"
        contents += f"nsteps                = {nsteps_nvt}\ncomm-mode             = Linear\n\n"
        contents += "; OUTPUT CONTROL OPTIONS\nnstxout               = 0\nnstvout               = 0\nnstfout               = 0\n"
        contents += "nstlog                = 5000\nnstenergy             = 5000\nnstxout-compressed    = 5000\n\n"
        contents += "; NEIGHBORSEARCHING PARAMETERS\ncutoff-scheme         = verlet\nns_type               = grid\nnstlist               = 20\n"
        contents += "rlist                 = 1.4\nrcoulomb              = 1.4\nrvdw                  = 1.4\nverlet-buffer-tolerance = 0.005\n\n"
        contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\ncoulombtype           = PME\nvdw_type              = cut-off\nfourierspacing        = 0.15\n"
        contents += "pme_order             = 4\newald_rtol            = 1e-05\n\n"
        contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\ntcoupl                = v-rescale\ntc-grps               = System\n"
        contents += f"tau_t                 = {tau_t_ps}\nref_t                 = {ref_t}\nPcoupl                = no\nPcoupltype            = isotropic\n"
        contents += "tau_p                 = 1.0\ncompressibility       = 4.5e-5\nref_p                 = 1.0\n\n"
        contents += "; GENERATE VELOCITIES FOR STARTUP RUN\ngen_vel               = no\n\n"
        contents += "; OPTIONS FOR BONDS\nconstraints           = hbonds\nconstraint_algorithm  = lincs\nunconstrained_start   = no\n"
        contents += "shake_tol             = 0.00001\nlincs_order           = 4\nlincs_warnangle       = 30\nmorse                 = no\nlincs_iter            = 2\n"
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(contents)

    def gen_npt_mdp_file(
        self,
        nsteps_npt=5000000,
        filename="npt.mdp",
        pression: Optional[float] = None,
        temperature: Optional[float] = None,
        dt_ps: float = 0.001,
        tau_t_ps: float = 1.0,
        tau_p_ps: float = 1.0,
    ):
        ref_p = 1.0 if pression is None else pression
        ref_t = self.temperature if temperature is None else temperature
        filepath = os.path.join(self.work_dir, filename)
        contents = "; npt.mdp - PEMD-Lite\n\n"
        contents += f"; RUN CONTROL PARAMETERS\nintegrator            = md\ndt                    = {dt_ps:.6f} \n"
        contents += f"nsteps                = {nsteps_npt}\ncomm-mode             = Linear\n\n"
        contents += "; OUTPUT CONTROL OPTIONS\nnstxout               = 0\nnstvout               = 0\nnstfout               = 0\n"
        contents += "nstlog                = 5000\nnstenergy             = 5000\nnstxout-compressed    = 5000\n\n"
        contents += "; NEIGHBORSEARCHING PARAMETERS\ncutoff-scheme         = verlet\nns_type               = grid\nnstlist               = 20\n"
        contents += "rlist                 = 1.4\nrcoulomb              = 1.4\nrvdw                  = 1.4\nverlet-buffer-tolerance = 0.005\n\n"
        contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\ncoulombtype           = PME\nvdw_type              = cut-off\nfourierspacing        = 0.15\n"
        contents += "pme_order             = 4\newald_rtol            = 1e-05\n\n"
        contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\ntcoupl                = v-rescale\ntc-grps               = System\n"
        contents += f"tau_t                 = {tau_t_ps}\nref_t                 = {ref_t}\nPcoupl                = Berendsen\nPcoupltype            = isotropic\n"
        contents += f"tau_p                 = {tau_p_ps}\ncompressibility       = 4.5e-5\nref_p                 = {ref_p}\n\n"
        contents += "; GENERATE VELOCITIES FOR STARTUP RUN\ngen_vel               = no\n\n"
        contents += "; OPTIONS FOR BONDS\nconstraints           = hbonds\nconstraint_algorithm  = lincs\nunconstrained_start   = no\n"
        contents += "shake_tol             = 0.00001\nlincs_order           = 4\nlincs_warnangle       = 30\nmorse                 = no\nlincs_iter            = 2\n"
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(contents)

    def commands_pdbtogro(self, packmol_pdb, *, box_length: Optional[float] = None, center: bool = False, distance: Optional[float] = None, output_gro="conf.gro"):
        input_path = self._resolve_path(packmol_pdb)
        output_path = self._resolve_path(output_gro)
        if center and distance is not None:
            self.commands = [f"{self._gmx()} editconf -f {input_path} -o {output_path} -c -d {distance}"]
        elif box_length is None:
            self.commands = [f"{self._gmx()} editconf -f {input_path} -o {output_path}"]
        else:
            self.commands = [f"{self._gmx()} editconf -f {input_path} -o {output_path} -box {box_length} {box_length} {box_length}"]
        return self

    def commands_grotopdb(self, gro_filename, pdb_filename):
        self.commands = [f"{self._gmx()} editconf -f {self._resolve_path(gro_filename)} -o {self._resolve_path(pdb_filename)}"]
        return self

    def commands_em(self, input_gro, output_str="em"):
        self.commands = [
            f"{self._gmx()} grompp -f {self._resolve_path(output_str + '.mdp')} -c {self._resolve_path(input_gro)} -p {self._resolve_path(self.top_filename)} -o {self._resolve_path(output_str + '.tpr')} -maxwarn 1",
            self._mdrun_cmd(output_str, em_stage=True),
        ]
        return self

    def commands_nvt(self, input_gro, output_str):
        self.commands = [
            f"{self._gmx()} grompp -f {self._resolve_path(output_str + '.mdp')} -c {self._resolve_path(input_gro)} -p {self._resolve_path(self.top_filename)} -o {self._resolve_path(output_str + '.tpr')} -maxwarn 1",
            self._mdrun_cmd(output_str),
        ]
        return self

    def commands_nvt_product(self, input_gro, output_str):
        self.commands = [
            f"{self._gmx()} grompp -f {self._resolve_path(output_str + '.mdp')} -c {self._resolve_path(input_gro)} -p {self._resolve_path(self.top_filename)} -o {self._resolve_path(output_str + '.tpr')} -maxwarn 1",
            self._mdrun_cmd(output_str),
        ]
        return self

    def commands_npt(self, input_gro, output_str):
        self.commands = [
            f"{self._gmx()} grompp -f {self._resolve_path(output_str + '.mdp')} -c {self._resolve_path(input_gro)} -p {self._resolve_path(self.top_filename)} -o {self._resolve_path(output_str + '.tpr')} -maxwarn 1",
            self._mdrun_cmd(output_str),
        ]
        return self

    def run_local(self, commands=None):
        os.makedirs(self.work_dir, exist_ok=True)
        commands = self.commands if commands is None else commands
        for cmd in commands:
            logger.info("Executing GROMACS command: %s", cmd)
            result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
            if result.stdout.strip():
                logger.info("GROMACS stdout tail:\n%s", result.stdout[-2000:])
            if result.stderr.strip():
                logger.warning("GROMACS stderr tail:\n%s", result.stderr[-2000:])
