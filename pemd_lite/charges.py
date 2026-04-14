from __future__ import annotations

import csv
import fcntl
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem

from .errors import LigParGenBackendError
from .project import Project

logger = logging.getLogger(__name__)

@dataclass
class ChargeTable:
    frame: pd.DataFrame


@dataclass
class ChargeResult:
    charge_table: ChargeTable
    ligpargen_dir: Path
    gmx_itp: Path
    gmx_gro: Path
    csv_path: Path
    ordered_pdb: Path
    input_mode: str


class ChargeBackend(ABC):
    @abstractmethod
    def generate_polymer_charges(
        self,
        *,
        project: Project,
        short_pdb: Path,
        short_mol: Chem.Mol,
        short_smiles: str,
    ) -> ChargeResult:
        raise NotImplementedError


def _parse_atoms_from_itp(itp_path: Path) -> list[dict[str, object]]:
    lines = itp_path.read_text(encoding="utf-8").splitlines()
    in_atoms = False
    rows: list[dict[str, object]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            in_atoms = line.lower() == "[ atoms ]"
            continue
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append(
            {
                "position": int(parts[0]) - 1,
                "atom": parts[4],
                "charge": float(parts[6]),
            }
        )
    return rows


def reconstruct_csv_from_itp(itp_path: Path, csv_path: Path) -> pd.DataFrame:
    rows = _parse_atoms_from_itp(itp_path)
    if not rows:
        raise LigParGenBackendError(f"Could not reconstruct charges from {itp_path}")
    logger.warning("LigParGen CSV missing; reconstructing charges from ITP: %s -> %s", itp_path, csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["position", "atom", "charge"])
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def _find_generated_file(search_dirs: list[Path], stems: list[str], ext: str) -> Path | None:
    suffixes = [f".gmx.{ext}", f".{ext}"]
    for base in search_dirs:
        for stem in stems:
            for suffix in suffixes:
                candidate = base / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
    return None


class LigParGenBackend(ChargeBackend):
    def __init__(self, charge_model: str = "CM1A-LBCC"):
        self.charge_model = charge_model

    def _ligpargen_base_cmd(self) -> list[str]:
        explicit = os.environ.get("PEMD_LIGPARGEN_EXEC", "").strip()
        if explicit:
            return [explicit]
        sibling = Path(sys.executable).resolve().parent / "ligpargen"
        if sibling.exists() and os.access(str(sibling), os.X_OK):
            return [str(sibling)]
        executable = shutil.which("ligpargen")
        if executable:
            return [executable]
        raise LigParGenBackendError(
            "Could not find LigParGen executable. Set PEMD_LIGPARGEN_EXEC or install the ligpargen command entrypoint."
        )

    def _run_resname(self, *, name: str, resname: str) -> str:
        """
        Generate a per-run residue name to reduce collisions on files that
        LigParGen/BOSS names from ``-r``. Many workflows use ``MOL`` for every
        polymer, so keeping ``-r MOL`` makes concurrent runs trample each other.
        """
        cleaned = "".join(ch for ch in resname.upper() if ch.isalnum()) or "MOL"
        if cleaned not in {"MOL", "UNK"}:
            return cleaned[:3]
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest().upper()
        mapped = f"P{digest[:2]}"
        logger.info("Using unique LigParGen resname fallback: input=%s mapped=%s", resname, mapped)
        return mapped

    def _run(self, cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env_bin = str(Path(sys.executable).resolve().parent)
        path_parts = [env_bin]
        path_parts.append(env.get("PATH", ""))
        env["PATH"] = ":".join(part for part in path_parts if part)
        lock_path = Path("/tmp/pemd_ligpargen.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            logger.info("Running LigParGen command in %s: %s", cwd, " ".join(cmd))
            result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)
            logger.info("LigParGen finished rc=%s", result.returncode)
            if result.stdout.strip():
                logger.info("LigParGen stdout tail:\n%s", result.stdout[-2000:])
            if result.stderr.strip():
                logger.warning("LigParGen stderr tail:\n%s", result.stderr[-2000:])
            return result

    def _ligpargen_dir(self, project: Project) -> Path:
        path = project.artifacts.ligpargen_dir(project.polymer.name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _output_paths(self, ligdir: Path, name: str) -> tuple[Path, Path, Path]:
        return (
            ligdir / f"{name}.gmx.itp",
            ligdir / f"{name}.gmx.gro",
            ligdir / f"{name}.csv",
        )

    def _ordered_pdb_from_gro(self, gmx_gro: Path, target: Path) -> Path:
        from . import io

        logger.info("Generating ordered PDB from GRO: %s -> %s", gmx_gro, target)
        io.convert_gro_to_pdb(gmx_gro, target)
        return target

    def _run_ligpargen(self, ligdir: Path, *, name: str, resname: str, charge: float, mode: str, payload: str) -> tuple[str, str, subprocess.CompletedProcess[str]]:
        run_resname = self._run_resname(name=name, resname=resname)
        cmd = self._ligpargen_base_cmd() + [
            "-n",
            name,
            "-p",
            str(ligdir),
            "-r",
            run_resname,
            "-c",
            str(int(charge)),
            "-cgen",
            self.charge_model,
        ]
        if mode == "pdb":
            cmd += ["-i", payload]
        elif mode == "smiles":
            cmd += ["-s", payload]
        else:
            raise ValueError(f"Unsupported LigParGen mode: {mode}")
        return mode, run_resname, self._run(cmd, ligdir)

    def _collect_outputs(
        self,
        *,
        ligdir: Path,
        name: str,
        resname: str,
        gmx_itp: Path,
        gmx_gro: Path,
        csv_path: Path,
    ) -> tuple[Path, Path, Path | None]:
        stems: list[str] = []
        for stem in (name, resname):
            if stem and stem not in stems:
                stems.append(stem)
        search_dirs = [ligdir, Path.cwd(), Path("/tmp")]

        found_itp = _find_generated_file(search_dirs, stems, "itp")
        found_gro = _find_generated_file(search_dirs, stems, "gro")
        found_csv = _find_generated_file(search_dirs, stems, "csv")

        logger.info(
            "Collecting LigParGen outputs: stems=%s search_dirs=%s found_itp=%s found_gro=%s found_csv=%s",
            stems,
            [str(path) for path in search_dirs],
            found_itp,
            found_gro,
            found_csv,
        )

        if found_itp and found_itp.resolve() != gmx_itp.resolve():
            gmx_itp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found_itp, gmx_itp)
            logger.warning("LigParGen ITP collected via fallback copy: %s -> %s", found_itp, gmx_itp)
        elif found_itp:
            gmx_itp = found_itp

        if found_gro and found_gro.resolve() != gmx_gro.resolve():
            gmx_gro.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found_gro, gmx_gro)
            logger.warning("LigParGen GRO collected via fallback copy: %s -> %s", found_gro, gmx_gro)
        elif found_gro:
            gmx_gro = found_gro

        if found_csv and found_csv.resolve() != csv_path.resolve():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found_csv, csv_path)
            logger.warning("LigParGen CSV collected via fallback copy: %s -> %s", found_csv, csv_path)
        elif found_csv:
            csv_path = found_csv

        return gmx_itp, gmx_gro, csv_path if csv_path.exists() else None

    def generate_polymer_charges(
        self,
        *,
        project: Project,
        short_pdb: Path,
        short_mol: Chem.Mol,
        short_smiles: str,
    ) -> ChargeResult:
        poly = project.polymer
        ligdir = self._ligpargen_dir(project)
        gmx_itp, gmx_gro, csv_path = self._output_paths(ligdir, poly.name)
        ordered_pdb = ligdir / f"{poly.name}_gmx_ordered.pdb"

        attempts = [("pdb", str(short_pdb.resolve())), ("smiles", short_smiles)]
        failures: list[str] = []
        input_mode = "pdb"
        for mode, payload in attempts:
            logger.info("LigParGen attempt starting: name=%s mode=%s", poly.name, mode)
            input_mode, run_resname, result = self._run_ligpargen(
                ligdir,
                name=poly.name,
                resname=poly.resname,
                charge=poly.charge,
                mode=mode,
                payload=payload,
            )
            gmx_itp, gmx_gro, found_csv = self._collect_outputs(
                ligdir=ligdir,
                name=poly.name,
                resname=run_resname,
                gmx_itp=gmx_itp,
                gmx_gro=gmx_gro,
                csv_path=csv_path,
            )
            if found_csv is not None:
                csv_path = found_csv
            if gmx_itp.exists() and gmx_gro.exists():
                if mode == "smiles":
                    logger.warning("LigParGen fallback succeeded with SMILES input for %s", poly.name)
                else:
                    logger.info("LigParGen succeeded with PDB input for %s", poly.name)
                break
            logger.warning("LigParGen attempt failed to materialize outputs: name=%s mode=%s rc=%s", poly.name, mode, result.returncode)
            failures.append(
                f"{mode}: rc={result.returncode}; stdout={result.stdout[-800:]}; stderr={result.stderr[-800:]}"
            )
        else:
            raise LigParGenBackendError("LigParGen did not produce gmx.itp/gmx.gro\n" + "\n".join(failures))

        if not csv_path.exists():
            frame = reconstruct_csv_from_itp(gmx_itp, csv_path)
        else:
            frame = pd.read_csv(csv_path)
            logger.info("Loaded LigParGen CSV directly: %s rows=%s", csv_path, len(frame))

        self._ordered_pdb_from_gro(gmx_gro, ordered_pdb)
        logger.info(
            "LigParGen outputs ready: itp=%s gro=%s csv=%s ordered_pdb=%s input_mode=%s",
            gmx_itp,
            gmx_gro,
            csv_path,
            ordered_pdb,
            input_mode,
        )
        return ChargeResult(
            charge_table=ChargeTable(frame=frame),
            ligpargen_dir=ligdir,
            gmx_itp=gmx_itp,
            gmx_gro=gmx_gro,
            csv_path=csv_path,
            ordered_pdb=ordered_pdb,
            input_mode=input_mode,
        )


class DatabaseBackend(ChargeBackend):
    def generate_polymer_charges(
        self,
        *,
        project: Project,
        short_pdb: Path,
        short_mol: Chem.Mol,
        short_smiles: str,
    ) -> ChargeResult:
        raise LigParGenBackendError("DatabaseBackend does not support polymer charge generation")
