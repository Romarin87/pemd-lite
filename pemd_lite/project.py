from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .errors import ProjectValidationError


@dataclass
class PolymerSpec:
    name: str
    resname: str
    repeating_unit: str
    left_cap: str
    right_cap: str
    length_short: int
    length_long: int
    numbers: int
    charge: float
    scale: float
    optimize_every_n_steps: int = 2


@dataclass
class SmallMoleculeSpec:
    key: str
    name: str
    resname: str
    charge: float
    scale: float
    numbers: int = 0
    ff_source: str = "database"
    smiles: Optional[str] = None


@dataclass
class RunSpec:
    relax_temperature: int = 1000
    production_temperature: int = 298
    production_pressure: float = 1.0
    density: float = 0.3
    short_chain_length_default: int = 4
    force_rebuild_short_chain: bool = False
    force_rebuild_long_chain: bool = False
    gpu: bool = False
    gmx_threads: int = 8
    enable_gpu_nonbonded: bool = True
    enable_gpu_bonded: bool = True
    enable_gpu_pme: bool = True
    add_length_scale: float = 1.0
    add_length_min_a: float = 100.0
    md_enable_em: bool = True
    md_enable_nvt: bool = False
    md_enable_npt: bool = True
    md_enable_production: bool = False
    md_nvt_steps: int = 200000
    md_npt_steps: int = 200000
    md_production_steps: int = 5000000


@dataclass
class ArtifactRegistry:
    root: Path
    polymer_name: str
    short_length: int
    long_length: int

    @property
    def md_dir(self) -> Path:
        return self.root / "MD_dir"

    @property
    def polymer_xml(self) -> Path:
        return self.root / f"{self.polymer_name}_charge_N{self.short_length}.xml"

    @property
    def polymer_short_pdb(self) -> Path:
        return self.polymer_chain_pdb(self.short_length)

    @property
    def polymer_long_pdb(self) -> Path:
        return self.polymer_chain_pdb(self.long_length)

    @property
    def relaxed_pdb(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_relax_N{self.long_length}.pdb"

    @property
    def pack_input(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_pack.inp"

    @property
    def pack_pdb(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_pack_cell.pdb"

    @property
    def topology(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_boxmd.top"

    @property
    def relax_topology(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_relax.top"

    @property
    def bonded_itp(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_bonded.itp"

    @property
    def nonbonded_itp(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_nonbonded.itp"

    @property
    def forcefield_top(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_forcefield_N{self.long_length}.top"

    @property
    def forcefield_gro(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_forcefield_N{self.long_length}.gro"

    @property
    def forcefield_pdb(self) -> Path:
        return self.md_dir / f"{self.polymer_name}_forcefield_N{self.long_length}.pdb"

    def polymer_chain_pdb(self, length: int) -> Path:
        return self.root / f"{self.polymer_name}_build_N{length}.pdb"

    def ligpargen_dir(self, name: Optional[str] = None) -> Path:
        target = name or self.polymer_name
        return self.root / f"ligpargen_{target}"


@dataclass
class Project:
    root: Path
    config_path: Path
    polymer: PolymerSpec
    small_molecules: List[SmallMoleculeSpec] = field(default_factory=list)
    run: RunSpec = field(default_factory=RunSpec)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def artifacts(self) -> ArtifactRegistry:
        return ArtifactRegistry(
            self.root,
            self.polymer.name,
            self.polymer.length_short,
            self.polymer.length_long,
        )

    def validate(self) -> None:
        if self.polymer.length_short < 3:
            raise ProjectValidationError(
                f"length_short={self.polymer.length_short} is unsupported for polymer charge transfer; use >= 3."
            )
        if self.polymer.length_long < self.polymer.length_short:
            raise ProjectValidationError("length_long must be >= length_short")
        if self.polymer.optimize_every_n_steps < 1:
            raise ProjectValidationError("optimize_every_n_steps must be >= 1")

    def molecule_specs(self) -> List[SmallMoleculeSpec]:
        return list(self.small_molecules)

    def existing_chain_paths(self) -> Dict[str, Path]:
        short_path = self.artifacts.polymer_chain_pdb(self.polymer.length_short)
        long_path = self.artifacts.polymer_chain_pdb(self.polymer.length_long)
        return {
            "short": short_path if short_path.exists() else Path(),
            "long": long_path if long_path.exists() else Path(),
        }


def _load_small_molecules(data: Dict[str, Any]) -> List[SmallMoleculeSpec]:
    out: List[SmallMoleculeSpec] = []
    for key, value in data.items():
        if key == "polymer" or not isinstance(value, dict):
            continue
        if "name" not in value or "resname" not in value:
            continue
        out.append(
            SmallMoleculeSpec(
                key=key,
                name=value["name"],
                resname=value["resname"],
                charge=float(value.get("charge", 0.0)),
                scale=float(value.get("scale", 1.0)),
                numbers=int(value.get("numbers", 0)),
                ff_source=str(value.get("ff_source", "database")),
                smiles=value.get("smiles"),
            )
        )
    return out


def load(path: Union[str, Path]) -> Project:
    input_path = Path(path).resolve()
    if input_path.is_dir():
        config_path = input_path / "md.json"
        root = input_path
    else:
        config_path = input_path
        root = input_path.parent
    if not config_path.exists():
        raise ProjectValidationError(f"Config file not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    polymer_data = data.get("polymer")
    if not isinstance(polymer_data, dict):
        raise ProjectValidationError("md.json must contain a 'polymer' object")

    length = polymer_data.get("length", [])
    if not isinstance(length, list) or len(length) < 2:
        raise ProjectValidationError("polymer.length must be a two-element list [short, long]")

    run_data = data.get("run", {}) if isinstance(data.get("run"), dict) else {}
    project = Project(
        root=root,
        config_path=config_path,
        polymer=PolymerSpec(
            name=str(polymer_data["name"]),
            resname=str(polymer_data["resname"]),
            repeating_unit=str(polymer_data["repeating_unit"]),
            left_cap=str(polymer_data.get("left_cap", "")),
            right_cap=str(polymer_data.get("right_cap", "")),
            length_short=int(length[0]),
            length_long=int(length[1]),
            numbers=int(polymer_data.get("numbers", 1)),
            charge=float(polymer_data.get("charge", 0.0)),
            scale=float(polymer_data.get("scale", 1.0)),
            optimize_every_n_steps=int(run_data.get("optimize_every_n_steps", 2)),
        ),
        small_molecules=_load_small_molecules(data),
        run=RunSpec(
            relax_temperature=int(run_data.get("relax_temperature", 1000)),
            production_temperature=int(run_data.get("production_temperature", 298)),
            production_pressure=float(run_data.get("production_pressure", 1.0)),
            density=float(run_data.get("density", 0.3)),
            short_chain_length_default=int(run_data.get("short_chain_length_default", 4)),
            force_rebuild_short_chain=bool(run_data.get("force_rebuild_short_chain", False)),
            force_rebuild_long_chain=bool(run_data.get("force_rebuild_long_chain", False)),
            gpu=bool(run_data.get("gpu", False)),
            gmx_threads=int(run_data.get("gmx_threads", 8)),
            enable_gpu_nonbonded=bool(run_data.get("enable_gpu_nonbonded", True)),
            enable_gpu_bonded=bool(run_data.get("enable_gpu_bonded", True)),
            enable_gpu_pme=bool(run_data.get("enable_gpu_pme", True)),
            add_length_scale=float(run_data.get("add_length_scale", 1.0)),
            add_length_min_a=float(run_data.get("add_length_min_a", 100.0)),
            md_enable_em=bool(run_data.get("md_enable_em", True)),
            md_enable_nvt=bool(run_data.get("md_enable_nvt", False)),
            md_enable_npt=bool(run_data.get("md_enable_npt", True)),
            md_enable_production=bool(run_data.get("md_enable_production", False)),
            md_nvt_steps=int(run_data.get("md_nvt_steps", 200000)),
            md_npt_steps=int(run_data.get("md_npt_steps", 200000)),
            md_production_steps=int(run_data.get("md_production_steps", 5000000)),
        ),
        raw_config=data,
    )
    project.validate()
    return project
