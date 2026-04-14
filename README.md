# PEMD-Lite

PEMD-Lite is a lightweight extraction of the PEMD workflow for polymer build, forcefield generation, packing, and GROMACS-driven MD preparation. This repository is organized as a standalone Python package so it can be published and versioned independently.

## What Is Included

- Polymer project loading and artifact naming
- Short- and long-chain polymer construction
- LigParGen-backed charge generation and repair helpers
- OPLS-AA XML and topology generation
- Single-chain relax and box estimation
- Packmol input generation
- GROMACS preparation and staged MD execution
- CSV/XLSX-driven batch project generation

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── docs/
│   └── environment.md
├── pemd_lite/
│   ├── __init__.py
│   ├── project.py
│   ├── polymer.py
│   ├── forcefield.py
│   ├── relax.py
│   ├── pack.py
│   ├── md.py
│   └── resources/
└── tests/
```

## Installation

PEMD-Lite depends on several scientific Python packages plus external executables that are not vendored by this repository.

```bash
pip install -e .
```

Required external tools in `PATH`:

- `ligpargen`
- `obabel`
- `packmol`
- `gmx_mpi`

Depending on your LigParGen setup, a BOSS runtime may also be required.

Target Python version: `3.7.12`.

The verified runtime does not keep `gromacs` inside the conda environment. Use an external GROMACS install and point PEMD-Lite at it with `PEMD_GMX_EXEC`, for example:

```bash
export PEMD_GMX_EXEC=/root/shared-nvme/soft/gromacs-2026.1/bin/gmx_mpi
```

## Quick Start

```python
from pemd_lite import load
from pemd_lite.polymer import PolymerBuilder

project = load("md.json")
polymer = PolymerBuilder(project).build_required()
print(polymer.long_pdb)
```

For the staged workflow API:

```python
from pemd_lite import run_until

result = run_until("path/to/project", "pack_cell")
print(result.pack.pack_pdb)
```

## Environment Notes

The currently deployed and verified software stack is documented in [docs/environment.md](docs/environment.md).
