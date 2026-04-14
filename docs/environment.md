# Environment Guide

This document records the deployed PEMD-Lite runtime that was verified on `2026-04-14` on the remote host `ackcs-00gjgmjp` (`ssh.zw1.paratera.com:2222`).

## Environment Manager

- Root conda installation: `/root/miniforge3`
- PEMD-Lite environment path: `/root/shared-nvme/pemd_lite_test_bwli/conda_envs/pemd_lite`
- Environment type: standalone conda prefix managed by `miniforge3`

## Verified Runtime Versions

### Python and packaging

- Python: `3.7.12`
- pip: `24.0`
- conda: `26.1.1` from `/root/miniforge3/bin/conda`

### Core Python packages from `conda-forge`

- `numpy==1.21.6`
- `pandas==1.3.5`
- `scipy==1.7.3`
- `rdkit==2022.09.1`
- `MDAnalysis==2.1.0`
- `ParmEd==3.4.3`
- `networkx==2.7`
- `openbabel==3.1.1`
- `packmol==21.2.1`
- `gromacs==2023.4`
- `cudatoolkit==11.8.0`

### Command-line tools exposed from the environment

- `gmx` -> GROMACS `2023.4-conda_forge`
- `obabel` -> reports `Open Babel 3.1.0`
- `packmol` -> available from the conda environment binary directory
- `ligpargen` -> `LigPargen 2.1`

## LigParGen Installation

The deployed environment uses the GitHub project:

- Source repository: [Isra3l/ligpargen](https://github.com/Isra3l/ligpargen)
- Package metadata homepage: `https://github.com/Isra3l/ligpargen`
- Installed package name reported by `pip show`: `LigPargen`
- Installed version: `2.1`
- Python requirement declared by that package: `==3.7.*`

Observed installation characteristics:

- `ligpargen` is installed into the PEMD-Lite conda environment's `site-packages`
- the console script lives at `/root/shared-nvme/pemd_lite_test_bwli/conda_envs/pemd_lite/bin/ligpargen`
- the source snapshot used on the machine includes a `setup.py`
- the checked source directory currently available on the machine does not contain `.git` metadata, so the exact commit was not recoverable from that path

## Recreating the Verified Stack

The closest public reconstruction of the verified environment is:

```bash
conda create -p /path/to/conda_envs/pemd_lite -c conda-forge \
  python=3.7.12 \
  pip=24.0 \
  numpy=1.21.6 \
  pandas=1.3.5 \
  scipy=1.7.3 \
  rdkit=2022.09.1 \
  mdanalysis=2.1.0 \
  parmed=3.4.3 \
  networkx=2.7 \
  openbabel=3.1.1 \
  packmol=21.2.1 \
  gromacs=2023.4 \
  cudatoolkit=11.8.0
```

Then install LigParGen from the GitHub source:

```bash
git clone https://github.com/Isra3l/ligpargen.git
cd ligpargen
git checkout v2.1
pip install .
```

## Compatibility Note

The currently verified deployed environment is Python `3.7.12`. If repository code is updated to rely on newer Python syntax or dependency versions, keep this document in sync and treat that as a new environment target rather than assuming compatibility with the deployed stack above.
