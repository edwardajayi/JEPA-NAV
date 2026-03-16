# JEPA-NAV

Vision-based indoor navigation using V-JEPA 2 and PointNav in Habitat-Sim (ReplicaCAD).

## Setup

- Create and activate the conda environment (see project docs for Habitat-Sim build).
- Ensure ReplicaCAD datasets and configs are under `data/`.

## Training

- Run PointNav training via the scripts in `scripts/`.
- For cluster runs, use the SLURM scripts in `scripts/slurm/` (e.g. submit the appropriate `.sh` for your config).

## Project layout

- `scripts/` — training and evaluation
- `scripts/slurm/` — cluster job scripts
- `data/` — datasets and scene configs
- `tex/` — paper/report LaTeX source
