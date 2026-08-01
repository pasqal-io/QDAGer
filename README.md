# QDAGer — README

Datasets along with their corresponding emulations, as well as the corresponding trained models can be found [here](https://zenodo.org/records/21719940). We use the same datasets and their splitting (train / val / test) used in [this project](https://github.com/structlearning/GraphEdX) for benchmarking.

---

## Getting started

### Installation

sym-graphs uses [Poetry](https://python-poetry.org/) for dependency management. Install it following the [official instructions](https://python-poetry.org/docs/#installation), then run from the repo root:

```bash
poetry install
```

**Not recommended:** Installing Poetry via pip (`pip install poetry`) can cause dependency conflicts when updating Poetry's own packages. Prefer the official installer.

If you have no other choice, run the following — but be aware that updating one of Poetry's own dependencies may break it and require troubleshooting:

```bash
source /path/to/your/venv
python3 -m pip install poetry
poetry install
```

</aside>

---

## Environment setup

Before running any experiment, set the following environment variable in your `~/.bashrc` or `~/.zshrc`:

```bash
export SYM_GRAPHS_ENV_ACTIVATE="/path/to/your/envs/sym_graphs/bin/activate"
```

This variable is used by the launcher scripts to activate your virtual environment. The repo contains no hardcoded personal paths.

---

## How to run

### Quantum features (`corr_dyn`) — Slurm

**1. Create the log folders:**

```bash
mkdir -p logs/equal logs/unequal
```

**2. Launch using the provided script** (example with the `aids` dataset):

```bash
bash expes/launcher.sh --dataset aids
```

This submits **4 Slurm jobs** (4 GPUs each) covering all combinations of settings:

| Setting | Model |
| --- | --- |
| equal | dir |
| equal | SH |
| unequal | dir |
| unequal | SH |

Config files are expected at:

```
configs/model_cfg/{equal,unequal}/{dir,SH}_<DATASET>.yaml
```

### Launch via Slurm — classical features (`hk_pe` or `rw_pe`)

```bash
bash expes/launcher_classical.sh --dataset aids
```

Same structure as above, but passes a `--features` argument to `main.py`. The active features list is controlled inside the script (`FEATURES_LIST`). Currently set to `hk_pe` only (`rw_pe` commented out).

Log files include the feature name as a suffix:

```
logs/{equal,unequal}/{model}_{dataset}.{out,err}_{features}
```

### Single-GPU run (no Slurm, no DDP)
Example with the `aids` dataset:
```bash
bash expes/launcher_single.sh --dataset aids
```

Uses plain `python main.py` (no `torch.distributed.run`), falls back to single-GPU training. 1 GPU, same config structure as `launcher.sh`.

---

## Why use the launcher scripts?

All launcher scripts handle:

- Config file validation before submission
- Consistent Slurm job naming (sanitized dataset and feature names)
- Sourcing the virtual environment via `$SYM_GRAPHS_ENV_ACTIVATE`
- Overwriting logs on reruns (no timestamp or job ID suffix)

| Script | Model | GPUs | Features arg |
| --- | --- | --- | --- |
| `launcher.sh` | QDAGer | 4 | no (default uses `corr_dyn`) |
| `launcher_classical.sh` | QDAGer | 4 | yes (`hk_pe`or `rw_pe`) |
| `launcher_single.sh` | QDAGer | 1 | no (default uses `corr_dyn`) |
