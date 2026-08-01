#!/usr/bin/env bash
# Launch example (from root):
# bash expes/launcher.sh --dataset aids

set -euo pipefail

if [[ "${1:-}" != "--dataset" || -z "${2:-}" ]]; then
  echo "Usage: $0 --dataset <DATASET_NAME>"
  exit 1
fi

DATASET="$2"
# Safer for Slurm job names (keeps logs + cfg using the original dataset string)
DATASET_JOBSAFE="${DATASET//[^A-Za-z0-9_]/_}"

ENV_ACTIVATE="${SYM_GRAPHS_ENV_ACTIVATE}"

if [[ -z "${ENV_ACTIVATE}" ]]; then
  echo "ERROR: SYM_GRAPHS_ENV_ACTIVATE is not set. Export it in your shell profile."
  exit 1
fi # custom path towards env activation if needed
PY_CMD_BASE="python -m torch.distributed.run --nproc_per_node=4 --standalone --nnodes=1 main.py"

SBATCH_COMMON=(
  --nodes=1
  --ntasks=1
  --gres=gpu:4
  --cpus-per-task=1
  --mem=128G
  --time=96:00:00
)

mkdir -p logs/equal logs/unequal

submit_one () {
  local setting="$1"   # equal|unequal
  local model="$2"     # dir|SH

  # NEW: cfg includes dataset suffix
  local cfg="configs/model_cfg/QDAGer/${setting}/${model}_${DATASET}.yaml"

  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: config not found: ${cfg}"
    exit 1
  fi

  # dir_eq / SH_eq / dir_uneq / SH_uneq
  local short
  if [[ "${setting}" == "equal" ]]; then
    short="${model}_eq"
  else
    short="${model}_uneq"
  fi

  # Slurm job name uses sanitized dataset to avoid special-char issues
  local job="${short}_${DATASET_JOBSAFE}"

  # EXACT filenames (no jobid, no timestamp) -> overwrites on reruns
  local err="logs/${setting}/${short}_${DATASET}.err"
  local out="logs/${setting}/${short}_${DATASET}.out"

  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="${job}" \
    --output="${out}" \
    --error="${err}" \
    --wrap "set -euo pipefail; source '${ENV_ACTIVATE}'; ${PY_CMD_BASE} --cfg '${cfg}' --dataset '${DATASET}'"
}

submit_one equal   dir
submit_one equal   SH
submit_one unequal dir
submit_one unequal SH
