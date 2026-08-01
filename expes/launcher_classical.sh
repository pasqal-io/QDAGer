#!/usr/bin/env bash
# Launch example (from root):
# bash expes/launcher.sh --dataset aids

set -euo pipefail

if [[ "${1:-}" != "--dataset" || -z "${2:-}" ]]; then
  echo "Usage: $0 --dataset <DATASET_NAME>"
  exit 1
fi

DATASET="$2"
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
  local features="$3"  # hk_pe|rw_pe

  local cfg="configs/model_cfg/QDAGer/${setting}/${model}_${DATASET}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "ERROR: config not found: ${cfg}"
    exit 1
  fi

  local short
  if [[ "${setting}" == "equal" ]]; then
    short="${model}_eq"
  else
    short="${model}_uneq"
  fi

  local features_jobsafe="${features//[^A-Za-z0-9_]/_}"

  # Job name can stay as-is (includes dataset + features)
  local job="${short}_${DATASET_JOBSAFE}_${features_jobsafe}"

  # ✅ features as SUFFIX for log files
  local err="logs/${setting}/${short}_${DATASET_JOBSAFE}.err_${features_jobsafe}"
  local out="logs/${setting}/${short}_${DATASET_JOBSAFE}.out_${features_jobsafe}"

  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="${job}" \
    --output="${out}" \
    --error="${err}" \
    --wrap "set -euo pipefail; source '${ENV_ACTIVATE}'; ${PY_CMD_BASE} --cfg '${cfg}' --dataset '${DATASET}' --features '${features}'"
}

# Run all current experiments for hk_pe and rw_pe (skip corr_dyn)
#FEATURES_LIST=(hk_pe rw_pe)
FEATURES_LIST=(hk_pe)

for feat in "${FEATURES_LIST[@]}"; do
  submit_one equal   dir "${feat}"
  submit_one equal   SH  "${feat}"
  submit_one unequal dir "${feat}"
  submit_one unequal SH  "${feat}"
done
