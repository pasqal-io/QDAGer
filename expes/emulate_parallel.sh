#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --array=0-3%4

NAMES=("aids" "mutagenicity" "ogbg-molhiv" "ogbg-molpcba" "yeast")
TYPES=("equal" "unequal")
MODES=("train" "val" "test")

name_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
type_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))

name="${NAMES[$name_idx]}"
type="${TYPES[$type_idx]}"


for mode in "${MODES[@]}"; do
  echo "$(date) | Running: name=$name, mode=$mode, type=$type"
  python ../sym_graphs/data_io/emulate.py --name "$name" --mode "$mode" --type "$type"
done
