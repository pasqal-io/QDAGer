#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=96:00:00

NAMES=("aids" "linux" "mutagenicity" "ogbg-code2" "ogbg-molhiv" "ogbg-molpcba" "yeast")
MODES=("train" "val" "test")

# Run "equal" and "unequal" in parallel as background processes
for TYPE in "equal" "unequal"; do
    (
        for name in "${NAMES[@]}"; do
            for mode in "${MODES[@]}"; do
                echo "$(date) | Running: name=$name, mode=$mode, type=$TYPE"
                python ../sym_graphs/data_io/emulate.py --name "$name" --mode "$mode" --type "$TYPE"
            done
        done
    ) &
done

wait
echo "All done."
