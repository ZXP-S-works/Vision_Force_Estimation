#!/bin/bash

starts=(30 30 30 10 10 10 5 5 5) # 10 10)
ends=(10 10 10 10 10 10 10 10 10) # 10 5)
seeds=(1 2 3 1 2 3 1 2 3)
for i in "${!starts[@]}"; do
    #echo "${array[i]}, ${int[i]}"
    HISTORY=${starts[i]}
    INTERVAL=${ends[i]}
    SEED=${seeds[i]}
    echo "${HISTORY}, ${INTERVAL}, ${SEED}"
    export HISTORY INTERVAL SEED
    sbatch --output final_${HISTORY}_int_${INTERVAL}_${SEED}_mlp.txt \
    --job-name final_${HISTORY}_int_${INTERVAL}_${SEED}_mlp\
    run_slurm_parameter.sbatch
    sleep 1
done 