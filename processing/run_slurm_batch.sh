#!/bin/bash

starts=(0 10 20 30 40 50 60 70 80 86)
ends=(10 20 30 40 50 60 70 80 86 92)
# starts=(0)
# ends=(5)

for i in "${!starts[@]}"; do
    #echo "${array[i]}, ${int[i]}"
    start=${starts[i]}
    end=${ends[i]}
    echo "${start}, ${end}"
    export start end
    sbatch --output s${start}_e${end}.txt \
    --job-name s${start}_e${end}\
    run_slurm_parameter.sbatch
    sleep 1
done 