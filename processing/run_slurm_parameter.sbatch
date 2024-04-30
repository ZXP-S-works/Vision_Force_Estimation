#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

module load miniconda
# using your anaconda environment
conda activate visual_force_conda
python misc.py -s=${start} -e=${end}