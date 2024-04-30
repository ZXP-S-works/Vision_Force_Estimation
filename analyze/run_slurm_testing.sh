#!/bin/bash

#SBATCH --job-name=testing.txt
#SBATCH --output=testing.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --time=0:10:00

module load miniconda
# using your anaconda environment
conda activate visual_force_conda
# python analyze_CNN.py  --test_data_dir=/vast/palmer/home.grace/yz2379/project/Data/experiment_data_1125_segmented/ --env_type=real_world_yz --n_history=20 --history_interval=1 --load_model=/vast/palmer/home.grace/yz2379/VisionForceEstimator/runs/Nov2316:43_1121_data_h20_int1/checkpoint/best_val.pt

python plot_error.py  --test_data_dir=/vast/palmer/home.grace/yz2379/project/Data/0120_data_segmented/ --env_type=real_world_yz --n_history=20 --history_interval=10 --load_model=/vast/palmer/home.grace/yz2379/VisionForceEstimator/runs/Jan2504:28_0120_data_h_20_int_10_random_posEnc/checkpoint/best_val.pt --test_csv_fn=force_prediction_result.csv --n_input_channel=5 --use_position_feature=True
