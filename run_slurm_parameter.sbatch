#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20480
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

module load miniconda
# using your anaconda environment
conda activate visual_force_conda

python main.py --train_data_dir=/vast/palmer/home.grace/yz2379/project/Data/final_data_segmented/ --valid_data_dir=/vast/palmer/home.grace/yz2379/project/Data/final_data_segmented/ --note=final_data_h_${HISTORY}_int_${INTERVAL}_seed_${SEED}_mlp --max_epoch=100 --env_type=real_world_yz --n_history=${HISTORY} --history_interval=${INTERVAL}  --train_csv_fn='train_result.csv' --valid_csv_fn='val_result.csv' --seed=${SEED} --network=mlpresnet
# python main.py --train_data_dir=/vast/palmer/home.grace/yz2379/project/Data/final_data_segmented/ --valid_data_dir=/vast/palmer/home.grace/yz2379/project/Data/final_data_segmented/ --note=final_data_h_${HISTORY}_int_${INTERVAL}_seed_${SEED}_segAug --max_epoch=100 --env_type=real_world_yz --n_history=${HISTORY} --history_interval=${INTERVAL} --train_csv_fn='train_result.csv' --valid_csv_fn='val_result.csv'  --seed=${SEED} --segmentation_aug=True #--use_position_feature=True --n_input_channel=5 --speed_augmentation=True --bs=64

# python main.py --train_data_dir=/vast/palmer/home.grace/yz2379/project/Data/0120_data_segmented/ --valid_data_dir=/vast/palmer/home.grace/yz2379/project/Data/0120_data_segmented/ --note=0120_data_h_${HISTORY}_int_${INTERVAL}_randomonly --max_epoch=100 --env_type=real_world_yz --n_history=${HISTORY} --history_interval=${INTERVAL} --train_csv_fn='random_only_train_result.csv' --valid_csv_fn='val_result.csv' --speed_augmentation=False

# python main.py --train_data_dir=/vast/palmer/home.grace/yz2379/project/Data/1013_wiping_data/all_train.pt --valid_data_dir=/vast/palmer/home.grace/yz2379/project/Data/1013_wiping_data/wiping_15.pt --test_data_dir=/vast/palmer/home.grace/yz2379/project/Data/1013_wiping_data/wiping_16.pt --note=1013_h_${HISTORY}_int_${INTERVAL} --max_epoch=100 --env_type=real_world_yz --n_history=${HISTORY} --history_interval=${INTERVAL} --grad_on_one_frame=False --gradient_accumulation=False

