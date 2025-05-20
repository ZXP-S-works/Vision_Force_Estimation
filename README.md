# VisionForceEstimator

This is the official repo for paper: Forces for Free: Vision-based Contact Force
Estimation with a Compliant Hand. 

## Hardware

We open source our gripper at https://github.com/grablab/openhand-hardware.

## Installation
See requirements.txt for all required packages. The installation is tested for Python version 3.11. 

## Data
Download our training and testing data from [Zenodo](doi.org/10.5281/zenodo.15453923). The testing data include all the static force prediction tasks. 

## Training

```
python main.py
--train_data_dir=/path/to/train_val
--valid_data_dir=/path/to/train_val
--test_data_dir=/path/to/test
--n_history=20
--history_interval=10
--train_csv_fn='train_result.csv'
--valid_csv_fn='val_result.csv'
--test_csv_fn='test_result.csv'
--seed=1
```

With the option to use segmentation augmentation: 
```
--segmentation_aug=True
```

Adjust the batch size if needed:
```
--bs=batch_size
```

The default model uses the transformer architecture to process history. To run the model with the MLP architecture:
```
--network=mlpresnet
```