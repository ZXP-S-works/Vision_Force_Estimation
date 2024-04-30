# VisionForceEstimator

This project aims to  use computer vision estimating the force that deforms a dexterous hand.

## ToDo:
1. Peg insertion task
2. Data augmentation through segmentation and augmentation

## Prerequisite

[Setup and tune Yale Open Hand (YOH)](https://github.com/XupengZhu-SW/openhand_node). This fork of YOH drive provides 
servo information.

## Realworld Data Collection

There are two options to collect ```train/valid/test``` data sets. These data sets are collected individually. I usually
 collect 5k/1k/2k for ```train/valid/test```.

### Option1: Manual data collection
This requires Realsense, Gamma FT sensor, YOH model O or T42.

You can manually move YOH to press the FT sensor and use
Realsense to capture the science. Through this way to generate data.
```
python real_workd_data/manual_collect_data.py
```
Inside function ```record_img_f```, the parameters ```dataset_size=50, save_dir='./data/', min_force=0.2``` 
controls dataset size, save directory, minimal recoding force.

### Option2: Using UR10 to collect data
This requires UR10, Realsense, Gamma FT sensor, YOH model O or T42.

The script will command UR10 to move YOH and press 
the FT sensor.

Warning: tune your UR10 to avoid collision.
```
python real_workd_data/ur_collect_data.py
```

## [Optional] Generate Data in Simulation

Generate data using Pybullet with one finger no tendon can be done by:
```
python simulaiton_data/generate_img_theta_force.py
--dataset_size=5000
--estimation_type=calculated_force
--img_size=128
```
This script will render RGBD images for a deformed YOH finger and calculate the force that could deform the finger to 
this shape.

## [Optional] Analyse YOH finger
Plot manipulability of a finger:
```
python simulaiton_data/theta_force_func.py
```

Plot force - joint displacement sensitivity of a finger:
```
python simulaiton_data/force_theta_func.py
```

## Training Vision Force Estimator:

Training ResNet (CNN) with real world data:
```
python main.py
--train_data_dir=data/0822_new_T42/real_world_10000.pt
--valid_data_dir=data/0822_new_T42/real_world_1000.pt
--test_data_dir=data/0822_new_T42/real_world_2000.pt
--n_hidden=32
--note=resnet
--network=resnet
```


Training MlpResNet (CNN + Memory) with real world data:
```
python main.py
--train_data_dir=data/0822_new_T42/real_world_10000.pt
--valid_data_dir=data/0822_new_T42/real_world_1000.pt
--test_data_dir=data/0822_new_T42/real_world_2000.pt
--bs=16
--n_hidden=32
--note=mlpresnet
--network=mlpresnet
```


Training TransformerResNet (CNN + Memory + Attention) with real world data:
```
python main.py
--train_data_dir=data/0822_new_T42/real_world_10000.pt
--valid_data_dir=data/0822_new_T42/real_world_1000.pt
--test_data_dir=data/0822_new_T42/real_world_2000.pt
--note=transformer
```

Training with simulation data:
```
python main.py 
--train_data_dir=data/0726_sim_theta_1_30/simulation_128_I_theta_f_5000.pt
--valid_data_dir=data/0726_sim_theta_1_30/simulation_128_I_theta_f_1000.pt
--n_hidden=32
--env_type=simulation_force
--note=simulation
--network=resnet
```

## Deploying Vision Force Estimator:
This requires Realsense, YOH, a trained model.

This scrip will load the trained Neural Network (NN) model in ```PATH_TO_YOUR_TRAINED_MODEL/29.pt``` and start estimate 
the force. Notice that the NN hyperparameters should be identical to the loaded model. You can check the hyperparameters
in ```MonthDayTime_Note/info/parameters```. The trained model usually located at 
```MonthDayTime_Note/info/checkpoint/?.pt```.

Waring: I didn't finish peg_insertion task. Use this script to move robot can lead to uncontrollable robot behaviour.
```
python peg_insertion.py
--load_model=PATH_TO_YOUR_TRAINED_MODEL/29.pt
--n_input_channel=3
--n_output_channel=2
--n_hidden=32
--env_type=real_world_xz
--note=T42
--resolution=128
--network=transformerresnet
--rot_aug=0
--trans_aug=0
--n_history=20
--history_interval=2
```
