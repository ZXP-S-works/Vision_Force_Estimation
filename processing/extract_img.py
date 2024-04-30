import sys
sys.path.append('../')
from utils.dataset import Dataset, ImgForce
from icecream import ic
import cv2
import numpy as np
import os

def extract_dataset(save_dir, data_dir):
    dataset = Dataset()
    dataset.load(data_dir, max_size=None)
    
    for i in range(len(dataset)):
        if i%10 == 0:
            print(i/len(dataset))
        dp = dataset[i]
        #print(dp[1],dp[2])
        img = dp.img
        img = np.uint8(np.swapaxes(img, 0, 2))
        img_fn = save_dir + f'rgb_{i}.png'
        cv2.imwrite(img_fn, img)

def plot_force(data_dir):
    dataset = Dataset()
    dataset.load(data_dir, max_size=None)
    forces = []
    for i in range(len(dataset)):
        dp = dataset[i]
        f = dp[2]
        ic(i, dp[1], f)


# save_dir = '/vast/palmer/home.grace/yz2379/Data/0801_raw_data/train/'
# checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0801_fixed_finger_2/real_world_1000.pt'

# save_dir = '/vast/palmer/home.grace/yz2379/Data/0801_raw_data/val/'
# checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0801_fixed_finger_2/real_world_1000.pt'
# extract_dataset(save_dir, checkpoint_path)

# checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0801_processed_data_1/real_world_5000.pt'
# save_dir = '/vast/palmer/home.grace/yz2379/Data/'
# extract_dataset(save_dir, checkpoint_path)

#save_dir = '/vast/palmer/home.grace/yz2379/Data/0811_raw_data/train/'
#checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0811_in_hand_sensor/real_world_5000.pt'
# save_dir = '/vast/palmer/home.grace/yz2379/Data/0811_raw_data/val/'
# checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0811_in_hand_sensor/real_world_1000.pt'

save_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'
checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0905_data/real_world_train_resized.pt'

extract_dataset(save_dir, checkpoint_path)
plot_force(checkpoint_path)