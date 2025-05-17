import numpy as np
import numpy.random as npr
import collections
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os, glob
import scipy
import sys, copy
import pickle, glob
import pandas as pd
import random 

ImgForce = collections.namedtuple('ImgForce', 'img, x, f, ur_pose, servo_info, record_flag')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def process_img(color_image):
    img = color_image.transpose(2, 0, 1)
    img = scipy.ndimage.zoom(img, (1, 0.25, 0.25), order=3)  # img to shape [160, 120]
    img = img.astype(np.float16)
    return img

def process_img_2(color_image):
    # img = np.concatenate((color_image, depth_image[:, :, None]), axis=-1).transpose(2, 0, 1)
    img = color_image.transpose(2, 0, 1)
    img = scipy.ndimage.zoom(img, (1, 0.25, 0.25), order=3)  # img to shape [160, 120]
    img = img.astype(np.float16)
    return img

class VisionForceDataset:
    def __init__(self, dataset_path, result_file_path, n_history=1, history_interval=1):
        self._history_interval = history_interval
        self._n_history = n_history
        self._storage = []
        results = pd.read_csv(os.path.join(dataset_path, result_file_path))
        results = results.drop_duplicates() # Remove duplicated headers from csv concatenation
        last_name = None
        debug = 0
        for row in results.itertuples(index=True, name='Pandas'):
            name, i = row.name, int(row.id)
            folder = os.path.join(dataset_path, name + '/')
            img_folder = os.path.join(folder, 'images/')
            force_folder = os.path.join(folder, 'forces/')
            
            force_fn = os.path.join(force_folder, f'{i}.pkl')
            img_fn = os.path.join(img_folder, f'{i}.png')
            try:
                if os.path.isfile(img_fn):
                    img = cv2.imread(img_fn)
                else:
                    img_fn = os.path.join(img_folder, f'{i:05d}.png')
                    img = cv2.imread(img_fn)
                img = cv2.resize(img, (160, 90))
                img = np.float16(np.swapaxes(img, 0, 2))
                dp = pickle.load(open(force_fn, 'rb'))
                ## at the beginning of a new run, set record_flag = False
                if last_name is None or last_name != name:
                    new_dp = ImgForce(img, dp[1], dp[2], dp[3], dp[4], False)
                else:
                    new_dp = ImgForce(img, dp[1], dp[2], dp[3], dp[4], dp[5])
                last_name = name
                self.add(new_dp)
            except Exception as e:
                print(str(e), name, i)
            if self.__len__() % 10000 == 0:
                print(f'Still loading data.. Loaded {self.__len__()} data points')
        print(f'Dataset Loaded with {len(self._storage)} data points')
        
    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def add(self, data):
        self._storage.append(data)
 
    def sample(self, batch_size, speed_augmentation):
        ## skip data that overlaps two grasps 
        batch = []
        while len(batch) < batch_size*self._n_history:
            #Not used for experiments in the paper
            if speed_augmentation:
                if np.random.random() < 0.5: #calling this make training very slow
                    # two types of augmentation, uniform scaling and augment one frame
                    # uniform scaling
                    if np.random.random() < 0.5:
                        interval = random.choice(list(np.arange(1, 21)))
                        batch_idx = npr.choice(self.__len__() - self._n_history * interval, 1, replace=False).tolist()[0]
                        tentative_pt = self._storage[batch_idx:batch_idx + self._n_history * interval:interval]
                        tentative_pt_whole_sequence = self._storage[batch_idx:batch_idx + self._n_history * interval]
                    # augment one frame
                    else:
                        N_of_frames_to_alter = random.choice(list(np.arange(-5,6)))
                        idx_of_frame_to_alter = random.choice(list(np.arange(1,self._n_history)))
                        batch_idx = npr.choice(self.__len__() - self._n_history * self._history_interval - 5, 1, replace=False).tolist()[0]
                        indeces = []
                        tmp_idx = batch_idx
                        for k in range(self._n_history):
                            if k == idx_of_frame_to_alter:
                                tmp_idx += max(1, N_of_frames_to_alter + self._history_interval)
                            else:
                                tmp_idx += self._history_interval
                            indeces.append(tmp_idx)
                        tentative_pt = [self._storage[pt_idx] for pt_idx in indeces] #self._storage[indeces]
                        tentative_pt_whole_sequence = self._storage[batch_idx:tmp_idx+1]
                else:
                    batch_idx = npr.choice(self.__len__() - self._n_history * self._history_interval, 1, replace=False).tolist()[0]
                    tentative_pt = self._storage[batch_idx:batch_idx + self._n_history * self._history_interval:self._history_interval]
                    tentative_pt_whole_sequence = self._storage[batch_idx:batch_idx + self._n_history * self._history_interval] 
            else:
                batch_idx = npr.choice(self.__len__() - self._n_history * self._history_interval, 1, replace=False).tolist()[0]
                tentative_pt = self._storage[batch_idx:batch_idx + self._n_history * self._history_interval:self._history_interval]
                tentative_pt_whole_sequence = self._storage[batch_idx:batch_idx + self._n_history * self._history_interval]
            has_overlap = False
            for frame in tentative_pt_whole_sequence:
                if frame.record_flag is False:
                    has_overlap = True
                    break
            if not has_overlap:
                batch.extend(tentative_pt)
        return batch

    def sample_continuous(self, batch_size):
        ## skip data that overlaps two grasps 
        batch = []
        batch_idx = 0
        while len(batch) < batch_size*self._n_history:
            tentative_pt = self._storage[batch_idx:batch_idx + self._n_history * self._history_interval:self._history_interval]
            tentative_pt_whole_sequence = self._storage[batch_idx:batch_idx + self._n_history * self._history_interval]
            has_overlap = False
            for frame in tentative_pt_whole_sequence:
                if frame.record_flag is False:
                    has_overlap = True
                    break
            if not has_overlap:
                batch.extend(tentative_pt)
            batch_idx += 1
        return batch

    def sample_by_index(self, batch_size, starting_idx):
        batch = []
        current_idx = starting_idx
        while len(batch) < batch_size*self._n_history:
            if current_idx > self.__len__() - self._n_history * self._history_interval:
                return batch, current_idx
            tentative_pt = self._storage[current_idx:current_idx + self._n_history * self._history_interval:self._history_interval]
            tentative_pt_whole_sequence = self._storage[current_idx:current_idx  + self._n_history * self._history_interval]
            current_idx += 1
            has_overlap = False
            for frame in tentative_pt_whole_sequence:
                if frame.record_flag is False:
                    has_overlap = True
                    break
            if not has_overlap:
                batch.extend(tentative_pt)
        return batch, current_idx

    def sample_by_index_verbose(self, batch_size, starting_idx):
        batch = []
        indeces = []
        current_idx = starting_idx
        while len(batch) < batch_size*self._n_history:
            if current_idx > self.__len__() - self._n_history * self._history_interval:
                return batch, current_idx, indeces
            tentative_pt = self._storage[current_idx:current_idx + self._n_history * self._history_interval:self._history_interval]
            tentative_pt_whole_sequence = self._storage[current_idx:current_idx  + self._n_history * self._history_interval]
            has_overlap = False
            for frame in tentative_pt_whole_sequence:
                if frame.record_flag is False:
                    has_overlap = True
                    break
            if not has_overlap:
                batch.extend(tentative_pt)
                indeces.append(current_idx)
            current_idx += 1

            
        return batch, current_idx, indeces
