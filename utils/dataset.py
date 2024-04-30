import numpy as np
import numpy.random as npr
import collections
import torch
import cv2
import matplotlib.pyplot as plt
try:
    from segment_anything import sam_model_registry, SamPredictor
    import onnxruntime
except:
    print('Segmentation packages not installed')
from tqdm import tqdm
import random
import os, glob
import scipy
import sys, copy
sys.path.append("../")
import pickle, glob
import pandas as pd
import random 
from icecream import ic

ImgForce = collections.namedtuple('ImgForce', 'img, x, f, ur_pose, servo_info, record_flag')
ImgForceMask = collections.namedtuple('ImgForceMask', 'img, x, f, mask, segmentedImg')

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
    # img = np.concatenate((color_image, depth_image[:, :, None]), axis=-1).transpose(2, 0, 1)
    img = color_image.transpose(2, 0, 1)
    img = scipy.ndimage.zoom(img, (1, 0.25, 0.25), order=3)  # img to shape [160, 120]
    # img = scipy.ndimage.zoom(img, (1, 0.5, 0.5), order=3)  # img to shape [160, 120]
    #img = scipy.ndimage.zoom(img, (1, 0.25, 0.25), order=3)  # img to shape [160, 120]
    img = img.astype(np.float16)
    return img

def process_img_2(color_image):
    # img = np.concatenate((color_image, depth_image[:, :, None]), axis=-1).transpose(2, 0, 1)
    img = color_image.transpose(2, 0, 1)
    img = scipy.ndimage.zoom(img, (1, 0.25, 0.25), order=3)  # img to shape [160, 120]
    img = img.astype(np.float16)
    return img


class Dataset:
    # modified from https://github.com/ZXP-S-works/SE2-equivariant-grasp-learning/blob/main/storage/buffer.py
    def __init__(self, size=0):
        self._history_interval = 1
        self._n_history = 1
        self._storage = []
        self._max_size = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size, speed_augmentation):
        ## skip data that overlaps two grasps 
        batch = []
        while len(batch) < batch_size*self._n_history:
            if speed_augmentation:
                # two types of augmentation, uniform scaling and augment one frame
                if np.random.random() < 0.5:
                    # uniform scaling
                    if np.random.random < 0.5:
                        interval = random.choice(list(np.arange(1, 21)))
                        # ic(interval)
                        batch_idx = npr.choice(self.__len__() - self._n_history * interval, 1, replace=False).tolist()[0]
                        tentative_pt = self._storage[batch_idx:batch_idx + self._n_history * interval:interval]
                        tentative_pt_whole_sequence = self._storage[batch_idx:batch_idx + self._n_history * interval]
                    # augment one frame
                    else:
                        N_of_frames_to_alter = random.choice(list(np.arange(-5,6)))
                        idx_of_frame_to_alter = random.choice(list(np.arange(1,self._n_history)))
                        indeces = []
                        tmp_idx = batch_idx
                        for k in range(self._n_history):
                            if k == idx_of_frame_to_alter:
                                tmp_idx += max(1, N_of_frames_to_alter + self._history_interval)
                            else:
                                tmp_idx += self._history_interval
                            indeces.append(tmp_idx)
                        tentative_pt = self._storage[indeces]
                        tentative_pt_whole_sequence = self._storage[batch_idx:tmp_idx+1]
                        # ic(indeces)
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

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx
        }

    def save(self, dir, name=''):
        checkpoint = self.getSaveState()
        torch.save(checkpoint, dir + name + '.pt')

    def load(self, dir, max_size=None, n_history=None, history_interval=None):
        checkpoint = torch.load(dir)
        self._max_size = checkpoint['max_size'] if max_size is None else int(max_size)
        self._storage = checkpoint['storage'][:self._max_size]
        self._n_history = n_history
        self._history_interval = history_interval

    def segmentAllDataONNX(self, segmentation_model, ONNX, BBs, points, debug = False):
        for i in tqdm(range(len(self._storage))):
            original_image, x, f, EE , servo_info = self._storage[i]
            image = cv2.cvtColor(np.uint8(np.swapaxes(original_image,0,2)), cv2.COLOR_BGR2RGB)
            segmentation_model.set_image(image)
            image_embedding = segmentation_model.get_image_embedding().cpu().numpy()
            total_mask = None
            for BB, points_set in zip(BBs, points):
                BB_reshaped = BB.reshape(2, 2)
                if debug:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    show_box(BB, plt.gca())
                    show_points(points_set, np.ones(len(points_set)), plt.gca())
                    plt.axis('off')
                    plt.show()

                # input_label = np.array([2., 3.])
                # onnx_coord = BB_reshaped[None, :, :]
                # onnx_label = input_label[None, :].astype(np.float32)
                points_input_label = np.ones(len(points_set))
                points_input = points_set
                BB_input = BB.reshape(2, 2)
                BB_input_label = np.array([2.,3.])

                onnx_coord = np.concatenate([points_input, BB_input],axis=0)[None, :, :]
                onnx_label = np.concatenate([points_input_label, BB_input_label], axis=0)[None, :].astype(np.float32)

                onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
                # Create an empty mask input and an indicator for no mask.
                onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
                }
                mask, _, low_res_logits = ort_session.run(None, ort_inputs)
                mask = mask > predictor.model.mask_threshold
                mask = mask[0,0,:,:]
                mask_int = np.uint8(mask.astype(int))
                if total_mask is None:
                    total_mask = mask_int
                else:
                    total_mask = total_mask | mask_int

            segmented_image = cv2.cvtColor(cv2.bitwise_and(image, image, mask = total_mask), cv2.COLOR_RGB2BGR)
            if debug:
                plt.figure(figsize=(10, 10))
                plt.imshow(segmented_image)
                plt.axis('off')
                plt.show()
            segmented_image = np.float16(np.swapaxes(segmented_image,0,2))
            augmented_data = ImgForceMask(original_image, x, f, EE , servo_info, np.swapaxes(total_mask,0,1), segmented_image)
            self._storage[i] = augmented_data
        return

    def augment_data(self, background_images, obj_images, obj_masks, multiplier = 1, obj_center = [80, 55], obj_dim_range = [40,90], debug = False):
        self._max_size = (multiplier+1)*self._max_size
        original_dataset_size = len(self._storage)        
        for tmp in range(multiplier):
            for i in tqdm(range(original_dataset_size)):
                #print(i)
                original_image, x, f, EE , servo_info, mask, segmented_image = self._storage[i]

                ##random background
                background_image = random.choice(background_images)
                _,x,y = original_image.shape
                aspect_ratio = y/x
                #cropping
                background_y, background_x, _ = background_image.shape
                if background_y/background_x > aspect_ratio:
                    cropping_x = background_x
                    cropping_y = background_x*aspect_ratio
                else:
                    cropping_y = background_y
                    cropping_x = background_y/aspect_ratio
                
                background_image = get_random_crop(background_image, int(cropping_y), int(cropping_x))
                background_image = np.uint8(cv2.resize(background_image, (x,y)))

                original_image_cv2_format = cv2.cvtColor(np.uint8(np.swapaxes(original_image,0,2)), cv2.COLOR_BGR2RGB)
                mask_cv2_format = np.swapaxes(mask,0,1) > 0.5

                augmented_image = background_image
                augmented_image[mask_cv2_format] = original_image_cv2_format.copy()[mask_cv2_format]

                #augmented_image = cv2.bitwise_and(original_image_cv2_format,background_image,mask = mask_cv2_format)
                if debug:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(augmented_image)
                    plt.axis('off')
                    plt.show()

                #random object
                random_obj_idx = random.choice(np.arange(len(obj_images)))
                obj_image = obj_images[random_obj_idx]
                obj_mask = obj_masks[random_obj_idx]
                ## random rotating object by 90 degress
                flag = random.choice([True, False])
                if flag:
                    obj_image = np.swapaxes(obj_image, 0 ,1)
                    obj_mask = np.swapaxes(obj_mask, 0 ,1)
                obj_size = random.choice([i for i in range(obj_dim_range[0], obj_dim_range[1])])
                obj_mask_cv2_format = obj_mask > 0.5
                if debug:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(obj_image)
                    plt.axis('off')
                    plt.show()

                if obj_image.shape[0] > obj_image.shape[1]:
                    obj_width = obj_size
                    obj_height = int(obj_image.shape[0]/obj_image.shape[1]*obj_width)
                else:
                    obj_height = obj_size
                    obj_width = int(obj_image.shape[1]/obj_image.shape[0]*obj_height)
                #print(obj_size, obj_width, obj_height)
                # randomize object center:
                obj_center_copy = copy.copy(obj_center)
                obj_center_copy[1] += random.choice([i for i in range(-15, 15)])

                UL = [max(obj_center_copy[0] - int(obj_height/2),0), max(obj_center_copy[1] - int(obj_width/2),0)]
                BR = [min(obj_center_copy[0] + int(obj_height/2),x), min(obj_center_copy[1] + int(obj_width/2),y)]
                image_patch = augmented_image[UL[0]:BR[0], UL[1]:BR[1]]
                #print(obj_size, obj_center_copy, UL, BR)
                if debug:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image_patch)
                    plt.axis('off')
                    plt.show()

                image_patch_upsampled = cv2.resize(image_patch, (obj_image.shape[1], obj_image.shape[0]))
                image_patch_upsampled[obj_mask_cv2_format] = obj_image[obj_mask_cv2_format]
                image_patch = cv2.resize(image_patch_upsampled, (image_patch.shape[1], image_patch.shape[0]))
                augmented_image[UL[0]:BR[0], UL[1]:BR[1]] = image_patch
                if debug:  
                    plt.figure(figsize=(10, 10))
                    plt.imshow(augmented_image)
                    plt.axis('off')
                    plt.show()

                #Optional: random lighting
                new_dp = ImgForce(augmented_image, x, f, EE , servo_info)
                self.add(new_dp)

        ## convert from ImgForceMask to ForceMast
        for i in tqdm(range(original_dataset_size)):
            original_image, x, f, EE , servo_info,_, _= self._storage[i]
            dp = ImgForce(original_image, x, f, EE , servo_info)
            self._storage[i] = dp
        return

    def resizeAllData(self):
        for i in tqdm(range(len(self._storage))):
            original_image, x, f, EE, servo_info = self._storage[i]
            original_image = scipy.ndimage.zoom(original_image.astype(np.float32), (1, 0.25, 0.25), order=3)
            original_image = original_image.astype(np.float16)
            self._storage[i] = ImgForce(original_image, x, f, EE, servo_info)
        return 

def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    if max_y <= 0:
        y = 0
    else:
        y = np.random.randint(0, max_y)
    if max_x <= 0:
        x = 0
    else:
        x = np.random.randint(0, max_x)
    
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop

"""Dataset that loads files from disk"""
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
                
            # debug += 1
            # if debug > 500:
            #     break
            if self.__len__() % 10000 == 0:
                print(self.__len__())
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
            if speed_augmentation:
                if np.random.random() < 0.5: #calling this make training very slow
                    # two types of augmentation, uniform scaling and augment one frame
                    # uniform scaling
                    if np.random.random() < 0.5:
                        interval = random.choice(list(np.arange(1, 21)))
                        # ic(interval)
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
                        # ic(N_of_frames_to_alter, indeces)
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


if __name__=='__main__':
    ## Segment images with finetuned SAM model
    sam_checkpoint =  '/vast/palmer/home.grace/yz2379/project/Data/sam_model_best.pth'
    model_type = "vit_h"
    device = "cuda"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device = device)

    dataset_cp_path = '/vast/palmer/home.grace/yz2379/project/Data/1006_data/all_train.pt'
    dataset = Dataset()
    dataset.load(dataset_cp_path, max_size=None)
    dataset.segmentAllData(sam_model=sam_model, debug = True)

    #dataset.save(dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/',  name='real_world_train_segmented')
    ## Convert data from 480 p to 160x120
    # dataset_cp_path = '/vast/palmer/home.grace/yz2379/Data/0905_data/real_world_val.pt'
    # dataset = Dataset()
    # dataset.load(dataset_cp_path, max_size=None)
    # dataset.resizeAllData()
    # dataset.save(dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/',  name='real_world_val_resized')

    # ## Segment Data 
    # sam_checkpoint = '/vast/palmer/home.grace/yz2379/Data/sam_vit_h_4b8939.pth'
    # model_type = "vit_h"
    # device = "cuda"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # predictor = SamPredictor(sam)

    # ## Use ONNX 
    # onnx_model_quantized_path = "/vast/palmer/home.grace/yz2379/Data/sam_onnx_quantized.onnx"
    # onnx_model_path = onnx_model_quantized_path
    # ## Note that: The ONNX model has a different input signature than SamPredictor.predict
    # ort_session = onnxruntime.InferenceSession(onnx_model_path, providers = ['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'])

    # dataset_cp_path = '/vast/palmer/home.grace/yz2379/Data/0905_data/real_world_train_resized.pt'
    # dataset = Dataset()
    # dataset.load(dataset_cp_path, max_size=None)
    # # BBs = [np.array([10, 0, 120, 80]),\
    # #     np.array([20, 70, 120, 160])]
    # #Yale dataset BB
    # BBs = [np.array([0, 0, 120, 90]),\
    #     np.array([0, 70, 120, 160])]
    # points = [np.array([[115, 15],[115, 30],[115, 42]]),
    #             np.array([[115, 120],[115, 127]])]

    # dataset.segmentAllData(predictor, ort_session, BBs, points = points, debug = False)
    # dataset.save(dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/',  name='real_world_train_segmented')

    # ## Augmentation
    # dataset_cp_path = '/vast/palmer/home.grace/yz2379/Data/0905_data/real_world_train_segmented.pt'
    # dataset = Dataset()
    # dataset.load(dataset_cp_path, max_size=None)
    # dataset._next_idx = len(dataset)

    # #background images
    # path_folder = '/vast/palmer/home.grace/yz2379/Data/MIT_Indoor/indoorCVPR_09/Images/office/'
    # img_fns = os.listdir(path_folder)
    # imgs = []
    # for fn in img_fns:
    #     img = cv2.imread(os.path.join(path_folder,fn))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     imgs.append(img)

    # #objects
    # obj_folder = '/vast/palmer/home.grace/yz2379/Data/ycb_processed/'
    # obj_imgs = []
    # obj_masks = []
    # for i in range(10):
    #     obj_imgs.append(cv2.imread(obj_folder + 'obj_' + str(i) + '.png'))
    #     obj_masks.append(np.load(obj_folder + 'mask_' + str(i) + '.npy'))
    # dataset.augment_data(background_images = imgs, obj_images = obj_imgs, obj_masks = obj_masks, multiplier = 3)
    # dataset.save(dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/',  name='real_world_train_augmented')

    # dataset_cp_path = '/vast/palmer/home.grace/yz2379/Data/0907_data/lab_background/train/real_world_5000.pt'
    # dataset = Dataset()
    # dataset.load(dataset_cp_path, max_size=None)
    # all_data = []
    # for i in range(len(dataset)):
    #     item = dataset[i]
    #     dp = item.servo_info['servo_loads'] + item.servo_info['servo_current_encoder'] + item.servo_info['servo_target_encoder']
    #     all_data.append(dp)

    #     original_image, x, f, EE , servo_info = item
    #     image = cv2.cvtColor(np.uint8(np.swapaxes(original_image,0,2)), cv2.COLOR_BGR2RGB)
    
    #     plt.figure(figsize=(10, 10))
    #     plt.title(f'{f[0]} {f[1]} {f[2]}')
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()

    # all_data = np.array(all_dat
    # print(np.max(all_data, axis = 1))
    # print(np.min(all_data, axis = 1))
