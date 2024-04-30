import sys
sys.path.append('../')
from utils.dataset import Dataset, ImgForce
import torch
from icecream import ic
import cv2
import numpy as np
import os
import glob, os
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

import collections
import torch
import random
import pandas as pd
import skimage

import copy
import shutil

ArucoForce = collections.namedtuple('ArucoForce', 'aruco, x, f, ur_pose, servo_info')

## concatenate datasets
def concatenate_datasets_and_resize(all_files):
    N = 0
    for file in all_files:
        dataset = Dataset()
        dataset.load(file, max_size=None)
        N += len(dataset)
    merged_dataset = Dataset(size = N)
    for file in all_files:
        dataset = Dataset()
        dataset.load(file, max_size=None)
        for i in tqdm(range(len(dataset))):
            dp = dataset[i]
            original_image, x, f, EE, servo_info, record_flag = dp
            # original_image = scipy.ndimage.zoom(original_image.astype(np.float32), (1, 0.25, 0.25), order=3)
            # original_image = original_image.astype(np.float16)
            if i == 0:
                new_dp = ImgForce(original_image, x, f, EE, servo_info, False)
            else:
                new_dp = ImgForce(original_image, x, f, EE, servo_info, record_flag)
            merged_dataset.add(new_dp)
    return merged_dataset

## extract images 
def extract_dataset(save_dir, data_dir, every = 1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = Dataset()
    dataset.load(data_dir, max_size=None)
    for i in range(len(dataset)):
        if i%every == 0:
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

def extract_aruco(data_dir, original_dataset):
    import apriltag
    detector = apriltag.Detector()
    #new_dataset = Dataset(size = old_dataset._max_size)
    for i in range(len(original_dataset)):
        img_fn = os.path.join(data_dir, f'{i}.png')
        img = cv2.imread(img_fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        print(results)
        #for pt in results:
        #    p
        exit()

def check_img_from_cp(path):
    dataset = Dataset()
    dataset.load(path, max_size=None)
    i = 0
    original_image, x, f, EE , servo_info, record_flag = dataset[i]
    #print(original_image.shape)
    print(f)
    image = cv2.cvtColor(np.uint8(np.swapaxes(original_image,0,2)), cv2.COLOR_BGR2RGB)
    # image = np.uint8(original_image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def calculate_dataset_stats(dataset_path):
    N_dp = 0
    folders = glob.glob(dataset_path)
    for folder in folders:
        force_folder = os.path.join(folder, 'forces/')
        all_force_files = glob.glob(force_folder)
        N_dp += len(all_force_files)
    return N_dp


## Concatenate all runs and split    
def create_splits_all_runs(dataset_path,  N_train, N_val = None, N_test = None, seed = 1, total_N = None):
    """create train and val splits"""
    folders = glob.glob(dataset_path)
    random.seed(seed)
    random.shuffle(folders)
    train_result = pd.DataFrame(columns = ['name', 'id'])
    val_result = pd.DataFrame(columns = ['name', 'id'])
    test_result = pd.DataFrame(columns = ['name', 'id'])
    if total_N is None:
        total_N = calculate_dataset_stats(dataset_path)

    # the last N_test are test data, and the N_val before these are val data
    if N_val is not None and N_test is not None:
        val_starting_idx = total_N - N_test - N_val
        test_starting_idx = total_N - N_test
    current_total_idx = 0
    for folder in folders:
        name = folder.split('/')[-1]
        force_folder = os.path.join(folder, 'forces/')
        all_force_files = glob.glob(force_folder)
        for i in range(len(all_force_files)):
            df = pd.DataFrame({'name': [name], 'action_id':[i]})
            if current_total_idx < N_train:
                train_result = pd.concat([train_result, df], axis=0, join='outer')
            elif current_total_idx >= val_starting_idx and current_total_idx < test_starting_idx:
                val_result = pd.concat([val_result, df], axis=0, join='outer')
            elif current_total_idx >= test_starting_idx:
                test_result = pd.concat([test_result, df], axis=0, join='outer')
            else:
                pass
            current_total_idx += 1
    
    return train_result, val_result, test_result


## Split within each run
def create_splits_per_run(dataset_path,  train_ratio, val_ratio, test_ratio):
    """create train and val splits"""
    folders = glob.glob(dataset_path + '*')
    train_result = pd.DataFrame(columns = ['name', 'id'])
    val_result = pd.DataFrame(columns = ['name', 'id'])
    test_result = pd.DataFrame(columns = ['name', 'id'])
    # the last N_test are test data, and the N_val before these are val data
    for folder in folders:
        print(folder)
        name = folder.split('/')[-1]
        # if 'wipe' in name:
        #     print(name)
        force_folder = os.path.join(folder, 'forces/')
        all_force_files = glob.glob(force_folder + '*')
        N_dp = len(all_force_files)
        for i in range(len(all_force_files)):
            df = pd.DataFrame({'name': [name], 'id':[i]})
            if i < N_dp*train_ratio:
                train_result = pd.concat([train_result, df], axis=0, join='outer')
            elif i >= N_dp*(1 - val_ratio - test_ratio) and i < N_dp*(1 - test_ratio):
                val_result = pd.concat([val_result, df], axis=0, join='outer')
            elif i >= N_dp*(1 - test_ratio):
                test_result = pd.concat([test_result, df], axis=0, join='outer')
            else:
                pass
    return train_result, val_result, test_result

def create_csv(folders):
    all_result = pd.DataFrame(columns = ['name', 'id'])
    for folder in folders:
        name = folder.split('/')[-1]
        print(name)
        force_folder = os.path.join(folder, 'forces/')
        print(force_folder)
        all_force_files = glob.glob(force_folder + '*')
        N_dp = len(all_force_files)
        print(N_dp)
        for i in range(len(all_force_files)):
            df = pd.DataFrame({'name': [name], 'id':[i]})
            all_result = pd.concat([all_result, df], axis=0, join='outer')
    return all_result 

def segment_dataset(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
                        debug = False, device = 'cuda', batch_size = 4):
    force_folder = os.path.join(folder, 'forces/*')
    all_force_files = glob.glob(force_folder )
    print(len(all_force_files))
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, 'images/'), exist_ok=True)
    sam_model.eval()
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    current_idx = 0
    while current_idx < len(all_force_files):
        ## assemble a single batch
        img_batch = None
        bbox_batch = None
        original_img_batch = []
        dps = []
        for i in range(batch_size):
            if current_idx >= len(all_force_files):
                break
            img = skimage.io.imread(os.path.join(folder, f'images/{current_idx}.png'))
            img_square = np.zeros((640,640,3), dtype = np.uint8)
            img_square[0:360,:,:] = img
            img = img_square
            img = cv2.resize(img,(256, 256))
            H, W, _ = img.shape
            img = np.uint8(img)
            original_img_batch.append(copy.deepcopy(img))
            
            img = sam_transform.apply_image(img)
            img_tensor = torch.as_tensor(img.transpose(2, 0, 1)).to(device)
            bbox = sam_transform.apply_boxes(bbox_raw, (H, W))
            box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4)
            if img_batch is None:
                img_batch = img_tensor.unsqueeze(0)
                bbox_batch = box_torch
            else:
                img_batch = torch.concat((img_batch, img_tensor.unsqueeze(0)), axis = 0)
                bbox_batch = torch.concat((bbox_batch, box_torch), axis = 0)
            current_idx += 1

        # print(current_idx)
        with torch.no_grad():
            input_img = sam_model.preprocess(img_batch) # (1, 3, 1024, 1024)
            ts_img_embedding = sam_model.image_encoder(input_img)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=bbox_batch,
                masks=None,
            )
            seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
                )
            seg_prob = torch.sigmoid(seg_prob)
            # convert soft mask to hard mask
            seg_prob = seg_prob.cpu().numpy().squeeze()
            seg = (seg_prob > 0.5).astype(np.uint8)

        ## save batch
        for i in range(len(original_img_batch)):
            if len(original_img_batch) > 1:
                segmented_img = cv2.bitwise_and(original_img_batch[i], original_img_batch[i], mask = seg[i])
            else:
                segmented_img = cv2.bitwise_and(original_img_batch[i], original_img_batch[i], mask = seg)
            if debug:
                plt.figure(figsize=(10, 10))
                plt.imshow(segmented_img)
                plt.axis('off')
                plt.show()
            #plt.savefig(f'{i}.png')
            segmented_img = cv2.cvtColor(segmented_img[0:144,:,:], cv2.COLOR_RGB2BGR) #256 * 9/16
            #exit()
            segmented_img = filter_segmented_image(segmented_img)
            cv2.imwrite(os.path.join(new_folder,f'images/{current_idx - len(original_img_batch) + i}.png'), \
                    segmented_img)

    ## Now copy paste the force files
    try:
        shutil.copytree(os.path.join(folder, 'forces/'), os.path.join(new_folder, 'forces/'))
    except:
        print('No force files found')


def segment_dataset_testing(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
                        debug = False, device = 'cuda', batch_size = 4):
    image_folder = os.path.join(folder, 'images/*')
    all_image_files = glob.glob(image_folder )
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, 'images/'), exist_ok=True)
    os.makedirs(os.path.join(new_folder, 'segmented_images/'), exist_ok=True)
    sam_model.eval()
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    current_idx = 0
    while current_idx < len(all_image_files):
        ## assemble a single batch
        img_batch = None
        bbox_batch = None
        original_img_batch = []
        dps = []
        for i in range(batch_size):
            if current_idx >= len(all_image_files):
                break
            img = skimage.io.imread(os.path.join(folder, f'images/{current_idx}.png'))
            img_square = np.zeros((640,640,3), dtype = np.uint8)
            img_square[0:360,:,:] = img
            img = img_square
            img = cv2.resize(img,(256, 256))
            H, W, _ = img.shape
            img = np.uint8(img)
            original_img_batch.append(copy.deepcopy(img))
            
            img = sam_transform.apply_image(img)
            img_tensor = torch.as_tensor(img.transpose(2, 0, 1)).to(device)
            bbox = sam_transform.apply_boxes(bbox_raw, (H, W))
            box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4)
            if img_batch is None:
                img_batch = img_tensor.unsqueeze(0)
                bbox_batch = box_torch
            else:
                img_batch = torch.concat((img_batch, img_tensor.unsqueeze(0)), axis = 0)
                bbox_batch = torch.concat((bbox_batch, box_torch), axis = 0)
            current_idx += 1
        print(current_idx)
        with torch.no_grad():
            input_img = sam_model.preprocess(img_batch) # (1, 3, 1024, 1024)
            ts_img_embedding = sam_model.image_encoder(input_img)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=bbox_batch,
                masks=None,
            )
            seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
                )
            seg_prob = torch.sigmoid(seg_prob)
            # convert soft mask to hard mask
            seg_prob = seg_prob.cpu().numpy().squeeze()
            seg = (seg_prob > 0.5).astype(np.uint8)

        ## save batch
        for i in range(len(original_img_batch)):
            if len(original_img_batch) > 1:
                segmented_img = cv2.bitwise_and(original_img_batch[i], original_img_batch[i], mask = seg[i])
            else:
                segmented_img = cv2.bitwise_and(original_img_batch[i], original_img_batch[i], mask = seg)

            segmented_img = cv2.cvtColor(segmented_img[0:144,:,:], cv2.COLOR_RGB2BGR) #256 * 9/16
            segmented_img = filter_segmented_image(segmented_img)
            original_img = original_img_batch[i]
            leading_zero_name = f"{current_idx - len(original_img_batch) + i:05d}"
            cv2.imwrite(os.path.join(new_folder,f'segmented_images/' + leading_zero_name + '.png'), \
                    segmented_img)
            cv2.imwrite(os.path.join(new_folder,f'images/' + leading_zero_name + '.png'), \
                cv2.cvtColor(original_img[0:144,:,:], cv2.COLOR_RGB2BGR))

def filter_segmented_image(img):
    import time
    start_time = time.time() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(len(cnts))
    #cnt = max(cnts, key=cv2.contourArea)
    cnts = sorted(cnts, key=cv2.contourArea)
    #print(len(cnts))
    # Output
    out = np.zeros(img.shape, np.uint8)
    cv2.drawContours(out, cnts[-4:], -1, (255,255,255), -1)
    out = cv2.bitwise_and(img, out)

    # print(f'Elapsed time is {time.time() - start_time}')
    # cv2.imshow('img', img)
    # cv2.imshow('inter',gray)
    # cv2.imshow('out', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return out


def filter_segmented_dataset(folder, new_folder, debug = False):
    img_folder = os.path.join(folder, 'images/*')
    all_image_files = glob.glob(img_folder )
    print(len(all_image_files))
    # os.makedirs(new_folder, exist_ok=True)
    # os.makedirs(os.path.join(new_folder, 'images/'), exist_ok=True)
    for i in range(len(all_image_files)):
        path = os.path.join(folder, f'images/{i}.png')
        img = cv2.imread(path)
        filter_segmented_image(img)
    return 


def extract_testing_force_data(save_folder, file):
    import pickle
    forces = []
    with (open(file, "rb")) as openfile:
        while True:
            try:
                forces.append(pickle.load(openfile))
            except EOFError:
                break
    forces = forces[0]
    ic(len(forces))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for idx, ft in enumerate(forces):
        data = ImgForce([], 0, ft, None, None, True)
        pickle.dump(data, open(os.path.join(save_folder, f'{idx}.pkl'), "wb"))

if __name__=="__main__":
    # ## extract image
    # save_dir = '/vast/palmer/home.grace/yz2379/project/Data/debug_2/'
    # data_dir = '/vast/palmer/home.grace/yz2379/project/Data/1006_data/all_val_segmented.pt'
    # extract_dataset(save_dir, data_dir, every = 10)
    # exit()

    # ## check image
    # path = '/vast/palmer/home.grace/yz2379/project/Data/1013_wiping_data/all_train.pt'
    # check_img_from_cp(path)
    # path = '/vast/palmer/home.grace/yz2379/project/Data/1013_wiping_data/all_train_segmented.pt'

    # check_img_from_cp(path)
    # path = '/vast/palmer/home.grace/yz2379/project/Data/1006_data/train_free_25_segmented_2.pt'
    # check_img_from_cp(path)
    # path = '/vast/palmer/home.grace/yz2379/project/Data/1006_data/train_free_40_segmented_2.pt'
    # check_img_from_cp(path)
    # path = '/vast/palmer/home.grace/yz2379/project/Data/1006_data/train_free_50_segmented_2.pt'
    # check_img_from_cp(path)
    # exit()

    ## Concatenate datasets:
    # folder_path = "/vast/palmer/home.grace/yz2379/project/Data/1019_assembled_data/*.pt"
    # all_files = glob.glob(folder_path)
    # all_files.sort()
    # folder = '/vast/palmer/home.grace/yz2379/project/Data/1019_assembled_data/'
    #print(all_files)
    # all_files = []
    # fns = []
    # for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
    #     fns.append('wiping_' + str(i))
    #for i in ['20','25','40','50']:
        # fns.append('test_lab_' + i)
        # fns.append('train_free_' + i)
    # fns += ['train_20_2','train_25_2','train_50_2']

    #for i in ['pear','sugar','jello']:
        # fns.append('test_' + i)
    # fns = ['wipe_40']

    # for fn in fns:
    #     #all_files.append( folder + fn  + '_segmented_b.pt')
    #     all_files.append( folder + fn  + '.pt')

    # merged_dataset = concatenate_datasets_and_resize(all_files[1:-1])
    # print(len(merged_dataset))
    # merged_dataset.save(folder, 'all_train')
    # merged_dataset = concatenate_datasets_and_resize(all_files[-1:])
    # print(len(merged_dataset))
    # merged_dataset.save(folder, 'all_val')

    ## Extract Images 
    # save_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train/'
    # checkpoint_path = '/vast/palmer/home.grace/yz2379/Data/0905_data/real_world_train.pt'
    # extract_dataset(save_dir, checkpoint_path)


    # # Extract aruco positions
    # data_dir = '/vast/palmer/home.grace/yz2379/Data/0911_data/train/'
    # old_dataset_cp_path = '/vast/palmer/home.grace/yz2379/Data/0911_data/train_data/real_world_train.pt'
    # old_dataset = Dataset()
    # old_dataset.load(old_dataset_cp_path, max_size=None)
    # extract_aruco(data_dir, old_dataset)

    # Split data, for new dataset that loads from disk
    # dataset_path = '/vast/palmer/home.grace/yz2379/project/Data/1126_data_segmented/'
    # total_N = calculate_dataset_stats(dataset_path)
    # train_result, val_result, test_result = \
    #    create_splits_per_run(dataset_path, train_ratio = 0.9, val_ratio = 0.1, test_ratio = 0.0)
    # train_result.to_csv(os.path.join(dataset_path,'train_result.csv'))
    # val_result.to_csv(os.path.join(dataset_path,'val_result.csv'))
    # # test_result.to_csv(os.path.join(dataset_path,'test_result.csv'))
    # exit()

    # #Split by run
    # dataset_path = '/vast/palmer/home.grace/yz2379/project/Data/final_data_segmented/'

    # # folders = []
    # all_training_folders = []
    # # for i in range(0,19):
    #     # all_training_folders.append(dataset_path + f'random_25_{i}')
    #     # all_training_folders.append(dataset_path + f'random_275_{i}')
    #     # all_training_folders.append(dataset_path + f'random_30_{i}')
    #     # all_training_folders.append(dataset_path + f'random_325_{i}')
    #     # all_training_folders.append(dataset_path + f'random_35_{i}')
    # for i in range(0,14):
    #     # all_training_folders.append(dataset_path + f'wipe_25_{i}')
    #     # all_training_folders.append(dataset_path + f'wipe_275_{i}')
    #     all_training_folders.append(dataset_path + f'wipe_30_{i}')
    #     # all_training_folders.append(dataset_path + f'wipe_325_{i}')
    #     # all_training_folders.append(dataset_path + f'wipe_35_{i}')

    # # random.shuffle(all_training_folders)
    # train_result = create_csv(all_training_folders[:])   
    # train_result.to_csv(os.path.join(dataset_path,'train_30wipe_result.csv')) 

    # folders = []
    # # for i in [19]:
    #     # folders.append(dataset_path + f'random_25_{i}')
    #     # folders.append(dataset_path + f'random_275_{i}')
    #     # folders.append(dataset_path + f'random_30_{i}')
    #     # folders.append(dataset_path + f'random_325_{i}')
    #     # folders.append(dataset_path + f'random_35_{i}')
    # for i in [14]:
    #     # folders.append(dataset_path + f'wipe_25_{i}')
    #     # folders.append(dataset_path + f'wipe_275_{i}')
    #     folders.append(dataset_path + f'wipe_30_{i}')
    #     # folders.append(dataset_path + f'wipe_325_{i}')
    #     # folders.append(dataset_path + f'wipe_35_{i}')
        
    # val_result = create_csv(folders)
    # val_result.to_csv(os.path.join(dataset_path,'val_30wipe_result.csv'))
    # folders = []
    # # for name in ['force_prediction_30', 'force_prediction_red']: # :,'sponge_0']
    # for name in ['peg_insertion']:
    #     folders.append(dataset_path + name)
    # test_result = create_csv(folders)
    # test_result.to_csv(os.path.join(dataset_path,'peg_insertion_result.csv'))

    # dataset_path = "/vast/palmer/home.grace/yz2379/project/Data/final_test_segmented/"
    # directories = [ f.path for f in os.scandir(dataset_path) if f.is_dir() ]
    # directories.sort()
    # names = []
    # for folder in directories:
    #     print(folder)
    #     name = folder.split('/')[-1]
    #     names.append(name)
    #     test_result = create_csv([folder])
    #     test_result.to_csv(os.path.join(dataset_path,name + '.csv'))
    # print(names)

    ## Force Prediction Testing
    folders = []
    dataset_path = "/vast/palmer/home.grace/yz2379/project/Data/test_data/"
    for name in ['white_BG_bigClamp', 'white_BG_paint', 'white_BG_jelloRed', 'white_BG_yellow',\
                    'lab_BG_bigClamp', 'lab_BG_paint', 'lab_BG_jelloRed', 'lab_BG_yellow']:
        folders.append(dataset_path + name)

    test_result = create_csv(folders)
    test_result.to_csv(os.path.join(dataset_path,'test_result.csv'))



    # ## Segment Images and Save 
    # from segment_anything.utils.transforms import ResizeLongestSide
    # from segment_anything import sam_model_registry, SamPredictor
    # import argparse
    # # sam_checkpoint =  '/vast/palmer/home.grace/yz2379/project/Data/sam_model_latest_b_finalAug.pth'
    # sam_checkpoint =  '/vast/palmer/home.grace/yz2379/project/Data/sam_model_latest_b_final.pth'
    # model_type = "vit_b"
    # device = "cuda"
    # sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam_model.to(device = device)

    # parser = argparse.ArgumentParser(description="segment images ")
    # parser.add_argument(
    # "-s",
    # "--start",
    # type=int,
    # default=0,)
    # parser.add_argument(
    # "-e",
    # "--end",
    # type=int,
    # default=1,)
    # args = parser.parse_args()

    # dataset_path = "/vast/palmer/home.grace/yz2379/project/Data/final_test/"
    # new_dataset_path =  "/vast/palmer/home.grace/yz2379/project/Data/final_test_segmented/"
    # directories = [ f.path for f in os.scandir(dataset_path) if f.is_dir() ]
    # directories.sort()
    # print(directories)
    # directories = directories[args.start:args.end]

    # for folder in directories:
    #     print(folder)
    #     name = folder.split('/')[-1]
    #     new_folder = os.path.join(new_dataset_path, name + '/')
    #     segment_dataset(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
    #         debug = False, device = 'cuda', batch_size = 3)
    #     #segment_dataset_testing(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
    #     #    debug = False, device = 'cuda', batch_size = 4)

    ## Filter segmented image
    # folder = '/vast/palmer/home.grace/yz2379/project/Data/experiment_data_1109/experiment_4_segmented/'
    # new_folder = None
    # filter_segmented_dataset(folder, new_folder, debug = False)

    # from segment_anything.utils.transforms import ResizeLongestSide
    # from segment_anything import sam_model_registry, SamPredictor
    # sam_checkpoint =  '/vast/palmer/home.grace/yz2379/project/Data/sam_model_latest_b.pth'
    # model_type = "vit_b"
    # device = "cuda"
    # sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam_model.to(device = device)
    # folder = '/vast/palmer/home.grace/yz2379/project/Data/experiment_data_1107/experiment_5/'
    # new_folder = '/vast/palmer/home.grace/yz2379/project/Data/experiment_data_Nov_testing/polygon/'
    # segment_dataset_testing(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
    #         debug = False, device = 'cuda', batch_size = 4)

    # #extract testing forces 
    # save_folder = '/vast/palmer/home.grace/yz2379/project/Data/final_test_slow/square_peg_smooth2/forces/'
    # file = '/vast/palmer/home.grace/yz2379/project/Data/final_test_slow/square_peg_smooth2/vision_ground_truth.pkl'
    # extract_testing_force_data(save_folder, file)

    # dataset_path = "/vast/palmer/home.grace/yz2379/project/Data/final_test/"
    # directories = [ f.path for f in os.scandir(dataset_path) if f.is_dir() ]
    # directories.sort()
    # for directory in directories:
    #     file = directory + '/vision_ground_truth.pkl'
    #     save_folder = directory + '/forces/'
    #     extract_testing_force_data(save_folder, file)extract_testing_force_data