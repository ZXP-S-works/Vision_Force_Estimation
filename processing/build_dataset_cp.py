import sys
sys.path.append('../')
from utils.dataset import Dataset, ImgForce
import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import os
import threading
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, SamPredictor
import torch
import copy
import skimage, glob
#ImgForce = collections.namedtuple('ImgForce', 'img, x, f')
ImgForce = collections.namedtuple('ImgForce', 'img, x, f, EE, servo_info, record_flag')

# data_dir = '/vast/palmer/home.grace/yz2379/Data/0911_data/train_data/'
# old_dataset_cp_path = '/vast/palmer/home.grace/yz2379/Data/0911_data/train_data/real_world_train.pt'
# new_dataset_cp_dir = '/vast/palmer/home.grace/yz2379/Data/0911_data/'
# new_dataset_cp_name = 'real_world_train'

def build_multiple_datasets(directories):
    for data_dir in directories:
        words = data_dir.split('/')
        fn = words[-1]
        old_dataset_cp_path = os.path.join(data_dir, 'real_world.pt')
        new_dataset_cp_dir = project_folder_path
        new_dataset_cp_name = fn
        build_dataset(data_dir, old_dataset_cp_path, new_dataset_cp_dir, new_dataset_cp_name)

def process_dataset_images(directories):
    for folder in directories:
        print(folder)
        img_folder = os.path.join(folder, 'images/')
        force_folder = os.path.join(folder, 'forces/')
        all_force_files = glob.glob(force_folder + '*')
        for i in range(len(all_force_files)):
            img_fn = os.path.join(img_folder, f'{i}.png')
            force_fn = os.path.join(force_folder, f'{i}.pkl')
            if not os.path.exists(os.path.join(folder, f'{i}.npy')):
                img = cv2.imread(img_fn)
                img = scipy.ndimage.zoom(img, (1/4, 1/4, 1), order=3)
                img = np.float16(np.swapaxes(img, 0, 2))
                np.save(os.path.join(folder, f'{i}'), img)
            else:
                img = np.load(os.path.join(folder, f'{i}.npy'))



def build_dataset(data_dir, old_dataset_cp_path, new_dataset_cp_dir, new_dataset_cp_name):
    print(data_dir)
    old_dataset = Dataset()
    old_dataset.load(old_dataset_cp_path, max_size=None)

    new_dataset = Dataset(size = old_dataset._max_size)
    for i in range(len(old_dataset)):
        if i%100 == 0:
            print(i)
        dp = old_dataset[i]
        img = cv2.imread(os.path.join(data_dir,f'{i}.png'))
        #img = img[:,100:580,:]
        img = scipy.ndimage.zoom(img, (1/4, 1/4, 1), order=3)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # print(img.shape)
        # exit()

        img = np.float16(np.swapaxes(img, 0, 2))
        new_dp = ImgForce(img, dp[1], dp[2], dp[3], dp[4], dp[5])
        new_dataset.add(new_dp)
    new_dataset.save(new_dataset_cp_dir, new_dataset_cp_name)

def alter_dataset(old_dataset_folder, cp_name):
    old_dataset = Dataset()
    old_dataset.load(old_dataset_folder + cp_name + ".pt", max_size=None)
    new_dataset = Dataset(size = old_dataset._max_size)
    for i in range(len(old_dataset)):
        # if i%100 == 0:
        #     print(i)
        dp = old_dataset[i]
        new_dp = ImgForce(dp[0], dp[1], dp[2], dp[3], dp[4], True)
        new_dataset.add(new_dp)
    print(old_dataset_folder, cp_name)
    #exit()
    new_dataset.save(old_dataset_folder, cp_name)

def build_dataset_segmented(data_dir, old_dataset_cp_path, new_dataset_cp_dir, new_dataset_cp_name, \
                sam_model, bbox_raw = np.array([0, 0, 256, 144]), debug = False, device = 'cuda', batch_size = 4):
    old_dataset = Dataset()
    old_dataset.load(old_dataset_cp_path, max_size=None)
    new_dataset = Dataset(size = old_dataset._max_size)
    sam_model.eval()
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    current_idx = 0
    print(new_dataset_cp_name)
    while current_idx < len(old_dataset):
        ## assemble a single batch
        img_batch = None
        bbox_batch = None
        original_img_batch = []
        dps = []
        for i in range(batch_size):
            # print(current_idx)
            if current_idx >= len(old_dataset):
                break
            dp = old_dataset[current_idx]
            dps.append(dp)
            img = skimage.io.imread(os.path.join(data_dir,f'{current_idx}.png'))
            img_square = np.zeros((640,640,3), dtype = np.uint8)
            img_square[0:360,:,:] = img
            img = img_square
            img = skimage.transform.resize(img,(256, 256),order=3,preserve_range=True,mode="constant",anti_aliasing=True)
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
            if debug:
                plt.figure(figsize=(10, 10))
                plt.imshow(segmented_img)
                plt.axis('off')
                plt.savefig('tmp.png')
            segmented_img = scipy.ndimage.zoom(segmented_img, (5/8, 5/8, 1), order=3)
            segmented_img = np.float16(np.swapaxes(segmented_img, 0, 2))
            dp = dps[i]
            new_dp = ImgForce(segmented_img, dp[1], dp[2], dp[3], dp[4], dp[5])
            new_dataset.add(new_dp)
    new_dataset.save(new_dataset_cp_dir, new_dataset_cp_name)
    # for i in range(len(old_dataset)):
    #     dp = old_dataset[i]
    #     #img = cv2.imread(os.path.join(data_dir,f'{i}.png'))
    #     img = skimage.io.imread(os.path.join(data_dir,f'{i}.png'))
    #     ## convert to square msg
    #     img_square = np.zeros((640,640,3), dtype = np.uint8)
    #     img_square[0:360,:,:] = img
    #     img = img_square

    #     if debug:
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(img/255)
    #         plt.axis('off')
    #         plt.show()
        
    #     img = skimage.transform.resize(img,(256, 256),order=3,preserve_range=True,mode="constant",anti_aliasing=True)
    #     H, W, _ = img.shape
    #     img = np.uint8(img)
    #     original_img = copy.deepcopy(img)
    #     img = sam_transform.apply_image(img)
    #     img_tensor = torch.as_tensor(img.transpose(2, 0, 1)).to(device)
    #     input_img = sam_model.preprocess(img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    #     assert input_img.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
    #     with torch.no_grad():
    #         ts_img_embedding = sam_model.image_encoder(input_img)
    #         bbox = sam_transform.apply_boxes(bbox_raw, (H, W))
    #         box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
    #         if len(box_torch.shape) == 2:
    #             box_torch = box_torch[:, None, :] # (B, 4) -> (B, 1, 4)
    #         sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
    #             points=None,
    #             boxes=box_torch,
    #             masks=None,
    #         )
    #         seg_prob, _ = sam_model.mask_decoder(
    #             image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
    #             image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
    #             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
    #             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
    #             multimask_output=False,
    #             )
    #         seg_prob = torch.sigmoid(seg_prob)
    #         # convert soft mask to hard mask
    #         seg_prob = seg_prob.cpu().numpy().squeeze()
    #         seg = (seg_prob > 0.5).astype(np.uint8)
    #     print(original_img.shape, seg.shape)
    #     segmented_img = cv2.bitwise_and(original_img, original_img, mask = seg)
    #     print(segmented_img)
    #     segmented_img = segmented_img[0:144,:,:] #256 * 9/16
    #     if debug:
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(segmented_img)
    #         plt.axis('off')
    #         plt.show()

    #     #I think I might have to swap the indeces back..
    #     segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
    #     segmented_img = scipy.ndimage.zoom(segmented_img, (5/8, 5/8, 1), order=3)
    #     if debug:
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(segmented_img)
    #         plt.axis('off')
    #         plt.show()

    #     segmented_img = np.float16(np.swapaxes(segmented_img, 0, 2))
    #     print(img.shape)
    #     new_dp = ImgForce(img, dp[1], dp[2], dp[3], dp[4], dp[5])
    #     new_dataset.add(new_dp)
    #     exit()
    # new_dataset.save(new_dataset_cp_dir, new_dataset_cp_name)



if __name__=='__main__':
    # ### Segment 
    # import argparse
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

    # class MyDataParallel(torch.nn.DataParallel):
    #     def __getattr__(self, name):
    #         try:
    #             return super().__getattr__(name)
    #         except AttributeError:
    #             return getattr(self.module, name)
    # print("Started")
    # # Build dataset with segmentation
    # sam_checkpoint =  '/vast/palmer/home.grace/yz2379/project/Data/sam_model_latest_b.pth'
    # model_type = "vit_b"
    # device = "cuda"
    # sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam_model = MyDataParallel(sam_model)

    # sam_model.to(device = device)

    # project_folder_path = '/vast/palmer/home.grace/yz2379/project/Data/1018_assembled_data/'
    # directories = [ f.path for f in os.scandir(project_folder_path) if f.is_dir() ]
    # directories.sort()

    # #directories = [directories[18]]
    # directories = directories[args.start:args.end]
    # #directories = [project_folder_path + 'wiping_8',project_folder_path + 'wiping_9' ]
    # for data_dir in directories:
    #     words = data_dir.split('/')
    #     fn = words[-1]
    #     old_dataset_cp_path = os.path.join(data_dir, 'real_world.pt')
    #     new_dataset_cp_dir = project_folder_path
    #     new_dataset_cp_name = fn
    #     build_dataset_segmented(data_dir, old_dataset_cp_path, new_dataset_cp_dir, new_dataset_cp_name + '_segmented_b', \
    #                             sam_model = sam_model, debug = False, batch_size = 2)

    # # exit()

    ## dataset building parallel version 
    N_threads = 10
    project_folder_path = '/vast/palmer/home.grace/yz2379/project/Data/1025_data/'
    directories = [ f.path for f in os.scandir(project_folder_path) if f.is_dir() ]
    directory_splits = np.array_split(directories, N_threads)

    threads = []
    for i in range(N_threads):
        #threads.append(threading.Thread(target=build_multiple_datasets, args = (directory_splits[i],)))
        threads.append(threading.Thread(target=process_dataset_images, args = (directory_splits[i],)))

    for thread in threads:
        print('flag')
        thread.start()
        
    
    print('started all')
    for thread in threads:
        thread.join()
    print('finished all')

    exit()

    ## Dataset building single thread
    project_folder_path = '/vast/palmer/home.grace/yz2379/project/Data/1006_data/'
    #directories = [ f.path for f in os.scandir(project_folder_path) if f.is_dir() ]
    #fns = ['wipe_40','dyn_40'] # ['train_20','train_25','train_40','train_50']
    fns = ['task_dataset_1']
    directories = []
    for fn in fns:
        data_dir = project_folder_path + fn + '/'
    directories.append(data_dir)
    for data_dir in directories:
        words = data_dir.split('/')
        fn = words[-1]
        if fn == '':
            fn = words[-2]
        old_dataset_cp_path = os.path.join(data_dir, 'real_world.pt')
        print(old_dataset_cp_path)
        print('fn', fn)
        new_dataset_cp_dir = project_folder_path
        new_dataset_cp_name = fn
        build_dataset(data_dir, old_dataset_cp_path, new_dataset_cp_dir, new_dataset_cp_name)

    # ## Dataset building single thread
    # project_folder_path = '/vast/palmer/home.grace/yz2379/project/Data/1013_wiping_data/'
    # #directories = [ f.path for f in os.scandir(project_folder_path) if f.is_dir() ]
    # directories = [project_folder_path + 'all_train/', project_folder_path + 'all_train_segmented/']
    # #fns = ['wipe_40','dyn_40'] # ['train_20','train_25','train_40','train_50']
    # #fns = ['task_dataset_1']
    # # directories = []
    # # for fn in fns:
    # #     data_dir = project_folder_path + fn + '/'
    # # directories.append(data_dir)
    # for data_dir in directories:
    #     words = data_dir.split('/')
    #     fn = words[-1]
    #     if fn == '':
    #         fn = words[-2]
    #     # old_dataset_cp_path = os.path.join(data_dir, 'real_world.pt')
    #     # print(old_dataset_cp_path)
    #     # print('fn', fn)
    #     # new_dataset_cp_dir = project_folder_path
    #     # new_dataset_cp_name = fn
    #     # build_dataset(data_dir, old_dataset_cp_path, new_dataset_cp_dir, new_dataset_cp_name)

    #     # ## alter dataset
    #     # alter_dataset(project_folder_path, fn )