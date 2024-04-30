import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

#ONNX stuff
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


import time, sys
sys.path.append('../')
from utils.dataset import Dataset, ImgForce
import sys

## Utility functions for debugging
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


def segment_img(save_dir, data_dir, N = 5000, vis = False, BBs = None, points = None, sam = None,\
    ONNX = None):
    predictor = SamPredictor(sam)

    ## Specify bb
    for i in range(N):
        if i%10 == 0:
            print(i/N)
        total_mask = None
        ## Segment one image with BB 
        image =  cv2.imread(data_dir + f'rgb_{i}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## Set image and predict mask
        predictor.set_image(image)
        
        if ONNX is not None:
            image_embedding = predictor.get_image_embedding().cpu().numpy()

        ## Use BB
        if BBs is not None:
            for BB in BBs:
                if ONNX is not None:
                    BB_reshaped = BB.reshape(2, 2)
                    input_label = np.array([2., 3.])
                    onnx_coord = BB_reshaped[None, :, :]
                    onnx_label = input_label[None, :].astype(np.float32)
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
                else:
                    masks, qualilties, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=BB[None, :],
                        multimask_output=False,
                    )
                    mask = masks[0]
                mask_int = np.uint8(mask.astype(int))
                if vis:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    show_mask(mask, plt.gca())
                    show_box(BB, plt.gca())
                    plt.axis('off')
                    plt.show()

                if total_mask is None:
                    total_mask = mask_int
                else:
                    total_mask = total_mask | mask_int
        ## use points 
        else:
            input_label = np.ones(len(points))
            if vis:
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                show_points(points, input_label, plt.gca())
                plt.axis('on')
                plt.show()  
            if not ONNX is None:
                onnx_coord = np.concatenate([points,np.array([[0.,0.]])],axis=0)[None, :, :]
                onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
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
            else:
                input_label = np.ones(len(points))
                masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=input_label,
                multimask_output=True,
                )
                idx = np.argmax(scores)
                mask = masks[idx]

            mask_int = np.uint8(mask.astype(int))
            if vis:
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(mask, plt.gca())
                show_points(points, input_label, plt.gca())
                plt.axis('off')
                plt.show()

            if total_mask is None:
                total_mask = mask_int
            else:
                total_mask = total_mask | mask_int

        ## Segmented image
        img_fn = save_dir + f'rgb_{i}.png'
        image = np.uint8(image)
        processed_image = cv2.bitwise_and(image,image,mask = total_mask)
        exit()
        cv2.imwrite(img_fn, processed_image)

        if vis:
            plt.figure(figsize=(10, 10))
            plt.imshow(processed_image)
            plt.show()

        ## Mask
        #np.save(save_dir + f'mask_{i}', total_mask)


def segment_img_both_ONNX(save_dir, data_dir, N = 5000, vis = False, points = None, BBs = None, sam = None,\
    ONNX = None):
    predictor = SamPredictor(sam)

    ## Specify bb
    for i in range(N):
        if i%10 == 0:
            print(i/N)
        ## Segment one image with BB 
        image =  cv2.imread(data_dir + f'rgb_{i}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## Set image and predict mask
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()

        points_input_label = np.ones(len(points))
        points_input = points

        N_BB = len(BBs)
        BB_input = BBs.reshape(N_BB*2, 2)
        BB_input_label = np.array([2.,3.]*N_BB)

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

        if vis:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            for BB in BBs:
                show_box(BB, plt.gca())
            show_points(points, points_input_label, plt.gca())
            plt.axis('off')
            plt.show()

        total_mask = mask_int

        ## Segmented image
        img_fn = save_dir + f'rgb_{i}.png'
        image = np.uint8(image)
        processed_image = cv2.bitwise_and(image,image,mask = total_mask)
        #cv2.imwrite(img_fn, processed_image)

        if vis:
            plt.figure(figsize=(10, 10))
            plt.imshow(processed_image)
            plt.show()

        exit()
        ## Mask
        #np.save(save_dir + f'mask_{i}', total_mask)

sam_checkpoint = "/vast/palmer/home.grace/yz2379/Data/sam_vit_h_4b8939.pth"
#sam_checkpoint = "/home/yifan/Data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

## Use ONNX 
onnx_model_quantized_path = "/vast/palmer/home.grace/yz2379/Data/sam_onnx_quantized.onnx"
onnx_model_path = onnx_model_quantized_path
## Note that: The ONNX model has a different input signature than SamPredictor.predict
ort_session = onnxruntime.InferenceSession(onnx_model_path, providers = ['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider'])


# save_dir = '/vast/palmer/home.grace/yz2379/Data/0801_processed_data_1/train/'
# data_dir = '/vast/palmer/home.grace/yz2379/Data/0801_raw_data/train/'

# save_dir = '/vast/palmer/home.grace/yz2379/Data/0801_processed_data_1/val/'
# data_dir = '/vast/palmer/home.grace/yz2379/Data/0801_raw_data/val/'

# save_dir = '/home/yifan/Data/0801_processed_data_1/val/'
# data_dir = '/home/yifan/Data/0801_raw_data/val/'

# save_dir = '/home/yifan/Data/0811_processed_data_1/val/'
# data_dir = '/home/yifan/Data/0811_raw_data/val/'



## segment with BBs
# save_dir = '/vast/palmer/home.grace/yz2379/Data/0811_processed_data_1/train/'
# data_dir = '/vast/palmer/home.grace/yz2379/Data/0811_raw_data/train/'

# segment_img(save_dir, data_dir, sam = sam, ONNX = ort_session, N = 5000, BBs = [np.array([20, 0, 120, 80]),\
#                                                         np.array([20, 70, 120, 160])], vis = False) #5000

# save_dir = '/vast/palmer/home.grace/yz2379/Data/0811_processed_data_1/val/'
# data_dir = '/vast/palmer/home.grace/yz2379/Data/0811_raw_data/val/'
# segment_img(save_dir, data_dir, sam = sam, ONNX = ort_session, N = 1000, BBs = [np.array([20, 0, 120, 80]),\
#                                                         np.array([20, 70, 120, 160])], vis = False) #5000

# save_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'
# data_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'
# segment_img(save_dir, data_dir, sam = sam, ONNX = ort_session, N = 1000, BBs = [np.array([0, 0, 120, 160])], vis = True) #5000

# save_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'
# data_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'

# Segement with points
# segment_img(save_dir, data_dir, sam = sam, ONNX = ort_session, N = 1000, points = np.array([[115, 15],\
#                                                     [115, 30],\
#                                                     [115, 42],\
#                                                     [100, 20],\
#                                                     [115, 120],\
#                                                     [115, 127],\
#                                                     [100, 149]
#                                                     ]), vis = True) #5000

## segment with both points and BBs all at once
# segment_img_both_ONNX(save_dir, data_dir, sam = sam, ONNX = ort_session, N = 5000, BBs = np.array([[20, 0, 120, 80],\
#                                                                                         [20, 70, 120, 160]]), 
#                                                                                         points = np.array([[110, 15],\
#                                                                                         [110, 27],\
#                                                                                         [110, 42],\
#                                                                                         [90, 31],\
#                                                                                         [92, 40],\
#                                                                                         [107, 125],\
#                                                                                         [107, 137],\
#                                                                                         [107, 149],\
#                                                                                         [92, 126],\
#                                                                                         [92, 115]]),vis = True) #5000
save_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'
data_dir = '/vast/palmer/home.grace/yz2379/Data/0905_data/train_raw_images/'
segment_img_both_ONNX(save_dir, data_dir, sam = sam, ONNX = ort_session, N = 5000, BBs = np.array([[0, 0, 120, 160]]), 
                                                                                        points =  np.array([[115, 15],\
                                                                                        [115, 30],\
                                                                                        [115, 42],\
                                                                                        [100, 20],\
                                                                                        [115, 120],\
                                                                                        [115, 127],\
                                                                                        [100, 149]
                                                                                        ]),vis = True) #5000


exit()
# save_dir = '/home/yifan/Data/0811_processed_data_1/train/'
# data_dir = '/home/yifan/Data/0811_raw_data/train/'
# segment_img(save_dir, data_dir, N = 5000, BBs = [np.array([20, 0, 120, 80]),\
#                                                         np.array([20, 70, 120, 160])]) #5000