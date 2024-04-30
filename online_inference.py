import os
import sys
import time
import copy
from threading import Thread

import torch
import numpy as np
import collections
from tqdm import tqdm
import cv2
from real_world_data.webcam import WebCamera
print("In online_inference: ", os.getcwd())
from utils.parameters import parse_args
from utils.dataset import Dataset, ImgForce, process_img
from model.vision_force_estimator import create_estimator
from real_world_data.ftsensors import FT_reading, FTSensor
#import skimage

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, SamPredictor

import matplotlib.pyplot as plt
device = "cuda:0"


def filter_segmented_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea)
    out = np.zeros(img.shape, np.uint8)
    cv2.drawContours(out, cnts[-4:], -1, (255,255,255), -1)
    out = cv2.bitwise_and(img, out)
    return out



class VFEstimator:
    def __init__(self, nn, cam, n_history, history_interval, Hz = 20, SAM = None):
        self.nn = nn
        self.cam = cam
        self.FT = FT_reading()
        self.stream = False
        self.history_interval = history_interval
        self.h = torch.zeros([n_history * history_interval, self.nn.h * 8]).to(device)
        self.Hz = Hz
        self.counter = 0
        self.n_history = n_history
        self.SAM = SAM
        if self.SAM is not None:
            self.SAM.eval()
            self.sam_transform = ResizeLongestSide(self.SAM.image_encoder.img_size)
            bbox_raw = np.array([0, 0, 256, 144])
            bbox = self.sam_transform.apply_boxes(bbox_raw, (256, 256))
            self.box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
            if len(self.box_torch.shape) == 2:
                self.box_torch = self.box_torch[:, None, :] # (B, 4) -> (B, 1, 4)
        print("Received history size:", self.n_history)

    def normalize_rgb(self, img):
        img /= 255.
        img -= 0.5
        return img

    def startStreaming(self):
        self.stream = True
        self.thread = Thread(target=self.receiveHandler)
        self.thread.daemon = True
        self.thread.start()

        print('VFEstimator started')

    def stopStreaming(self):
        self.stream = False
        time.sleep(0.1)

    def receiveHandler(self):

        while self.stream:
            start_time = time.time()
            color_image, time_stamp = self.cam.get_rgb_frames()
            #color_image = cv2.imread('/home/grablab/Downloads/0.png')
            time_stamp = 0.
            
            if self.SAM is None:
                img = process_img(color_image)
                img = np.swapaxes(img, 1, 2)
                img = self.normalize_rgb(torch.tensor(img, dtype=torch.float)[:3].unsqueeze(0))
                img = img.to(device)
                print(img.shape)
            else:
                with torch.no_grad(): 
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    img_square = np.zeros((640,640,3), dtype = np.uint8)
                    img_square[0:360,:,:] = color_image
                    #img = skimage.transform.resize(img_square,(256, 256),order=3,preserve_range=True,mode="constant",anti_aliasing=True)
                    img = cv2.resize(img_square,(256, 256))
                    img = np.uint8(img)
                    original_img = copy.deepcopy(img)
                    img = self.sam_transform.apply_image(img)
                    img = torch.as_tensor(img.transpose(2, 0, 1)).to(device)
                    input_img = self.SAM.preprocess(img.unsqueeze(0))
                    ts_img_embedding = self.SAM.image_encoder(input_img)
                    sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
                        points=None,
                        boxes=self.box_torch,
                        masks=None,
                    )
                    seg_prob, _ = self.SAM.mask_decoder(
                        image_embeddings=ts_img_embedding.to(device), # (B, 256, 64, 64)
                        image_pe=self.SAM.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                        multimask_output=False,
                        )
                    seg_prob = torch.sigmoid(seg_prob)
                    # convert soft mask to hard mask
                    seg_prob = seg_prob.cpu().numpy().squeeze()
                    seg = (seg_prob > 0.5).astype(np.uint8)
                    #segmented img
                    img = cv2.bitwise_and(original_img, original_img, mask = seg)

                    ## need to scale image properly
                    img = cv2.cvtColor(img[0:144,:,:], cv2.COLOR_RGB2BGR)
                    img = filter_segmented_image(cv2.resize(img, (160, 90)))

                    # cv2.imshow('', img)
                    # cv2.waitKey(0) 
                    # cv2.destroyAllWindows() 

                    img = np.float16(np.swapaxes(img, 0, 2))
                    
                    img = self.normalize_rgb(torch.tensor(img, \
                        dtype=torch.float)[:3].unsqueeze(0))
                    img = img.to(device)
                
            with torch.no_grad():
                h = self.nn.forward_cnn(img)
                # self.h is a IFIO stack of CNN latent features
                self.h[-1:], self.h[:-1] = h, self.h[1:].clone()
                f, _ = self.nn.forward_tf(self.h[::self.history_interval], None)
            # print(img.shape, f.tolist())
            self.FT = FT_reading(f.tolist(), time_stamp)
            elapsed_time = time.time() - start_time
            # print(1/elapsed_time)
            if elapsed_time < 1/self.Hz:
                time.sleep(1/self.Hz - elapsed_time)
            self.counter += 1


    def getForce(self):
        if self.counter < self.n_history*self.history_interval:
            print("Memory:", self.counter, "/", self.n_history*self.history_interval)
            return None
        else:
            return self.FT.measurement


if __name__=="__main__":
    # Use such as: python online_inference.py --load_model=/home/grablab/Downloads/1002_history_10_best_val.pt

    args, hyper_parameters = parse_args()
    cam = WebCamera()
    #cam = None
    model_path = '/home/grablab/Downloads/1121_h20_int1_Hz10.pt'
    Hz = 10
    nn = create_estimator(args)
    nn.loadModel(model_path)
    nn.network.eval()
    USE_SEG = True
    if USE_SEG:
        sam_checkpoint =  '/home/grablab/Downloads/sam_model_latest_b.pth'
        model_type = "vit_b"
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam_model.to(device = device)
        vfe = VFEstimator(nn.network, cam, args.n_history, args.history_interval, SAM = sam_model, Hz = 10)
    else:
        vfe = VFEstimator(nn.network, cam, args.n_history, args.history_interval, Hz = 10)
    vfe.startStreaming()

    # Loop and print out the timing between different readings.
    for _ in range(10000):
        while True:
            v, t = vfe.FT.measurement, vfe.FT.timestamp
            dt = time.time() - t
            #print(v,t)
            #print("value: ", np.round(v, 1))
            #print("value: ", np.round(v, 1))
            #print('sensing t: ', np.round(dt, 3))
            time.sleep(0.2)
    vfe.stopStreaming()
