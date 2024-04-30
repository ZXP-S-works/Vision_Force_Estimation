import os
import sys
import time
import copy
import torch
import numpy as np
import collections
from tqdm import tqdm
import sys
sys.path.append("../")
from utils.parameters import parse_args
from utils.logger import Logger
from utils.dataset import Dataset, ImgForce, VisionForceDataset
from model.vision_force_estimator import create_estimator
import torch.nn as nn
from icecream import ic
from model.cnn import ResBlock

device='cuda'

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
def analyze_CNN():
    args, hyper_parameters = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    # setup estimator
    estimator = create_estimator(args)
    estimator.train()
    if args.load_model:
        estimator.loadModel(args.load_model)

    # get all the conv layers 
    conv_layers = []
    # get all the model children as list
    model_children = list(estimator.network.pre_conv.children()) + \
        list(estimator.network.conv_down.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        # ic(model_children[i])
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            # model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        # model_weights.append(child.weight)
                        conv_layers.append(child)
        elif type(model_children[i]) == nn.MaxPool2d:
            counter+=1
            # model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == ResBlock:
            for child in model_children[i].layer1.children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    # model_weights.append(child.weight)
                    conv_layers.append(child)
            # ic(conv_layers)
            for child in model_children[i].layer2.children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    # model_weights.append(child.weight)
                    conv_layers.append(child)
            # ic(conv_layers)
            # if model_children[i].upscale is not None:
            #     for child in model_children[i].upscale.children():
            #         if type(child) == nn.Conv2d:
            #             counter+=1
            #             # model_weights.append(child.weight)
            #             conv_layers.append(child)
            # ic(conv_layers)
            # exit()
    print(f"Total convolution layers: {counter}")
    print("conv_layers")
    ic(conv_layers)
    ## get image 
    if args.load_from_disk:
        if args.test_data_dir is not None:
            test_data_dir = args.test_data_dir
            fn = 'test_result.csv'
        else:
            test_data_dir = args.valid_data_dir
            fn = args.valid_csv_fn
        test_set = VisionForceDataset(test_data_dir, fn, \
                    n_history=args.n_history, history_interval=args.history_interval )
    batch, current_idx, indeces_batch = test_set.sample_by_index_verbose(1, 0)
    image, x, f, _ = estimator.load_batch(batch, is_train=False)
    image = image[0].unsqueeze(0)
    #ic(image.shape)
    
    ## generate features 
    outputs = []
    names = []
    for layer in conv_layers[0:]:
        ic(str(layer))
        ic(image.shape)
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)


    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from icecream import ic
    analyze_CNN()
