import os, pickle, glob, sys
import numpy as np
sys.path.append('../')
from utils.dataset import Dataset, ImgForce
import matplotlib.pyplot as plt
from natsort import natsorted
from check_data import extract_dataset
import shutil

datasets_dir = '/home/grablab/VisionForceEstimator/real_world_data/1024_data/'
save_dir = '/home/grablab/VisionForceEstimator/real_world_data/1024_testing_data/'

dataset_paths = os.listdir(datasets_dir)
print("Datasets:", dataset_paths)
for dataset_path in dataset_paths:
    print("creating", dataset_path)
    dataset_pt_path = os.path.join(datasets_dir, dataset_path, 'real_world.pt')
    

    #move all images
    source = os.path.join(datasets_dir, dataset_path, 'images')
    destination = os.path.join(save_dir,dataset_path)
    os.makedirs(destination, exist_ok=True)
    # gather all files
    allfiles = os.listdir(source)
    #print(source)
    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.copy(src_path, dst_path)

    # grab all pickle files
    pickle_list = glob.glob(os.path.join(datasets_dir, dataset_path, 'forces/*.pkl'))
    pickle_list = natsorted(pickle_list)
    dataset_pt = Dataset(len(pickle_list))
    # iterate over all pickle files and create dataset
    for pickle_file in pickle_list:
        data = pickle.load(open(pickle_file, 'rb'))
        dataset_pt.add(data)
    # save dataset
    print(destination)
    dataset_pt.save(destination + '/', name = 'real_world')
    dataset_pt.save(os.path.join(datasets_dir, dataset_path) + '/', name = 'real_world')
    forces, times, record_flags = extract_dataset(dataset_pt_path)
    
    plt.plot(np.array(forces)[:-1,1], label='x')
    plt.plot(np.array(forces)[:-1,2], label='z')
    plt.legend()
    plt.savefig(os.path.join(datasets_dir, dataset_path, 'data_visualization.png'))
    plt.clf()