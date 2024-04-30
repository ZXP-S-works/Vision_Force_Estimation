import sys
sys.path.append('../')
from utils.dataset import Dataset, ImgForce
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import glob

'''
Plot experiment data
'''
# # paths to predicted forces and ground truth forces
# No = 25
# No = input('No:')
# No = int(No)
# force_path = f'/home/grablab/VisionForceEstimator/real_world_data/experiment_data/experiment_{No}/vision_recorded_forces.pkl'
# ground_truth_path = f'/home/grablab/VisionForceEstimator/real_world_data/experiment_data/experiment_{No}/vision_ground_truth.pkl'

# # force_path = f'/home/grablab/VisionForceEstimator/real_world_data/1116_filtering_data/experiment_{No}/vision_recorded_forces.pkl'
# # ground_truth_path = f'/home/grablab/VisionForceEstimator/real_world_data/1116_filtering_data/experiment_{No}/vision_ground_truth.pkl'

# # force_path =  '/vast/palmer/home.grace/yz2379/project/Data/experiment_data_1109/experiment_6/vision_recorded_forces.pkl'
# # ground_truth_path =  '/vast/palmer/home.grace/yz2379/project/Data/experiment_data_1109/experiment_6/vision_ground_truth.pkl'

# # create lists to hold the data
# forces = []
# ground_truths = []


# # read data from the pickle files
# with (open(force_path, "rb")) as openfile:
#     while True:
#         try:
#             forces.append(pickle.load(openfile))
#         except EOFError:
#             break

# with (open(ground_truth_path, "rb")) as openfile:
#     while True:
#         try:
#             ground_truths.append(pickle.load(openfile))
#         except EOFError:
#             break

# def filter_force(x, history = 3):
#     history_list = []
#     new_x = []
#     for pt in x:
#         if len(history_list) < history:
#             history_list.append(pt)
#         else:
#             history_list.pop(0)
#             history_list.append(pt)
#         new_x.append(np.average(history_list))
#     return new_x

# # print(forces)
# # plot forces
# # print(np.array(forces).shape)
# filtered_force_Fx = np.array(filter_force(np.array(forces[0])[:,1]))
# filtered_force_Fz = np.array(filter_force(np.array(forces[0])[:,2]))                           

# plt.plot(filtered_force_Fx - filtered_force_Fx[0], 'r:', label='predicted Fx')
# plt.plot(filtered_force_Fz - filtered_force_Fz[0], 'b:', label='predicted Fz')
# plt.plot(filter_force(np.array(ground_truths[0])[:,1]) - filter_force(np.array(ground_truths[0])[:,1])[0], 'r', label='ground truth Fx')
# plt.plot(filter_force(np.array(ground_truths[0])[:,2]) - filter_force(np.array(ground_truths[0])[:,2])[0], 'b', label='ground truth Fz')
# plt.legend(loc="upper left")
# plt.show()


'''
Check collected data
'''
import pickle
#folder = './experiment_data/experiment_data_peg_insertion_1/square_peg/forces/'
folder = '/vast/palmer/home.grace/yz2379/project/Data/final_data_segmented/random_25_0/forces/'
fns = glob.glob(folder + "*")

forces = []
for i in range(len(fns)):
    with (open(folder + f'{i}.pkl', "rb")) as openfile:
        dp = pickle.load(openfile)
    f = dp[2]
    forces.append(f)
print(forces)
for i, force in enumerate(forces):
    print(i, force)
plt.plot(np.array(forces)[:,1], c = 'r')
plt.plot(np.array(forces)[:,2], c = 'b')
plt.show()

# # import pickle
# root_folder = './experiment_data/'
# folders = [f.path for f in os.scandir(root_folder) if f.is_dir() ]
# for folder in folders:
#     #folder = './experiment_data/experiment_data_peg_insertion_1/square_peg/forces/'
#     folder = folder + '/forces/'
#     fns = glob.glob(folder + "*")
#     name = folder.split('/')[-3]
#     forces = []
#     for i in range(len(fns)):
#         with (open(folder + f'{i}.pkl', "rb")) as openfile:
#             dp = pickle.load(openfile)
#         f = dp[2]
#         forces.append(f)
#     # print(forces)
#     # for i, force in enumerate(forces):
#     #     print(i, force)
#     plt.plot(np.array(forces)[:,1], c = 'r')
#     plt.plot(np.array(forces)[:,2], c = 'b')
#     plt.title(name)
#     plt.show()



exit()

## Replay Segmentation
import sys
sys.path.append('../')
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry, SamPredictor
import torch, cv2, copy
import skimage, scipy


def segment_dataset(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
                        debug = False, device = 'cuda', batch_size = 4):
    image_folder= os.path.join(folder, 'images/')
    all_image_files = glob.glob(image_folder + '*')
    os.makedirs(new_folder, exist_ok=True)
    print(image_folder)
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
            if debug:
                plt.figure(figsize=(10, 10))
                plt.imshow(segmented_img)
                plt.axis('off')
                plt.savefig('tmp.png')
            cv2.imwrite(os.path.join(new_folder,f'{current_idx - len(original_img_batch) + i}.png'), \
                    segmented_img)
            

sam_checkpoint =  '/home/grablab/Downloads/sam_model_latest_b_final.pth'
model_type = "vit_b"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device = 'cuda')
sam_model.eval()
# folder = f'/home/grablab/VisionForceEstimator/real_world_data/experiment_data/experiment_{No}/'
# new_folder = f'/home/grablab/VisionForceEstimator/real_world_data/experiment_data/experiment_{No}_segmented/'

folder = f'/home/grablab/VisionForceEstimator/real_world_data/experiment_data/lab_BG_yellow/'
new_folder = f'/home/grablab/VisionForceEstimator/real_world_data/experiment_data/white_BG_yellow_seg/'

segment_dataset(folder, new_folder, sam_model, bbox_raw = np.array([0, 0, 256, 144]), \
                        debug = False, device = 'cuda', batch_size = 4)
