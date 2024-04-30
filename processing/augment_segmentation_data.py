import cv2 
import glob, copy, os, random
from icecream import ic
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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


debug = False  # False 
obj_dim_range = [125,250]
obj_center = [100, 320]
original_folder = '/vast/palmer/home.grace/yz2379/project/Data/final_data_segmentation_data/'
new_folder = '/vast/palmer/home.grace/yz2379/project/Data/final_data_segmentation_data_augmented/'
original_images_folder = original_folder + 'images/'
original_masks_folder = original_folder + 'labels/'
new_images_folder = new_folder + 'images/'
new_masks_folder = new_folder + 'labels/'

os.makedirs(new_images_folder, exist_ok = True)
os.makedirs(new_masks_folder, exist_ok = True)

original_images_fns = glob.glob(original_images_folder + '*')
original_images_fns.sort()
original_masks_fns = glob.glob(original_masks_folder + '*')
original_masks_fns.sort()

## 
original_images = [] 
original_masks = []
for image_fn, mask_fn in zip(original_images_fns, original_masks_fns):
    original_images.append(cv2.imread(image_fn))
    original_masks.append(cv2.imread(mask_fn))


## Background images 
background_images = []
path_folder = '/vast/palmer/home.grace/yz2379/project/Data/MIT_Indoor/indoorCVPR_09/Images/office/'
img_fns = os.listdir(path_folder)
for fn in img_fns:
    img = cv2.imread(os.path.join(path_folder,fn))
    if img is not None:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        background_images.append(img)

path_folder = '/vast/palmer/home.grace/yz2379/project/Data/MIT_Indoor/indoorCVPR_09/Images/corridor/'
img_fns = os.listdir(path_folder)
for fn in img_fns:
    img = cv2.imread(os.path.join(path_folder,fn))
    if img is not None:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        background_images.append(img)

path_folder = '/vast/palmer/home.grace/yz2379/project/Data/MIT_Indoor/indoorCVPR_09/Images/meeting_room/'
img_fns = os.listdir(path_folder)
for fn in img_fns:
    img = cv2.imread(os.path.join(path_folder,fn))
    if img is not None:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        background_images.append(img)


## Object images 
obj_folder = '/vast/palmer/home.grace/yz2379/project/Data/ycb_processed/'
obj_images = []
obj_masks = []
for i in range(10):
    obj_images.append(cv2.cvtColor(cv2.imread(obj_folder + 'obj_' + str(i) + '.png'), cv2.COLOR_BGR2RGB))
    obj_masks.append(np.load(obj_folder + 'mask_' + str(i) + '.npy'))


augmented_image_counter = 0
## First duplicate and reorder original images 
for image, mask in zip(original_images, original_masks):
    cv2.imwrite(new_images_folder + f'{augmented_image_counter}.png', image)
    cv2.imwrite(new_masks_folder + f'{augmented_image_counter}.png', mask)
    augmented_image_counter += 1

## Augment with random background and images 
multiplier = 90
no_obj_multipliers = 30
for tmp in range(multiplier):
    for idx in range(len(original_images)):
        original_image = original_images[idx]
        original_mask = original_masks[idx]
        if len(original_mask.shape) == 3:
            original_mask = original_mask[:, :, 0]
        original_mask = original_mask > 0
        # if debug:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(original_mask)
        #     plt.axis('off')
        #     plt.show()

        ##random background
        background_image = random.choice(background_images)
        y, x,_ = original_image.shape
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
        original_image_cv2_format = original_image
        mask_cv2_format = original_mask #np.swapaxes(original_mask,0,1) > 0.5

        augmented_image = background_image
        augmented_image[mask_cv2_format] = original_image_cv2_format.copy()[mask_cv2_format]

        augmented_mask = copy.deepcopy(original_masks[idx])
        #augmented_image = cv2.bitwise_and(original_image_cv2_format,background_image,mask = mask_cv2_format)
        if debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(augmented_image)
            plt.axis('off')
            plt.show()

        ## have a batch of images without objects
        if tmp >= no_obj_multipliers:
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

            # ic(obj_image.shape, obj_width, obj_height)
            #print(obj_size, obj_width, obj_height)
            # randomize object center:
            obj_center_copy = copy.copy(obj_center)
            #obj_center_copy[1] += random.choice([i for i in range(-15, 15)])
            obj_center_copy[1] += random.choice([i for i in range(-200, 200)])
            UL = [max(obj_center_copy[0] - int(obj_height/2),0), max(obj_center_copy[1] - int(obj_width/2),0)]
            BR = [min(obj_center_copy[0] + int(obj_height/2),y), min(obj_center_copy[1] + int(obj_width/2),x)]
            # ic(obj_center_copy, UL, BR)

            image_patch = augmented_image[UL[0]:BR[0], UL[1]:BR[1]]
            #print(obj_size, obj_center_copy, UL, BR)
            # if debug:
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(image_patch)
            #     plt.axis('off')
            #     plt.show()

            image_patch_upsampled = cv2.resize(image_patch, (obj_image.shape[1], obj_image.shape[0]))
            image_patch_upsampled[obj_mask_cv2_format] = obj_image[obj_mask_cv2_format]
            image_patch = cv2.resize(image_patch_upsampled, (image_patch.shape[1], image_patch.shape[0]))
            augmented_image[UL[0]:BR[0], UL[1]:BR[1]] = image_patch

            augmented_mask_patch = augmented_mask[UL[0]:BR[0], UL[1]:BR[1]]
            augmented_mask_patch_upsampled = cv2.resize(augmented_mask_patch, (obj_image.shape[1], obj_image.shape[0]))
            augmented_mask_patch_upsampled[obj_mask_cv2_format] = np.zeros(obj_image.shape)[obj_mask_cv2_format]
            augmented_mask_patch = cv2.resize(augmented_mask_patch_upsampled, (image_patch.shape[1], image_patch.shape[0]))
            augmented_mask[UL[0]:BR[0], UL[1]:BR[1]] = augmented_mask_patch


            if debug:  
                plt.figure(figsize=(10, 10))
                plt.imshow(augmented_image)
                plt.axis('off')
                plt.show()

                plt.figure(figsize=(10, 10))
                plt.imshow(augmented_mask)
                plt.axis('off')
                plt.show()

        cv2.imwrite(new_images_folder + f'{augmented_image_counter}.png', augmented_image)
        cv2.imwrite(new_masks_folder + f'{augmented_image_counter}.png', augmented_mask)
        augmented_image_counter += 1
        
