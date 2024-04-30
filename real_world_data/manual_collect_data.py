import os
from tracemalloc import start

from ftsensors import FTSensor
import numpy as np
import time
try:
    from realsense import RGBDCamera, visualize_data
except:
    print('Pyrealsense2 not installed')
from webcam import WebCamera
import sys
sys.path.append('../')
from utils.dataset import Dataset, ImgForce, process_img
from tqdm import tqdm
import matplotlib.pyplot as plt
from FT_client import FTClient
from Franka_client import FrankaClient
import cv2
import pickle
from real_world_utils import gravityCompensator
from CONSTANTS import ft_ip

DEBUG = False
plot = False
# ft_ip = 'http://192.168.0.103:80'


def record_img_f(dataset_size=50, save_dir='./data/', min_force=0.2,
                 stop_event=None, stop_lock=None,
                 pause_event=None, pause_lock=None,
                 t_img_f=0, rtde_r=None, T=None):
    # dataset
    dataset = Dataset(dataset_size)
    num_img = 0
    pbar = tqdm(total=dataset_size)
    fs = []

    # force-torque sensor
    ft_sensor = FTSensor()
    ft_sensor.startStreaming()

    # RGBD camera
    # rgbd_cam = RGBDCamera()
    # rgbd_cam.start_streaming()
    rgbd_cam = WebCamera()

    pause = False

    while num_img < dataset_size:
        #color_image, depth_image, time_stamp = rgbd_cam.get_rgb_frames(plot=plot)
        color_image, time_stamp = rgbd_cam.get_rgb_frames(plot=plot)
        # color_image = color_image[120:360, 160:480, :]

        # from FT buffer find the FT data that is closest to the image timestamp
        ft_data = ft_sensor.buffer.get(time_stamp, lag=t_img_f)
        if ft_data is not None:
            ft, ts = ft_data
            d_f = (ft ** 2).sum()
            d_f = np.sqrt(d_f)
            fs.append(ft)

            if pause_lock is not None:
                with pause_lock:
                    pause = True if pause_event.is_set() else False

            if d_f > min_force and not pause:
                print(d_f)
                img = process_img(color_image)
                # visualize_data(img, ft, time_stamp, ts)
                try:
                    ur_pose = None if rtde_r is None else rtde_r.getActualTCPPose()
                except:
                    ur_pose = None if rtde_r is None else rtde_r.get_EE_transform()  ## Franka 
                servo_info = None if T is None else T.readServoInfos()
                data = ImgForce(img, time_stamp, ft, ur_pose, servo_info)
                dataset.add(data)
                num_img += 1
                pbar.update(1)
                # time.sleep(0.05)
    dataset.save(save_dir, name='real_world_' + str(dataset_size))

    if stop_lock is not None:
        with stop_lock:
            stop_event.set()

    if DEBUG:
        ff = np.stack(fs, axis=1).T
        plt.figure()
        plt.title('Force VS Time')
        plt.plot(ff[:, 0], label='x', alpha=0.7)
        plt.plot(ff[:, 1], label='y', alpha=0.7)
        plt.plot(ff[:, 2], label='z', alpha=0.7)
        plt.legend()
        plt.xlabel('time step')
        plt.ylabel('f (N)')
        plt.savefig(os.path.join(save_dir, 'real_world_' + str(dataset_size) + '.pdf'))
        plt.show()
        batch = dataset.sample_continuous(50)
        for data in batch:
            img, ft, ts = data.img, data.f, data.x
            visualize_data(img, ft, ts, 0)

        return True


def record_img_f_panda(dataset_size=50, save_dir='./data/', min_force=0.2,
                 stop_event=None, stop_lock=None,
                 pause_event=None, pause_lock=None,
                 t_img_f=0, T=None, save_raw_img = False, Hz = 20): #, ft_sensor = None):
    print('-------------')
    print('-------------')

    # dataset
    dataset = Dataset(dataset_size)
    num_img = 0
    pbar = tqdm(total=dataset_size)
    fs = []

    ## There would be a small amount of time misalignment between camera and ft sensor 

    # RGBD camera
    # rgbd_cam = RGBDCamera()
    # rgbd_cam.start_streaming()
    rgbd_cam = WebCamera()
    
    ft_sensor = FTClient('http://172.16.0.64:8080')
    # ft_sensor.zero_ft_sensor()
    # ft_sensor.start_ft_sensor()

    # try: 
    #     # rtde_r = FrankaClient('http://172.16.0.1:8080')
    # except:
    rtde_r = None
    
    pause = False

    while num_img < dataset_size:
        loop_start_time = time.time()
        self.record_frequency = 20
        #color_image, depth_image, time_stamp = rgbd_cam.get_rgb_frames(plot=plot)

        # print(color_image.shape)
        ft_data = ft_sensor.read_ft_sensor()

        #print(ft_data)
        if ft_data is not None:
            ft = np.array(ft_data)
            d_f = (ft ** 2).sum()
            d_f = np.sqrt(d_f)
            fs.append(ft)
            if pause_lock is not None:
                with pause_lock:
                    pause = True if pause_event.is_set() else False
                    if pause:
                        print('PAUSE ON')
            #if d_f > min_force and not pause:
            # still same while pausing
            if d_f > min_force:
                # print(d_f)
                if not save_raw_img:
                    img = process_img(color_image)
                else:
                    img = []
                # visualize_data(img, ft, time_stamp, ts)
                ur_pose = None if rtde_r is None else rtde_r.get_EE_transform()  ## Franka 
                servo_info = None if T is None else T.readServoInfos()
                data = ImgForce(img, time_stamp, ft, ur_pose, servo_info, not pause)
                dataset.add(data)
                if save_raw_img:
                    cv2.imwrite(os.path.join(save_dir, f'{num_img}.png'), color_image)
                num_img += 1
                pbar.update(1)
            elapsed_time = time.time() - loop_start_time
            if elapsed_time < 1/Hz:
                time.sleep(1/Hz - elapsed_time)
            print(num_img)

    ## Webcam needs release
    rgbd_cam.shutdown()

    save_path = os.path.join(save_dir, 'real_world.pt')
    if os.path.exists(save_path):
        fn = 'real_world_' + str(int(time.time()))
    else:
        fn = 'real_world'
    dataset.save(save_dir, name= fn)
    if stop_lock is not None:
        with stop_lock:
            stop_event.set()

    if DEBUG:
        ff = np.stack(fs, axis=1).T
        plt.figure()
        plt.title('Force VS Time')
        plt.plot(ff[:, 0], label='x', alpha=0.7)
        plt.plot(ff[:, 1], label='y', alpha=0.7)
        plt.plot(ff[:, 2], label='z', alpha=0.7)
        plt.legend()
        plt.xlabel('time step')
        plt.ylabel('f (N)')
        plt.savefig(os.path.join(save_dir, 'real_world_' + str(dataset_size) + '.pdf'))
        plt.show()
        batch = dataset.sample_continuous(50)
        for data in batch:
            img, ft, ts = data.img, data.f, data.x
            visualize_data(img, ft, ts, 0)

        return True

def record_img_f_raw(save_dir='./data/',
                 stop_event=None, stop_lock=None,
                 pause_event=None, pause_lock=None,
                 Hz = 10, use_cam = True, controller = None): 
    print('-------------')
    if use_cam:
        cam = WebCamera()
    ft_sensor = FTClient(ft_ip)
    pause = False
    last_pause_status = False
    img_num = 0
    record_dataset_force_dir = save_dir + 'forces/'
    record_dataset_img_dir = save_dir + 'images/'
    os.makedirs(record_dataset_force_dir, exist_ok = True)
    os.makedirs(record_dataset_img_dir, exist_ok = True)

    gravity_compensator = gravityCompensator()
    first_pic_time = None
    while True:
        with stop_lock:
            if stop_event.is_set():
                print('RECORDING ENDED')
                break
        loop_start_time = time.time()
        if use_cam:
            color_image, time_stamp = cam.get_rgb_frames()  
        else:
            color_image = None
            time_stamp =None  
        ft_data = ft_sensor.read_ft_sensor()
        if ft_data is not None and color_image is not None:
            ft = np.array(ft_data)
            with pause_lock:
                pause = True if pause_event.is_set() else False
                if pause:
                    print('PAUSE ON')
                    pass
            if not pause:
                if first_pic_time is None:
                    first_pic_time = time.time()
                if pause != last_pause_status: #first timestep after unpausing 
                    record_flag = False
                else:
                    record_flag = True
                # compensate for gravity
                if controller is not None:
                    ft = gravity_compensator.get_compensated_force(ft, controller)

                data = ImgForce([], time_stamp, ft, None, None, record_flag)
                pickle.dump(data, open(os.path.join(record_dataset_force_dir, f'{img_num}.pkl'), "wb"))
                if use_cam:
                    cv2.imwrite(os.path.join(record_dataset_img_dir, f'{img_num}.png'), color_image)
                img_num += 1
                if img_num%100 == 0:
                    print(time.time() - first_pic_time, img_num)
            elapsed_time = time.time() - loop_start_time
            if elapsed_time < 1/Hz:
                time.sleep(1/Hz - elapsed_time)
            last_pause_status = pause
                
    ## Webcam needs release
    if use_cam:
        cam.shutdown()


if __name__ == '__main__':
    from FT_client import FTClient
    from Franka_client import FrankaClient
    record_img_f_panda(dataset_size = 500, save_dir = './random_test_data/', min_force=0.0, save_raw_img = True)
