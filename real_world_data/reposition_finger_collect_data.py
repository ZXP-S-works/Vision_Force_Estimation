from cProfile import label
import time
import numpy as np
# from openhand_node.hands import Model_O, Model_T42
from manual_collect_data import record_img_f_raw
import threading
from tqdm import tqdm
from peg_in_hole import PegInHoleTask, set_joint_config_linear
from franka_collect_data import rand_move, fixed_move, rand_move_orientation
from real_world_utils import set_EE_transform_linear, T_42_controller, play_audible_alert
from ftsensors import FTSensor
from FT_client import FTClient
from Franka_client import FrankaClient
from ur_controller.ur5.ur5_wrapper import ur5Wrapper
from utils.dataset import Dataset

from klampt.math import vectorops as vo
from klampt.math import se3, so3
import copy, math
import os, json, random
from icecream import ic
from CONSTANTS import t_pickup_high, t_pickup, t_random_start, t_fixed_start, ft_ip, gripper_port, finger_zero_positions, UR5_ip

angle_range = 6/180*math.pi #4/180*math.pi #15/180*math.pi
trans_range = 0.001 #0.005 #0.015
def randomize_orientation_position(controller, randomize_z = False):
    rotation_arm = [0, 0, 0.15] # check this 
    current_T = controller.get_EE_transform()
    angle = (random.random() - 0.5)*2*angle_range
    #z = (random.random() - 0.5)*2*trans_range
    if randomize_z:
        z = -random.random()*0.01
    else:
        z = 0 #math.fabs(angle)/math.pi*180*-0.002
    ic(z)
    y = (random.random() - 0.5)*2*trans_range
    delta_R = so3.from_rotation_vector([angle,0,0])
    target_R = so3.mul(delta_R, current_T[0])
    target_t = vo.add(vo.sub(current_T[1], rotation_arm), vo.add(so3.apply(delta_R, rotation_arm), [0,y,z]))
    set_EE_transform_linear(controller, (target_R, target_t), 0.01)
    time.sleep(1)
    return 

ft_ip = ft_ip
gripper_port = gripper_port
#R_default= [math.sqrt(2)/2, -math.sqrt(2)/2, 0, -math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0.0, 0.0, -1] #Franka
# t_pickup_high = [0.3, 0.166, 0.35]
# t_pickup = [0.3, 0.166, 0.28] # might need to be adjusted
# t_wipe_start = [0.4, -0.02, 0.28]
# t_random_start = [0.41, -0.025, 0.27]
R_default = [0, 0, -1, math.sqrt(2)/2, math.sqrt(2)/2, 0, math.sqrt(2)/2, -math.sqrt(2)/2, 0]



if __name__=='__main__':
    dataset_dir = './final_data/'
    N_wiping = 15 #20
    use_slope = True #False #
    N_random_move = 0 #10
    N_fixed_peg_random_move = 0 #15
    time_per_run = 3*60 #4*60 #
    max_joint_v = 0.1
    note = 'wipe_35' #'sponge' #
    task_name = "wipe"
    if use_slope:
        workspace_size = 0.1
    else:
        workspace_size = 0.06
    task_args = {"is_randomizing_force":True, "randomizing_vel":False, "random_vel_range":[0.25, 1],\
                 "randomizing_control":True, "random_kp_Fz_logrange": [np.log(0.0001), np.log(0.005)],\
                "random_kp_Fx_logrange": [np.log(0.001), np.log(0.01)],\
                "kp_Fz_2_speedcap_ratio_logrange": [np.log(0.25), np.log(3)],\
                "kp_Fx_2_speedcap_ratio_logrange": [np.log(1), np.log(1)],\
                'workspace_size': workspace_size}
    # task_args = {"is_randomizing_force":False, "randomizing_vel":False, "random_vel_range":[0.25, 1],\
    #              "randomizing_control":False, "random_kp_Fz_logrange": [np.log(0.0001), np.log(0.005)],\
    #             "random_kp_Fx_logrange": [np.log(0.001), np.log(0.01)],\
    #             "kp_Fz_2_speedcap_ratio_logrange": [np.log(0.25), np.log(3)],\
    #             "kp_Fx_2_speedcap_ratio_logrange": [np.log(1), np.log(1)]}
    start_position_file = "../ur5_start_position.json"
    start_position_high_file = "../ur5_start_position_high.json"
    wipe_start_position_file = "../ur5_wipe_start_position.json"
    wipe_start_position_high_file = "../ur5_wipe_start_position_high.json"

    ## Setup
    print("Connecting to server...")
    # controller = FrankaClient('http://172.16.0.1:8080')
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)
    
    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=True)
    time.sleep(1)
    print("Model T42 is initialized")

    ## Wiping data collection
    for i in range(N_wiping):
        if i <= 9:
            continue
        # Recording
        save_dir = dataset_dir + note + f'_{i}/'
        os.makedirs(save_dir, exist_ok=True) 
        stop_event = threading.Event()
        stop_lock = threading.Lock()
        pause_event = threading.Event()
        pause_lock = threading.Lock()
        record = threading.Thread(target=record_img_f_raw,
                kwargs={"save_dir": save_dir,
                        "stop_event": stop_event, "stop_lock": stop_lock,
                        "pause_event": pause_event, "pause_lock": pause_lock,
                        "Hz": 10}) 

        # Release gripper
        T.release()

        play_audible_alert()
        # Pause running before the peg is put back into the slot
        val = input('Have you put back the peg?')

        # Go to initial Position
        set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
        time.sleep(2)

        ft_driver = FTClient(ft_ip)
        ft_driver.zero_ft_sensor()
        ft_driver.start_ft_sensor()

        # Go to pickup location high
        set_EE_transform_linear(controller, (R_default, t_pickup_high), 0.03)
        time.sleep(0.5)

        # Go to pickup location
        set_EE_transform_linear(controller, (R_default, t_pickup), 0.03)
        time.sleep(0.5)

        # Randomize orientation
        randomize_orientation_position(controller, randomize_z=True)

        #Close gripper
        T.close()

        # Go to pickup location high
        set_EE_transform_linear(controller, (R_default, t_pickup_high), 0.03)
        time.sleep(0.5)

        # Start recording
        with pause_lock:
            pause_event.set()
        record.start()
        if use_slope:
            PegInHoleTask(controller, T, wipe_start_position_high_file, dataset_dir, task_name, task_args,
                stop_event, stop_lock, pause_event, pause_lock, mode = 'collection', task_timer = time_per_run)
        else:
            # wiping
            PegInHoleTask(controller, T, wipe_start_position_file, dataset_dir, task_name, task_args,
                    stop_event, stop_lock, pause_event, pause_lock, mode = 'collection', task_timer = time_per_run)
        
        

    ## Random Move Data Collection
    for i in range(N_random_move):
        if i <= 13:
            continue
        # Recording
        save_dir = dataset_dir + note + f'_{i}/'
        os.makedirs(save_dir, exist_ok=True) 
        stop_event = threading.Event()
        stop_lock = threading.Lock()
        pause_event = threading.Event()
        pause_lock = threading.Lock()
        record = threading.Thread(target=record_img_f_raw,
                kwargs={"save_dir": save_dir,
                        "stop_event": stop_event, "stop_lock": stop_lock,
                        "pause_event": pause_event, "pause_lock": pause_lock,
                        "Hz": 10}) 

        # Release gripper
        T.release()

        # Go to initial Position
        # set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
        set_joint_config_linear(controller, json.load(open(start_position_high_file, "r")))
        print("At initial position")
        time.sleep(2)


        ft_sensor = FTClient(ft_ip)
        # calibrate FT sensor
        ft_sensor.zero_ft_sensor()
        ft_sensor.start_ft_sensor()

        # Go to random collect initial position
        set_EE_transform_linear(controller, (R_default, t_random_start), 0.03)
        time.sleep(0.5)


        # Randomize orientation
        randomize_orientation_position(controller)

        # val = input('Have you put back the peg?')
        # time.sleep(3)

        # Close gripper
        T.close()
        # controller.close()
        # exit()
        # Go to random collect initial position
        set_EE_transform_linear(controller, (R_default, t_random_start), 0.03)
        time.sleep(1)


        # Start recording
        with pause_lock:
            pause_event.set()
        record.start()

        # rand move
        rand_move(controller, T, stop_event, stop_lock, pause_event, pause_lock, total_time = time_per_run,
                  x_range=[-0.015,0.015],z_range=[-0.02,0.01]) #z_range=[-0.015,0.005]

        # play_audible_alert()

    ## Random Move Data Collection
    for i in range(N_fixed_peg_random_move):
        if i <= 2:
            continue
        # Recording
        save_dir = dataset_dir + note + f'_{i}/'
        os.makedirs(save_dir, exist_ok=True) 
        stop_event = threading.Event()
        stop_lock = threading.Lock()
        pause_event = threading.Event()
        pause_lock = threading.Lock()
        record = threading.Thread(target=record_img_f_raw,
                kwargs={"save_dir": save_dir,
                        "stop_event": stop_event, "stop_lock": stop_lock,
                        "pause_event": pause_event, "pause_lock": pause_lock,
                        "Hz": 10, "controller": controller}) 

        # Release gripper
        T.release()

        # Go to initial Position
        # set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
        set_joint_config_linear(controller, json.load(open(start_position_high_file, "r")))
        print("At initial position", controller.get_EE_transform())
        time.sleep(2)

        ft_sensor = FTClient(ft_ip)
        # calibrate FT sensor
        ft_sensor.zero_ft_sensor()
        ft_sensor.start_ft_sensor()

        # Go to random collect initial position
        set_EE_transform_linear(controller, (R_default, t_fixed_start), 0.03)
        time.sleep(0.5)

        # Randomize orientation
        randomize_orientation_position(controller)

        # Close gripper
        T.close()

        # Go to random collect initial position
        set_EE_transform_linear(controller, (R_default, t_fixed_start), 0.03)
        time.sleep(1)


        # Start recording
        with pause_lock:
            pause_event.set()
        record.start()

        # rand move
        rand_move_orientation(controller, T, stop_event, stop_lock, pause_event, pause_lock, total_time = 4*60,
                x_range=[-0.01,0.01],z_range=[-0.015,0.005], angle_range=[-15/180*math.pi, 15/180*math.pi]) #z_range=[-0.015,0.005]
                #   x_range=[-0.015,0.015],z_range=[-0.02,0.01], angle_range=[-15/180*math.pi, 15/180*math.pi]) #z_range=[-0.015,0.005]

    T.release()
    controller.close()

    play_audible_alert()
    time.sleep(0.5)
    play_audible_alert()