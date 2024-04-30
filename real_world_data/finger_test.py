from cProfile import label
import time
import numpy as np
# from openhand_node.hands import Model_O, Model_T42
from manual_collect_data import record_img_f_raw
import threading
from tqdm import tqdm
from peg_in_hole import PegInHoleTask, set_joint_config_linear
from franka_collect_data import set_EE_transform_linear, rand_move, fixed_move

from real_world_utils import T_42_controller,play_audible_alert
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
from CONSTANTS import t_pickup_high, t_pickup, t_random_start, t_fixed_start, R_default, finger_zero_positions, UR5_ip, gripper_port, ft_ip

angle_range = 15/180*math.pi
trans_range = 0.015

#R_default= [math.sqrt(2)/2, -math.sqrt(2)/2, 0, -math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0.0, 0.0, -1] #Franka
# t_pickup_high = [0.3, 0.166, 0.35]
# t_pickup = [0.3, 0.166, 0.28] # might need to be adjusted
# t_wipe_start = [0.4, -0.02, 0.28]
# t_random_start = [0.41, -0.025, 0.27]


start_position_file = "../ur5_start_position.json"
start_position_labview_file = "../ur5_start_position_labview.json"
start_position_high_file = "../ur5_start_position_high.json"
wipe_start_position_file = "../ur5_wipe_start_position.json"
VFE_params = {'model_path':'/home/grablab/Downloads/final_h20_int10_hz10.pt',
              'n_history':20, 'history_interval':10}

def randomize_orientation_position(controller):
    rotation_arm = [0, 0, 0.15] # check this 
    current_T = controller.get_EE_transform()
    angle = (random.random() - 0.5)*2*angle_range
    #z = (random.random() - 0.5)*2*trans_range
    z = 0 #math.fabs(angle)/math.pi*180*-0.002
    y = (random.random() - 0.5)*2*trans_range
    delta_R = so3.from_rotation_vector([angle,0,0])
    target_R = so3.mul(delta_R, current_T[0])
    target_t = vo.add(vo.sub(current_T[1], rotation_arm), vo.add(so3.apply(delta_R, rotation_arm), [0,y,z]))
    set_EE_transform_linear(controller, (target_R, target_t), 0.01)
    time.sleep(1)
    return 


def wipe():
    dataset_dir = './test/'
    N_wiping = 1 #10 #10
    N_random_move = 0 #10
    max_joint_v = 0.1
    note = '1231_test'
    task_name = "wipe"
    task_args = {"is_randomizing_force":False, "randomizing_vel":False, "randomizing_control":False, 'workspace_size': 0.16}

    ## Setup
    print("Connecting to server...")
    #controller = FrankaClient('http://172.16.0.1:8080')
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)
    
    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)
    time.sleep(1)
    print("Model T42 is initialized")
    

    ## Wiping data collection
    for i in range(N_wiping):

        # Release gripper
        T.release()       

        # Pause running before the peg is put back into the slot
        val = input('(Wiping)Have you put back the peg?')

        print('Going to initial position')
        # Go to initial Position
        set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
        time.sleep(2)

        print('Calibrating FT sensor')
        ft_driver = FTClient(ft_ip)
        ft_driver.zero_ft_sensor()
        ft_driver.start_ft_sensor()

        # print('Going to pickup position')
        # # Go to pickup location high
        # set_EE_transform_linear(controller, (R_default, t_pickup_high), 0.03)
        # time.sleep(0.5)

        # # Go to pickup location
        # set_EE_transform_linear(controller, (R_default, t_pickup), 0.03)
        # time.sleep(0.5)

        #Close gripper
        T.close()

        # # Go to pickup location high
        # set_EE_transform_linear(controller, (R_default, t_pickup_high), 0.03)
        # time.sleep(0.5)

        # wiping
        PegInHoleTask(controller, T, wipe_start_position_file, dataset_dir, task_name, task_args,
                None, None, None, None, mode = 'legacy', task_timer = 2*60, VFE_params = VFE_params )
        
    controller.close()

def peg_in_hole_2d():
    dataset_dir = './1114_data/'
    N_wiping = 1 #10
    N_random_move = 0 #10
    max_joint_v = 0.1
    note = 'test_insert'
    task_name = "insert_object_2D"
    task_args = {}
    start_position_file = "../ur5_start_position.json"
    wipe_start_position_file = "../ur5_wipe_start_position.json"

    ## Setup
    print("Connecting to server...")
    #controller = FrankaClient('http://172.16.0.1:8080')
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)

    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)
    time.sleep(1)
    print("Model T42 is initialized")

    ## Wiping data collection
    for i in range(N_wiping):
        # Recording
        # save_dir = dataset_dir + note + f'_{i}/'
        # os.makedirs(save_dir, exist_ok=True) 
        # stop_event = threading.Event()
        # stop_lock = threading.Lock()
        # pause_event = threading.Event()
        # pause_lock = threading.Lock()
        # record = threading.Thread(target=record_img_f_raw,
        #         kwargs={"save_dir": save_dir,
        #                 "stop_event": stop_event, "stop_lock": stop_lock,
        #                 "pause_event": pause_event, "pause_lock": pause_lock,
        #                 "Hz": 10}) 

        # Release gripper
        T.release()
        
        print('Going to initial position')
        # Go to initial Position
        set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
        # time.sleep(2)

        # Pause running before the peg is put back into the slot
        # val = input('Have you put back the peg?')
        time.sleep(2)

        print('Calibrating FT sensor')
        ft_driver = FTClient(ft_ip)
        ft_driver.zero_ft_sensor()
        ft_driver.start_ft_sensor()

        input('Ready to close?')

        #Close gripper
        T.close()
        time.sleep(1)

        # wiping
        #PegInHoleTask(controller, T, wipe_start_position_file, dataset_dir, task_name, task_args,
        #        None, None, None, None, mode = 'legacy', task_timer = 120)
        PegInHoleTask(controller, T, start_position_file, dataset_dir, task_name, task_args,
                None, None, None, None, mode = 'legacy', task_timer = 120)
        
def track_position_force_trajectory():
    dataset_dir = './1116_data/'
    N = 1 #10
    max_joint_v = 0.1
    task_name = "track_force_position_trajectory"
    # forces = [[0.], [0.8]]
    # positions = [[0.,0.], [0.,-0.07]]

    # Positions are specified in the world frame
    # first stroke
    # forces = [[1.0], [0.8], [0.5], [0.4], [0.3], [0.2],[-0.1]]
    # positions = [[0.,0.],[0.0,0.03],[0.0075,0.06], [0.015,0.09],\
    #              [0.03,0.12], [0.04, 0.135], [0.045, 0.1425]]

    # second stroke
    forces = [[0.4], [0.4], [0.5], [1.0], [0.8], [0.3],[-0.1]]
    positions = [[0.,0.],[0.0,0.03],[-0.01,0.06], [-0.02,0.09],\
                 [-0.04,0.12], [-0.07, 0.13], [-0.073, 0.13]]

    task_args = {"positions": positions, "forces": forces}
    start_position_file = "../start_position_0.json"
    write_start_position_file = "../write_start_position.json"

    ## Setup
    print("Connecting to server...")
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)
    
    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)
    time.sleep(1)
    print("Model T42 is initialized")

    for i in range(N):

        # Release gripper
        T.release()     

        print('Going to initial position')
        # Go to initial Position
        set_joint_config_linear(controller, json.load(open(wipe_start_position_file, "r")), max_joint_v)
        time.sleep(2)

        print('Calibrating FT sensor')
        ft_driver = FTClient(ft_ip)
        ft_driver.zero_ft_sensor()
        ft_driver.start_ft_sensor()

        #Close gripper
        T.close()

        PegInHoleTask(controller, T, wipe_start_position_file, dataset_dir, task_name, task_args,
                None, None, None, None, mode = 'legacy', task_timer = 600, VFE_params = VFE_params, no_delta=False)
    
def measure():
    dataset_dir = './experiment_data_1113/'
    N = 1 #10
    max_joint_v = 0.1
    task_name = "measure"
    task_args = {}
    start_position_file = "../start_position_0.json"

    ## Setup
    print("Connecting to server...")
    #controller = FrankaClient('http://172.16.0.1:8080')
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)

    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)
    time.sleep(1)
    print("Model T42 is initialized")

    ## Wiping data collection
    for i in range(N):
        # Release gripper
        T.release()   

        print('Going to initial position')
        # Go to initial Position
        set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
        time.sleep(2)

        print('Calibrating FT sensor')
        ft_driver = FTClient('http://172.16.0.64:8080')
        ft_driver.zero_ft_sensor()
        ft_driver.start_ft_sensor()

        #Close gripper
        T.close()

        PegInHoleTask(controller, T, start_position_file, dataset_dir, task_name, task_args,
                None, None, None, None, mode = 'collection', task_timer = 600, VFE_params = VFE_params)

def friction_test():
    dataset_dir = './experiment_data/'
    max_joint_v = 0.1

    save_dir = dataset_dir + f'friction_0/'
    os.makedirs(save_dir, exist_ok=True) 
    stop_event = threading.Event()
    stop_lock = threading.Lock()
    pause_event = threading.Event()
    pause_lock = threading.Lock()
    record = threading.Thread(target=record_img_f_raw,
            kwargs={"save_dir": save_dir,
                    "stop_event": stop_event, "stop_lock": stop_lock,
                    "pause_event": pause_event, "pause_lock": pause_lock,
                    "Hz": 10, "use_cam":True}) 

    ## Setup
    print("Connecting to server...")
    # controller = FrankaClient('http://172.16.0.1:8080')
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)
    
    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)
    time.sleep(1)
    print("Model T42 is initialized")


    # Release gripper
    T.release() 

    # Go to initial Position
    #set_joint_config_linear(controller, json.load(open(start_position_file, "r")))
    set_joint_config_linear(controller, json.load(open(start_position_high_file, "r")))
    time.sleep(2)

    ft_sensor = FTClient(ft_ip)
    # calibrate FT sensor
    ft_sensor.zero_ft_sensor()
    ft_sensor.start_ft_sensor()

    # Close gripper
    T.close()
    time.sleep(1)

    # Start recording
    with pause_lock:
        pause_event.set()
    record.start()

    # rand move
    fixed_move(controller, T, stop_event, stop_lock, pause_event, pause_lock)

    with stop_lock:
        stop_event.set()
    controller.close()
    return 
    
def grab_and_move():
    dataset_dir = './experiment_data/'
    max_joint_v = 0.1

    save_dir = dataset_dir + f'lab_BG_yellow2/'
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

    ## Setup
    print("Connecting to server...")
    # controller = FrankaClient('http://172.16.0.1:8080')
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = UR5_ip)
    
    # start robot
    controller.initialize()
    controller.start()

    # create hand control, first release then close
    print("Conneting to Model_T42 hand...")
    T = T_42_controller(finger_zero_positions, port=gripper_port, data_collection_mode=False)    
    time.sleep(1)
    print("Model T42 is initialized")

    # Release gripper
    T.release() 
    # Go to initial Position
    set_joint_config_linear(controller, json.load(open(start_position_labview_file, "r")))
    # set_joint_config_linear(controller, json.load(open(start_position_file, "r")))
    time.sleep(2)

    ft_sensor = FTClient(ft_ip)
    # calibrate FT sensor
    ft_sensor.zero_ft_sensor()
    ft_sensor.start_ft_sensor()

    # Close gripper
    T.close()

    # Start recording
    with pause_lock:
        pause_event.set()
    record.start()
    with pause_lock:
        pause_event.clear()

    start_time = time.time()
    while time.time() - start_time < 20:
        time.sleep(1)
    play_audible_alert()
    (R, current_t) = controller.get_EE_transform()
    cmd = None
    while True:
        ic(current_t)
        delta_z = float(input('Desired delta z: '))
        delta_y = float(input('Desired delta y: '))
        if math.fabs(delta_z) > 0.05 or math.fabs(delta_y) > 0.05:
            print('Warning: delta z or y too large')
            continue
        else:
            target_t = [current_t[0], current_t[1]+delta_y, current_t[2]+delta_z]
        ic(target_t)
        flag = input('To stop, type q: ')
        if flag == 'q':
            break
        set_EE_transform_linear(controller, (R_default, target_t), 0.01)
        time.sleep(1)

        (R, current_t) = controller.get_EE_transform()

    with stop_lock:
        stop_event.set()
    controller.close()
    return 

if __name__=='__main__':
    # wipe()
    track_position_force_trajectory()
    # measure()
    # peg_in_hole_2d()
    # friction_test()
    # grab_and_move()