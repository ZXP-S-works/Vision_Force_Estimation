from cProfile import label
import time
import numpy as np
from openhand_node.hands import Model_O, Model_T42
from manual_collect_data import record_img_f, record_img_f_panda, record_img_f_raw
import threading
from tqdm import tqdm

from real_world_data.ftsensors import FTSensor
from FT_client import FTClient
from Franka_client import FrankaClient

from klampt.math import vectorops as vo
from klampt.math import se3, so3
import copy, math
import os
from icecream import ic
from real_world_utils import set_joint_config_linear, set_EE_transform_linear, set_EE_transform_trap, calculate_rotate_R

#J_HOME = [1.30, -2.01, 2.58, -2.14, -1.58, -0.277]
J_ABOVE_WORKSPACE= [0.1355788823558541, -0.4248259986638436, -0.1287394373428151, -2.579566111849772, -0.06686140052246786, 2.1762822047604455, 0.8316315718762212]
# C_WORKSPACE_CENTER = [0.076, -0.899, 0.065, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.074, -0.965, 0.09, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.074, -0.965, 0.03, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.071, -0.897, 0.05, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.071, -0.897, 0.08, 0, -3.142, 0]
ang = 45/180*math.pi
C_WORKSPACE_CENTER = [[math.sin(ang), -math.sin(ang), 0, \
        -math.sin(ang), -math.sin(ang), 0, \
        0, 0, -1], \
        [0.395, -0.005+0.0254/2, 0.27]] #0.25 0.26 0.3 #+0.0254/2
GRASP_HEIGHTS = [0, -0.02, -0.04] #[0, -0.01, -0.02, -0.04] #, -0.06]
FINGER_LATERAL_SHIFT_BOUND = [-0.05,0.05]
FINGER_NEUTRAL_BOUND = [0.29, 0.34]
MIN_XTRANS_RANGE = -0.02 # 0.02
MAX_XTRANS_RANGE = 0.02 # 0.02
MIN_ZTRANS_RANGE = -0.015
MAX_ZTRANS_RANGE = 0.015
ZLIFT = 0.015
ROT_RANGE = np.deg2rad(0)
CLOSE_AMT = 0.2
UR_MAX_SPEED = 0.05
SLEEP_AMT = 1

def gripper_close(T, amt):
    T.moveMotor(0, amt)
    T.moveMotor(1, amt)

def rand_move(controller, T, stop_event, stop_lock, pause_event, pause_lock, total_time = 60, \
              x_range=[-0.02,0.02],z_range=[-0.015,0.015]):

    # random_move_speed_range = [0.0001,0.004]
    # random_move_speed_range_log = [math.log(0.0001), math.log(0.004)]
    random_move_speed_range_log = [math.log(0.001), math.log(0.004)]
    max_time = 15
    with pause_lock:
        pause_event.clear()

    start_time = time.time()
    ## Starting EE transform
    next_pos = controller.get_EE_transform()

    while time.time() - start_time < total_time:
        rand_pos = copy.deepcopy(next_pos)
        rand_pos[1][1] += np.random.uniform(x_range[0], x_range[1])
        rand_pos[1][2] += np.random.uniform(z_range[0],z_range[1])
        speed = math.exp(np.random.uniform(random_move_speed_range_log[0], random_move_speed_range_log[1]))
        accel = speed/2
        #set_EE_transform_trap(controller, rand_pos, 0.004, 0.002, max_time=max_time)
        set_EE_transform_trap(controller, rand_pos, speed, accel, max_time)
        time.sleep(1)
        #exit()

    with stop_lock:
        stop_event.set()

def rand_move_orientation(controller, T, stop_event, stop_lock, pause_event, pause_lock, total_time = 60, \
              x_range=[-0.02,0.02],z_range=[-0.015,0.015], angle_range=[-10/180*math.pi, 10/180*math.pi]):

    random_move_speed_range = [0.0001,0.005]
    random_move_speed_range_log = [math.log(0.0001), math.log(0.004)]
    random_rotation_speed_range_log = [math.log(0.01), math.log(0.05)]
    max_time = 15
    with pause_lock:
        pause_event.clear()

    start_time = time.time()
    ## Starting EE transform
    next_pos = controller.get_EE_transform()

    current_angle = 0.

    while time.time() - start_time < total_time:
        rand_pos = copy.deepcopy(next_pos)
        rand_pos[1][1] += np.random.uniform(x_range[0], x_range[1])
        rand_pos[1][2] += np.random.uniform(z_range[0],z_range[1])
        target_angle = np.random.uniform(angle_range[0],angle_range[1])
        target_R = calculate_rotate_R(current_R=rand_pos[0], delta_alpha=target_angle)
        rand_pos = (target_R, rand_pos[1])
        speed = math.exp(np.random.uniform(random_move_speed_range_log[0], random_move_speed_range_log[1]))
        omega = math.exp(np.random.uniform(random_rotation_speed_range_log[0], random_rotation_speed_range_log[1]))
        accel = speed/2
        #set_EE_transform_trap(controller, rand_pos, 0.004, 0.002, max_time=max_time)
        set_EE_transform_linear(controller, rand_pos, max_trans_v = speed, max_rotation_w = omega, max_time=max_time)
        time.sleep(1)
        #exit()

    with stop_lock:
        stop_event.set()


def fixed_move(controller, T, stop_event, stop_lock, pause_event, pause_lock):
    max_time = 15
    with pause_lock:
        pause_event.clear()

    start_time = time.time()
    ## Starting EE transform
    next_pos = controller.get_EE_transform()
    Xs = np.array([0.01,0,0.01,0,-0.01,0,-0.01,0])*0.2
    Zs = np.array([0.01,0,-0.01,0,0.01,0,-0.01,0])*0.2
    wait_times = [1, 3,1,3,1,3,1,3]
    for i in range(8):
        rand_pos = copy.deepcopy(next_pos)
        ## debugging 
        rand_pos[1][1] += Xs[i]
        rand_pos[1][2] += Zs[i]
        set_EE_transform_trap(controller, rand_pos, 0.01,0.005, max_time)
        time.sleep(wait_times[i])

    with stop_lock:
        stop_event.set()
    return

if __name__ == '__main__':
    # UR10e initialization
    robot = FrankaClient('http://172.16.0.1:8080')
    robot.initialize()
    robot.start()
    print('-------')
    # ft sensor 
    ft_sensor = FTClient('http://172.16.0.64:8080')


    # YOH T42 initialization
    T = Model_T42(port='/dev/ttyUSB0', s1=1, s2=2, dyn_model='XM', s1_min=0.35, s2_min=0.04)
    print("YOH is initialized")
    T.release()
    time.sleep(SLEEP_AMT)

    ##Data collection initialization
    data_size = 2000 # 10000
    save_dir = './1011_data/experiment/'

    WIPING = False
    SIN_MOVE = False
    SQUARE_MOVE = True
    RANDOM_MOVE = False
    RANDOM_GRASP_Z = False
    RANDOM_MOVE_TRANS = False
    PICKUP = False
    if PICKUP:  
        assert RANDOM_MOVE_TRANS == True
        assert RANDOM_MOVE == True

    os.makedirs(save_dir, exist_ok=True) 
    dataset_full = threading.Event()
    dataset_full_lock = threading.Lock()
    pause = threading.Event()
    pause_lock = threading.Lock()
    record = threading.Thread(target=record_img_f_panda,
                              kwargs={"dataset_size": data_size, "save_dir": save_dir, "min_force": 0.0,
                                      "stop_event": dataset_full, "stop_lock": dataset_full_lock,
                                      "pause_event": pause, "pause_lock": pause_lock,
                                      "T": None, "save_raw_img": True, "Hz": 20}) #, "ft_sensor": ft_sensor}) "rtde_r": robot, 

    print("Dataset is initialized")

    # Move from home to above_workspace
    q = robot.get_joint_config()
    err1 = np.linalg.norm(np.asarray(J_ABOVE_WORKSPACE) - np.asarray(q))
    if err1 > 0.8:
        raise NotImplementedError('Robot is far from known pose.')
    set_joint_config_linear(robot, J_ABOVE_WORKSPACE)

   
    # Move to workspace
    #set_EE_transform_linear(robot, copy.deepcopy(C_WORKSPACE_CENTER))
    set_EE_transform_trap(robot, copy.deepcopy(C_WORKSPACE_CENTER), 0.02,0.01)
    
    # with pause_lock:
    #     pause.set()
    # time.sleep(1)

    next_pos = copy.deepcopy(C_WORKSPACE_CENTER)
    time.sleep(2)

    # calibrate FT sensor
    ft_sensor.zero_ft_sensor()
    ft_sensor.start_ft_sensor()

    if not PICKUP:
        # close gripper
        gripper_close(T, 0.30)
        time.sleep(1)

    # start recording
    record.start()

    init_move = True

    ## Sinusoidal movements
    if SIN_MOVE:
        next_pos = copy.deepcopy(C_WORKSPACE_CENTER)
        set_EE_transform_linear(robot, next_pos,0.01)
        time.sleep(2)
        gripper_close(T, 0.30)
        time.sleep(SLEEP_AMT)

        dt = 0.01
        start_time = time.time()
        radius = 0.01
        frequency = 1/10*2*math.pi
        while 1:
            current_time = time.time() - start_time
            #loop_starttime = time.time()
            rand_pos = copy.deepcopy(next_pos)
            rand_pos[1][1] += radius*math.sin(current_time*frequency)
            rand_pos[1][2] += radius*math.cos(current_time*frequency)
            #set_EE_traWnsform_linear(robot, rand_pos, 0.01)
            robot.set_EE_transform(rand_pos)        
            time.sleep(dt)
            with dataset_full_lock:
                if dataset_full.is_set():
                    break

    ## Wiping 
    if WIPING:
        # with pause_lock:
        #     pause.set()
        # rand_pos = copy.deepcopy(next_pos)
        # rand_pos[1][2] += 0.02
        # set_EE_transform_trap(robot, rand_pos, 0.001, 0.001)        
        # time.sleep(0.5)                
        # with pause_lock:
        #     pause.clear()
        rand_pos = copy.deepcopy(next_pos)
        rand_pos[1][2] -= 0.02
        set_EE_transform_trap(robot, rand_pos, 0.001, 0.001)       
        time.sleep(0.5)

        rand_pos = copy.deepcopy(next_pos)
        rand_pos[1][2] -= 0.02
        rand_pos[1][1] += 0.03
        set_EE_transform_trap(robot, rand_pos, 0.001, 0.001)   
        time.sleep(0.5)

        rand_pos = copy.deepcopy(next_pos)
        rand_pos[1][2] -= 0.02
        rand_pos[1][1] -= 0
        set_EE_transform_trap(robot, rand_pos, 0.001, 0.001)   
        time.sleep(0.5)

  
    ## Square movements
    if SQUARE_MOVE:
        Xs = [MIN_XTRANS_RANGE,MIN_XTRANS_RANGE,MAX_XTRANS_RANGE,MAX_XTRANS_RANGE, MIN_XTRANS_RANGE,MAX_XTRANS_RANGE,MAX_XTRANS_RANGE,MIN_XTRANS_RANGE,MIN_XTRANS_RANGE]
        Zs = [MIN_ZTRANS_RANGE, MAX_ZTRANS_RANGE, MAX_ZTRANS_RANGE, MIN_ZTRANS_RANGE, MIN_ZTRANS_RANGE, MIN_ZTRANS_RANGE, MAX_ZTRANS_RANGE, MAX_ZTRANS_RANGE,MIN_ZTRANS_RANGE]
        for i in range(len(Xs)):
            rand_pos = copy.deepcopy(next_pos)
            rand_pos[1][1] += Xs[i]
            rand_pos[1][2] += Zs[i]
            set_EE_transform_trap(robot, rand_pos, 0.003, 0.001)
            time.sleep(0.5)

    if PICKUP:
        with pause_lock:
            pause.set()
        gripper_close(T, 0.30)
        time.sleep(SLEEP_AMT)
        next_pos[1][2] += ZLIFT
        set_EE_transform_trap(robot, next_pos, 0.002, 0.001)
        time.sleep(0.5)
        with pause_lock:
            pause.clear()

    # Randomly move within workspace, i.e., CENTER -+TRANS_RANG, -+ROT_RANGE
    if RANDOM_MOVE:
        for j in range(10000):
            if j % 10 == 0:
                # every 10 cycle, place-pick the peg at random height but a fixed X- location
                # then move the peg to workspace_center
                if RANDOM_GRASP_Z:
                    with pause_lock:
                        pause.set()
                    
                    T.release()
                    time.sleep(SLEEP_AMT)

                    next_pos = copy.deepcopy(C_WORKSPACE_CENTER)
                    change_amount = GRASP_HEIGHTS[j // 10 % len(GRASP_HEIGHTS)]

                    next_pos[1][2] += change_amount

                    set_EE_transform_trap(robot, next_pos,0.005,0.002)
                    time.sleep(2)
                    gripper_close(T, 0.30)
                    # shift_value = np.random.uniform(FINGER_LATERAL_SHIFT_BOUND[0], FINGER_LATERAL_SHIFT_BOUND[1])
                    # neutral_value = np.random.uniform(FINGER_NEUTRAL_BOUND[0], FINGER_NEUTRAL_BOUND[1])
                    # T.moveMotor(0, neutral_value + shift_value) # shift left and right
                    # T.moveMotor(1, neutral_value - shift_value)

                    time.sleep(SLEEP_AMT)
                    if init_move:
                        with pause_lock:
                            pause.clear()
                        init_move = False


                    ## pickup 
                    # next_pos[1][2] += ZLIFT
                    # set_EE_transform_trap(robot, next_pos, 0.002, 0.001)

                    time.sleep(0.5)
                    with pause_lock:
                        pause.clear()
            else:
                if RANDOM_MOVE_TRANS:
                    # Randomly move within workspace, i.e., last_grasp_loc -+TRANS_RANG, -+ROT_RANGE
                    rand_pos = copy.deepcopy(next_pos)
                    rand_pos[1][1] += np.random.uniform(MIN_XTRANS_RANGE, MAX_XTRANS_RANGE)
                    rand_pos[1][2] += np.random.uniform(MIN_ZTRANS_RANGE, MAX_ZTRANS_RANGE)
                    set_EE_transform_trap(robot, rand_pos, 0.003, 0.001)
                time.sleep(0.5)
            with dataset_full_lock:
                if dataset_full.is_set():
                    break

    with dataset_full_lock:
        if not dataset_full.is_set():
            print("Robot stopped before data collection.")
    record.join()

    # Open hand, move to home
    T.release()
    time.sleep(1)
    set_joint_config_linear(robot, J_ABOVE_WORKSPACE)

    # shutdown
    # robot.shutdown()


    ### Check data 
    from check_data import extract_dataset
    import matplotlib.pyplot as plt
    checkpoint_path = save_dir + 'real_world.pt'
    forces, times, record_flags = extract_dataset(checkpoint_path)
    plt.plot(np.array(forces)[:-1,1:3], 'b')
    # plt.legend()
    plt.show()
