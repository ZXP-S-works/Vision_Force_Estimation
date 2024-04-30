import time

import rtde_receive
import rtde_control
import numpy as np
from openhand_node.hands import Model_O, Model_T42
from manual_collect_data import record_img_f
import threading
from tqdm import tqdm

from real_world_data.ftsensors import FTSensor
from real_world_data.realsense import RGBDCamera
from utils.dataset import Dataset

J_HOME = [1.30, -2.01, 2.58, -2.14, -1.58, -0.277]
J_MID_TO_HOME = [1.41, -1.52, 2.20, -2.26, -1.58, -0.17]
J_ABOVE_WORKSPACE = [1.46, -1.09, 1.63, -2.12, -1.58, -0.12]
# C_WORKSPACE_CENTER = [0.076, -0.899, 0.065, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.074, -0.965, 0.09, 0, -3.142, 0]
C_WORKSPACE_CENTERS = [[0.074, -0.965, 0.03, 0, -3.142, 0],
                       [0.074, -0.965, 0.055, 0, -3.142, 0],
                       [0.074, -0.965, 0.068, 0, -3.142, 0],
                       [0.074, -0.965, 0.055, 0, -3.142, 0]]
TRANS_RANGE = 0.005
ROT_RANGE = np.deg2rad(3)
MIN_SPEED = 0.02
MAX_SPEED = 0.1
MIN_ACC = 0.01
MAX_ACC = 0.05

if __name__ == '__main__':
    # UR10e initialization
    ip = "192.168.1.100"  # IP for the UR10e robot
    rtde_c = rtde_control.RTDEControlInterface(ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    print("Robot is initialized.")

    # YOH O initialization
    # T = Model_O('/dev/ttyUSB0', 4, 1, 2, 3, 'XM', 0.0, 0.36, -0.11, 0.05)
    T = Model_T42(port='/dev/ttyUSB0', s1=10, s2=11, dyn_model='XM', s1_min=-0.035, s2_min=-0.05, motorDir=[-1, -1])

    print("YOH is initialized")

    # Data collection initialization
    data_size = 10000
    dataset_full = threading.Event()
    dataset_full_lock = threading.Lock()
    record = threading.Thread(target=record_img_f,
                              kwargs={"dataset_size": data_size, "save_dir": '../data/', "min_force": 0,
                                      "event": dataset_full, "lock": dataset_full_lock})
    print("Dataset is initialized")

    # Move from home to above_workspace
    while True:
        ans = input('Robot is going to move, are you ready? (Y/N)')
        if ans in ['Y', 'y']:
            break
        time.sleep(1)
    # q = rtde_r.getActualQ()
    # err = np.linalg.norm(np.asarray(J_HOME) - np.asarray(q))
    # if err > 0.3:
    #     raise NotImplementedError('Robot is far from home pose.')
    # rtde_c.moveJ(J_MID_TO_HOME)
    q = rtde_r.getActualQ()
    err1 = np.linalg.norm(np.asarray(J_ABOVE_WORKSPACE) - np.asarray(q))
    err2 = np.linalg.norm(np.asarray(J_MID_TO_HOME) - np.asarray(q))
    if err1 > 0.3 and err2 > 0.3:
        raise NotImplementedError('Robot is far from known pose.')
    rtde_c.moveJ(J_ABOVE_WORKSPACE)
    while True:
        ans = input('Is YOH model O mounted? (Y/N)')
        if ans in ['Y', 'y']:
            break
        time.sleep(1)

    # Move to workspace, close gripper
    rtde_c.moveL(C_WORKSPACE_CENTERS[0], speed=0.05, acceleration=0.1)
    record.start()
    time.sleep(1)
    # T.pinch_close(0.5)
    T.close(0.3)
    time.sleep(2)

    # # impulse to test the system
    # time.sleep(1)
    # next_pos = np.asarray(C_WORKSPACE_CENTER)
    # next_pos[0] += 0.005
    # # T.pinch_close(0.5)
    # T.close(0.3)
    # rtde_c.moveL(next_pos.tolist(), speed=0.5, acceleration=0.5)
    # time.sleep(4)

    # Randomly move within workspace, i.e., CENTER -+TRANS_RANG, -+ROT_RANGE
    for j in range(10000):
        next_pos = np.asarray(C_WORKSPACE_CENTERS[j // 10 % 4])  # every 10 waypoint move to another height
        next_pos[0] += np.random.uniform(-TRANS_RANGE, TRANS_RANGE)
        next_pos[2] += np.random.uniform(-TRANS_RANGE, TRANS_RANGE)
        next_pos[4] += np.random.uniform(-ROT_RANGE, ROT_RANGE)
        # T.pinch_close(0.5)
        if j % 10 == 0:  # open-close gripper
            T.release()
            time.sleep(0.5)
            rtde_c.moveL(next_pos.tolist(), speed=MAX_SPEED, acceleration=MAX_ACC)
        else:  # move the gripper to next location with random speed/acc
            T.close(0.3)
            speed = np.random.uniform(MIN_SPEED, MAX_SPEED)
            acceleration = np.random.uniform(MIN_ACC, MAX_ACC)
            rtde_c.moveL(next_pos.tolist(), speed=speed, acceleration=acceleration)
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
    rtde_c.moveJ(J_ABOVE_WORKSPACE)
    while True:
        ans = input('Is YOH model O unmounted? (Y/N)')
        if ans in ['Y', 'y']:
            break
        time.sleep(1)
    # rtde_c.moveJ(J_MID_TO_HOME)
    # rtde_c.moveJ(J_HOME)
