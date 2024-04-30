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
# C_WORKSPACE_CENTER = [0.074, -0.965, 0.03, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.071, -0.897, 0.05, 0, -3.142, 0]
# C_WORKSPACE_CENTER = [0.071, -0.897, 0.08, 0, -3.142, 0]
C_WORKSPACE_CENTER = [0.069, -0.897, 0.105, 0, -3.142, 0]
GRASP_HEIGHTS = [0, -0.02, -0.04, -0.02]
XTRANS_RANGE = 0.02
ZTRANS_RANGE = 0.01
ZLIFT = 0.015
ROT_RANGE = np.deg2rad(0)
CLOSE_AMT = 0.3
UR_MAX_SPEED = 0.05
SLEEP_AMT = 1

if __name__ == '__main__':
    # UR10e initialization
    ip = "192.168.1.100"  # IP for the UR10e robot
    rtde_c = rtde_control.RTDEControlInterface(ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    print("Robot is initialized.")

    # YOH O initialization
    # T = Model_O('/dev/ttyUSB0', 4, 1, 2, 3, 'XM', 0.0, 0.36, -0.11, 0.05)
    # YOH T42 initialization
    T = Model_T42(port='/dev/ttyUSB0', s1=10, s2=11, dyn_model='XM', s1_min=-0.25, s2_min=-0.44, motorDir=[-1, -1])

    print("YOH is initialized")

    # Data collection initialization
    data_size = 100
    dataset_full = threading.Event()
    dataset_full_lock = threading.Lock()
    pause = threading.Event()
    pause_lock = threading.Lock()
    record = threading.Thread(target=record_img_f,
                              kwargs={"dataset_size": data_size, "save_dir": '../data/', "min_force": 0.0,
                                      "stop_event": dataset_full, "stop_lock": dataset_full_lock,
                                      "pause_event": pause, "pause_lock": pause_lock,
                                      "rtde_r": rtde_r, "T": T})
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
    rtde_c.moveL(C_WORKSPACE_CENTER, speed=UR_MAX_SPEED, acceleration=0.1)
    record.start()
    # time.sleep(1)
    # T.pinch_close(0.5)
    next_pos = np.asarray(C_WORKSPACE_CENTER)
    # T.close(CLOSE_AMT)
    time.sleep(2)

    # # impulse to test the system
    # time.sleep(1)
    # next_pos = np.asarray(C_WORKSPACE_CENTER)
    # next_pos[0] += 0.005
    # # T.pinch_close(0.5)
    # T.close(CLOSE_AMT)
    # rtde_c.moveL(next_pos.tolist(), speed=0.5, acceleration=0.5)
    # time.sleep(4)

    # Randomly move within workspace, i.e., CENTER -+TRANS_RANG, -+ROT_RANGE
    for j in range(10000):
        if j % 10 == 0:
            # every 10 cycle, place-pick the peg at random height but a fixed X- location
            # then move the peg to workspace_center
            # with pause_lock:
            #     pause.set()
            rtde_c.moveL(next_pos.tolist(), speed=UR_MAX_SPEED, acceleration=0.01)
            time.sleep(SLEEP_AMT)
            T.release()
            time.sleep(SLEEP_AMT)
            next_pos = np.asarray(C_WORKSPACE_CENTER)
            next_pos[2] += GRASP_HEIGHTS[j // 10 % len(GRASP_HEIGHTS)]
            rtde_c.moveL(next_pos.tolist(), speed=UR_MAX_SPEED, acceleration=0.01)
            time.sleep(SLEEP_AMT)
            T.close(CLOSE_AMT)
            time.sleep(SLEEP_AMT)
            next_pos[2] += ZLIFT
            rtde_c.moveL(next_pos.tolist(), speed=UR_MAX_SPEED, acceleration=0.01)
            time.sleep(SLEEP_AMT)
            # with pause_lock:
            #     pause.clear()
        else:
            # Randomly move within workspace, i.e., last_grasp_loc -+TRANS_RANG, -+ROT_RANGE
            rand_pos = next_pos.copy()
            rand_pos[0] += np.random.uniform(-XTRANS_RANGE, XTRANS_RANGE)
            rand_pos[2] += np.random.uniform(-ZTRANS_RANGE, 0)
            rand_pos[4] += np.random.uniform(-ROT_RANGE, ROT_RANGE)
            rtde_c.moveL(rand_pos.tolist(), speed=UR_MAX_SPEED, acceleration=0.01)
        T.close(CLOSE_AMT)
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
