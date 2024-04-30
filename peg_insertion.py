import os
import sys
import time
import copy
from threading import Thread

from openhand_node.hands import Model_O, Model_T42
import rtde_control
import rtde_receive
import torch
import numpy as np
import collections
from tqdm import tqdm
from real_world_data.ur_collect_data import *
from real_world_data.ftsensors import FT_reading, FTSensor
from real_world_data.realsense import RGBDCamera
from utils.parameters import parse_args
from utils.logger import Logger
from utils.dataset import Dataset, ImgForce, process_img
from model.vision_force_estimator import create_estimator

C_PEG_LOCATION = C_WORKSPACE_CENTER
C_X_NOISE = 0.05
C_Z_HEIGHT = 0.1
MAX_MOVEMENT = 0.03

# ToDo: 1 force control; 2 coordinates

class KControl:
    def __init__(self, ur_c, ur_r, ft):
        self.goal_fxfz = np.asarray([0, 0])
        self.kp = 0.001
        self.ur_c, self.ur_r = ur_c, ur_r
        self.ft = ft
        self.force_tolerance = 0.2

    def set_fxfz_goal(self, fxfz):
        # notice that fx goal is absolute, while fz goal has +- sign
        self.goal_fxfz = fxfz

    def f_error(self):
        # when there is no X force, move towards +X
        if self.ft.FT.measurement[0] < self.force_tolerance:
            fx_error = 1
        else:
            fx_error = min(self.goal_fxfz[0] - self.ft.FT.measurement[0],
                           -self.goal_fxfz[0] - self.ft.FT.measurement[0])
        fz_error = self.goal_fxfz[1] - self.ft.FT.measurement[1]
        return np.asarray(fx_error, fz_error)

    def move(self):
        while np.linalg.norm(self.f_error()) > self.force_tolerance:
            current_pos = self.ur_r.getActualTCPPose()
            assert current_pos[3:] == [0, 0, 0]
            dxz = self.f_error() * self.kp
            target_pos = current_pos
            target_pos[0, 2] += dxz
            # safety checking weather target_pos is too far from peg location
            assert np.linalg.norm(target_pos - np.asarray(C_PEG_LOCATION)) < 0.05
            self.ur_c.moveL(target_pos, speed=UR_MAX_SPEED, acceleration=0.01)
        print("force goal reached, force error: ", self.f_error())


class VFEstimator:
    def __init__(self, nn, cam, n_history, history_interval):
        self.nn = nn
        self.cam = cam
        self.FT = FT_reading()
        self.stream = False
        self.history_interval = history_interval
        self.h = torch.zeros([n_history * history_interval, self.nn.h * 8])

    def startStreaming(self):
        self.stream = True
        self.thread = Thread(target=self.receiveHandler)
        self.thread.daemon = True
        self.thread.start()

        print('VFEstimator started')

    def stopStreaming(self):
        self.stream = False
        time.sleep(0.1)

    def receiveHandler(self):
        while self.stream:
            color_image, depth_image, time_stamp = self.cam.get_rgb_frames(plot=True)
            img = process_img(color_image)
            img = torch.tensor(img, dtype=torch.float)[:3].unsqueeze(0)
            h = self.nn.forward_cnn(img)
            # self.h is a IFIO stack of CNN latent features
            self.h[-1:], self.h[:-1] = h, self.h[1:].clone()
            with torch.no_grad():
                f, _ = self.nn.forward_tf(self.h[::self.history_interval])
            self.FT = FT_reading(f.tolist(), time_stamp)


def insert_peg():
    args, hyper_parameters = parse_args()

    # setup camera
    rgbd_cam = RGBDCamera()
    rgbd_cam.start_streaming()

    # setup estimator
    nn = create_estimator(args)
    nn.loadModel(args.load_model)
    nn.network.eval()
    vfe = VFEstimator(nn.network, rgbd_cam, args.n_history, args.history_interval)
    vfe.startStreaming()

    # Loop and print out the timing between different readings.
    for _ in range(10000):
        while True:
            v, t = vfe.FT.measurement, vfe.FT.timestamp
            dt = time.time() - t

            print("value: ", np.round(v, 1))
            print('sensing t: ', np.round(dt, 3))
            time.sleep(0.2)
    vfe.stopStreaming()

    # setup ft sensor if needed
    ft_sensor = FTSensor()
    ft_sensor.startStreaming()

    # UR10e initialization
    ip = "192.168.1.100"  # IP for the UR10e robot
    rtde_c = rtde_control.RTDEControlInterface(ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    print("Robot is initialized.")

    # initialize controller
    controller = KControl(rtde_c, rtde_r, ft_sensor)

    # YOH model T42 initialization
    T = Model_T42(port='/dev/ttyUSB0', s1=10, s2=11, dyn_model='XM', s1_min=-0.25, s2_min=-0.44, motorDir=[-1, -1])

    # insert peg
    # grasp peg
    while True:
        ans = input('Robot is going to move, are you ready? (Y/N)')
        if ans in ['Y', 'y']:
            break
        time.sleep(1)
    q = rtde_r.getActualQ()
    err1 = np.linalg.norm(np.asarray(J_ABOVE_WORKSPACE) - np.asarray(q))
    err2 = np.linalg.norm(np.asarray(J_MID_TO_HOME) - np.asarray(q))
    if err1 > 0.3 and err2 > 0.3:
        raise NotImplementedError('Robot is far from known pose.')
    rtde_c.moveJ(J_ABOVE_WORKSPACE)
    rtde_c.moveL(C_PEG_LOCATION, speed=UR_MAX_SPEED)
    T.close(CLOSE_AMT)

    # move to random location along X-
    lift = C_PEG_LOCATION
    lift[2] += C_Z_HEIGHT
    rtde_c.moveL(lift, speed=UR_MAX_SPEED)
    random_x = lift
    random_x[0] += np.random.uniform(-C_X_NOISE, C_X_NOISE)
    rtde_c.moveL(random_x, speed=UR_MAX_SPEED)

    # goal: fz = 1N
    controller.set_fxfz_goal(np.asarray([0, 1]))
    controller.move()

    # keep: fz = 1N
    # goal: fx = 1N
    controller.set_fxfz_goal(np.asarray([1, 1]))
    controller.move()

    # insert
    # goal: fz = 2N
    controller.set_fxfz_goal(np.asarray([0, 2]))
    controller.move()
    T.release()
    time.sleep(1)

    # move above workspace
    current_pos = rtde_r.getActualTCPPose()
    current_pos[2] += C_Z_HEIGHT
    rtde_c.moveL(current_pos, speed=UR_MAX_SPEED)


if __name__ == '__main__':
    insert_peg()
