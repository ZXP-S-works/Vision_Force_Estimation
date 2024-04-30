from xmlrpc.client import ServerProxy
from threading import Thread, Lock
import threading
import time
import os
import numpy as np
from klampt.math import so3, se3, vectorops as vo

dirname = os.path.dirname(__file__)

class FrankaClient:
    def __init__(self, address = 'http://127.0.0.1:8080'):
        self.s = ServerProxy(address)
        #self.shut_down = False

    def initialize(self):
        self.s.initialize()

    def start(self):
        self.s.start()

    def shutdown(self):
        self.s.shutdown()

    def get_joint_config(self):
        return self.s.get_joint_config()

    def get_joint_velocity(self):
        return self.s.get_joint_velocity()

    def get_joint_torques(self):
        return self.s.get_joint_torques()

    def get_EE_transform(self, tool_center = se3.identity()):
        return self.s.get_EE_transform(tool_center)

    def get_EE_velocity(self):
        #print("Requesting the EE velocity")
        return self.s.get_EE_velocity()

    def get_EE_wrench(self):
        return self.s.get_EE_wrench()

    def set_joint_config(self, q):
        self.s.set_joint_config(q)

    def set_EE_transform(self, T):
        self.s.set_EE_transform(T)

    def set_EE_velocity(self, v):
        self.s.set_EE_velocity(v)

if __name__=="__main__":
    import json, math
    from franka_collect_data import set_EE_transform_linear, set_joint_config_linear
    ft_arm_driver = FrankaClient('http://172.16.0.1:8080')
    ft_arm_driver.initialize()
    ft_arm_driver.start()
    time.sleep(1)

    # R_EE_World = [math.sqrt(2)/2,
    #              math.sqrt(2)/2,
    #              0,
    #              math.sqrt(2)/2,
    #              -math.sqrt(2)/2,
    #              0,
    #              0.0,
    #              0.0,
    #              -1]

    # R,t = ft_arm_driver.get_EE_transform()
    # print(R)
    # R_gripper = so3.mul(R, so3.inv(R_EE_World))
    # heading = math.atan2(R_gripper[1], R_gripper[0])/math.pi*180
    # print(R_gripper, heading)

    start_position_file = "../start_position_0.json"
    set_joint_config_linear(ft_arm_driver, json.load(open(start_position_file, "r")))

    time.sleep(1)
    print(ft_arm_driver.get_joint_config())
    print(ft_arm_driver.get_EE_transform())

    target_EE_T = [[math.sqrt(2)/2,
                 -math.sqrt(2)/2,
                 0,
                 -math.sqrt(2)/2,
                 -math.sqrt(2)/2,
                 0,
                 0.0,
                 0.0,
                 -1],
                [0.39795164623192886, -0.0034064403438991927, 0.4]]
    set_EE_transform_linear(ft_arm_driver, target_EE_T, 0.01)
    time.sleep(1)
    print(ft_arm_driver.get_EE_transform())
    print(ft_arm_driver.get_joint_config())

    # target_EE_T = [[math.sqrt(2)/2,
    #              -math.sqrt(2)/2,
    #              0,
    #              -math.sqrt(2)/2,
    #              -math.sqrt(2)/2,
    #              0,
    #              0.0,
    #              0.0,
    #              -1],
    #             [0.3, 0.1, 0.35]]
    
    # set_EE_transform_linear(ft_arm_driver, target_EE_T, 0.01)
    # time.sleep(1)
    # print(ft_arm_driver.get_joint_config())


    # target_EE_T = [[0.6645038933047369,
    #              -0.7472355826620318,
    #              -0.008576700215157426,
    #              -0.7472053266578609,
    #              -0.6642220758584929,
    #              -0.02220886661569875,
    #              0.0,
    #              0.0,
    #              -1],
    #             [0.3, 0.1, 0.29]]

    # set_EE_transform_linear(ft_arm_driver, target_EE_T, 0.01)
    # time.sleep(1)
    # print(ft_arm_driver.get_joint_config())

    #ft_arm_driver.shutdown()

