from openhand_node.hands import Model_O, Model_T42
import time, math
import numpy as np
from klampt.math import vectorops as vo
from klampt.math import so3, se3
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle, os
from icecream import ic

def play_audible_alert():
    """Play sound to alert user when needed
    """
    duration = 0.5  # seconds
    freq = 550  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

class T_42_controller:
    def __init__(self, finger_offsets, port = '/dev/ttyUSB0', data_collection_mode = False):
        self.T = Model_T42(port=port, s1=1, s2=2, dyn_model='XM', s1_min=0, s2_min=0.5)
        self.data_collection_mode = data_collection_mode
        self.finger_offsets = finger_offsets

    def release(self):
        # print(0.06 + self.finger_offsets[0], 0.06 + self.finger_offsets[1])
        self.T.moveMotor(0, 0.06 + self.finger_offsets[0]) #right finger, viewing from camera 
        self.T.moveMotor(1, 0.06 + self.finger_offsets[1]) #left finger, viewing from camera 

        time.sleep(1)

    def close(self):
        if self.data_collection_mode:
            if np.random.random()>0.5:
                self.T.moveMotor(0, self.finger_offsets[0] - 0.055) # 0.29
                self.T.moveMotor(1, self.finger_offsets[1] - 0.055) # 0.56
            else:
                self.T.moveMotor(1, self.finger_offsets[1] - 0.055) 
                self.T.moveMotor(0, self.finger_offsets[0] - 0.055) 
        else:
            self.T.moveMotor(0, self.finger_offsets[0] - 0.055) # 0.29
            self.T.moveMotor(1, self.finger_offsets[1] - 0.055) # 0.56
        time.sleep(1)

    def move_to_zero_positions(self):
        self.T.moveMotor(0, self.finger_offsets[0]) 
        self.T.moveMotor(1, self.finger_offsets[1])


def rotate_EE(controller, alpha):
    rotation_arm = [0, 0, 0.15] # check this 
    current_T = controller.get_EE_transform()
    angle = alpha
    if math.fabs(angle) > 46/180*math.pi:
        print('Angle too big.. aborting ')
        return 
    delta_R = so3.from_rotation_vector([angle,0,0])
    target_R = so3.mul(delta_R, current_T[0])
    set_EE_transform_linear(controller, (target_R, current_T[1]), 0.01)
    time.sleep(1)
    return 

def calculate_rotate_R(current_R, delta_alpha):
    angle = delta_alpha
    delta_R = so3.from_rotation_vector([angle,0,0])
    target_R = so3.mul(delta_R, current_R)
    return target_R

def set_joint_config_linear(robot, target_q, max_v = 0.2):
    current_q = robot.get_joint_config()
    max_distance = np.max(np.abs(np.array(target_q) - np.array(current_q)))
    total_time = float(max_distance)/max_v
    start_time = time.time()
    dt = 0.01
    while time.time() - start_time <= total_time:
        elapsed_time = time.time() - start_time
        robot.set_joint_config(vo.interpolate(current_q, target_q, elapsed_time/total_time))
        time.sleep(dt)
    robot.set_joint_config(target_q)

def set_EE_transform_linear(robot, target_T, max_trans_v = 0.1, max_rotation_w = 0.05, max_time = math.inf):
    current_EE = robot.get_EE_transform()
    distance = np.linalg.norm(np.array(target_T[1]) - np.array(current_EE[1]))
    rotation_distance = math.fabs(so3.distance(target_T[0], current_EE[0]))
    total_time = max(float(distance)/max_trans_v, float(rotation_distance)/max_rotation_w)
    start_time = time.time()
    dt = 0.01
    while time.time() - start_time <= total_time:
        elapsed_time = time.time() - start_time
        robot.set_EE_transform(se3.interpolate(current_EE, target_T, elapsed_time/total_time))
        time.sleep(dt)
        if time.time() - start_time > max_time:
            break
    robot.set_EE_transform(target_T)

def set_EE_transform_trap(robot, target_T, max_trans_v = 0.1, max_a = 0.1, max_time = 100):
    ## assume orientation remain the same 
    current_EE = robot.get_EE_transform()
    dt = 0.01 
    milestones = []
    timestamps = []
    #Determine if the distance will reach max velocity
    distance = float(np.linalg.norm(np.array(target_T[1]) - np.array(current_EE[1])))
    displace_vector = vo.unit(vo.sub(target_T[1], current_EE[1]))
    init_trans = current_EE[1]

    #under linear acceleration
    time_to_reach_max_v = max_trans_v/max_a
    distance_ramping = time_to_reach_max_v*max_trans_v/2
    if distance_ramping*2 >= distance:
        mid_point_time = math.sqrt(distance/2/max_a)
        current_time = 0
        while current_time < mid_point_time*2:
            timestamps.append(current_time)
            if current_time < mid_point_time:
                dist = current_time*current_time*max_a/2
            else:
                dist = distance - (mid_point_time*2 - current_time)**2*max_a/2
            T = (target_T[0], vo.add(init_trans,vo.mul(displace_vector, dist)))
            milestones.append(T)
            current_time += dt
        timestamps.append(mid_point_time*2)
        milestones.append(target_T)
    else:
        max_v_time = (distance - distance_ramping*2)/max_trans_v
        total_time = 2*time_to_reach_max_v + max_v_time
        current_time = 0
        while current_time < total_time:
            timestamps.append(current_time)
            if current_time <= time_to_reach_max_v:
                dist = current_time*current_time*max_a/2
            elif current_time > time_to_reach_max_v and current_time <= time_to_reach_max_v + max_v_time:
                dist = distance_ramping + (current_time - time_to_reach_max_v)*max_trans_v
            else:

                c = total_time - current_time
                dist = distance - c**2*max_a/2
            T = (target_T[0], vo.add(init_trans,vo.mul(displace_vector, dist)))
            milestones.append(T)
            current_time += dt

        timestamps.append(total_time)
        milestones.append(target_T)
    # for T in milestones:
    #     print(T[1])
    for (T, t) in zip(milestones, timestamps):
        if t > max_time:
            break
        robot.set_EE_transform(T)
        time.sleep(dt)


class gravityCompensator:
    def __init__(self, Fy_gpr_path = './Fy_gpr.pkl', Fz_gpr_path = './Fz_gpr.pkl') -> None:
        
        with open(Fy_gpr_path, 'rb') as f:
            self.Fy_gpr = pickle.load(f)
        with open(Fz_gpr_path, 'rb') as f:
            self.Fz_gpr = pickle.load(f)

    def get_compensated_force(self, ft, controller):
        T_EE = controller.get_EE_transform()
        Rx = T_EE[0][0:3]
        # ic(Rx)
        angle = math.atan2(Rx[1], -Rx[2])
        # ic(angle)
        Fy_offset = self.Fy_gpr.predict(np.array([[angle]]))[0]
        Fz_offset = self.Fz_gpr.predict(np.array([[angle]]))[0]
        # ic(Fy_offset, Fz_offset)
        # ic(ft[1:3])
        # ic(ft)
        return np.array([ft[0], ft[1] - Fy_offset, ft[2] - Fz_offset] + list(ft[3:6]))

