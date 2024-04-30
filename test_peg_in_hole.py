# from franka_python_controller import FrankaController
# from franka_python_controller.motionUtils import GlobalCollisionHelper
from real_world_data.FT_client import FTClient
from real_world_data.Franka_client import FrankaClient
from klampt import RobotModel, vis
from klampt import WorldModel,RobotModel
from klampt.model import ik,collide
from klampt.math import so3, se3, vectorops as vo
from openhand_node.hands import Model_T42
# from icecream import ic
import copy, math, time
import json
import pickle
from pprint import pprint
import numpy as np
from scipy.spatial import KDTree
# import PyKDL
import random
from termcolor import colored
import colorama
colorama.init()
# from real_world_data.Franka_client import FrankaClient
import sys
from online_inference import VFEstimator
from real_world_data.webcam import WebCamera
from model.vision_force_estimator import create_estimator
from utils.parameters import parse_args
from utils.dataset import Dataset, ImgForce
import cv2, os
import threading
import simple_pid

class PegInHoleTask():

    ## TODO:
    ## - Detect changes in contact formation: monitoring forces AND torque direction changes?
    ## - Trigger full contact analysis when change is detected.
    ## - Record pose when performing analysis to allow navigating from one point to another later
    ## Implement direction switching within an exploration plan linked to the evolution of the available DOFs

    def __init__(self, controller, start_position_file, dataset_dir, task_name, task_args, mode = 'collection'):
        print("Initializing PegInHole...")
        ## Contact regulation setting
        self.maintain_contact = True # Activate or deactivate the contact regulation task entirely
        self.maintain_override = True # If True, then loosing contact will stop main task until contact is re-established
        self.standard_gravity_target =(None, None, -1) #-0.5
        self.current_contact_target = self.standard_gravity_target # Set the contact target for the contact regulation task
        self.base_current_contact_target = list(self.standard_gravity_target)
        self.base_z_target = self.base_current_contact_target[2]
        self.wall_current_contact_target = list(self.standard_gravity_target)
        ## Task setting: This task will execute secondary to the Contact regulation task
        # self.current_task = {"name":"still", "args":{}}
        # self.current_task = {"name":"slide", "args":{"slide_axis":"y"}}
        # self.current_task = {"name":"slide_and_climb", "args":{"slide_axis":"-x"}}
        # self.current_task = {"name":"insert_object_2D", "args":{}}
        # self.current_task = {"name":"insert2D", "args":{}} # ONGOING
        # self.current_task = {"name":"wipe", "args":{"is_randomizing_force":False}}
        # self.current_task = {"name":"touch_walls", "args":{"is_randomizing_force":True, "mode": "contact", "is_stick_to_wall": True}}
        self.current_task = {"name": task_name, "args": task_args}

        ## General settings
        global_gain = 0.5 # 0.75 in data collection, 0.5 in execution
        control_mode = "pid"
        task_timer = 2*60

        # Create pid controller
        if control_mode == "pid":
            pid_scale = 0.4
            pid_P = -0.1*pid_scale # -0.03 during data collection 0.07
            pid_I = 0 #-0.005*pid_scale
            pid_D = -0.005*pid_scale
            self.pid_z = simple_pid.PID(pid_P, pid_I, pid_D, setpoint=None)
            self.pid_z.sample_time = 0.1  # Update every 0.1 seconds
            self.pid_y = simple_pid.PID(pid_P, pid_I, pid_D, setpoint=None)
            self.pid_y.sample_time = 0.1  # Update every 0.1 seconds

        self.no_delta = True
        # self.no_delta = False

        # self.lock = True
        self.lock = False

        self.controller = controller
        ## Always keep to TRUE: none of the recent tasks have been tested with the world space.
        # self.project_all_in_effector_space = False
        self.project_all_in_effector_space = True

        #self.debug = False
        self.debug = True
        self.force_source = []
        self.force_source.append('ATI') # between ATI, franka, and Vision Force Estimator (VFE)
        # self.force_source = 'franka' # between ATI, franka, and vision
        self.force_source.append('VFE') # Get the force estim   ate from the Vision Force Estimation model

        self.record_forces = True
        self.record_CF = False
        self.record_frequency = 10 #20
        self.record_timer = time.time()
        self.record_dataset_size = 200
        self.record_num_img = 0
        self.recorded_forces = []
        self.recorded_forces_ground_truth = []
        self.record_every = 100
        # self.record_dataset_dir = "/home/grablab/VisionForceEstimator/real_world_data/task_data/task_dataset_" + str(time.time())
        counter = 1
        folder_name = dataset_dir + task_name  # "/home/grablab/VisionForceEstimator/real_world_data/1026_test_data/40_"
        self.record_dataset_dir = folder_name + str(counter)
        while os.path.exists(self.record_dataset_dir):
            self.record_dataset_dir = folder_name + str(counter)
            counter += 1

        self.record_dataset_force_dir = os.path.join(self.record_dataset_dir, 'forces')
        self.record_dataset_img_dir = os.path.join(self.record_dataset_dir, 'images')

        counter = 0
        test_record_file_folder = f"/home/grablab/VisionForceEstimator/real_world_data/experiment_data_{counter}/"
        while os.path.exists(test_record_file_folder):
            counter += 1
            test_record_file_folder = f"/home/grablab/VisionForceEstimator/real_world_data/experiment_data_{counter}/"
        os.makedirs(test_record_file_folder)
        self.record_file = test_record_file_folder + 'vision_recorded_forces.pkl'
        self.record_file_ground_truth = test_record_file_folder + "vision_ground_truth.pkl"
        self.record_file_CF = "/data_experiments/forces/vels_FTs.pkl"
        self.recorded_delta_z = []

        os.makedirs(self.record_dataset_dir, exist_ok=True)
        os.makedirs(self.record_dataset_force_dir, exist_ok=True)
        os.makedirs(self.record_dataset_img_dir, exist_ok=True)
        print("Folders created.")
        # Determines how to compute the initial offset (assuming a free-floating start position)
        self.averages = [0, 0, 0, 0, 0, 0]
        self.averages_GT = [0, 0, 0, 0, 0, 0]
        self.nb_samples = 100
        self.last_forces = []
        self.last_forces_GT = []
        self.delta_from_floating = [0, 0, 0, 0, 0, 0]

        print("Moving to initial position...")
        set_joint_config_linear(controller, json.load(open(start_position_file, "r")))
        #controller.set_joint_config()
        time.sleep(2)
        print("Done")

        # create hand control, first release then close
        print("Conneting to Model_T42 hand...")
        T = Model_T42(port='/dev/ttyUSB0', s1=1, s2=2, dyn_model='XM', s1_min=0.35, s2_min=0)
        time.sleep(1)
        print("Model T42 is initialized")
        # T.release()
        T.moveMotor(0, 0)
        T.moveMotor(1, 0)
        print('Release done')
        time.sleep(2)

        if 'ATI' in self.force_source:
            print("Connecting to FT sensor...")
            self.ft_driver = FTClient('http://172.16.0.64:8080')
            self.ft_driver.zero_ft_sensor()
            self.ft_driver.start_ft_sensor()
            for i in range(20):
                start_time = time.time()
                print(self.ft_driver.read_ft_sensor())
                time.sleep(0.1)
            print("Done")

        # up_timer = 3
        # start_time_up = time.time()
        # while time.time() - start_time_up < up_timer:
        #     self.controller.set_EE_velocity([0, 0, 0.009, 0, 0, 0])

        T.moveMotor(0, 0.29)
        T.moveMotor(1, 0.37)

        time.sleep(2)
        print("Done")

        print("Arm initialized and positioned")

        self.cam = WebCamera()
        if 'VFE' in self.force_source:
            print("Starting VFE model...")
            # Starting the Vision Force Estimator
            args, hyper_parameters = parse_args()

            # Model to run for force predictions
            model_name = 'model7'

            if model_name == 'model1':
                model_path = '/home/grablab/Downloads/1002_history_10_best_val.pt'

                n_history = 10
                history_interval = 1

            elif model_name == 'model2':
                model_path = '/home/grablab/Downloads/h_15_interval_4_best_val.pt'

                n_history = 15
                history_interval = 4
            elif model_name == 'model3':
                model_path = '/home/grablab/Downloads/h_8_interval_4_best_val.pt'

                n_history = 8
                history_interval = 4
                Hz = 20
            elif model_name == 'model4':
                model_path = '/home/grablab/Downloads/h_20_int_1_Hz10.pt'

                n_history = 20
                history_interval = 1
                Hz = 10
            elif model_name == 'model5':
                model_path = '/home/grablab/Downloads/1008_h_20_int_1_Hz10.pt'

                n_history = 20
                history_interval = 1
                Hz = 10
            elif model_name == 'model6':
                model_path = '/home/grablab/Downloads/1009_h_20_int_1_Hz10.pt'

                n_history = 20
                history_interval = 1
                Hz = 10
            elif model_name == 'model7':
                model_path = '/home/grablab/Downloads/1025_h20_int1_10Hz.pt'

                n_history = 20
                history_interval = 1
                Hz = 10
            else:
                raise Exception('Unknown model name: ' + model_name)

            args.n_history = n_history
            args.history_interval = history_interval
            args.env_type = 'real_world_yz'
            nn = create_estimator(args)
            nn.loadModel(model_path)
            nn.network.eval()
            self.vfe = VFEstimator(nn=nn.network, cam=self.cam, n_history=n_history, history_interval=history_interval, Hz = Hz)
            self.vfe.startStreaming()

            while self.vfe.getForce() is None:
                #print("Filling VFE memory, waiting a force estimate...")
                pass
            print("Done, received a force estimate")
            print("VFE model started.")

        self.force_data = None
        self.force_data_GT = None

        self.rate = 20 #40 during data collection
        self.t = 0
        self.last_contact_time = 99
        self.move = True
        # self.move = False

        # Not sure why?
        #time.sleep(4)

        # Gathering the current force values, computing the average and considering that as initial offset
        if not self.no_delta:
            while(len(self.last_forces) < self.nb_samples):
                self.read_forces()
                time.sleep(0.05)

        for force in self.last_forces[:self.nb_samples-1]:
            for i in range(len(self.averages)):
                self.averages[i] += force[i]
        for i in range(len(self.averages)):
            self.averages[i] /= self.nb_samples

        for force in self.last_forces_GT[:self.nb_samples-1]:
            for i in range(len(self.averages_GT)):
                self.averages_GT[i] += force[i]
        for i in range(len(self.averages_GT)):
            self.averages_GT[i] /= self.nb_samples

        print("Found initial floating z:", self.averages)
        print("Found initial floating z GT:", self.averages_GT)

        self.read_forces()
        print("Forces after calibration:")
        pprint(self.delta_from_floating)

        # Initializing the task "memory" variables
        self.task_state = None
        self.task_timer = time.time()
        self.task_vars = {"i":0}
        self.task_finished = False

        self.tasks_data = {}
        self.contact_formation_data = []
        self.current_vels = [0, 0, 0, 0, 0, 0]
        self.current_delta_forces = [0, 0, 0, 0, 0, 0]

        task_start_time = time.time()

        # Starting main loop
        while True:
            loop_start_time = time.time()
            self.read_forces()
            print("\n"*100, "*"*10)
            print("Task completion:", int(time.time()-task_start_time), "/", task_timer)
            #print("Delta from floating:")
            #pprint(self.delta_from_floating) # Current forces with initial offset subtracted

            speed_cap = 0.0025*global_gain*2
            speed_cap_z = 0.1*global_gain # data collectoin 0.04*global_gain
            speed_cap_rot = 0.001*0.4
            speed_cap_rot_z = 0.001*0.4*3
            gains = [speed_cap, speed_cap, speed_cap_z, speed_cap_rot, speed_cap_rot, speed_cap_rot_z]
            target_motion_vels = [0, 0, 0, 0, 0, 0]

            ## Gathering the velocities from the regulation and main task
            # Collect the velocities from the regulation task, if applicable
            if self.maintain_contact:
                contact_vels = self.maintain_side_contact_pid(target_forces=self.current_contact_target)
            else:
                contact_vels = [None, None, None]

            # Collect the velocities from the main task
            target_motion_vels = self.get_target_motion(self.current_task["name"], self.current_task["args"], gains=gains)

            # If 'override' is active, erase the main task velocities.
            if not self.maintain_contact or not self.maintain_override or (self.maintain_override and self.last_contact_time < 1):
                pass
            else:
                target_motion_vels = [0, 0, 0, 0, 0, 0] # Cancelling task velocities
                print("OVERRIDE - Seeking contact!")
                # Skipping task until contact has been made.

            ## Assembling velocites : every 'not None' velocity component from the regulation overwrites the associated component from the main task
            # Consequence: regulation command has priority over the main task.
            vels = copy.copy(target_motion_vels)
            if self.maintain_contact:
                for i, contact_vel in enumerate(contact_vels):
                    if self.current_contact_target[i] is not None and contact_vel is not None:
                        vels[i] = contact_vel

            ## Post-processing the velocities
            # Scaling the velocities to ensure no component is higher than its limit (if so: the entire vector is reduced iteratively until all limits are enforced)
            final_vels = self.scale_vels(vels, cap=speed_cap, cap_z=speed_cap_z, cap_rot=speed_cap_rot, cap_rot_z=speed_cap_rot_z)

            self.current_vels = copy.copy(final_vels)
            # Projecting the velocities from world frame to effector frame
            # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
            (t, R) = self.get_pos()

            self.contact_formation_data.append((self.current_vels, self.delta_from_floating, time.time(), (t, R)))
            #print("Velocities:")
            #pprint(["{:.2f}".format(x) for x in self.current_vels])
            #print("Forces:")
            #pprint([round(x, 2) for x in self.force_data])
            print("Force Readings:")
            #pprint(["{:.2f}".format(x) for x in self.delta_from_floating])
            print("X :", "{:.2f}".format(self.delta_from_floating[1]), " - ", "Z :", "{:.2f}".format(self.delta_from_floating[2]))
            print("Error forces:")
            error_thr = 0.2
            #print([colored(str(round(gt_x - x, 2)), "green") if abs(gt_x - x) < error_thr else colored(str(round(gt_x - x, 2)), "red") for x, gt_x in zip(self.force_data_GT, self.force_data)])
            #pprint(["{:.2f}".format(gt_x - x) for x, gt_x in zip(self.force_data_GT, self.force_data)])
            print("X :", "{:.2f}".format(self.force_data_GT[1]-self.force_data[1]), " - ", "Z :", "{:.2f}".format(self.force_data_GT[2]-self.force_data[2]))
            if self.record_CF:
                if len(self.contact_formation_data) % self.record_every == 0:
                    pickle.dump(self.contact_formation_data, open(self.record_file_CF, "wb"))
            if self.project_all_in_effector_space:
                final_vels = self.project_velocities_to_effector_space(final_vels)

            # If a force is already high in a given direction, prevent velocities that will increase it further
            final_vels = self.cancel_velocities_already_in_opposition(final_vels)


            if self.debug:
                print("-"*10)
                # print("Forces:", self.force_data - [self.average_x, self.average_y, self.average_z])
                print("contact speed:", contact_vels)
                print("task_speeds:", target_motion_vels)
                print("Final vels:", final_vels)

            # Emergency stop: if a force is over a max threshold, stop the program!
            # If ok, publishing the velocities to the controller
            max_forces = 8
            if self.move and self.are_forces_safe(self.force_data_GT, max_forces=max_forces):
                # self.publishJointVelocity_jog(final_vels)
                if not self.lock:
                    self.controller.set_EE_velocity(final_vels)
                else:
                    self.controller.set_EE_velocity([0, 0, 0, 0, 0, 0])
            else:
                # if not self.are_forces_safe(self.force_data, max_forces=max_forces):
                if not self.are_forces_safe(self.force_data_GT, max_forces=max_forces):
                    print("Forces are unsafe, stopping motion!")
                    self.controller.set_EE_velocity([0, 0, 0, 0, 0, 0])
                    # return
            # self.r.sleep()
            elapsed_time = time.time() - loop_start_time
            if elapsed_time < 1/self.rate:
                time.sleep(1/self.rate - elapsed_time)
            else:
                time.sleep(0.000001)

            if time.time() - task_start_time > task_timer:
                break

        # Coming back to initial position
        set_joint_config_linear(controller, json.load(open(start_position_file, "r")))
        #controller.set_joint_config()
        time.sleep(2)
        # T.moveMotor(0, 0)
        # T.moveMotor(1, 0)
        # print("I'm done with this!")

    def reset_contact_estimation(self):
        print("Resetting history data for CF analysis")
        self.contact_formation_data = []


    def detect_resistance(self, ignore=[False, False, False, False, False, False]):
        """
        Check all current forces and flag those that are high enough as a sign of mechanical constraint
        """
        threshold_F = 0.5
        threshold_T = 0.05
        thresholds = [
                    threshold_F,
                    threshold_F,
                    threshold_F,
                    threshold_T,
                    threshold_T,
                    threshold_T
                    ]
        found_resistance = [[False, False], [False, False], [False, False], [False, False], [False, False], [False, False]]
        # print(found_resistance)
        for i in range(len(self.delta_from_floating)):
            if not ignore[i]:
                if self.delta_from_floating[i] < -thresholds[i]:
                    found_resistance[i][0] = True
                if self.delta_from_floating[i] > thresholds[i]:
                    found_resistance[i][1] = True
        # print("resistance result:", found_resistance)
        return found_resistance

    def _getPos(self):
        return self.arm_group.get_current_joint_values()

    def _movePos(self, pos, blocking=True):
        self._commandArmJointPos(pos, blocking=blocking)

    def _startPos(self, blocking=False , idx=0):
        if idx==0:
            self.arm_start_config = [0.1,0.6,0.015,1.837,0.039,0.58,1.55]

        elif idx==1:
            self.arm_start_config = [
                                        0.43159645167725286,
                                        0.7899865307098849,
                                        -0.4105545797555427,
                                        1.894466273038767,
                                        -0.24140429615540532,
                                        0.5600611299275257,
                                        1.8786897994969158
                                    ]
        elif idx == 2:
            self.arm_start_config = [0.10022007814186187,
                                    0.7022916998272191,
                                    0.014732710727812497,
                                    1.981050966399423,
                                    0.03811230617324119,
                                    0.3909278873869387,
                                    1.5471321302064291]


        elif idx == 3: #Andy's favorite position
            self.arm_start_config = [0.10661166475805205,
                                     0.7012599959344819,
                                     0.022030647930444076,
                                     1.8573098495099813,
                                     0.059619665673493485,
                                     0.5455911049696354,
                                     1.5538105462287042]

        elif idx == 4: #Andy's new favorite position
            self.arm_start_config = [0.09780953690375586,
                                     0.5648850313760823,
                                     0.02298850218828947,
                                     1.7392785499976695,
                                     0.04633570833510236,
                                     0.7954560168108017,
                                     1.563571308107414]


        else: # Default position
            self.arm_start_config = [0.1,0.6,0.015,1.837,0.039,0.58,1.55]
        self._commandArmJointPos(self.arm_start_config, blocking=blocking)

    def _commandArmJointPos (self, pos, blocking=False):
        self.arm_group.set_joint_value_target(pos)
        plan = self._plan_execution()
        self.arm_group.execute(plan, blocking)

    def _plan_execution(self): #this keeps track of a timer
        tic = time.time()
        plan = self.arm_group.plan()
        d = time.time()-tic
        self.planning_time+=d
        self.planning_actions+=1
        return plan

    def reset_counters(self):
        self.planning_time = 0.
        self.planning_actions = 0
        self.start_time = time.time()
        self.hand_actions = 0

    # def _initializeArm(self):
    #     self.reset_counters()
    #     self.arm_group = MoveGroupCommander("arm")
    #     self._pos_tolerance = 0.0
    #     self._ortn_tolerance = 0.0
    #     self.planner_type = 'RRTConnectkConfigDefault'
    #     self.arm_group.set_planner_id(self.planner_type)

    def project_velocities_to_effector_space(self, vels):
        test = True

        if not test:
            # From world to effector
            (trans1, rot1)  = self.listener.lookupTransform('wam/base_link', 'wam/wrist_palm_stump_link', rospy.Time(0))
            rotation = Rotation.from_quat(rot1)
            rotation_debug = Rotation.from_euler("z",0, degrees=True )

            debug = False
            if debug:
                rotation = rotation_debug
            # print("Applying rotation:", rotation.as_euler("xyz", degrees=True))
            T = [vels[0], vels[1], vels[2]]
            R = [vels[3], vels[4], vels[5]]

            # Something is wrong, a 90deg rotation around the z axis seem to solve the issue. Should be looked into at some point, probably a dump mistake.
            rot_T = rotation_debug.apply(rotation.apply(T))
            rot_R = rotation_debug.apply(rotation.apply(R))

            # print("Before rotation:", vels)
            # print("After rotation:", rot_T, rot_R)

            return [rot_T[0], rot_T[1], rot_T[2], rot_R[0], rot_R[1], rot_R[2]]
        else:
            return [vels[0], vels[1], -vels[2], vels[3], vels[4], vels[5]]
    # def rotate_force_readings(self, forces):
    #     try:
    #         if not self.project_all_in_effector_space:
    #             # From sensor to world
    #             (trans1, rot1)  = self.listener.lookupTransform('wam/base_link', 'sensor_space', rospy.Time(0))
    #         else:
    #             # From sensor to effector
    #             rotation_debug = Rotation.from_euler("z",0, degrees=True )
    #             (trans1, rot1)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'sensor_space', rospy.Time(0))
    #         rotation = Rotation.from_quat(rot1)
    #         # print("Applying rotation:", rotation.as_euler("xyz", degrees=True))
    #         forces = self.force_data[0:3]
    #         torques = self.force_data[3:6]
    #         if self.project_all_in_effector_space:
    #             aligned_forces = rotation_debug.apply(rotation.apply(forces))
    #             aligned_torques = rotation_debug.apply(rotation.apply(torques))
    #         else:
    #             aligned_forces = rotation.apply(forces)
    #             aligned_torques = rotation.apply(torques)

    #         # print("Initial forces:", forces)
    #         # print("Rotated forces:", aligned_forces)
    #         # print("Rotated torques:", aligned_forces)
    #         # To avoid breaking the contact regulation code when switching back and forth from world to effector
    #         # aligned_forces[2]*=-1

    #         result = [
    #             aligned_forces[0],
    #             aligned_forces[1],
    #             aligned_forces[2],
    #             aligned_torques[0],
    #             aligned_torques[1],
    #             aligned_torques[2],
    #             ]

    #         return result
    #     except Exception as e:
    #         print("Rotation failed:", e)
    #         return forces

    def cancel_velocities_already_in_opposition(self, vels):
        forces_safety_threshold = 5
        for i, vel in enumerate(vels):
            if vel < 0 and self.delta_from_floating[i] < -forces_safety_threshold:
                print("Cancelling velocity:", -(i+1))
                vels[i] = 0
            if vel > 0 and self.delta_from_floating[i] > forces_safety_threshold:
                print("Cancelling velocity:", i+1)
                vels[i] = 0
        return vels

    def are_forces_safe(self, forces, max_forces=10, max_torques=1):
        # pprint(forces)
        return abs(forces[0]) < max_forces and abs(forces[1]) < max_forces and abs(forces[2]) < max_forces and abs(forces[3]) < max_torques and abs(forces[4]) < max_torques and abs(forces[5]) < max_torques

    def scale_vels(self, vels, cap, cap_z, cap_rot, cap_rot_z):
        final_vels = vels
        while(abs(final_vels[0]) > cap or abs(final_vels[1]) > cap):
            # print("X/Y Speeds too high, scaling down...")
            final_vels[0] *= 0.9
            final_vels[1] *= 0.9

        while(abs(final_vels[2])) > cap_z:
            # print("Z Speeds too high, scaling down...", final_vels[2], "/", cap_z)
            final_vels[2] *= 0.9

        while(abs(final_vels[3]) > cap_rot or abs(final_vels[4]) > cap_rot):
            # print("ROT X/Y Speeds too high, scaling down...")
            final_vels[3] *=0.9
            final_vels[4] *=0.9

        while abs(final_vels[5]) > cap_rot_z:
            # print("ROT Z Speeds too high, scaling down...")
            final_vels[5] *=0.9

        return final_vels

    def get_target_motion(self, task_type="still", args=None, gains=[0.001, 0.001, 0.001, 0, 0, 0]):
        """
        Collect the velocities associated to a given main task
        Velocities are [tx, ty, tz, rx, ry, rz] (for the lateral axis orientation, refer to the markings on the robot hand)
        """
        print("Executing task:", task_type)
        vels = [0, 0, 0, 0, 0, 0]
        if task_type == "still":
            vels = [0, 0, 0, 0, 0, 0]
        if task_type == "down":
            vels = [0, 0, 1, 0, 0, 0]
        if task_type == "up":
            vels = [0, 0, -1, 0, 0, 0]
        if task_type == "left":
            vels = [-1, 0, 0, 0, 0, 0]
        if task_type == "right":
            vels = [1, 0, 0, 0, 0, 0]
        if task_type == "forward":
            vels = [0, 1, 0, 0, 0, 0]
        if task_type == "backward":
            vels = [0, -1, 0, 0, 0, 0]
        if task_type == "circle":
            vels = self.draw_cirle(axis=args["axis"])
        if task_type == "orient_towards":
            # print("Orienting")
            vels = self.orient_towards()
        if task_type == "slide":
            vels = self.slide(axis=args["slide_axis"])
        if task_type == "slide_and_orient":
            if abs(self.force_data[0]-self.average_x) > 1 or abs(self.force_data[1]-self.average_y) > 1:
                vels = self.orient_towards()
            else:
                print("Sliding!")
                if args["slide_axis"] == "x":
                    vels = [1, 0, 0, 0, 0, 0]
                if args["slide_axis"] == "-x":
                    vels = [-1, 0, 0, 0, 0, 0]
                if args["slide_axis"] == "y":
                    vels = [0, 1, 0, 0, 0, 0]
                if args["slide_axis"] == "-y":
                    vels = [0, -1, 0, 0, 0, 0]
        if task_type == "slide_and_climb":
            vels = self.explore_laterally(axis=args["slide_axis"])
        if task_type == "test_contact":
            vels = self.test_contact(frozen_axis=args["frozen_axis"], initial_contact_target=args["initial_contact_target"])
        if task_type == "align_with_plane_below":
            vels = self.align_with_plane_below()
        if task_type == "slide_and_test":
            vels = self.slide_and_test(args["slide_axis"])
        if task_type == "direct_control":
            vels = self.direct_control()
        if task_type == "explore_hole":
            vels = self.explore_hole(args["start_direction"])
        if task_type == "rotate_against_edge":
            vels = self.rotate_against_edge()
        if task_type == "insert_object":
            vels = self.insert_object()
        if task_type == "insert_object_2D":
            vels = self.insert_object_2D()
        if task_type =="explore_hole_andy":
            vels = self.explore_hole_andy()
        if task_type == "wipe":
            vels = self.wipe(args["is_randomizing_force"])
        if task_type == "touch_walls":
            vels = self.touch_walls(args["is_randomizing_force"], args["mode"], args["is_stick_to_wall"])
        if task_type == "reposition_fingers":
            vels = self.reposition_fingers(args["target_location"], args["target_orientation"])
        if task_type == "random_move":
            vels = self.random_move()

        return self.apply_gains(vels, gains)

    def wipe(self, is_randomizing_force=False):
        """
        Moves arm back and forth. Changes direction based on position. (Mei)
        Arg(s):
            is_randomizing_force : bool
                Whether the contact force is randomized while switching directions
        Returns:
            vels : list[float]
                Velocity commands to send to the arm.
        """
        vels = [0., 0., 0., 0., 0., 0.]
        self.record_forces = True

        # Randomize contact force during wiping
        if is_randomizing_force:
            rand_force_range = [-0.3, -2.0] #[-0.3, -1.2]
            change_period = 10 # in seconds

        # Initizlize task
        if "is_started" not in self.task_vars:
            self.task_vars.update({"is_started":True})
            # Calculate workspace
            center_point = self.get_pos()[0][1]
            workspace_size = 0.1 # m

            self.task_vars.update({"boundary_positive":center_point+workspace_size/2, "boundary_negative":center_point-workspace_size/2})
            # Start task by moving down
            self.task_vars.update({"task":"down"})
            self.task_vars.update({"force_timer":time.time()})

        print("Wiping - " + self.task_vars["task"])

        # Task state machine
        if self.task_vars["task"] == "down":
            # Change conatct target to moving down
            self.current_contact_target = self.standard_gravity_target
            self.maintain_override = True
            # Transition from down to slide
            if self.last_contact_time < 0.1:
                self.task_vars["task"] = "slide_positive"

        elif self.task_vars["task"] == "slide_positive":
            # self.current_contact_target = self.standard_gravity_target # sets constant force target
            # print("contact target" + str(self.current_contact_target))
            self.maintain_override = False
            vels = [0., 1., 0., 0., 0., 0.]
            # Check if boundary is reached, if so, change direction
            if self.get_pos()[0][1] > self.task_vars["boundary_positive"]:
                self.task_vars["task"] = "slide_negative"

        elif self.task_vars["task"] == "slide_negative":
            # self.current_contact_target = self.standard_gravity_target
            # print("contact target" + str(self.current_contact_target))
            self.maintain_override = False
            vels = [0., -1., 0., 0., 0., 0.]
            # Check if boundary is reached, if so, change direction
            if self.get_pos()[0][1] < self.task_vars["boundary_negative"]:
                self.task_vars["task"] = "slide_positive"

        else:
            raise Exception("Unsupported subtask within back and forth: " + self.task_vars["task"])

        # Change contacyt force target based on time period
        if is_randomizing_force:
            if time.time() - self.task_vars["force_timer"] > change_period:
                self.base_current_contact_target  = [None, None, random.uniform(rand_force_range[0], rand_force_range[1])]
                #self.current_contact_target = [None, None, random.uniform(rand_force_range[0], rand_force_range[1])]
                self.task_vars["force_timer"] = time.time()
            self.current_contact_target = copy.deepcopy(self.base_current_contact_target)
            self.current_contact_target[2] += float(0.3*np.sin((time.time() - self.task_vars["force_timer"])/(random.uniform(0.5, 4)*np.pi)))

        return vels

    def touch_walls(self, is_randomizing_force=False, mode="contact", is_stick_to_wall=False):
        """
        Moves arm back and forth. Changes direction when hitting a wall. (Mei)
        Arg(s):
            is_randomizing_force : bool
                Whether the contact force is randomized while switching directions
            mode : str
                Mode for the touch walls. The mode is between "contact", and
                "float". "contact" touches the bottom while moving back and forth,
                "float" is the same as "contact" except that the peg is floating in the air.
            is_stick_to_wall : bool
                Whether the pegs maintains contact with the walls for a period of time
                before switching direction.
        Returns:
            vels : list[float]
                Velocity commands to send to the arm.
        """
        vels = [0., 0., 0., 0., 0., 0.]
        float_height = 0.3

        self.record_forces = True

        # Randomize contact force during wiping
        if is_randomizing_force:
            bottom_rand_force_range = [-0.3, -1.2]
            wall_rand_force_range = [0.2, 0.5]
            bottom_change_period = 15 # in seconds
            wall_change_period = 5 # in seconds

        if is_stick_to_wall:
            stick_period = 10 # in seconds

        # Initizlize task
        if "is_started" not in self.task_vars:
            self.task_vars.update({"is_started":True})
            # Setup initial wall contact forces
            initial_wall_contact_force = 0.5
            self.task_vars.update({"boundary_positive":initial_wall_contact_force, "boundary_negative":-initial_wall_contact_force})
            # Start task by moving down
            self.task_vars.update({"task":"down"})
            self.task_vars.update({"bottom_change_timer":time.time()})
            self.task_vars.update({"wall_contact_timer":time.time()})
            self.task_vars.update({"wall_change_timer":time.time()})

        print("Touching walls - " + self.task_vars["task"])

        def set_wall_target(direction = 1):
            if mode == "contact":
                z_target = -0.5
            elif mode == 'float':
                z_target = None
            self.current_contact_target =[None, direction*random.uniform(wall_rand_force_range[0], wall_rand_force_range[1]), z_target]

        # State machine
        if self.task_vars["task"] == "down":
            # Move downward
            self.current_contact_target = self.standard_gravity_target
            self.maintain_override = True
            # Switch to new state based on mode
            if mode == "contact":
                if self.last_contact_time < 0.1:
                    self.task_vars["task"] = "move_positive"
            elif mode == "float":
                if self.get_pos()[0][2] < float_height:
                    self.current_contact_target = [None, None, None]
                    self.task_vars["task"] = "move_positive"
            else:
                raise Exception("Unsupported mode: " + mode)
        elif self.task_vars["task"] == "move_positive":
            # print("contact target" + str(self.current_contact_target))
            self.current_contact_target[1] = None
            self.maintain_override = False
            vels = [0., 1., 0., 0., 0., 0.]
            # Check if boundary is reached, if so, switch state based on if sticking to walls
            if self.delta_from_floating[1] > self.task_vars["boundary_positive"]:
                if is_stick_to_wall:
                    self.task_vars["task"] = "wall_touch_positive"
                    self.task_vars["wall_contact_timer"] = time.time()
                    self.task_vars["wall_change_timer"] = time.time()
                    if is_randomizing_force:
                        set_wall_target(1)
                else:
                    self.task_vars["task"] = "move_negative"
        elif self.task_vars["task"] == "move_negative":
            # print("contact target" + str(self.current_contact_target))
            self.current_contact_target[1] = None
            self.maintain_override = False
            vels = [0., -1., 0., 0., 0., 0.]
            # Check if boundary is reached, if so, switch state based if sticking to walls
            if self.delta_from_floating[1] < self.task_vars["boundary_negative"]:
                if is_stick_to_wall:
                    self.task_vars["task"] = "wall_touch_negative"
                    self.task_vars["wall_contact_timer"] = time.time()
                    self.task_vars["wall_change_timer"] = time.time()
                    if is_randomizing_force:
                        set_wall_target(-1)
                else:
                    self.task_vars["task"] = "move_positive"
        elif self.task_vars["task"] == "wall_touch_positive":
            if is_randomizing_force:
                if time.time() - self.task_vars["wall_change_timer"] > wall_change_period:
                    set_wall_target(1)
                    self.task_vars["wall_change_timer"] = time.time()
                print("contact target" + str(self.current_contact_target))
            else:
                if mode == "contact":
                    self.current_contact_target = [None, 0.5, -0.5]
                elif mode == "float":
                    self.current_contact_target = [None, 0.5, None]
                print("contact target" + str(self.current_contact_target))
            if time.time() - self.task_vars["wall_contact_timer"] > stick_period:
                self.task_vars["task"] = "move_negative"
        elif self.task_vars["task"] == "wall_touch_negative":
            if is_randomizing_force:
                if time.time() - self.task_vars["wall_change_timer"] > wall_change_period:
                    if is_randomizing_force:
                        set_wall_target(-1)
                    self.task_vars["wall_change_timer"] = time.time()
                print("contact target" + str(self.current_contact_target))
            else:
                if mode == "contact":
                    self.current_contact_target = [None, -0.5, -0.5]
                elif mode == "float":
                    self.current_contact_target = [None, -0.5, None]
                print("contact target" + str(self.current_contact_target))
            if time.time() - self.task_vars["wall_contact_timer"] > stick_period:
                self.task_vars["task"] = "move_positive"
        else:
            raise Exception("Unsupported subtask within back and forth: " + self.task_vars["task"])

        # Change contacyt force target based on time period
        if is_randomizing_force and mode == 'contact':
            if time.time() - self.task_vars["bottom_change_timer"] > bottom_change_period:
                self.base_z_target = random.uniform(bottom_rand_force_range[0], bottom_rand_force_range[1])
                self.task_vars["bottom_change_timer"] = time.time()
            current_z_target = self.base_z_target + float(0.3*np.sin((time.time() - self.task_vars["bottom_change_timer"])/(random.uniform(0.5, 4)*np.pi)))
            self.current_contact_target = list(self.current_contact_target[0:2]) + [current_z_target]

        return vels

    def reposition_fingers(self, location, orientation):
        '''
        Change grasp postion of the fingers. The hand would release, grasp at a new postions, then regrasp the object.

        Arg(s):
            location : list[float] 3 x 1
                Target location for the reorientation. Defaults to the starting location.
            orientation : list[float] 6 x 1
                Target postion for the end effector to reorient. Coordinates defined in world space.
        Return:
            Commanded velocities to the arm
        '''
        vels = [0., 0., 0., 0., 0., 0.]

        # pause recording
        self.record_forces = False

        # move to defined location

        # release gripper

        #
        return vels

    def random_move(self):
        pass

    def pick_exploration_direction(self):
        # IPython.embed()
        KD = KDTree(self.task_vars['positions'])
        candidates = []
        for _ in range(40):
            candidates.append([np.random.uniform(self.task_vars["workspace_bounds"][0][0],self.task_vars["workspace_bounds"][0][1]),np.random.uniform(self.task_vars["workspace_bounds"][1][0],self.task_vars["workspace_bounds"][1][1])])

        max_idx= 0
        for i in range(1,len(candidates)):
            if KD.query(candidates[i])[0] > KD.query(candidates[max_idx])[0]:
                max_idx = i

        return candidates[max_idx]

    def explore_hole_andy(self):

        '''
        Task:
        - (1) Explore: Go in a given lateral direction
            - If edge of workspace is reached, pick a new direction and start over at (1)
            - If a contact is detected, switch to (2)
        - (2) Insert: Regulate to center the object, push down and spiral the fingers
        '''

        vels = [0, 0, 0, 0, 0, 0]

        keep_within_workspace = True

        inhand_manipulation = True

        # (trans1, rot1)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (trans1, rot1) = self.get_pos()

        file_path = "/home/grablab/Documents/data_experiments/forces/hole_exploration.pkl"

        z_target = -1.1 # -0.8 #this was 0.5

        if "is_started" not in self.task_vars:
            if inhand_manipulation:
                self.inHandType.publish("stop") #reset the grasp



            self.current_contact_target = (-0.5, -0.5, z_target)

            self.task_vars.update({"is_started":True})
            self.task_vars.update({"contact_started":False})
            self.maintain_override = True
            self.task_vars.update({"recorded":[]})
            self.task_vars.update({"positions":[]})
            self.task_vars.update({"global_timer":time.time()})
            self.task_vars['target'] = [0,0]

            #Calculate workspace
            center_point = [trans1[0], trans1[1]]
            workspace_size = 0.01 # cm
            self.task_vars.update({"workspace_bounds":[[center_point[0]-workspace_size/2, center_point[0]+workspace_size/2], [center_point[1]-workspace_size/2, center_point[1]+workspace_size/2]]})
            self.task_vars['target'] = copy.copy(center_point)

            #self.task_vars.update({"task":"explore"})
            self.task_vars.update({"task":"initial"})
            pickle.dump([[], self.task_vars["workspace_bounds"], [], self.task_vars['target']], open(file_path, "wb"))
            self.task_vars.update({"global_timer":time.time()})
            self.task_vars.update({'recently_inside_workspace':True})
            self.task_vars.update({'contact_target_for_insertion':copy.copy(self.current_contact_target)})
            self.task_vars.update({"time_outside_workspace": 1e12}) #large if inside workspace, timer if outside
            self.task_timer = time.time()
            time.sleep(0.5)

        print("Sub task:", self.task_vars["task"])

        if self.task_vars["task"] == "initial":
            if time.time() - self.task_timer > 5:
                self.current_contact_target = (-0.5, 0, z_target)
                self.task_vars.update({"task":"explore"})
            else:
                self.current_contact_target = [-0.5, 0,None]

            new_record = self.task_vars["positions"]
            new_record.append((trans1[0], trans1[1]))
            self.task_vars.update({"positions":new_record})



        if self.task_vars["task"] == "explore":
            if keep_within_workspace:
                print("T:", trans1[0], "/", trans1[1])
                print(self.task_vars["workspace_bounds"])
                randomness = 0.25
                target_lateral = 0.8

                outside_workspace = (trans1[0] < self.task_vars["workspace_bounds"][0][0] and self.current_contact_target[0] > 0) or \
                                    (trans1[0] >  self.task_vars["workspace_bounds"][0][1] and self.current_contact_target[0] < 0) or \
                                    (trans1[1] < self.task_vars["workspace_bounds"][1][0] and self.current_contact_target[1] > 0) or \
                                    (trans1[1] >  self.task_vars["workspace_bounds"][1][1] and self.current_contact_target[1] < 0)

                if outside_workspace and self.task_vars['recently_inside_workspace'] or time.time()- self.task_vars['time_outside_workspace']> 4:
                    target = self.pick_exploration_direction()

                    self.task_vars['target'] = target
                    target.append(0)
                    diff = np.asarray(trans1)-np.asarray(target)
                    if abs(diff[0])> abs(diff[1]):
                        mult = abs(0.5/diff[0])
                    else:
                        mult = abs(0.5/diff[1])
                    self.current_contact_target = [mult*diff[0], mult*diff[1], -0.7]
                    #self.current_contact_target = [0.5, 0, -0.7]
                    self.task_vars.update({'contact_target_for_insertion':copy.copy(self.current_contact_target)})
                    self.task_vars.update({'recently_inside_workspace':False})
                    self.task_vars['time_outside_workspace'] = time.time()

                #This is so we only change once while we are outside the workspace
                if not outside_workspace:
                    self.task_vars.update({'recently_inside_workspace':True})
                    self.task_vars['time_outside_workspace'] = 1e12


            # According to the current target, we set the grasp configuration to angle the object in the same direction as the motion
            if inhand_manipulation:
                if abs(self.current_contact_target[0]) > abs( self.current_contact_target[1]):
                    if self.current_contact_target[0] > 0:
                        self.inHandType.publish("left")
                    else:
                        self.inHandType.publish("right")
                else:
                    if self.current_contact_target[1] > 0:
                        self.inHandType.publish("up")
                    else:
                        self.inHandType.publish("down")
            #
            # if not self.task_vars["contact_started"]:
            #     self.task_timer = time.time()

            if self.last_contact_time < 1 and not self.task_vars["contact_started"]:
                self.task_timer = time.time()
                self.task_vars.update({"contact_started":True})
                print("Started task timer")
            # if self.last_contact_time > 1 and self.task_vars["contact_started"]:
            #     self.task_vars.update({"contact_started":False})

            print("task timer:", time.time()-self.task_timer)
            lateral_strength = 0.5
            if np.linalg.norm(np.asarray([self.delta_from_floating[0], self.delta_from_floating[1]])) > lateral_strength:
            #if self.last_contact_time < 0.1:
                self.current_task = {"name":"insert_object", "args":{}}


            new_record = self.task_vars["positions"]
            new_record.append((trans1[0], trans1[1]))
            self.task_vars.update({"positions":new_record})

            if time.time() - self.task_vars["global_timer"] > 1:
                pickle.dump([self.task_vars["recorded"], self.task_vars["workspace_bounds"], self.task_vars["positions"],self.task_vars['target'] ], open(file_path, "wb"))
                self.task_vars.update({"global_timer":time.time()})

        return vels

    #New version of the old one, hopefully we can get it to work this time.
    #Works pretty well for the pear, still working on for the triangle.
    def insert_object(self):
        """
        Task:
        - (1) Go down until contact
        - (2) Go right until side contact
        - (3) Go left until side contact
        - (4) Rotate until you find get a torque that is too high
        - (5) Align and insert
        """

        vels = [0, 0, 0, 0, 0, 0]
        insertion_done = False
        # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (t, R) = self.get_pos()
        lateral_strength = 0.5
        downward_strength = 0.5


        # Initializing the task. Starting with 'down' task
        if "state" not in self.task_vars:
            self.task_vars.update({"state":"down"})
            self.task_vars. update({"starting_position": t})
            self.maintain_override = False

        print('********************************')
        print('Task: ', 'insert_object')
        print("Sub-task:", self.task_vars["state"])

        # Sub-task: Go down, and get a contact | Do nothing until self.last_contact is ~ 0
        if self.task_vars["state"] == "down":
            self.current_contact_target = self.standard_gravity_target
            if abs(self.delta_from_floating[2]) > abs(self.standard_gravity_target[2]): #downward_strength: # Regulation finished, switch to slide1
                self.task_vars.update({"state":"slide1"})
            else:
                vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work


        # Sub-task: Go left, keep down, until catching the edge of the hole.
        if self.task_vars["state"] == "slide1":
            # IPython.embed()
            if 'contact_target_for_insertion' in self.task_vars.keys():
                self.current_contact_target = self.task_vars['contact_target_for_insertion']

                unit_vector_1 =np.asarray(self.delta_from_floating[:2]) / np.linalg.norm(np.asarray(self.delta_from_floating[:2]))
                unit_vector_2 = np.asarray(self.current_contact_target[:2]) / np.linalg.norm(np.asarray(self.current_contact_target[:2]))
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)
                print('ANGLE1: ', angle)
                print('NORM1: ', np.linalg.norm(np.asarray(self.delta_from_floating[:2])))
                if abs(angle)<0.3 and np.linalg.norm(np.asarray(self.delta_from_floating[:2]))> lateral_strength:
                    self.task_vars.update({"state":"slide2"})

            else:
                # self.inHandType.publish("left")
                self.current_contact_target = (lateral_strength, None, -downward_strength)
                if abs(self.delta_from_floating[0]) > lateral_strength: # Regulation finished, switch to slide1
                    self.task_vars.update({"state":"slide2"})
                else:
                    vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        # Sub-task: Go up
        if self.task_vars["state"] == "slide2":
            if 'contact_target_for_insertion' in self.task_vars.keys():
                self.current_contact_target = [-self.task_vars['contact_target_for_insertion'][1]+self.task_vars['contact_target_for_insertion'][0], self.task_vars['contact_target_for_insertion'][0]+self.task_vars['contact_target_for_insertion'][1], self.task_vars['contact_target_for_insertion'][2]]


                unit_vector_1 =np.asarray(self.delta_from_floating[:2]) / np.linalg.norm(np.asarray(self.delta_from_floating[:2]))
                unit_vector_2 = np.asarray(self.current_contact_target[:2]) / np.linalg.norm(np.asarray(self.current_contact_target[:2]))
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)
                print('ANGLE2: ', angle)
                print('NORM2: ', np.linalg.norm(np.asarray(self.delta_from_floating[:2])))

                if abs(angle)<0.5 and np.linalg.norm(np.asarray(self.delta_from_floating[:2]))> lateral_strength/1.5:
                    self.task_vars.update({"state":"rotate"})
                    self.past_contact_target = copy.copy(self.current_contact_target)

            else:
                self.current_contact_target = (lateral_strength, lateral_strength, -downward_strength)
                if abs(self.delta_from_floating[1]) > lateral_strength: # Regulation finished, switch to rotate
                    self.task_vars.update({"state":"rotate"})
                    self.past_contact_target = copy.copy(self.current_contact_target)

                else:
                    vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        # Sub-task: Go up
        if self.task_vars["state"] == "rotate":
            self.current_contact_target = [np.clip(self.past_contact_target[0], -0.4, 0.4),np.clip(self.past_contact_target[1], -0.4, 0.4), -1.1] #self.past_contact_target[2]]
            vels = [0, 0, 0, 0, 0, 1] # Do nothing, let the regulation work
            #if abs(self.delta_from_floating[5]) > 0.02:# Regulation finished, switch to slide1
            if self.delta_from_floating[5] > 0.05:# Regulation finished, switch to slide1 USED 0.01 for the triangle
                self.task_vars.update({"state":"insert"})


        if self.task_vars["state"] == "insert":
            rots = self.getObjectAngle()
            if rots[0] is not None:
                print('Rotations about axes: ', rots)
                try:
                    #Always use the hand rotations
                    if abs(rots[1])> abs(rots[0]):
                        if rots[1]<-0.0:
                            self.inHandType.publish("right_delta")
                        elif rots[1]>0.0:
                            self.inHandType.publish("left_delta")
                    else:
                        if rots[0]<-0.0:
                            self.inHandType.publish("up_delta")
                        elif rots[1]>0.0:
                            self.inHandType.publish("down_delta")
                except:
                    pass
                    # rospy.logwarn('ANDY YOU DID SOMEHTING BAD ')
                    # IPython.embed()

                #regulate_motion = "translation" #this worked fine for the pear
                regulate_motion = "rotation"

                if regulate_motion == "translation":
                    tar = 1.0
                    self.current_contact_target = [0, 0, -1.0] #use this for the normal stuff
                    # self.current_contact_target = [self.current_contact_target[0], self.current_contact_target[1], -1.5]

                    if rots[1]<-0.02:
                        self.current_contact_target[0] = -tar
                    elif rots[1]>0.02:
                        self.current_contact_target[0] = tar
                    if rots[0]<-0.02:
                            self.current_contact_target[1] = -tar
                    elif rots[0]>0.02:
                            self.current_contact_target[1] = tar
                elif regulate_motion == "rotation":
                    tar = 1.0
                    self.current_contact_target = [0, 0, -0.8] #use this for the normal stuff
                    if rots[1]<-0.0:
                        vels[3] = -1
                        # self.current_contact_target[0] = -tar
                    elif rots[1]>0.0:
                        vels[3] = 1
                        # self.current_contact_target[0] = tar
                    if rots[0]<-0.0:
                        vels[4] = 1
                        # self.current_contact_target[1] = -tar
                    elif rots[0]>0.0:
                        vels[4] = -1
                        # self.current_contact_target[1] = tar
            else:
                # self.inHandType.publish("spiral")
                self.current_contact_target = [0, 0, -1.]

        return vels

    def insert_object_2D(self):
        """
        Task:
        - (1) Go down until contact
        - (2) Go right until side contact
        - (3) insert (go down)
        """

        vels = [0, 0, 0, 0, 0, 0]
        insertion_done = False
        (t, R) = self.get_pos()
        lateral_strength = 1
        downward_strength = 2


        # Initializing the task. Starting with 'down' task
        if "state" not in self.task_vars:
            self.task_vars.update({"state":"down"})
            self.task_vars. update({"starting_position": t})
            self.maintain_override = False

        print('********************************')
        print('Task: ', 'insert_object')
        print("Sub-task:", self.task_vars["state"])

        # Sub-task: Go down, and get a contact | Do nothing until self.last_contact is ~ 0
        if self.task_vars["state"] == "down":
            self.current_contact_target = self.standard_gravity_target
            if abs(self.delta_from_floating[2]) > abs(self.standard_gravity_target[2]): #downward_strength: # Regulation finished, switch to slide1
                self.task_vars.update({"state":"slide1"})
            else:
                vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        # Sub-task: Go left, keep down, until catching the edge of the hole.
        if self.task_vars["state"] == "slide1":
            # IPython.embed()
            if 'contact_target_for_insertion' in self.task_vars.keys():
                self.current_contact_target = self.task_vars['contact_target_for_insertion']

                unit_vector_1 =np.asarray(self.delta_from_floating[:2]) / np.linalg.norm(np.asarray(self.delta_from_floating[:2]))
                unit_vector_2 = np.asarray(self.current_contact_target[:2]) / np.linalg.norm(np.asarray(self.current_contact_target[:2]))
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)
                #print('ANGLE1: ', angle)
                #print('NORM1: ', np.linalg.norm(np.asarray(self.delta_from_floating[:2])))
                if abs(angle)<0.3 and np.linalg.norm(np.asarray(self.delta_from_floating[:2]))> lateral_strength:
                    self.task_vars.update({"state":"slide2"})

            else:
                self.current_contact_target = (lateral_strength, None, -downward_strength)
                if abs(self.delta_from_floating[0]) > lateral_strength: # Regulation finished, switch to slide1
                    self.task_vars.update({"state":"insert"})
                else:
                    vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        if self.task_vars["state"] == "insert":
            # In this setting, we cannot act on the object rotation by actuating the fingers, so we just go down and push awas from both sides
            self.current_contact_target = [0, 0, self.standard_gravity_target[2]] #use this for the normal stuff
            vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        return vels

    #Original, sorta working
    def rotate_against_edge(self):
        """
        Task:
        - (1) Go down until contact
        - (2) Go right until side contact
        - (3) Start rotating aroung z - if jammed (torque on rz), pull up
        - (4) If contact is lost, get back, and restart at (3)
        - (5) If altitude get low enough, start inserting: regulate lateral forces to 0, push down, and spiral fingers
        """

        vels = [0, 0, 0, 0, 0, 0]
        insertion_done = False

        # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (t, R) = self.get_pos()
        # (t, R)  = self.listener.lookupTransform('tag_10', 'wam/base_link', rospy.Time(0))
        print('ALTITUDE: ')
        print(t)
        print('-----------------')

        if "ref_altitude" in self.task_vars:
            print("Altitude delta:", self.task_vars["ref_altitude"] - t[2])

        # offset_altitude = 0.005 #works decently for the cylinder

        #offset_altitude = 0.014 # Works with pear ## OLD
        offset_altitude = 0.005 # Works with pear
        # offset_altitude = 0.01 # Works with triangle/rectangle  (not really)

        lateral_strength = 0.4

        # Initializing the task. Starting with 'down' task
        if "state" not in self.task_vars:
            self.task_vars.update({"state":"down"})
            # self.task_vars.update({"state":"insert"})
            self.task_vars.update({"initial_contact":False})
            self.maintain_override = False
            self.current_contact_target = self.standard_gravity_target

        print("Sub-task:", self.task_vars["state"])
        print("Initial contact:", self.task_vars["initial_contact"])

        # Sub-task: Go down, and get a contact | Do nothing until self.last_contact is ~ 0
        if self.task_vars["state"] == "down":
            if self.last_contact_time < 0.5 and not self.task_vars["initial_contact"]:
                # Regulation finished, switch to keep_edge. Start timer to prevent switching to 'going_back' immediately.
                self.task_vars.update({"state":"keep_edge"})
                self.task_timer = time.time()

            else:
                vels = [0, 0, 0, 0, 0, 0] # Do nothing, let the regulation work

        # Sub-task: Go left, keep down, until catching the edge of the hole.
        if self.task_vars["state"] == "keep_edge":

            self.inHandType.publish("left")
            self.current_contact_target = (lateral_strength, lateral_strength, -1)

            if self.task_vars["initial_contact"]:
                if "ref_altitude" not in self.task_vars:
                    if "alt_timer" not in self.task_vars:
                        self.task_vars.update({"alt_timer":time.time()})
                        self.task_vars.update({"alt_hist":[]})
                    else:
                        if time.time() - self.task_vars["alt_timer"] < 2:
                            all_alts = self.task_vars["alt_hist"]
                            all_alts.append(t[2])
                            self.task_vars.update({"alt_hist":all_alts})
                            return vels
                        else:
                            ref_alt = 0
                            for alt in self.task_vars["alt_hist"]:
                                ref_alt += alt
                            ref_alt /= len(self.task_vars["alt_hist"])

                            print(ref_alt)
                            #IPython.embed()
                            self.task_vars.update({"ref_altitude":ref_alt})
                else:
                    vels[5] = 1
                #self.current_contact_target = (lateral_strength, lateral_strength , -1)


            if not self.task_vars["initial_contact"] and time.time() - self.task_timer > 5 and self.last_contact_time < 0.1:
                self.task_vars.update({"initial_contact":True})


            if self.task_vars["initial_contact"] and self.last_contact_time > 10:
                print("Edge lost, going back to retrieve it!")
                self.task_vars.update({"state":"going_back"})
                self.task_timer = time.time()
                # self.task_timer = time.time()

            if "ref_altitude" in self.task_vars and self.task_vars["ref_altitude"]-t[2] > offset_altitude \
                or abs(self.delta_from_floating[5]) > 0.06: #or self.delta_from_floating[0]>0.5:

                self.task_vars.update({"state":"insert"})
                self.task_timer = time.time()

        # Condition to stop going back is broken: stops WAY TOO SOON.
        if self.task_vars["state"] == "going_back":
            # self.inHandType.publish("right")
            self.current_contact_target = (-lateral_strength, None, -0.5)
            # if self.last_contact_time <0.1:
            if time.time() - self.task_timer > 10:
                print("Contact re-establish with other edge of hole.")
                self.task_vars.update({"state":"keep_edge"})
                self.task_vars.update({"initial_contact":False})
            vels[5] = 1

        if abs(self.delta_from_floating[5]) > 0.06:
            # Getting in a 'jammed' situation. Needs to pull out, keep rotating and enter again.
            print("JAMMED!! Pulling out...")
            self.current_contact_target = (None, None, None)
            vels = [0, 0, -1, 0, 0, 0]


        if self.task_vars["state"] == "insert":
            # IPython.embed()
            rots = self.getObjectAngle()
            print('Rotations about axes: ', rots)

            #Always use the hand rotations
            if abs(rots[1])> abs(rots[0]):
                if rots[1]<-0.0:
                    self.inHandType.publish("right_delta")
                elif rots[1]>0.0:
                    self.inHandType.publish("left_delta")
            else:
                if rots[0]<-0.0:
                    self.inHandType.publish("up_delta")
                elif rots[1]>0.0:
                    self.inHandType.publish("down_delta")

            if time.time() - self.task_timer < 5:
                # self.current_contact_target = [None, None, None]
                self.current_contact_target = [0.5, 0, -0.5]
                # self.inHandType.publish("open")

                #vels = [0, 0, 0, 0, 0, -1]
            #when out of limits, use the arm every 5 seconds or so.
            else:

                # regulate_motion = "translation"
                regulate_motion = "rotation"

                if regulate_motion == "translation":
                    tar = 1.5
                    self.current_contact_target = [0, 0, -tar]
                    if rots[1]<-0.0:
                        self.current_contact_target[0] = -tar
                    elif rots[1]>0.0:
                        self.current_contact_target[0] = tar
                    if rots[0]<-0.0:
                            self.current_contact_target[1] = -tar
                    elif rots[0]>0.0:
                            self.current_contact_target[1] = tar
                elif regulate_motion == "rotation":
                    if rots[1]<-0.0:
                        vels[3] = -1
                    elif rots[1]>0.0:
                        vels[3] = 1
                    if rots[0]<-0.0:
                        vels[4] = 1
                    elif rots[0]>0.0:
                        vels[4] = -1
                # vels = [0, 0, 0, 0, 0, 1]
                # self.inHandType.publish("spiral")

            if time.time() - self.task_timer > 10:
                self.task_timer = time.time()

            # if self.task_vars["ref_altitude"]-t[2] > 0.045:
            #     IPython.embed()

        return vels


    def getObjectAngle(self):
        try:
            # (t, R)  = self.listener.lookupTransform('tag_10', 'wam/wrist_palm_stump_link', rospy.Time(0))
            (t, R) = self.get_pos()
            rot = PyKDL.Rotation.Quaternion(*R)
            rots = list(rot.GetRPY())
            rots[0]= rots[0]+math.pi
            rots[0] = (rots[0]+math.pi)%(2*math.pi) - math.pi
            return rots
        except:
            return [None, None, None]


    def explore_hole(self, start_direction):
        '''
        Task:
        - (1) Explore: Go in a given lateral direction
            - If edge of workspace is reached, pick a new direction and start over at (1)
            - If a contact is detected, switch to (2)
        - (2) Insert: Regulate to center the object, push down and spiral the fingers
        '''
        # self.current_contact_target = [None, None, None]
        # return [0, 0, 0, 0, 0, 1]



        vels = [0, 0, 0, 0, 0, 0]

        keep_within_workspace = True
        # keep_within_workspace = False

        randomize_new_direction = True
        # randomize_new_direction = False

        inhand_manipulation = True

        center_point = [-0.027, -0.64]
        workspace_size = 0.01 # cm
        workspace_bounds = [[center_point[0]-workspace_size/2, center_point[0]+workspace_size/2], [center_point[1]-workspace_size/2, center_point[1]+workspace_size/2]]

        # (trans1, rot1)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (trans1, rot1)  = self.get_pos()
        print("Workspace bounds")
        pprint(workspace_bounds)
        print("T:", trans1)

        file_path = "/home/grablab/Documents/data_experiments/forces/hole_exploration.pkl"


        if "is_started" not in self.task_vars:
            if inhand_manipulation:
                self.inHandType.publish("stop")
            z_target = -0.5
            # z_target = None
            if start_direction == "-x":
                self.current_contact_target = (-0.5, 0, z_target)
            if start_direction == "x":
                self.current_contact_target = (0.5, 0, z_target)
            if start_direction == "-y":
                self.current_contact_target = (0, -0.5, z_target)
            if start_direction == "y":
                self.current_contact_target = (0, 0.5, z_target)
            if start_direction == "xy":
                self.current_contact_target = (0.5, 0.5, z_target)
            self.task_vars.update({"is_started":True})
            self.task_vars.update({"contact_started":False})
            self.maintain_override = True
            self.task_vars.update({"recorded":[]})
            self.task_vars.update({"positions":[]})
            self.task_vars.update({"global_timer":time.time()})

            self.task_vars.update({"task":"explore"})

            pickle.dump([[], workspace_bounds, []], open(file_path, "wb"))
            self.task_vars.update({"global_timer":time.time()})

        print("Sub task:", self.task_vars["task"])

        if self.task_vars["task"] == "explore":
            if keep_within_workspace:
                print("T:", trans1[0], "/", trans1[1])
                randomness = 0.25
                if trans1[0] < workspace_bounds[0][0] and self.current_contact_target[0] > 0:
                    print("-x bound reached, inverting target")
                    self.current_contact_target = (-self.current_contact_target[0]+random.uniform(-randomness,randomness), self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])
                if trans1[0] > workspace_bounds[0][1] and self.current_contact_target[0] < 0:
                    print("+x bound reached, inverting target")
                    self.current_contact_target = (-self.current_contact_target[0]+random.uniform(-randomness,randomness), self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])

                if trans1[1] < workspace_bounds[1][0] and self.current_contact_target[1] > 0:
                    print("-y bound reached, inverting target")
                    self.current_contact_target = (self.current_contact_target[0]+random.uniform(-randomness,randomness), -self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])
                if trans1[1] > workspace_bounds[1][1] and self.current_contact_target[1] < 0:
                    print("+y bound reached, inverting target")
                    self.current_contact_target = (self.current_contact_target[0]+random.uniform(-randomness,randomness), -self.current_contact_target[1]+random.uniform(-randomness,randomness), self.current_contact_target[2])


            # According to the current target, we set the grasp configuration to angle the object in the same direction as the motion
            if inhand_manipulation:
                if abs(self.current_contact_target[0]) > abs( self.current_contact_target[1]):
                    if self.current_contact_target[0] > 0:
                        self.inHandType.publish("left")
                    else:
                        self.inHandType.publish("right")
                else:
                    if self.current_contact_target[1] > 0:
                        self.inHandType.publish("up")
                    else:
                        self.inHandType.publish("down")

            if not self.task_vars["contact_started"]:
                self.task_timer = time.time()

            if self.last_contact_time < 1 and not self.task_vars["contact_started"]:
                self.task_timer = time.time()
                self.task_vars.update({"contact_started":True})
                print("Started task timer")
            if self.last_contact_time > 1 and self.task_vars["contact_started"]:
                self.task_vars.update({"contact_started":False})

            print("task timer:", time.time()-self.task_timer)
            if self.task_vars["contact_started"] and time.time() - self.task_timer > 1:
                max_target = 0.5
                print("Equilibrium reached, switching direction")
                print("torques:", self.delta_from_floating[3], self.delta_from_floating[4])
                ratio = abs(self.delta_from_floating[3])/(abs(self.delta_from_floating[3])+abs(self.delta_from_floating[4]))
                print("Ratio:", ratio)

                # X velocity is driven by the amount of Y-axis torque, and Y vel by X-torque
                new_y = abs(self.delta_from_floating[3])/self.delta_from_floating[3] # Getting the sign of current direction
                # print("new x sign:", new_y)
                new_y *= ratio*max_target
                # print("new x:", new_y)

                new_x = abs(self.delta_from_floating[4])/self.delta_from_floating[4] # Getting the sign of current direction
                new_x *= -(1-ratio)*max_target

                new_x_target = new_x
                new_y_target = new_y

                if randomize_new_direction:
                    amount = 0.5

                    new_x_target += random.uniform(-amount, amount)
                    new_y_target += random.uniform(-amount, amount)

                    new_x_target = np.clip(new_x_target, -max_target, max_target)
                    new_y_target = np.clip(new_y_target, -max_target, max_target)

                # new_x = None
                # self.current_contact_target = (
                #     -self.current_contact_target[0] if self.current_contact_target[0] is not None else None,
                #     -self.current_contact_target[1] if self.current_contact_target[1] is not None else None,
                #     self.current_contact_target[2]
                #                               )
                self.current_contact_target = (new_x_target, new_y_target, self.current_contact_target[2])
                print("new target:", self.current_contact_target)
                # time.sleep(9999999999999999999)
                self.task_timer = time.time()

                self.task_vars.update({"last_torque_direction":(self.delta_from_floating[3], self.delta_from_floating[4])})


                new_record = self.task_vars["recorded"]
                # new_record.append((trans1, (self.delta_from_floating[0], self.delta_from_floating[1])))
                new_record.append((trans1, (new_x, new_y)))
                self.task_vars.update({"recorded":new_record})

                # Switching to new insertion task: TESTING
                self.task_vars.update({"task":"insert"})
                self.inHandType.publish("spiral")

            new_record = self.task_vars["positions"]
            new_record.append((trans1[0], trans1[1]))
            self.task_vars.update({"positions":new_record})

            if time.time() - self.task_vars["global_timer"] > 1:
                pickle.dump([self.task_vars["recorded"], workspace_bounds, self.task_vars["positions"]], open(file_path, "wb"))
                self.task_vars.update({"global_timer":time.time()})


        if self.task_vars["task"] == "insert":
            self.current_contact_target = [0, 0, -0.5]

        return vels

    def slide(self, axis):
        """
        Task: Push down and sideways, making it slide across a plane
        """
        vels = [0, 0, 0, 0, 0, 0]

        if axis == "x":
            vels = [1, 0, 0, 0, 0, 0]
        if axis == "-x":
            vels = [-1, 0, 0, 0, 0, 0]
        if axis == "y":
            vels = [0, 1, 0, 0, 0, 0]
        if axis == "-y":
            vels = [0, -1, 0, 0, 0, 0]

        return vels

    def dist(self, a, b):
        return math.sqrt( (a - b)**2 + (a - b)**2 )

    def slide_and_test(self, axis):
        """
        Task:
        - (1) Push down and sideways, making it slide across a plane. After a time, start (2)
        - (2) Run a contact test procedure
        """
        vels = [0, 0, 0, 0, 0, 0]
        print("Task: Slide and test")
        print("Current sub-task:", self.task_state)
        print("task time:")
        self.current_contact_target = self.standard_gravity_target
        self.maintain_override = True

        test_interval = 5
        testing_complete = True

        if self.task_state is None:
            self.task_timer = time.time()
            self.task_state = "test_contact"

        if time.time() - self.task_timer > test_interval:
            self.task_state = "test_contact"
            testing_complete = False

        if self.task_state == "test_contact":
            vels, test_done = self.test_contact(down_first=False)
            if test_done:
                testing_complete = True
                self.task_timer = time.time()
                self.task_state = "slide"
                vels = [0, 0, 0, 0, 0, 0]

        if self.task_state == "slide":
            vels = self.slide(axis)

        return vels


    def direct_control(self, safe=True):
        """
        Task: Wait for keyboard input to create the velocity
        - W: up
        - S: down
        - A: left
        - D: right
        - Q: left rotation
        - E: right rotation
        """
        vels = [0, 0, 0, 0, 0, 0]
        self.maintain_contact = False

        forces_safety_threshold = 10
        # (t, R)  = self.listener.lookupTransform('wam/wrist_palm_stump_link', 'wam/base_link', rospy.Time(0))
        (t, R)  = self.get_pos()
        print('ALTITUDE: ')
        print(t)
        print('-----------------')

        if "w" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[2] > -forces_safety_threshold:
                vels[2] = -1
        if "s" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[2] < forces_safety_threshold:
                vels[2] = 1
        if "a" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[0] > -forces_safety_threshold:
                vels[0] = -1
        if "d" in self.current_pressed_keys:
            if not safe or self.delta_from_floating[0] < forces_safety_threshold:
                vels[0] = 1
        if "q" in self.current_pressed_keys:
            if not safe or (abs(self.delta_from_floating[0]) < forces_safety_threshold and abs(self.delta_from_floating[1]) < forces_safety_threshold and abs(self.delta_from_floating[2]) < forces_safety_threshold):
                vels[4] = -1
        if "e" in self.current_pressed_keys:
            if not safe or (abs(self.delta_from_floating[0]) < forces_safety_threshold and abs(self.delta_from_floating[1]) < forces_safety_threshold and abs(self.delta_from_floating[2]) < forces_safety_threshold):
                vels[4] = 1


        return vels

    def test_contact(self, frozen_axis="y", initial_contact_target=[None, None, None]):
        """
        Task: Perform an array of motions, and detect for each the mechanical constraints (from the force readings)
        """

        vels = [0, 0, 0, 0, 0, 0]
        self.current_contact_target = initial_contact_target

        # loop = True
        loop = False

        if self.task_state is None: # Setting the initial sub-task
            if initial_contact_target != [None, None, None]:
                self.task_state = "initial_contact"
            else:
                self.task_state = "test_contact"
                self.maintain_override = False
                self.maintain_contact = False

        if self.task_state  == "initial_contact":
            self.maintain_override = True

            if self.last_contact_time < 1:
                self.task_state = "test_contact"
                self.maintain_override = False
                self.maintain_contact = False

        if self.task_finished:
            print("Task finished")
            print("Found resistances:")
            pprint(self.task_vars["resistances"])
            time.sleep(999999999)
            return vels#, self.task_finished

        if self.task_state  == "test_contact":
            self.maintain_contact = False
            if "initPos" not in self.task_vars:
                self.task_vars.update({"initPos":self._getPos()})
                self.task_vars.update({"motion_idx":0})
                self.task_vars.update({"resistances":[[False, False], [False, False], [False, False], [False, False], [False, False], [False, False]]})
                self.task_timer = time.time()

            motions = ["x", "-x", "y", "-y", "z", "-z", "rz", "-rz"]#, "ry", "-ry"]
            # motions = ["rz"]
            # motions = ["x", "y", "z"]
            motion_duration = 5

            # Current motion done, starting new one and resetting timer
            if time.time() - self.task_timer > motion_duration:
                self._movePos(self.task_vars["initPos"])
                if self.task_vars["motion_idx"] < len(motions)-1:
                    self.task_vars.update({"motion_idx": self.task_vars["motion_idx"]+1})
                else:
                    if loop:
                        print("Resetting motion idx to 0")
                        self.task_vars.update({"motion_idx": -1})
                        print(self.task_vars)
                    else:
                        print("All motion performed - Returning zero vels")

                        self.task_finished = True

                    return vels#, self.task_finished
                self.task_timer = time.time()

            # Perform a series of motion, returning to the initial position after each test.
            print("Executing sub-task:", motions[self.task_vars["motion_idx"]])
            current_motion = motions[self.task_vars["motion_idx"]]


            if current_motion == "x":
                vels[0] = 1
                ignores = [False, True, True, True, False, True]
            if current_motion == "-x":
                vels[0] = -1
                ignores = [False, True, True, True, False, True]
            if current_motion == "y":
                vels[1] = 1
                ignores = [True, False, True, False, True, True]
            if current_motion == "-y":
                vels[1] = -1
                ignores = [True, False, True, False, True, True]
            if current_motion == "z":
                vels[2] = 1
                ignores = [True, True, False, True, True, True]
            if current_motion == "-z":
                vels[2] = -1
                ignores = [True, True, False, True, True, True]
            if current_motion == "ry":
                vels[4] = 1
                # ignores = [True, True, False, True, True, True]
            if current_motion == "-ry":
                vels[4] = -1
            if current_motion == "rz":
                vels[5] = 1
                ignores = [True, True, True, True, True, False]
            if current_motion == "-rz":
                vels[5] = -1
                ignores = [True, True, True, True, True, False]

            new_resistance = self.detect_resistance(ignore=ignores)
            print("Real-time Resistance:")
            pprint(new_resistance)
            print("Already found resistances:")
            pprint(self.task_vars["resistances"])
            for i, resistance in enumerate(new_resistance):
                if True in resistance:
                    print("Found new resistance on axis:", i)
                    if resistance[0]:
                        self.task_vars["resistances"][i][0] = True
                    if resistance[1]:
                        self.task_vars["resistances"][i][1] = True
            # if self.task_vars["resistances"][0][0]:
            #     time.sleep(999999999999999999)
            # # Interrupting the current motion if a resistance is found to avoid wear
            # for resistance_axis in new_resistance:
            #     if "True" in resistance_axis:
            #         print("INTERRUPTING MOTION!")
            #         self.task_vars.update({"motion_idx": self.task_vars["motion_idx"]+1})
            #         self.task_timer = time.time()

        return vels#, self.task_finished

    def align_with_plane_below(self):
        """
        Task: try to regulate a side contact to 0 by rotating against it
        """
        vels = [0, 0, 0, 0, 0, 0]
        self.maintain_contact = True
        self.current_contact_target = self.standard_gravity_target
        self.maintain_override = True

        if abs(self.delta_from_floating[3]) > 0.1:
            vels[3] = self.delta_from_floating[3]*10
        # if abs(self.delta_from_floating[4]) > 0.01:
        #     vels[3] = -self.delta_from_floating[4]*10

        return vels

    def explore_laterally(self, axis):
        """
        Task:
        - (1) Go down until contact
        - (2) Slide until side contact (go to (3)) or bottom contact is lost (go to (1))
        - (3) Climb while maintaining side contact, then switch back to (2)
        """
        vels = [0, 0, 0, 0, 0, 0]
        print("current subtask:", self.task_state)

        if self.last_contact_time < 2:
            self.task_timer = time.time()

        if self.task_state is None: # Setting the initial sub-task
            self.task_state = "down"

        if self.task_state  == "down":
            self.maintain_override = False
            vels = [0, 0, 1, 0, 0, 0]

            if self.last_contact_time < 1:
                self.task_state = "sliding"
                self.maintain_override = True

        if self.task_state  == "sliding":
            if axis == "x":
                vels = [1, 0, 0, 0, 0, 0]
            if axis == "-x":
                vels = [-1, 0, 0, 0, 0, 0]
            if axis == "y":
                vels = [0, 1, 0, 0, 0, 0]
            if axis == "-y":
                vels = [0, -1, 0, 0, 0, 0]

            # Transition to Sliding
            # If a new side contact is detected, we record the current contact force as a target, and try to maintain that while going up
            diff_x = self.force_data[0]-self.averages[0]
            diff_y = self.force_data[1]-self.averages[1]
            force_offset = 1
            if abs(diff_x) > force_offset or abs(diff_y) > force_offset:
                if abs(diff_x) > force_offset:
                    target_x = abs(diff_x)/diff_x*0.5
                else:
                    target_x = 0
                if abs(diff_y) > force_offset:
                    target_y = abs(diff_y)/diff_y*0.5
                else:
                    target_y = 0
                print("Switching sub-task to Climbing")
                self.current_contact_target = (target_x, target_y, None)
                self.task_state = "climbing"
                # self.current_task = {"name":"up", "args":{}}
                self.maintain_override = True

            # Transition to Down
            if self.last_contact_time > 2 and time.time() - self.task_timer > 3:
                self.task_state = "down"
                self.current_contact_target = self.standard_gravity_target
                self.maintain_override = False


        if self.task_state == "climbing":
            vels = [0, 0, -1, 0, 0, 0]
            if self.last_contact_time > 2:
                self.task_state = "sliding"
                self.last_contact_time = 0
                self.current_contact_target = self.standard_gravity_target
                self.maintain_override = False
                self.task_timer = time.time()
        print("Task timer:", time.time()-self.task_timer)
        return vels

    def orient_towards(self):
        """
        Task: Assuming the object hit a planar surface with an angle, we try to regulate the side contact to 0 by rotating and translating back (trying to rotate around the object)
        """
        print("Re-orienting!")
        # delta_from_floating_x = self.force_data[0] - self.average_x
        # delta_from_floating_y = self.force_data[1] - self.average_y

        # print("deltas from floating:", delta_from_floating_x, "/", delta_from_floating_y)

        target_lat = 0.5
        # target_y = 1.0*0.4

        # Rotation around fingertips: 20 lat for 1 rot
        # Rotation around half cylinder: 28/1 (lat/rot)
        gain_lateral = 40
        gain_rot = 1
        global_gain = 2
        if self.project_all_in_effector_space:
            gain_rot *=-1
        gain_z = 0.3
        vel_z = -0.1

        # return [gain_lateral, 0, 0, 0, gain_rot, 0]

        if abs(self.delta_from_floating[1]) > target_lat:
            print("Regulating lateral forces, target=", target_lat)
            scale =self.delta_from_floating[1]*global_gain
            vel_y = -gain_lateral*scale
            vel_rx = gain_rot*scale
            # vel_rx = delta_from_floating_y/abs(delta_from_floating_y)*gain_rot
            print("vel y:", vel_y)
            print("rot rx:", vel_rx)
        else:
            vel_rx = 0
            vel_y = 0

        if abs(self.delta_from_floating[0]) > target_lat:
            print("Regulating lateral forces, target=", target_lat)
            scale = self.delta_from_floating[0]*global_gain
            vel_x = -scale*gain_lateral
            vel_ry = -scale*gain_rot
            # vel_ry = -delta_from_floating_x/abs(delta_from_floating_x)*gain_rot
            print("vel x:", vel_x)
            print("rot ry:", vel_ry)
            # if delta_from_floating_y > 0:
            #     vel_y = -1*(1+abs(delta_from_floating_y))
            # else:
            #     vel_y = 1*(1+abs(delta_from_floating_y))

        else:
            vel_ry = 0
            vel_x = 0
        return [vel_x, vel_y, vel_z, vel_rx, vel_ry, 0]


    def apply_gains(self, vels, gains):
        for i in range(len(vels)):
            vels[i] *= gains[i]
        return vels


    def maintain_side_contact(self, target=None, gain = 1):
        """
        Regulation task: Try to reach the lateral force targets by acting on the related velocities
        """
        print("Current contact target:", target)
        print("Last contact:", self.last_contact_time)
        # print("Average z/current_z", self.average_z, self.force_data[2])

        #if self.project_all_in_effector_space:
        #    self.delta_from_floating[2] *=-1
            # self.delta_from_floating *=-1
            # target *=-1
        # self.recorded_delta_z.append(self.delta_from_floating)
        # print("Forces:")
        # print("Current delta from floating:", self.delta_from_floating)
        # print("z force target:", target)

        precision = 0.1

        if (target[0] is None or self.dist(self.delta_from_floating[0], target[0]) < precision) and (target[1] is None or self.dist(self.delta_from_floating[1], target[1]) < precision) and (target[2] is None or self.dist(self.delta_from_floating[2], target[2]) < precision):
            self.last_contact_time = 0
        else:
            self.last_contact_time += 1/self.rate

        vels = [None, None, None]
        for i, vel_target in enumerate(target):
            if vel_target is not None:
                if vel_target is not None and self.force_data is not None:
                    vel = (vel_target-self.delta_from_floating[i])*gain
                # if self.last_contact_time > 5:
                if self.last_contact_time > math.inf:
                    print("No recent contact, speeding up")
                    vel *= 4
                else:
                    vel *= 0.5
                if self.project_all_in_effector_space and i == 2:
                    vel *=-1

                if i == 0 or i == 1:
                    vel *= 2

                vels[i] = vel

        return vels

    def maintain_side_contact_pid(self, target_forces=None, gain = 1):
        """
        Regulation task: Try to reach the lateral force targets by acting on the related velocities (Mei)
        """
        print("Current contact target pid:", target_forces)
        print("Last contact pid:", self.last_contact_time)

        precision = 0.1

        # Keep track of contact time
        if (target_forces[0] is None or self.dist(self.delta_from_floating[0], target_forces[0]) < precision) and (target_forces[1] is None or self.dist(self.delta_from_floating[1], target_forces[1]) < precision) and (target_forces[2] is None or self.dist(self.delta_from_floating[2], target_forces[2]) < precision):
            self.last_contact_time = 0
        else:
            self.last_contact_time += 1/self.rate

        # Update target velocities
        vels = [None, None, None]
        if target_forces[1] is not None:
            self.pid_y.setpoint = target_forces[1]
            vels[1] = self.pid_y(self.delta_from_floating[1])
        if target_forces[2] is not None:
            self.pid_z.setpoint = target_forces[2]
            vels[2] = self.pid_z(self.delta_from_floating[2])

        return vels

    def draw_cirle(self, axis=None, inverse_radius_size=3.14/1000*2):
        """
        Task: Slide across an horizontal planar surface with a cicular motion
        """
        if axis == "x":
            vel_x = 0
            vel_y = math.cos(self.t)
            vel_z = math.sin(self.t)
        if axis == "y":
            vel_x = math.cos(self.t)
            vel_y = 0
            vel_z = math.sin(self.t)
        if axis == "z":
            vel_x = math.cos(self.t)
            vel_y = math.sin(self.t)
            vel_z = 0

        self.t += inverse_radius_size
        scale = 100
        vel_x *= scale
        vel_y *= scale
        vel_z *= scale
        # print("vels: ", vel_y, vel_z)
        # self.publishJointVelocity_jog([0.,vel_y,vel_z, 0, 0, 0])
        return [vel_x, vel_y, vel_z, 0, 0, 0]

    #velocity of the 7 joints of the robot
    # def publishJointVelocity(self, vel):
    #     if len(vel) !=7:
    #         rospy.logerr('Velocity vector not of size 7')
    #         return
    #     vel = self._check_vels(vel)
    #     msg = RTJointVel()
    #     msg.velocities = vel
    #     self._robot_vel_publisher.publish(msg)


    # def publishJointVelocity_jog(self, vel):
    #     if len(vel) !=6:
    #         rospy.logerr('Velocity vector not of size 6')
    #         return

    #     msg = JogFrame()
    #     header = Header()
    #     header.frame_id = "wam/base_link"
    #     header.stamp = rospy.Time.now()
    #     msg.header = header
    #     msg.group_name = "arm"
    #     msg.link_name = "wam/wrist_palm_link"
    #     # msg.link_name = "obj_frame"
    #     msg.avoid_collisions = True
    #     vec_T = Vector3()
    #     vec_T.x = vel[0]
    #     vec_T.y = vel[1]
    #     vec_T.z = vel[2]
    #     msg.linear_delta = vec_T
    #     vec_R = Vector3()
    #     vec_R.x = vel[3]
    #     vec_R.y = vel[4]
    #     vec_R.z = vel[5]
    #     msg.angular_delta = vec_R
    #     # pprint(msg)
    #     self._robot_vel_publisher_jog.publish(msg)


    def _cart_vels(self, xyz_vel = [0.,0.,0.], rpy_vel = [0.,0.,0.]):
        move = xyz_vel+rpy_vel
        j = self._robot_joint_state
        Jac = np.asarray(self.arm_group.get_jacobian_matrix(j))
        Jac_inv = np.linalg.pinv(Jac)
        vel = np.matmul(Jac_inv ,np.asarray(move).reshape(-1,1))
        return vel


    def _check_vels(self, vel):
        thresh = 0.4
        vels =  [min(max(v, -thresh), thresh) for v in vel]
        mini = 0.15
        # vels[0] = min(max(vels[0], mini), -mini)
        # return [min(max(v, mini), -mini) for v in vels]
        return vels

    # def _getPose(self, object = 'simple_base', origin = 'world'):
    #     (trans1, rot1)  = self.listener.lookupTransform(origin, object, rospy.Time(0))
    #     return PyKDL.Frame(PyKDL.Rotation.Quaternion(*rot1), PyKDL.Vector(*trans1))


    ##########################################
    # callbacks
    ##########################################

    def testCallback(self, msg):
        self.current_task = {"name":"insert_object", "args":{}}

    def testCallback2(self, msg):
        self.task_vars.update({"state":"insert"})

    def openDone(self, msg):
        self.inHandType.publish("open")
        self.task_vars.update({"state":"done"})
        self._startPos(blocking=True, idx=4) #there are three of these set up


    def joint_state_callback(self, msg):
        self._robot_joint_state = list(msg.position)

    def read_forces(self):
        history_size = 2 # For computing the smoothed forces

        # pprint(req)

        # self.force_data = [req.wrench.force.x, req.wrench.force.y, req.wrench.force.z, req.wrench.torque.x, req.wrench.torque.y, req.wrench.torque.z]
        if 'franka' in self.force_source:
            self.force_data = self.controller.get_EE_wrench()
            self.force_data_GT = copy.copy(self.force_data)
        if 'ATI' in self.force_source:
            ## TODO: add proper transform
            self.force_data = self.ft_driver.read_ft_sensor()
            for i in range(3):
                self.force_data[i] *= -1
            self.force_data_GT = copy.copy(self.force_data)
            # print("\nGT form ATI:")
            # pprint(self.force_data_GT)
        if 'VFE' in self.force_source:
            print("VFE estimate:")
            pprint(self.vfe.getForce())
            self.force_data = [0, self.vfe.getForce()[0][0], self.vfe.getForce()[0][1], 0, 0, 0]
        # print("Raw force data:", self.force_data)
        if self.project_all_in_effector_space:
            self.force_data[2] *=1
        # self.force_data = self.rotate_force_readings(self.force_data)

        if self.record_forces:
            dataset = Dataset(self.record_dataset_size)
            # record at designated frequency
            if time.time() - self.record_timer > 1 / self.record_frequency:
                self.record_timer = time.time()
                # recording run time forces
                self.recorded_forces.append(self.force_data)
                if self.force_data_GT is not None:
                    self.recorded_forces_ground_truth.append(self.force_data_GT)
                if len(self.recorded_forces) % self.record_every == 0:
                    pickle.dump(self.recorded_forces, open(self.record_file, "wb"))
                    pickle.dump(self.recorded_forces_ground_truth, open(self.record_file_ground_truth, "wb"))
                    # print("Forces saved")

                # recording dataset (images and pickle files)
                color_image, time_stamp = self.cam.get_rgb_frames()

                ur_pose = None if self.controller is None else self.controller.get_EE_transform()  ## Franka
                servo_info = None #if T is None else T.readServoInfos()
                data = ImgForce([], time_stamp, np.array(self.force_data_GT), ur_pose, servo_info, True  )
                pickle.dump(data, open(os.path.join(self.record_dataset_force_dir, f'{self.record_num_img}.pkl'), "wb"))
                cv2.imwrite(os.path.join(self.record_dataset_img_dir, f'{self.record_num_img}.png'), color_image)
                self.record_num_img += 1
        raw_force = True
        if not raw_force:
            print('flag')
            # Smoothing the readings
            if len(self.last_forces) < self.nb_samples:
                self.last_forces.append(self.force_data)
            else:
                self.last_forces.pop(0)
                self.last_forces.append(self.force_data)
            # print(self.last_forces)
            averages = [0, 0, 0, 0, 0, 0]
            for force in self.last_forces[-history_size:-1]:
                for i in range(len(averages)):
                    averages[i] += force[i]
            for i in range(len(averages)):
                averages[i] /= history_size
            self.force_data = averages

            # Smoothing the readings
            if len(self.last_forces_GT) < self.nb_samples:
                self.last_forces_GT.append(self.force_data_GT)
            else:
                self.last_forces_GT.pop(0)
                self.last_forces_GT.append(self.force_data_GT)
            # print(self.last_forces)
            averages_GT = [0, 0, 0, 0, 0, 0]
            for force in self.last_forces_GT[-history_size:-1]:
                for i in range(len(averages_GT)):
                    averages_GT[i] += force[i]
            for i in range(len(averages_GT)):
                averages_GT[i] /= history_size
            self.force_data_GT = averages_GT

            # Subtracting the calibration readings
            for i in range(len(self.force_data)):
                if not self.no_delta:
                    self.delta_from_floating[i] = self.force_data[i]-self.averages[i]
                else:
                    self.delta_from_floating[i] = self.force_data[i]
        else:
            for i in range(len(self.force_data)):
                self.delta_from_floating[i] = self.force_data[i]

        if self.project_all_in_effector_space:
            self.delta_from_floating[2] *=-1

    def on_press(self, key):
        self.current_pressed_keys.add(key.char)
        # print('{0} pressed'.format(key))

    def on_release(self, key):
        self.current_pressed_keys.discard(key.char)
        # print('{0} release'.format(key))

    def get_vels(self):
        vels = controller.get_EE_velocity(tool_center = se3.identity())
        return ( (vels[0], vels[1], vels[2]), (vels[3], vels[4], vels[5]))

    def get_pos(self):
     (R, t) = controller.get_EE_transform(tool_center = se3.identity())
     return (t, R)

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

def run_peg_in_hole(dataset_dir, task_name, task_args):
    '''
    Shorter version of main function in peg_in_hole
    '''
    reset_start_position = False

    start_position_file = "start_postion_0.json"

    # controller instance
    print("Connecting to server...")
    controller = FrankaClient('http://172.16.0.1:8080')

    # start robot
    controller.initialize()
    controller.start()

    print("DONE")

    # get current states
    current_config = controller.get_joint_config()

    if reset_start_position:
        json.dump(current_config, open(start_position_file, "w"))

    PegInHoleTask(controller, start_position_file, dataset_dir, task_name, task_args)

if __name__ == "__main__":
    # reset_start_position = True
    reset_start_position = False

    start_position_file = "start_postion_0.json"
    # Klampt library for kinematics
    # world_fn = "./models/franka_world.xml"
    # EE_link = 'tool_link'
    # world = WorldModel()
    # world.readFile(world_fn)
    # robot_model = world.robot(0)
    # collider = collide.WorldCollider(world)
    # collision_checker = GlobalCollisionHelper(robot_model, collider)
    # params = {'address': "172.16.0.2"} ## TBD, joint stiffness can also be set here

    # controller instance
    print("Connecting to server...")
    controller = FrankaClient('http://172.16.0.1:8080')

    # start robot
    controller.initialize()
    controller.start()

    print("DONE")

    # get current states
    current_config = controller.get_joint_config()

    if reset_start_position:
        json.dump(current_config, open(start_position_file, "w"))

    current_joint_velocity = controller.get_joint_velocity()
    current_joint_torques = controller.get_joint_torques()
    # ic(current_config, current_joint_torques, current_joint_velocity)

    current_EE_transform = controller.get_EE_transform(tool_center = se3.identity()) #transform of tool center,
        #specified in the tool frame (last frame of the robot)
        #also this is in the klampt se3 format, (R, t), where R is the column major format of the 3x3 rot matrix
        #and t is the translation
    current_EE_velocity = controller.get_EE_velocity()
    current_EE_wrench = controller.get_EE_wrench() # This is in the world frame but can be easily transformed with
        # the current_EE_transform

    # ic(current_EE_transform, current_EE_velocity, current_EE_wrench)
    #set_joint_config_linear(controller, json.load(open(start_position_file, "r")))
    #controller.set_joint_config()
    #time.sleep(2)

    PegInHoleTask(controller, start_position_file, '', task_name="wipe", task_args={"is_randomizing_force":False})

    # control the robot
    # current_config[3] += 0.2
    # controller.set_joint_config(current_config,{})
    # time.sleep(2)

    # send a sinw wave motion to joint 5
    control_params = {} #can alternatively specify different params
    # current_time = 0.
    # while current_time < 3:
    #     target_q = copy.copy(current_config)
    #     target_q[4] += math.sin(current_time)*0.1
    #     controller.set_joint_config(target_q, control_params)
    #     time.sleep(0.01)
    #     current_time += 0.01
    # time.sleep(1)

    # # Set EE transform, in the world frame
    #current_EE_transform = controller.get_EE_transform(tool_center = se3.identity()) #transform of tool center,
    # current_EE_transform[1][2] += 0.02 #move up 2 cm
    #controller.set_EE_transform(current_EE_transform, control_params)
    #time.sleep(10)

    # Set EE velocity, in the world frame
    #controller.set_EE_velocity([0,0,-0.01,0,0,0], control_params)
    #time.sleep(5)
    # controller.set_EE_velocity([0,0,0,0.05,0,0], control_params)
    # time.sleep(5)

    # shutdown the robot
    #controller.close()
