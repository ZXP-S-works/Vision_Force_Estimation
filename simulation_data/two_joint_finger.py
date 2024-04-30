import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from simulation_data.theta_force_func import ThetaForceFunc

# WORKSPACE = np.asarray([[0, 0.15],
#                         [0, 0.15],
#                         [0, 0.05]])
WORKSPACE = np.asarray([[0., 0.2],
                        [0., 0.2],
                        [0, 0.05]])
IMG_SIZE = 128
PIX_SIZE = (WORKSPACE[:2, 1] - WORKSPACE[:2, 0]) / IMG_SIZE
DEBUG = False


# Notes: I find that I cannot set joint_stiffness > 1. Otherwise, it leads to 0 force reading.
#        Even with the same location of the peg, the force reading/ finger configuration is different. (Stochastic?)


class Finger:
    def __init__(self):
        # self.base_pos = [0.0, 0.0, 0.05]
        # self.base_orn = [1.57, 0, 2.355]
        self.base_pos = [WORKSPACE[0, 1], 0.05, 0.05]
        self.base_orn = [1.57, 0, 4.71]

        self.tool_link = 0
        self.max_force = 1000
        self.joint_stiffness = 0.06

        self.id = p.loadURDF("./ohp_model_t_model/urdf/two_link_finger_mesh.urdf",
                             basePosition=self.base_pos,
                             baseOrientation=p.getQuaternionFromEuler(self.base_orn),
                             useFixedBase=True
                             )

        # disable velocity controller so that we can use torque control
        for idx in range(2):
            p.setJointMotorControl2(self.id, idx, p.VELOCITY_CONTROL, force=0)

    def update_joint(self):
        joint_reading = []
        for idx in range(2):
            joint_reading.append(self.spring_joint(idx))
        return np.asarray(joint_reading)

    def position_joint(self, joint_readings:list):
        joint_reading = []
        for idx in range(2):
            p.setJointMotorControl2(self.id, idx, p.POSITION_CONTROL, targetPosition=joint_readings[idx], force=10)
            joint_reading.append(p.getJointState(self.id, idx)[0])
        return np.asarray(joint_reading)

    def spring_joint(self, idx):
        # joint_reading = p.getJointState(self.id, idx)[0] % (2 * np.pi)
        # joint_reading = joint_reading if joint_reading < np.pi else joint_reading - 2 * np.pi
        joint_reading = p.getJointState(self.id, idx)[0]
        spring_torque = -joint_reading * self.joint_stiffness
        # print(idx, p.getJointState(self.id, idx)[0])
        p.setJointMotorControl2(self.id, idx, p.TORQUE_CONTROL, force=spring_torque)
        return joint_reading


class Sensor:
    def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far, sensor_type='rgb'):
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraUpVector=cam_up_vector,
            cameraTargetPosition=target_pos,
        )

        self.sensor_type = sensor_type  # for depth only; choice: rgb.
        assert self.sensor_type == 'rgb'
        self.near = near
        self.far = far
        self.fov = np.degrees(2 * np.arctan((target_size / 2) / self.far))
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, 1, self.near, self.far)

    def normalize_256(self, img, channel_normalize):
        """

      :param img: in shape b x 3 x h x w
      :return:
      """
        img = img / 255
        img -= 0.5
        img = img / 10 if channel_normalize else img
        return img

    def gray_scal(self, img):
        img[:] = np.expand_dims(img[:].mean(0), 0)
        return img

    def getImage(self, size):
        # renderer = pb.ER_TINY_RENDERER if self.sensor_type == 'd' else pb.ER_BULLET_HARDWARE_OPENGL
        renderer = p.ER_TINY_RENDERER
        image_arr = p.getCameraImage(width=size, height=size,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     renderer=renderer)
        depth_img = np.array(image_arr[3])
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)
        depth_img = np.abs(depth - np.max(depth)).reshape(size, size)
        if self.sensor_type == 'd':
            obs = depth_img
        elif self.sensor_type in ['rgb']:
            obs = np.array(image_arr[2]).astype(np.uint8)
            obs = np.moveaxis(obs, -1, 0)[:-1]
        else:
            raise NotImplementedError
        return obs


def create_sliders(hrange=0.035):
    motorsIds = []
    for idx, param in enumerate(['posX', 'posY']):
        motorsIds.append(p.addUserDebugParameter(param, -hrange, hrange, 0))
    return motorsIds


def read_sliders(sliders):
    values = []
    for sider in sliders:
        values.append(p.readUserDebugParameter(sider))
    return values


class Cylinder:
    def __init__(self):
        self.anchor = [0.1, 0.1, 0]
        self.pose = self.anchor.copy()
        # self.pose[1] += 0 if DEBUG else np.random.uniform(0, 0.04)
        self.orie = p.getQuaternionFromEuler([0, 0, 0])
        self.id = p.loadURDF("./urdf/cylinder.urdf",
                             basePosition=self.pose,
                             baseOrientation=self.orie,
                             useFixedBase=False
                             )
        self.max_force = 1000

        self.base_constraint = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.pose,
            childFrameOrientation=self.orie
        )

    def apply_action(self, u):
        new_pos = self.anchor.copy()
        new_pos[0] += u[0]
        new_pos[1] += u[1]
        p.changeConstraint(self.base_constraint, jointChildPivot=new_pos, maxForce=self.max_force)

    # def apply_action(self, u):
    #     target_pose = np.asarray(self.anchor.copy())
    #     target_pose[:2] += np.asarray(u)
    #     current_pose = np.asarray(p.getConstraintInfo(self.base_constraint)[7])
    #     delta = target_pose - current_pose
    #     max_vel = 0.0001
    #     delta = delta.clip(-max_vel, max_vel)
    #     new_pos = current_pose + delta
    #     p.changeConstraint(self.base_constraint, jointChildPivot=new_pos.tolist(), maxForce=self.max_force)


def get_contact_force(bodyA, bodyB):
    contacts = p.getContactPoints(bodyA, bodyB)
    force = []
    position = []
    for contact in contacts:
        #        for i in range(len(contact)):
        #            print(i, contact[i])
        position.append((np.asarray(contact[5]) + np.asarray(contact[6])) / 2)
        normal_force = contact[9] * np.asarray(contact[7])
        friction_force = contact[10] * np.asarray(contact[11]) + contact[12] * np.asarray(contact[13])
        force.append(normal_force + friction_force)
    #        print('friction', friction_force)
    force = np.asarray(force).sum(0)
    position = np.asarray(position).mean(0)
    return force, position


def setup_camera(workspace, sensor_type='rgb'):
    # Setup camera
    ws_size = max(workspace[0][1] - workspace[0][0], workspace[1][1] - workspace[1][0])
    cam_pos = [workspace[0].mean(), workspace[1].mean(), 10]
    target_pos = [workspace[0].mean(), workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    sensor = Sensor(cam_pos, cam_up_vector, target_pos, ws_size,
                    cam_pos[2] - 1, cam_pos[2], sensor_type=sensor_type)
    return sensor.getImage


def descretize_pose(x):
    return (x[:2] - WORKSPACE[:2, 0]) // PIX_SIZE


def visualize_data(img, x, f):
    plt.figure()
    plt.imshow(img[:3].transpose(1, 2, 0) + 0.5)
    plt.title('F = ' + str(np.round(np.sqrt(f[0] ** 2 + f[1] ** 2), 2)) + 'N')
    plt.quiver(x[1], x[0], f[1], f[0], scale=10, color='r', angles='xy')
    plt.show()


if __name__ == '__main__':

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    # p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    robot = Finger()
    sliders = create_sliders()
    peg = Cylinder()
    camera = setup_camera(WORKSPACE)
    # us = np.asarray([[0, 0],
    #                 [-0.035, 0.035]])
    joint_readings = []
    fs = []
    f_from_calculations = []
    length = 100

    for i in range(length):
        u = i / length * np.asarray([-0.035, 0.035])
        # u = us[i % 2]

        for j in range(1):
            for _ in range(500):
                if DEBUG:
                    u = read_sliders(sliders)
                peg.apply_action(u)
                joint_reading = robot.update_joint()
                p.stepSimulation()
            img = camera(IMG_SIZE)
            force, position = get_contact_force(robot.id, peg.id)
            if not np.isnan(position).all():
                f = np.round(np.sqrt(force[0] ** 2 + force[1] ** 2), 2)
                fs.append(np.asarray([f, force[0], force[1]]))
                joint_readings.append(joint_reading)
                print('---------------------')
                print('F = ' + str(f) + 'N')
                print('pose', position)
                des_pose = descretize_pose(position)
                # visualize_data(img, des_pose, force)
                cfg = ThetaForceFunc(np.deg2rad(-15), np.deg2rad(-45), joint_reading[0], joint_reading[1])
                cfg.initial_guess = np.asarray([0., 0.])
                f_from_calculation = cfg.f()
                f_from_calculations.append(np.asarray([np.linalg.norm(f_from_calculation), f_from_calculation[0], f_from_calculation[1]]))
    p.disconnect()

    fs = np.asarray(fs)
    f_from_calculations = np.asarray(f_from_calculations)
    joint_readings = np.asarray(joint_readings)
    plt.figure()
    plt.plot(fs[:], label='force')
    plt.plot(f_from_calculations[:], label='calculation')
    plt.plot(joint_readings[:, 0], label='joint 0')
    plt.plot(joint_readings[:, 1], label='joint 1')
    plt.legend()
    plt.xlabel('simulation step')
    plt.ylabel('N or radius')
    plt.show()
