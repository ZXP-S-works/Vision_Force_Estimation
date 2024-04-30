import argparse

import matplotlib.pyplot as plt
import numpy.random as npr
from two_joint_finger import *
from utils.dataset import Dataset, ImgForce
from tqdm import tqdm
from utils.parameters import strToBool


def pars_args():

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset_size', type=float, default=5e3)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='../data/')
    parser.add_argument('--debug', type=strToBool, default=False)
    parser.add_argument('--estimation_type', type=str, default='force',
                        choices=['force', 'joints', 'calculated_force'])

    return parser.parse_args()


if __name__ == '__main__':
    args = pars_args()
    physicsClient = p.connect(p.DIRECT)  # p.DIRECT for non-graphical version
    # physicsClient = p.connect(p.GUI)  # p.GUI for graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = Finger()
    sliders = create_sliders(hrange=3.1415)
    camera = setup_camera(WORKSPACE)
    dataset = Dataset(args.dataset_size)
    num_img = 0
    pbar = tqdm(total=args.dataset_size)
    forces, joint_readings = [], []
    u0 = np.asarray([np.deg2rad(30), np.deg2rad(30)])
    # u0 = np.asarray([np.deg2rad(30), np.deg2rad(90)])

    while num_img < args.dataset_size:
        du = np.random.uniform([0, 0], [np.deg2rad(30), np.deg2rad(30)])
        # u += u0
        # u = u.tolist()
        # du = np.asarray([np.deg2rad(60), np.deg2rad(-45)])

        for _ in range(50):
            if args.debug:
                u = read_sliders(sliders)
            p.stepSimulation()
            joint_reading = robot.position_joint((du + u0).tolist())
        if args.estimation_type.find('calculated') != -1:
            cfg = ThetaForceFunc(joint_reading[0], joint_reading[1])
            calculated_force = cfg.force(du[0], du[1])
            if np.linalg.norm(calculated_force) > 10 or np.linalg.norm(calculated_force) < 0.01:
                continue
            if args.estimation_type == 'calculated_force':
                img = camera(args.img_size)
                data = ImgForce(img, joint_reading, calculated_force)
                forces.append(calculated_force)
                joint_readings.append(joint_reading)
            # if np.linalg.norm(calculated_force) > 30:
            # plt.figure()
            # plt.imshow(img.transpose(1,2,0))
            # plt.title(calculated_force)
            # plt.show()
        else:
            raise NotImplementedError
        dataset.add(data)
        num_img += 1
        pbar.update(1)
    p.disconnect()

    forces = np.asarray(forces)
    plt.figure()
    plt.scatter(forces[:, 0], forces[:, 1])
    plt.xlabel(r'$F_x$')
    plt.ylabel(r'$F_y$')
    plt.title('Force')
    plt.axis('equal')
    plt.show()
    joint_readings = np.asarray(joint_readings) * 180 / 3.1415
    plt.figure()
    plt.scatter(joint_readings[:, 0], joint_readings[:, 1], c=np.linalg.norm(forces, axis=1))
    plt.colorbar()
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title('force magnitude')
    plt.axis('equal')
    plt.show()
    plt.figure()
    plt.scatter(joint_readings[:, 0], joint_readings[:, 1], c=np.arctan2(forces[:, 1], forces[:, 0]) * 180 / 3.142)
    plt.colorbar()
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title('force orientation')
    plt.axis('equal')
    plt.show()

    dataset.save(args.save_dir, name='simulation_' + str(args.img_size) + '_I_theta_f_' + str(int(args.dataset_size)))
