import argparse
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
    physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # planeId = p.loadURDF("plane.urdf")
    robot = Finger()
    sliders = create_sliders()
    peg = Cylinder()
    camera = setup_camera(WORKSPACE)
    dataset = Dataset(args.dataset_size)
    num_img = 0
    pbar = tqdm(total=args.dataset_size)
    forces = []

    while num_img < args.dataset_size:
        u = np.random.uniform([-0.035, 0], [0, 0.035])

        for _ in range(50):
            if args.debug:
                u = read_sliders(sliders)
            peg.apply_action(u)
            p.stepSimulation()
            joint_reading = robot.update_joint()
        force, position = get_contact_force(robot.id, peg.id)
        force_magnitude = np.sqrt(force[0] ** 2 + force[1] ** 2) if not np.isnan(position).all() else 0
        if force_magnitude > 0.01:
            des_pose = descretize_pose(position)
            if args.estimation_type == 'force':
                img = camera(args.img_size)
                data = ImgForce(img, des_pose, force[:2])
            elif args.estimation_type == 'joints':
                img = camera(args.img_size)
                data = ImgForce(img, des_pose, joint_reading)
            elif args.estimation_type.find('calculated') != -1:
                cfg = ThetaForceFunc(np.deg2rad(-15), np.deg2rad(-45), joint_reading[0], joint_reading[1])
                cfg.initial_guess = np.asarray([0., 0.])
                calculated_force = cfg.f()
                if np.linalg.norm(calculated_force) > 10 or np.linalg.norm(calculated_force) < 0.05 ** 2:
                    continue
                if args.estimation_type == 'calculated_force':
                    forces.append(force)
                    img = camera(args.img_size)
                    data = ImgForce(img, joint_reading, calculated_force)
            else:
                raise NotImplementedError
            dataset.add(data)
            num_img += 1
            pbar.update(1)
    p.disconnect()

    forces = np.asarray(forces)
    plt.figure()
    plt.scatter(forces[:, 0], forces[:, 1])
    plt.show()

    dataset.save(args.save_dir, name='simulation_' + str(args.img_size) + '_' + str(int(args.dataset_size)))
