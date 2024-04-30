from FT_client import FTClient
from ur_controller.ur5.ur5_wrapper import ur5Wrapper
import math, time, json
from real_world_utils import set_EE_transform_linear, set_joint_config_linear, rotate_EE, gravityCompensator
from icecream import ic
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle, os
import threading
from manual_collect_data import record_img_f_raw


def collect_data(save_folder = './'):
    start_position_file = "../ur5_start_position.json"
    gravity = 6.0
    max_joint_v = 0.1

    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = '192.168.0.101')
    # start robot
    controller.initialize()
    controller.start()

    set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
    time.sleep(2)

    ft_driver = FTClient('http://192.168.0.103:80')
    ft_driver.zero_ft_sensor()
    ft_driver.start_ft_sensor()
    time.sleep(1)


    N_points = 121
    angles = np.linspace(-40/180*math.pi, 40/180*math.pi, N_points)
    #total_angle = 0
    all_forces = []
    current_angle = 0
    for angle in angles:
        ic(angle)
        delta_angle = angle - current_angle
        rotate_EE(controller, delta_angle)
        time.sleep(0.5)
        current_angle = angle
        # total_angle += angle

        total_force = np.zeros(3)
        N_avg = 20
        for _ in range(N_avg):
            ft = ft_driver.read_ft_sensor()
            force = ft[0:3]
            total_force += np.array(force)
            time.sleep(0.001)
        avg_force =total_force/N_avg
        ic(avg_force)
        all_forces.append(avg_force)
        #Fz_compensated = gravity*(1 - math.cos(total_angle))
        # ic(total_angle, 1 - math.cos(total_angle))
        #Fy_compensated = gravity*math.sin(total_angle)
        # print(ft[0:3], Fy_compensated, Fz_compensated)


    controller.close()
    # ft_driver = FTClient('http://192.168.0.103:8080')
    # ft_driver.zero_ft_sensor()
    # ft_driver.start_ft_sensor()

    np.save(save_folder + 'fixed_peg_Y', np.array(all_forces))
    np.save(save_folder + 'fixed_peg_X', angles)

    return 

def test_compensation():
    start_position_file = "../ur5_start_position.json"
    gravity = 6.0
    max_joint_v = 0.1

    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = '192.168.0.101')
    # start robot
    controller.initialize()
    controller.start()

    set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
    time.sleep(2)


    ft_driver = FTClient('http://192.168.0.103:80')
    ft_driver.zero_ft_sensor()
    ft_driver.start_ft_sensor()
    time.sleep(1)

    # gravity_compensator = gravityCompensator()
    # rotate_EE(controller, 20/180*math.pi)
    # time.sleep(0.5)

    # print(gravity_compensator.get_compensated_force(ft = ft_driver.read_ft_sensor(), controller=controller))

    # controller.close()

    save_dir = './compensation_test/'
    os.makedirs(save_dir, exist_ok=True) 
    stop_event = threading.Event()
    stop_lock = threading.Lock()
    pause_event = threading.Event()
    pause_lock = threading.Lock()
    record = threading.Thread(target=record_img_f_raw,
            kwargs={"save_dir": save_dir,
                    "stop_event": stop_event, "stop_lock": stop_lock,
                    "pause_event": pause_event, "pause_lock": pause_lock,
                    "Hz": 10, "controller":controller}) 

    with pause_lock:
        pause_event.clear()
    record.start()

    rotate_EE(controller, 20/180*math.pi)
    time.sleep(0.2)
    rotate_EE(controller, -20/180*math.pi)
    time.sleep(0.2)
    rotate_EE(controller, -20/180*math.pi)

    with stop_lock:
        stop_event.set()

    controller.close()

def train_GPR(X,Y,x_plot):
    X = np.expand_dims(X,-1)
    Y = np.expand_dims(Y,-1)
    x_plot = np.expand_dims(x_plot, -1)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
    kernel = 1.0 * RBF(length_scale=0.01, length_scale_bounds=(1e-3, 10)) + WhiteKernel(
        noise_level=1, noise_level_bounds=(1e-5, 1e-1)
        )
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gpr.fit(X, Y)
    y_mean, y_std = gpr.predict(x_plot, return_std=True)
    plt.plot(X, Y, label="Expected signal")
    plt.scatter(x=X[:], y=Y, color="black", alpha=0.4, label="Observations")
    plt.errorbar(x_plot, y_mean, y_std)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    _ = plt.title(
        (
            f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
            f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
        ),
        fontsize=8,
    )
    plt.show()
    return gpr

def train_model(save_folder = './'):
    X = np.load(save_folder + 'fixed_peg_X.npy')
    Y = np.load(save_folder + 'fixed_peg_Y.npy')
    x_plot = angles = np.linspace(-40/180*math.pi, 40/180*math.pi, 300)
    Fy_gpr = train_GPR(X,Y[:,1],x_plot)
    Fz_gpr = train_GPR(X,Y[:,2],x_plot)

    with open('Fy_gpr.pkl','wb') as f:
        pickle.dump(Fy_gpr,f)

    with open('Fz_gpr.pkl','wb') as f:
        pickle.dump(Fz_gpr,f) 

    return 

def calibrate_rotation():
    start_position_file = "../ur5_start_position_high.json"
    gravity = 6.0
    max_joint_v = 0.1

    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = '192.168.0.101')
    # start robot
    controller.initialize()
    controller.start()

    set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
    time.sleep(2)

    # rotate_EE(controller, -20/180*math.pi)
    # ic(20/180*math.pi)
    # time.sleep(0.5)
    # T_EE = controller.get_EE_transform()
    # Rx = T_EE[0][0:3]
    # ic(Rx)
    # angle = math.atan2(Rx[1], -Rx[2])
    # ic(angle)

    

    print(controller.get_EE_transform())

    R = [0,0,-1, math.sin(45/180*math.pi), math.sin(45/180*math.pi), 0, math.sin(45/180*math.pi), -math.sin(45/180*math.pi), 0]

    set_EE_transform_linear(controller, (R, controller.get_EE_transform()[1]))
    time.sleep(2)

    print(controller.get_EE_transform())
    print(controller.get_joint_config())

    controller.close()

def random_test():
    start_position_file = "../ur5_start_position.json"
    gravity = 6.0
    max_joint_v = 0.1

    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = '192.168.0.101')
    # start robot
    controller.initialize()
    controller.start()

    set_joint_config_linear(controller, json.load(open(start_position_file, "r")), max_joint_v)
    time.sleep(2)

    rotate_EE(controller, 20/180*math.pi)
    time.sleep(0.5)

    controller.close()

    
if __name__=='__main__':
    # collect_data()
    # train_model()
    # test_compensation()
    # calibrate_rotation()
    # random_test()
    start_position_file = "../ur5_start_position.json"
    controller = ur5Wrapper(world_fn = 'ur_controller/data/ur5_with_gripper_world.xml', address = '192.168.0.101')
    # start robot
    controller.initialize()
    controller.start()
    # R = [0,0,-1, math.sin(45/180*math.pi), math.sin(45/180*math.pi), 0, math.sin(45/180*math.pi), -math.sin(45/180*math.pi), 0]
    # t = controller.get_EE_transform()[1]
    # print(controller.get_joint_config())
    # target_t = [t[0],t[1],t[2]+0.02]
    # set_EE_transform_linear(controller, (R,target_t), 0.005)
    # time.sleep(1)

    # from CONSTANTS import max_joint_v
    set_joint_config_linear(controller, json.load(open(start_position_file, "r")), 0.1)
    time.sleep(2)
    print(controller.get_EE_transform())
    controller.close()