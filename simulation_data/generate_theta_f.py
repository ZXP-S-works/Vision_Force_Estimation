import matplotlib.pyplot as plt
import numpy as np
from simulation_data.theta_force_func import ThetaForceFunc
plt.rc('text', usetex=True)  # Enable LaTeX rendering

if __name__ == '__main__':

    error = np.deg2rad(1)
    f_errors = []
    fs = []
    pos = []
    resolution = 100
    df = np.zeros(resolution)
    theta1 = np.deg2rad(45)

    d1s = np.linspace(0, 0.45, resolution)

    _, axe0 = plt.subplots(1, 1)
    _, axe1 = plt.subplots(1, 1)
    _, axe2 = plt.subplots(1, 1)
    for theta2 in [0, 15, 30, 45, 60, 75, 90]:
        theta2 = np.deg2rad(theta2)
        for idx in range(resolution):
            d1 = d1s[idx] * np.pi
            cfg = ThetaForceFunc(theta1, theta2, d1)
            # Solve the implicit function
            f0 = cfg.f()

            d1 += error
            cfg = ThetaForceFunc(theta1, theta2, d1)
            # Solve the implicit function
            f1 = cfg.f()

            f_error = np.linalg.norm(f0 - f1)

            fs.append(f0)
            pos.append(cfg.position())
            f_errors.append(f_error)

        fs = np.asarray(fs)
        axe0.plot(fs[:, 0], fs[:, 1], label=r'$\theta_2=$'+str(np.rint(np.rad2deg(theta2))))
        fs = []

        pos = np.asarray(pos)
        axe1.plot(pos[:, 0], pos[:, 1], label=r'$\theta_2=$'+str(np.rint(np.rad2deg(theta2))))
        pos = []

        f_errors = np.asarray(f_errors)
        axe2.plot(d1s, f_errors, label=r'$\theta_2=$'+str(np.rint(np.rad2deg(theta2))))
        f_errors = []

    axe0.set_title('force')
    axe0.set_xlabel('x (N)')
    axe0.set_ylabel('y (N)')
    axe1.set_title('position of the tip')
    axe1.set_xlabel('x (m)')
    axe1.set_ylabel('y (m)')
    axe2.set_title('force error')
    axe2.set_xlabel(r'$d\theta_1$ ($\pi$)')
    axe2.set_ylabel('N')
    # plt.figure()
    # plt.plot(df)
    #
    # fs = np.asarray(fs)
    # plt.figure()
    # plt.scatter(fs[:, 0], fs[:, 1])

    axe0.legend()
    axe1.legend()
    axe2.legend()

    axe2.set_yscale('log')
    plt.show()
    # f_errors = np.asarray(f_errors).mean()
    # fs = np.linalg.norm(np.asarray(fs), axis=1).mean()
    # print('f_errors ', f_errors)
    # print('fs ', fs)
    # print('f error percentage ', f_errors / fs * 100)
