import matplotlib.pyplot as plt
from scipy.optimize import root
import numpy as np
from simulation_data.theta_force_func import ThetaForceFunc


if __name__ == '__main__':

    f = 0.01
    resolution = 200
    d1s, d2s = np.zeros((resolution, resolution)), np.zeros((resolution, resolution))
    theta1 = np.deg2rad(0)

    for row in np.arange(0, resolution):
        for col in np.arange(0, resolution):
            theta2 = col / resolution - 0.5
            theta2 *= 2 * np.pi
            theta_f = row / resolution - 0.5
            theta_f *= 2 * np.pi
            fc = np.asarray([f * np.cos(theta_f), f * np.sin(theta_f)])
            cfg = ThetaForceFunc(theta1, theta2)
            d1s[row, col], d2s[row, col] = cfg.dtheta(fc)

    d1s, d2s = np.abs(np.flip(d1s, axis=0)) / f, np.abs(np.flip(d2s, axis=0)) / f

    plt.figure()
    plt.imshow(d1s)
    plt.xlabel(r'$\theta2$')
    plt.xticks([0, resolution / 2, resolution], [r'$-\pi$', r'$0$', r'$\pi$'])
    plt.ylabel(r'$\theta_1^F$')
    plt.yticks([0, resolution / 2, resolution], [r'$\pi$', r'$0$', r'$-\pi$'])
    plt.title(r'$|d\theta_1/df|$')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(d2s)
    plt.xlabel(r'$\theta2$')
    plt.xticks([0, resolution / 2, resolution], [r'$-\pi$', r'$0$', r'$\pi$'])
    plt.ylabel(r'$\theta_1^F$')
    plt.yticks([0, resolution / 2, resolution], [r'$\pi$', r'$0$', r'$-\pi$'])
    plt.title(r'$|d\theta_2/df|$')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(d1s + d2s)
    plt.xlabel(r'$\theta2$')
    plt.xticks([0, resolution / 2, resolution], [r'$-\pi$', r'$0$', r'$\pi$'])
    plt.ylabel(r'$\theta_1^F$')
    plt.yticks([0, resolution / 2, resolution], [r'$\pi$', r'$0$', r'$-\pi$'])
    plt.title(r'$|d\theta_1/df| + |d\theta_2/df|$')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(d1s * d2s)
    plt.xlabel(r'$\theta2$')
    plt.xticks([0, resolution / 2, resolution], [r'$-\pi$', r'$0$', r'$\pi$'])
    plt.ylabel(r'$\theta_1^F$')
    plt.yticks([0, resolution / 2, resolution], [r'$\pi$', r'$0$', r'$-\pi$'])
    plt.title(r'$|d\theta_1/df| * |d\theta_2/df|$')
    plt.colorbar()
    plt.show()

    plt.figure()
    for theta2 in [0, 30, 60, 90]:
        plt.plot((d1s + d2s)[-1:0:-1, np.rint((theta2 / 180 + 1) / 2 * resolution).astype(int)].reshape(-1),
                 label=r'$\theta_2=$' + str(theta2))
    plt.ylabel(r'$|d\theta_1/df| + |d\theta_2/df|$')
    plt.xlabel(r'$\theta_1^F$')
    plt.xticks([0, resolution / 2, resolution], [r'$-\pi$', r'$0$', r'$\pi$'])
    plt.legend()
    plt.show()

    plt.figure()
    for theta2 in [0, 30, 60, 90]:
        plt.plot((d1s * d2s)[-1:0:-1, np.rint((theta2 / 180 + 1) / 2 * resolution).astype(int)].reshape(-1),
                 label=r'$\theta_2=$' + str(theta2))
    plt.ylabel(r'$|d\theta_1/df| * |d\theta_2/df|$')
    plt.xlabel(r'$\theta_1^F$')
    plt.xticks([0, resolution / 2, resolution], [r'$-\pi$', r'$0$', r'$\pi$'])
    plt.legend()
    plt.show()
