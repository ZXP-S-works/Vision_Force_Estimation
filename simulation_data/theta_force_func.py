from matplotlib import pyplot as plt
from scipy.optimize import root
import numpy as np


class ThetaForceFunc:
    def __init__(self, theta1, theta2, l1=0.1, l2=0.03):
        # finger shape
        self.k1, self.k2 = 0.07, 0.07
        # jacobian from https://asmorgan24.github.io/Files/Other/Papers/Yale/Morgan_TRO2021.pdf
        Ja = l1 * np.sin(theta1)
        Jb = l2 * np.sin(theta1 + theta2)
        Jd = l1 * np.cos(theta1)
        Je = l2 * np.cos(theta1 + theta2)
        self.J = np.asarray([[-Ja - Jb, -Jb],
                             [Jd + Je, Je]])

    def force(self, d1, d2):
        """given current finger configuration, return the contact force"""
        tao = np.asarray([self.k1 * d1, self.k2 * d2])
        fc = np.matmul(np.linalg.inv(self.J.T), tao.reshape(2, 1))
        return fc

    def omega(self):
        JTJ = np.matmul(self.J.T, self.J)
        w = np.linalg.det(JTJ)
        w = np.sqrt(w)
        return w

    def dtheta(self, fc):
        """given initial finger configuration, apply a small force, return the joint displacement"""
        tao = np.matmul(self.J.T, fc)
        d1 = tao[0] / self.k1
        d2 = tao[1] / self.k2
        return d1, d2


if __name__ == '__main__':

    f = 0.01
    resolution = 200
    l_total = 0.1
    l2s, theta2s = np.zeros((resolution * resolution)), np.zeros((resolution * resolution))
    omegas = np.zeros((resolution * resolution))
    theta1 = np.deg2rad(0)

    for row in np.arange(0, resolution):
        for col in np.arange(0, resolution):
            theta2 = col / resolution
            theta2 *= np.pi
            ratio = row / resolution + 0.1  # 0.5 to 1.5
            l1 = l_total * (0.35 / (0.35 + ratio))
            l2 = l_total - l1
            cfg = ThetaForceFunc(theta1, theta2, l1, l2)
            l2s[row * resolution + col] = l2
            theta2s[row * resolution + col] = theta2
            omegas[row * resolution + col] = cfg.omega()

    plt.figure()
    x = l2s * np.cos(theta2s)
    y = l2s * np.sin(theta2s)
    plt.scatter(x, y, c=omegas)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.title(r'$\omega$')
    plt.colorbar()
    plt.show()
