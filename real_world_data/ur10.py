import rtde_receive
import rtde_control
import numpy as np

J_HOME = [1.30, -2.01, 2.58, -2.14, -1.58, -0.277]
J_MID_TO_HOME = [1.41, -1.52, 2.20, -2.26, -1.58, -0.17]
J_ABOVE_WORKSPACE = [1.46, -1.09, 1.63, -2.12, -1.58, -0.12]
C_WORKSPACE_CENTER = [0.076, -0.899, 0.1, 0, -3.142, 0]
TRANS_RANGE = 0.01
ROT_RANGE = np.deg2rad(30)


if __name__ == '__main__':
    ip = "192.168.1.100"  # IP for the UR10e robot
    rtde_c = rtde_control.RTDEControlInterface(ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)

    # d = rtde_r.getActualTCPPose()  # get current pose
    # d[2] += 0.05  # add 0.1m to z axis translation
    # rtde_c.moveL(d)  # move to new pose
    # j = rtde_r.getActualQ()

    # Move from home to above_workspace
    j = rtde_r.getActualQ()
    err = np.linalg.norm(np.asarray(j) - np.asarray(J_HOME))
    if err > 0.3:
        raise NotImplementedError('Robot is too far from home!')
    rtde_c.moveJ(J_HOME)
    rtde_c.moveJ(J_MID_TO_HOME)
    rtde_c.moveJ(J_ABOVE_WORKSPACE)
    # Move to workspace
    rtde_c.moveL(C_WORKSPACE_CENTER)
    # Randomly move within workspace, i.e., CENTER -+TRANS_RANG, -+ROT_RANGE
    for i in range(10):
        next_pos = np.asarray(C_WORKSPACE_CENTER)
        next_pos[0] += np.random.uniform(-TRANS_RANGE, TRANS_RANGE)
        next_pos[2] += np.random.uniform(-TRANS_RANGE, TRANS_RANGE)
        next_pos[4] += np.random.uniform(-ROT_RANGE, ROT_RANGE)
        rtde_c.moveL(next_pos.tolist())
    # Move to home
    rtde_c.moveJ(J_ABOVE_WORKSPACE)
    rtde_c.moveJ(J_MID_TO_HOME)
    rtde_c.moveJ(J_HOME)

