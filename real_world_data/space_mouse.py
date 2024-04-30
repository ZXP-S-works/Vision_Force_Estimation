# Copyright (c) 2023 Boston Dynamics AI Institute, Inc.
# All rights reserved.

""" Script to teleoperate the UR Robots"""
import pyspacemouse
import rtde_control, rtde_receive
import argparse
import time
import IPython

TRANSLATION_SPEED_COEFFICIENT = 0.10  # maximum this value for m/s
ORIENTATION_SPEED_COEFFICIENT = 0.06  # maximum this value for rad/s
DEFAULT_MAX_ACCELERATION = 0.35  # a bit above the UR default of 0.25

DEBUG = False


# Connect with robot and control in velocity mode
class URRobotController():

    def __init__(self, ip='192.178.1.50'):
        self.ip = ip
        self.controller = rtde_control.RTDEControlInterface(self.ip)
        self.receiver = rtde_receive.RTDEReceiveInterface(self.ip)

    def velocityControl(self, velocities=[0, 0, 0, 0, 0, 0]):
        self.controller.speedL(velocities, DEFAULT_MAX_ACCELERATION, 0.1)


# Building this class within script for documentation purposes
class SpaceNavState():

    def __init__(self, SpaceNavReading):
        # timespamp last button was pressed
        self.time = SpaceNavReading.t

        # x, y, z traslation of the spacemouse
        self.x = SpaceNavReading.x
        self.y = SpaceNavReading.y
        self.z = SpaceNavReading.z

        # roll, pitch, yaw rotation of the space mouse
        self.roll = SpaceNavReading.roll
        self.pitch = SpaceNavReading.pitch
        self.yaw = SpaceNavReading.yaw

        # [list] of whether or not buttons on mouse are pressed
        self.buttons = SpaceNavReading.buttons


parser = argparse.ArgumentParser(description='Teleoperation interface to control the robot and pinch gripper.')
parser.add_argument('--ip_address', '-i', type=str, help='IP address of the robot', default='192.168.1.100')
parser.add_argument('--sleep_time', type=float, help='Sleep between each robot update', default=0.002)
parser.add_argument('--disable_rotation', '-r', default=False, action='store_true',
                    help='Make the robot only translate and not rotate.')
parser.add_argument('--disable_translation', '-t', default=False, action='store_true',
                    help='Make the robot only rotate and not translate.')
args = parser.parse_args()

if __name__ == "__main__":

    ip = args.ip_address

    success = pyspacemouse.open()
    robot = URRobotController(ip)

    print('[LOG] Robot controller up and running... begin space navigator input.')

    if success:
        try:
            while True:
                s = pyspacemouse.read()
                state = SpaceNavState(s)

                # Setup velocity vector for control
                v = []
                # Determine translation velocity
                if args.disable_translation:
                    v.extend([0.0, 0.0, 0.0])
                else:
                    v.extend([state.x * TRANSLATION_SPEED_COEFFICIENT, state.y * TRANSLATION_SPEED_COEFFICIENT,
                              state.z * TRANSLATION_SPEED_COEFFICIENT])

                # Determine orientation velocity
                if args.disable_rotation:
                    v.extend([0.0, 0.0, 0.0])
                else:
                    v.extend([-state.pitch * ORIENTATION_SPEED_COEFFICIENT * 7,
                              state.roll * ORIENTATION_SPEED_COEFFICIENT * 7,
                              -state.yaw * ORIENTATION_SPEED_COEFFICIENT * 7])
                    # v.extend([state.roll*ORIENTATION_SPEED_COEFFICIENT*7,
                    # state.pitch*ORIENTATION_SPEED_COEFFICIENT*7, state.yaw*ORIENTATION_SPEED_COEFFICIENT*7])

                if DEBUG:
                    print(f"Velocities: {v}")
                    IPython.embed()

                robot.velocityControl(v)
                time.sleep(args.sleep_time)

        except KeyboardInterrupt:
            pass
    else:
        print(
            '[ERR] failure to acqure space mouse link. Ensure you have an open port, e.g. <sudo chmod 666 /dev/hidraw8>')
