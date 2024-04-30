from klampt.model.trajectory import Trajectory, RobotTrajectory
import matplotlib.pyplot as plt
import yaml
import numpy as np
from klampt.math import vectorops as vo
import math
from icecream import ic

def visualize_trajectory(forcePositionTrajectory):
    dt = 1
    size_scale = 20
    times = np.arange(0, forcePositionTrajectory.total_time(), dt)
    ic(forcePositionTrajectory.total_time())
    for t in times:
        force, position, heading = forcePositionTrajectory.get_force_position_heading(t)
        #print(position)
        plt.scatter(position[0], position[1], c = 'b', s = max(0.001,force[0])*size_scale)
    plt.axis('equal')
    plt.show()

class forcePositionTrajectory:
    def __init__(self, forces, positions, max_speed, max_rot_speed, xy_offset, scale = 0.3) -> None:
        assert len(forces) == len(positions)
        times = [0]
        heading_vector = np.array(positions[1]) - np.array(positions[0])
        initial_heading = math.atan2(heading_vector[1], heading_vector[0])
        headings = [[initial_heading]]
        current_time = 0
        for i in np.arange(1, len(forces), 1):
            translation_time = \
                np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))/(max_speed*scale)
            if i == 1:
                rotation_time = 0
                last_heading = initial_heading
            else:
                heading_vector = np.array(positions[i]) - np.array(positions[i-1])
                heading = math.atan2(heading_vector[1], heading_vector[0])
                rotation_time = math.fabs(heading - last_heading)/(max_rot_speed*scale)
                last_heading = heading
            headings.append([last_heading])
            ic(rotation_time, translation_time)
            current_time += max(rotation_time, translation_time)
            times.append(current_time)
        ic(headings)
        offset_positions = []
        for pos in positions:
            offset_positions.append(vo.add(pos, xy_offset))
        self.force_trajectory = Trajectory(times, forces)
        self.position_trajectory = Trajectory(times, offset_positions)
        self.heading_trajectory = Trajectory(times, headings)

    def get_force_position_heading(self, t):
        return self.force_trajectory.eval(t), self.position_trajectory.eval(t),\
            self.heading_trajectory.eval(t)

    def total_time(self):
        return self.force_trajectory.endTime()

if __name__=='__main__':
    # forces = [[0.5], [0.3], [0.2], [0.1], [0.0], [-0.1],[-0.4]]
    # positions = [[0.,0.],[0.0,0.03],[0.0075,0.06], [0.015,0.09],\
    #              [0.03,0.12], [0.04, 0.135], [0.041, 0.14]]

    forces = [[0.1], [0.1], [0.2], [0.5], [0.3], [0.0],[-0.3]]
    positions = [[0.,0.],[0.0,0.03],[-0.01,0.06], [-0.02,0.09],\
                 [-0.04,0.12], [-0.07, 0.13], [-0.071, 0.13]]

    # forces = []
    # positions = []
    # for i in np.arange(0,3,0.01):
    #     forces.append([i+0.1])
    #     positions.append([math.cos(i)*10, math.sin(i)*10])
    #     print([math.cos(i)*10, math.sin(i)*10])
    # print(forces, positions)
    forcePositionTrajectory = forcePositionTrajectory(forces, positions, max_speed = 0.00125, max_rot_speed = 0.01,\
                                            xy_offset=[0,0])
    

    print(forcePositionTrajectory.get_force_position_heading(0))
    visualize_trajectory(forcePositionTrajectory)

    # with open('data.yml', 'w') as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False)


