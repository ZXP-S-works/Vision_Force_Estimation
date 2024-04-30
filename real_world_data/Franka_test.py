from Franka_client import FrankaClient  
import time, math

params = {'impedance': [6000, 6000, 6000, 5000, 5000, 4000, 4000]} #{'impedance': [3000, 3000, 3000, 2500, 2500, 2000, 2000]}
arm_driver = FrankaClient('http://172.16.0.1:8080')
arm_driver.initialize()
arm_driver.start()

print(arm_driver.get_joint_config())
print(arm_driver.get_EE_transform())

ang = 45/180*math.pi
arm_driver.set_EE_transform([[math.sin(ang), -math.sin(ang), 0, \
        -math.sin(ang), -math.sin(ang), 0, \
        0, 0, -1], \
        [0.4, 0, 0.4]])
time.sleep(5)
print(arm_driver.get_joint_config())
arm_driver.shutdown()

