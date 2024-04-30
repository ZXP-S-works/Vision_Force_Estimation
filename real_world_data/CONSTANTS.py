import math
# t_pickup_high = [-0.485, -0.178, 0.44] 
# t_pickup = [-0.485, -0.178, 0.35] 
# # t_wipe_start = [0.4, -0.02, 0.28]
# t_random_start = [-0.57, 0.0075, 0.34]
# t_fixed_start = [-0.57, 0.0075, 0.34]

t_pickup_high = [-0.485, -0.19, 0.22] 
t_pickup = [-0.485, -0.19, 0.13] 
# t_wipe_start = [0.4, -0.02, 0.28]
t_random_start = [-0.585, 0.00, 0.155] # this is for 35 mm #[-0.585, 0.00, 0.1615]
t_fixed_start = [-0.56, 0.0075, 0.13]
R_default = [0, 0, -1, math.sqrt(2)/2, math.sqrt(2)/2, 0, math.sqrt(2)/2, -math.sqrt(2)/2, 0]
ft_ip = 'http://192.168.0.103:80'
gripper_port = '/dev/ttyUSB1'
finger_zero_positions = [0.075, 0.18] #[0.065, 0.175] #[0.0775, 0.1845] #[0.08, 0.1875] #[0.085, 0.19] (2nd, current calibration pic) #[0.095, 0.2] (initial)
UR5_ip = '192.168.0.101'