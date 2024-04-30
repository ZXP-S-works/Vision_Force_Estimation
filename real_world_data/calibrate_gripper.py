from CONSTANTS import finger_zero_positions, gripper_port
from real_world_utils import set_EE_transform_linear, T_42_controller
import cv2, time
from webcam import WebCamera

T = T_42_controller(finger_offsets=finger_zero_positions, port=gripper_port)
T.release()
T.close()
# exit()
T.move_to_zero_positions()
time.sleep(1)

zero_image = cv2.imread('gripper_calibration_img.png')
cam = WebCamera()
color_image, time_stamp = cam.get_rgb_frames()  
# cv2.imwrite('gripper_calibration_img.png', color_image)

cv2.imshow('New Img', color_image)
cv2.imshow('Calibration Img', zero_image)
cv2.waitKey(0)


