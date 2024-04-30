import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

class WebCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(-1)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # set camera focal length
        

    def get_rgb_frames(self, plot=False):
        for _ in range(100):
            ret, color_frame = self.cap.read()
            time_stamp = time.time()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                continue
            
            color_image = np.asanyarray(color_frame)
            if plot:
                visualize(color_image)
            return color_image, time_stamp #depth_image

    def shutdown(self):
        self.cap.release()

def visualize(color_image):
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(1)


if __name__ == "__main__":
    rgb_cam = WebCamera()
    color_image, time_stamp = rgb_cam.get_rgb_frames(plot=True)
    rgb_cam.shutdown()
           