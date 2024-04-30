## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
import matplotlib.pyplot as plt
###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time


def visualize(color_image, depth_image):
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)


class RGBDCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = self.config.resolve(pipeline_wrapper)

        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        # found_rgb = False
        # for s in device.sensors:
        #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
        #         found_rgb = True
        #         break
        # if not found_rgb:
        #     print("The demo requires Depth camera with Color sensor")
        #     exit(0)

        #self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        serial_num = '935722061082'
        self.config.enable_device(serial_num.encode('utf-8'))
        #self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 8)
        # if device_product_line == 'L500':
        #     self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        # else:
        #     self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def start_streaming(self):
        self.pipeline.start(self.config)

    def stop_streaming(self):
        self.pipeline.stop()

    def get_rgb_frames(self, plot=False):
        for _ in range(100):
            # Wait for a coherent pair of frames: depth and color
            # sensor_timestamp is the idea choice, but to obtain it, we need custom build realsense SDK.
            # https://support.intelrealsense.com/hc/en-us/community/posts/360033435954-D435-get-timestamp-of-each-frame
            # https://github.com/IntelRealSense/librealsense/issues/12058#issuecomment-1661954386
            frames = self.pipeline.wait_for_frames()
            time_stamp = frames.get_timestamp() / 1e3
            ## For custom build pyrealsense2
            # t = time.time()
            # t1 = frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
            # t2 = frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp)
            # t3 = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            # t4 = frames.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
            # t5 = frames.get_timestamp()
            # print(t, t1, t2, t3, t4, t5)
            ## For pyrealsense2
            # t = time.time()
            # t1 = t - frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp) / 1e3
            # t2 = t - frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp) / 1e3
            # t3 = t - frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival) / 1e3
            # t5 = t - frames.get_timestamp() / 1e3
            # print(t, t1, t2, t3, t5)

            #depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            #if not depth_frame or not color_frame:
            if not color_frame:
                continue

            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if plot:
                visualize(color_image, depth_image)

            return color_image, time_stamp #depth_image


def visualize_data(img, ft, time_stamp, ts):
    plt.figure()
    # ToDo unify img RGB format (int?)
    plt.imshow(img[:3].transpose(1, 2, 0).astype(float) / 255)
    plt.title('fx {:.02f}, fz {:.02f}, t_img {:.02f}, t_f {:.02f}'.format(ft[0], ft[2], time_stamp, ts))
    plt.show()


if __name__ == "__main__":
    rgbd_cam = RGBDCamera()
    # Start streaming
    rgbd_cam.start_streaming()

    try:
        while True:
            color_image, depth_image, time_stamp = rgbd_cam.get_rgb_frames(plot=True)
            color_image = color_image.transpose(2, 0, 1)
            # visualize_data(color_image, [0, 0, 0], time_stamp, 0)
    finally:
        # Stop streaming
        rgbd_cam.stop_streaming()
