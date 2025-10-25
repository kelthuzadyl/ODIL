import numpy as np

RGB_TOPIC_NAME = "/right_cam/right_cam/color/image_rect_raw"
DEPTH_TOPIC_NAME = "/right_cam/right_cam/aligned_depth_to_color/image_raw"
HANDEYE_TRANSFORM = np.load("config/handeye_4x4.npy")
K = np.load("config/intrinsic_3x3.npy")
