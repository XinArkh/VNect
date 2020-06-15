#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator


#################
### Functions ###
#################
def joints_iter_gen():
    """
    a generator to yield joints iteratively, supporting 3d animation plot
    """
    global joints_3d
    while 1:
        yield joints_3d


def my_exit(camera_capture):
    """
    exit opencv environment
    """
    try:
        camera_capture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        raise


##################
### Parameters ###
##################
# camera serial number (int) or video path (str)
# video = 0
video = './pic/test_video.mp4'
# whether apply transposed matrix (when camera is flipped)
# T = True
T = False
# placeholder
joints_3d = np.zeros((21, 3))
# vnect input image size
box_size = 368
# parent joint indexes of each joint (for plotting the skeletal lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8,
                 9, 14, 11, 12, 14, 14, 1, 4, 7, 10,
                 13]


#######################
### Initializations ###
#######################
# initialize VNect estimator
estimator = VNectEstimator()
# catch the video stream
camera_capture = cv2.VideoCapture(video)
assert camera_capture.isOpened(), 'Video stream not opened: %s' % str(video)
W_img, H_img = (int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# use a simple HOG method to initialize bounding box
hog = HOGBox()
success, frame = camera_capture.read()
rect = 0, 0, W_img, H_img
while success and cv2.waitKey(1) == -1:
    if T:
        # if not calling .copy(), an unexpected bug occurs
        # mirror
        # frame = np.transpose(frame, axes=[1, 0, 2]).copy()
        # no mirror
        frame = np.rot90(frame, 3).copy()
    choose, rect = hog(frame)
    if choose:
        break
    success, frame = camera_capture.read()
# bounding box params
x, y, w, h = rect


#################
### Main Loop ###
#################
## trigger any keyboard events to stop the loop ##

# start 3d plotting
utils.plot_3d_init(joint_parents, joints_iter_gen)

success, frame = camera_capture.read()
while success and cv2.waitKey(1) == -1:
    if T:
        # mirror
        # frame = np.transpose(frame, axes=[1, 0, 2])
        # no mirror
        frame = np.rot90(frame, 3)
    # crop bounding box from the raw frame
    frame_cropped = frame[y: y + h, x: x + w, :]
    # vnect estimation
    joints_2d, joints_3d = estimator(frame_cropped)
    # 2d plotting
    joints_2d[:, 0] += y
    joints_2d[:, 1] += x
    frame_draw = utils.draw_limbs_2d(frame.copy(), joints_2d, joint_parents, [x, y, w, h])
    frame_draw = utils.img_scale(frame_draw, 1024 / max(W_img, H_img))
    cv2.imshow('2D Prediction', frame_draw)
    # update bounding box
    y_min = (np.min(joints_2d[:, 0]))
    y_max = (np.max(joints_2d[:, 0]))
    x_min = (np.min(joints_2d[:, 1]))
    x_max = (np.max(joints_2d[:, 1]))
    buffer_x = 0.8 * (x_max - x_min + 1)
    buffer_y = 0.2 * (y_max - y_min + 1)
    x, y = (max(int(x_min - buffer_x / 2), 0),
            max(int(y_min - buffer_y / 2), 0))
    w, h = (int(min(x_max - x_min + buffer_x, W_img - x)),
            int(min(y_max - y_min + buffer_y, H_img - y)))
    # update frame
    success, frame = camera_capture.read()

my_exit(camera_capture)
