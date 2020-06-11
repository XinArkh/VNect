#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import time
import numpy as np
import multiprocessing as mp
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator


##################
### Parameters ###
##################
# the input camera serial number in the PC (int), or PATH to input video (str)
# video = 0
video = './pic/test_video.mp4'
# whether apply transposed matrix (when camera is flipped)
# T = True
T = False
# vnect input image size
box_size = 368
# parent joint indexes of each joint (for plotting the skeletal lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8,
                 9, 14, 11, 12, 14, 14, 1, 4, 7, 10,
                 13]


#######################
### Initializations ###
#######################
def init():
    # initialize VNect estimator
    global estimator
    estimator = VNectEstimator()
    # catch the video stream
    global camera_capture
    camera_capture = cv2.VideoCapture(video)
    assert camera_capture.isOpened(), 'Video stream not opened: %s' % str(video)
    global W_img, H_img
    W_img, H_img = (int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


################
### Box Loop ###
################
# use a simple HOG method to initialize bounding box
def hog_box():
    hog = HOGBox()
    ## click the mouse when ideal bounding box appears ##
    success, frame = camera_capture.read()
    # initialize bounding box as the maximum rectangle
    rect = [0, 0, W_img, H_img]
    while success and cv2.waitKey(1) == -1:
        if T:
            # if not calling .copy(), an unexpected bug occurs
            frame = np.transpose(frame, axes=[1, 0, 2]).copy()
        choose, rect = hog(frame)
        if choose:
            break
        success, frame = camera_capture.read()
    # return the bounding box params (x, y, w, h)
    return rect


#################
### Main Loop ###
#################
## trigger any keyboard events to stop the loop ##
def main(q_start3d, q_joints):
    init()
    x, y, w, h = hog_box()
    q_start3d.put(True)
    success, frame = camera_capture.read()
    while success and cv2.waitKey(1) == -1:
        if T:
            frame = np.transpose(frame, axes=[1, 0, 2]).copy()
        # crop bounding box from the raw frame
        frame_cropped = frame[y:y + h, x:x + w, :]
        # vnect estimation
        joints_2d, joints_3d = estimator(frame_cropped)
        q_joints.put(joints_3d)
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

    # angles_file.close()
    try:
        camera_capture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        raise


if __name__ == '__main__':
    q_start3d = mp.Queue()
    q_joints = mp.Queue()
    ps_main = mp.Process(target=main, args=(q_start3d, q_joints))
    ps_plot3d = mp.Process(target=utils.plot_3d,
                           args=(q_start3d, q_joints, joint_parents),
                           daemon=True)
    ps_main.start()
    ps_plot3d.start()
    ps_main.join()
