#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator


# the input camera serial number of the PC (int), or PATH to input video (str)
video = './pic/test_video.mp4'
# the side length of the bounding box
box_size = 368
# whether apply transposed matrix (when camera is flipped)
T = False
# parent joint indexes of each joint (for plotting the skeleton lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]


def imshow_3d(ax_3d):
    ax_3d.clear()
    ax_3d.view_init(-90, -90)
    ax_3d.set_xlim(-500, 500)
    ax_3d.set_ylim(-500, 500)
    ax_3d.set_zlim(-500, 500)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])
    white = (1.0, 1.0, 1.0, 0.0)
    ax_3d.w_xaxis.set_pane_color(white)
    ax_3d.w_yaxis.set_pane_color(white)
    ax_3d.w_xaxis.line.set_color(white)
    ax_3d.w_yaxis.line.set_color(white)
    ax_3d.w_zaxis.line.set_color(white)
    utils.draw_limbs_3d(ax_3d, joints_3d, joint_parents)
    # the following line is unnecessary with matplotlib 3.0.0, but ought to be activated
    # under matplotlib 3.0.2 (other versions not tested)
    # plt.pause(0.00001)


def my_exit(cameraCapture):
    try:
        cameraCapture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        raise


# catch the video stream
cameraCapture = cv2.VideoCapture(video)
assert cameraCapture.isOpened(), 'Video stream not opened: %s' % str(video)

# use HOG method to initialize bounding box
hog = HOGBox()

success, frame = cameraCapture.read()
rect = None
while success and cv2.waitKey(1) == -1:
    choose, rect = hog(frame)
    if choose:
        break
    success, frame = cameraCapture.read()

x, y, w, h = rect
fig = plt.figure()
ax_3d = plt.axes(projection='3d')
plt.ion()
ax_3d.clear()
plt.show()

# initialize VNect estimator
estimator = VNectEstimator(T=T)

# main loop
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1:
    # crop bounding box from the raw frame
    frame_cropped = frame[y:y+h, x:x+w, :]
    joints_2d, joints_3d = estimator(frame_cropped)
    ## plot ##
    # 2d plotting
    frame_square = utils.img_scale_squareify(frame_cropped, box_size)
    frame_square = utils.draw_limbs_2d(frame_square, joints_2d, joint_parents)
    cv2.imshow('2D Prediction', frame_square)
    # 3d plotting
    imshow_3d(ax_3d)
    success, frame = cameraCapture.read()

my_exit(cameraCapture)
