#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import time
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
# the input camera serial number in the PC (int), or PATH to input video (str)
# video = 0
video = './pic/test_video.mp4'
# whether apply transposed matrix (when camera is flipped)
# T = True
T = False

## vnect params ##
# vnect input image size
box_size = 368
# parent joint indexes of each joint (for plotting the skeletal lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

# placeholder
joints_3d = np.zeros((21, 3))


#######################
### Initializations ###
#######################
# initialize VNect estimator
estimator = VNectEstimator()

# open a txt file to save angle data
# angles_file = open('angles.txt', 'w')

# catch the video stream
camera_capture = cv2.VideoCapture(video)
assert camera_capture.isOpened(), 'Video stream not opened: %s' % str(video)

# use a simple HOG method to initialize bounding box
hog = HOGBox()

################
### Box Loop ###
################
## click the mouse when ideal bounding box appears ##
success, frame = camera_capture.read()
# initialize bounding box as the maximum rectangle
rect = 0, 0, frame.shape[1], frame.shape[0]
while success and cv2.waitKey(1) == -1:
    # .copy() to prevent an unexpected bug
    frame = np.transpose(frame, axes=[1, 0, 2]).copy() if T else frame
    choose, rect = hog(frame)
    if choose:
        break
    success, frame = camera_capture.read()
# the final static bounding box params
x, y, w, h = rect


#################
### Main Loop ###
#################
## trigger any keyboard events to stop the loop ##

# start 3d skeletal animation plotting
utils.plot_3d_init(joint_parents, joints_iter_gen)

t = time.time()
success, frame = camera_capture.read()
while success and cv2.waitKey(1) == -1:
    # crop bounding box from the raw frame
    frame = np.transpose(frame, axes=[1, 0, 2]).copy() if T else frame
    frame_cropped = frame[y:y + h, x:x + w, :]
    # vnect estimating process
    joints_2d, joints_3d = estimator(frame_cropped)

    # 2d plotting
    frame_square = utils.img_scale_squareify(frame_cropped, box_size)
    frame_square = utils.draw_limbs_2d(frame_square, joints_2d, joint_parents)
    cv2.imshow('2D Prediction', frame_square)

    # write angle data into txt
    # angles_file.write('%f %f %f %f %f %f %f %f\n' % tuple([a for a in angles]))

    success, frame = camera_capture.read()

# angles_file.close()
my_exit(camera_capture)
