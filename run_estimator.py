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
# video = './pic/test_video.mp4'
video = 0
# video = './pic/action5.mp4'
# the side length of the bounding box
box_size = 368
# whether apply transposed matrix (when camera is flipped)
# T = False
T = True


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
hog = HOGBox(T=T)

success, frame = cameraCapture.read()
rect = None
while success and cv2.waitKey(1) == -1:
    choose, rect = hog(frame)
    if choose:
        break
    success, frame = cameraCapture.read()

x, y, w, h = rect
# initialize VNect estimator
estimator = VNectEstimator(T=T)
# main loop
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1:
    # crop bounding box from the raw frame
    frame_cropped = frame[y:y+h, x:x+w, :] if not T else frame[x:x+w, y:y+h, :]
    joints_2d, joints_3d = estimator(frame_cropped)
    print(np.linalg.norm(joints_3d[4]-joints_3d[3]), np.linalg.norm(joints_3d[7]-joints_3d[6]))
    success, frame = cameraCapture.read()

my_exit(cameraCapture)
