#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from joints2angles import joints2angles
from src.hog_box import HOGBox
from src.estimator import VNectEstimator

import serial


# the input camera serial number of the PC (int), or PATH to input video (str)
# video = 0
video = './pic/angle.mp4'
# the side length of the bounding box
box_size = 368
# whether apply transposed matrix (when camera is flipped)
# T = True
T = False


def my_exit(cameraCapture):
    try:
        cameraCapture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        raise


# initialize serial connection
ser = serial.Serial('COM3', 9600, timeout=0)

j2a = joints2angles()

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
    angles = j2a(joints_3d)
    msg = '%d,%d,%d,%d,%d,%d,%d,%d' % angles
    ser.write(msg.encode())
    success, frame = cameraCapture.read()

my_exit(cameraCapture)
