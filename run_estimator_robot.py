#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import time
import serial
import threading
import numpy as np
from src.hog_box import HOGBox
from src import utils
from src.ros_talker import RosTalker
from src.estimator import VNectEstimator
from src.joints2angles import Joints2Angles


#################
### Functions ###
#################
def joints_iter_gen():
    global joints_3d
    while 1:
        yield joints_3d


def ser_thread_func(COM='COM6', baudrate=9600, freq=20):
    global angles_r
    ser = serial.Serial(COM, baudrate, timeout=0)
    plock = threading.Lock()
    period = 1 / freq
    count = 0
    while True:
        plock.acquire()
        ser.write('START\r\n'.encode())
        for a in angles_r:
            msg = str(a) + '\r\n'
            ser.write(msg.encode())
        plock.release()
        print('Send serial message NO.%d:' % count, angles_r)
        count += 1
        time.sleep(period)


def my_exit(camera_capture):
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
video = 0
# video = './pic/wx/1/wx1.mp4'
# whether apply transposed matrix (when camera is flipped)
# T = True
T = False

## serial params ##
COM = 'COM6'
baudrate = 9600
freq = 20

## vnect params ##
# vnect input image size
box_size = 368
# parent joint indexes of each joint (for plotting the skeletal lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

# placeholders
joints_3d = np.zeros((21, 3))
angles_r = [0, 0, 0, 0]


#######################
### Initializations ###
#######################
# initialize VNect estimator
estimator = VNectEstimator()

## choose ros or serial connection solution ##
# initialize ros connection
# ros_talker = RosTalker(host='10.13.106.70', yumi=False)

# initialize serial connection in a sub thread
# ser_thread = threading.Thread(target=ser_thread_func, args=(COM, baudrate, freq), daemon=True)

# open a txt file to save angle data
# angles_file = open('angles.txt', 'w')

# initialize joints-to-angles calculation class
j2a = Joints2Angles()

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
# start serial sending thread if use serial
# ser_thread.start()

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

    # parse joint points into angles
    angles = j2a(joints_3d)

    # strategy: send message only if there is at least one angle shift > 15°
    # if not np.any(angles - np.deg2rad(5) >= 0):
    #     success, frame = camera_capture.read()
    #     continue
    # strategy: time interval between two messages must not smaller than 600ms
    # if time.time() - t < 0.6:
    #     success, frame = camera_capture.read()
    #     continue
    # else:
    #     t = time.time()

    # send ros message
    # ros_talker.send(angles)

    # update angle variables for serial to write
    s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r = angles
    angles_r = [s0_r, s1_r, e0_r, e1_r]

    # write angle data into txt
    # angles_file.write('%f %f %f %f %f %f %f %f\n' % tuple([a for a in angles]))

    success, frame = camera_capture.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.GaussianBlur(frame, (21, 21), 0)

# angles_file.close()
my_exit(camera_capture)
