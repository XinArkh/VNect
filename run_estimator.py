#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import serial
from src.hog_box import HOGBox
from src.ros_talker import RosTalker
from src.estimator import VNectEstimator
from src.joints2angles import joints2angles

# the input camera serial number of the PC (int), or PATH to input video (str)
# video = 0
video = './pic/angle.mp4'
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


# initialize ros connection
ros_talker = RosTalker(host='10.13.106.70', yumi=True)

# initialize serial connection
# ser = serial.Serial('COM3', 9600, timeout=0)

# initialize joints-to-angles calculation class
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
    # print(angles)
    ros_talker(angles)
    # write to serial interface
    # ser.write(b'ARM\r\n')
    # for s in [str(a)+'\r\n' for a in angles]:
    #     ser.write(s.encode())
    success, frame = cameraCapture.read()

my_exit(cameraCapture)
