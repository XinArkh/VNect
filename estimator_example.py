#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utils
from estimator import VNectEstimator


# the input camera serial number of the PC (int), or PATH to input video (str)
video = './test_src/action5.mp4'
# the side length of the bounding box
box_size = 368
# whether apply transposed matrix (when camera is flipped)
T = False
# parent joint indexes of each joint (for plotting the skeleton lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
# mouse click flag
clicked = False


def on_mouse(event, x, y, flags, param):
    """
    attain mouse clicking message
    """
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


def draw_BB_rect(img, rect):
    """
    draw bounding box in the BB initialization window, and record current rect (x, y, w, h)
    """
    global W, H
    x, y, w, h = rect
    offset_w = int(0.4 / 2 * W)
    offset_h = int(0.2 / 2 * H)
    pt1 = (np.max([x - offset_w, 0]), np.max([y - offset_h, 0]))
    pt2 = (np.min([x + w + offset_w, W]), np.min([y + h + offset_h, H]))
    # print(pt1, pt2)
    cv2.rectangle(img, pt1, pt2, (28, 76, 242), 4)
    rect = [np.max([x - offset_w, 0]),  # x
            np.max([y - offset_h, 0]),  # y
            np.min([x + w + offset_w, W]) - np.max([x - offset_w, 0]),  # w
            np.min([y + h + offset_h, H]) - np.max([y - offset_h, 0])]  # h
    return rect


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


def myExit(cameraCapture):
    try:
        cameraCapture.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        raise


# catch the video stream
cameraCapture = cv2.VideoCapture(video)
assert cameraCapture.isOpened(), 'Video stream not opened: %s' % str(video)

# frame width and height
W = int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
if T:
    W, H = H, W


# use HOG method to initialize bounding box
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
box_init_window_name = 'Bounding Box Initialization'
cv2.namedWindow(box_init_window_name)
cv2.setMouseCallback(box_init_window_name, on_mouse)

success, frame = cameraCapture.read(); frame = frame.T if T else frame
rect = None
while success and cv2.waitKey(1) == -1:
    found, w = hog.detectMultiScale(frame)
    if len(found) > 0:
        rect = draw_BB_rect(frame, found[np.argmax(w)])
    scale = 400 / H
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    cv2.imshow(box_init_window_name, frame)

    if clicked:
        clicked = False
        cv2.destroyWindow(box_init_window_name)
        break

    success, frame = cameraCapture.read(); frame = frame.T if T else frame

x, y, w, h = rect
fig = plt.figure()
ax_3d = plt.axes(projection='3d')
plt.ion()
ax_3d.clear()
plt.show()

# initialize VNect estimator
estimator = VNectEstimator()

# main loop
success, frame = cameraCapture.read(); frame = frame.T if T else frame
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
    success, frame = cameraCapture.read(); frame = frame.T if T else frame

myExit(cameraCapture)
