#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import numpy as np


class HOGBox:
    """
    a simple HOG-method-based human tracking box
    """
    # mouse click flag
    clicked = False

    def __init__(self, T=False):
        self.T = T
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._box_init_window_name = 'Bounding Box Initialization'
        cv2.namedWindow(self._box_init_window_name)
        cv2.setMouseCallback(self._box_init_window_name, self._on_mouse)

    def __call__(self, img):
        img = img.T if self.T else img
        H, W = img.shape[:2]
        rect = [0, 0, W, H]
        found, w = self.hog.detectMultiScale(img)
        if len(found) > 0:
            rect = self.draw_rect(img, found[np.argmax(w)], H, W)
        scale = 400 / H
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imshow(self._box_init_window_name, img)

        if self.clicked:
            cv2.destroyWindow(self._box_init_window_name)
        return self.clicked, rect

    def _on_mouse(self, event, x, y, flags, param):
        """
        attain mouse clicking message
        """
        if event == cv2.EVENT_LBUTTONUP:
            self.clicked = True

    @staticmethod
    def draw_rect(img, rect, H, W):
        """
        draw bounding box in the BB initialization window, and record current rect (x, y, w, h)
        """
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
