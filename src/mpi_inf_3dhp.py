#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import re
import cv2
import math
import h5py
import numpy as np
import pandas as pd
import utils


class Mpi_Inf_3dhp:
    # all joints in mpi_inf_3dhp
    all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',  # 5
                       'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',  # 12
                       'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',  # 17
                       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',  # 23
                       'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']
    # joint indexes used in vnect (totally 21 joints)
    vnect_ids = [i - 1 for i in [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7, 18, 13, 28, 23]]
    img_size = 2048  # frame size in mpi_inf_3dhp dataset
    box_size = 368
    heatmap_size = 46
    hm_factor = box_size / heatmap_size

    def __init__(self, bpath, subjects=None, if_train_set=True):
        # select training set or test set
        self.if_train_set = if_train_set
        # load data index path
        self.list_path = os.path.join(bpath, 'train.txt' if self.if_train_set else 'test.txt')  # frame list
        self.annot_path = os.path.join(bpath, 'annots.h5')  # annotations
        # select subjects
        subjects = [1, 2, 3, 4, 5, 6, 7, 8] if subjects is None else subjects
        self.subjects = ['S%d' % n for n in subjects]  # ['S1', 'S2' ... 'S8']
        # load frame path list
        self.df = pd.read_csv(self.list_path, sep=' ', header=None)  # load path data
        self.df = self.df.loc[self.df[0].isin(self.subjects), 1].sample(frac=1)  # select and suffle
        # frame annotation
        self.annots = h5py.File(self.annot_path, 'r')

    def load_data(self, batch_size, joints=None):
        joints = self.vnect_ids if joints is None else joints  # joints is a list showing which joints to choose
        # init x (images)  shape: (batch_size, 368, 368, 3)
        batch_x = np.zeros((batch_size, self.box_size, self.box_size, 3), dtype=np.uint8)
        # init y (heatmaps)  shape: (batch_size, 46, 46, 21*4)
        batch_y = np.zeros((batch_size, self.heatmap_size, self.heatmap_size, len(joints) * 4), dtype=np.float32)
        load_num = 0
        while load_num < batch_size:
            fpath = self.df.sample(1).iloc[0]
            S, Seq, video, frame = self.parse_path(fpath, self.if_train_set)
            h5_path_2 = '/S{0}/Seq{1}/annot2/video_{2}'.format(S, Seq, video)
            h5_path_univ3 = '/S{0}/Seq{1}/univ_annot3/video_{2}'.format(S, Seq, video)
            coords = self.annots[h5_path_2][frame, :].reshape((-1, 2))
            pole_left = np.min(coords[:, 0])
            pole_right = np.max(coords[:, 0])
            pole_top = np.min(coords[:, 1])
            pole_bottom = np.max(coords[:, 1])
            margin_w = int((pole_right - pole_left) * 0.2)
            margin_h = int((pole_bottom - pole_top) * 0.1)
            # (y_2_0, x_2_0): upper-left point of the bounding box
            y_2_0 = pole_top - margin_h
            x_2_0 = pole_left - margin_w
            h = pole_bottom + margin_h - y_2_0
            w = pole_right + margin_w - x_2_0
            # make sure of enough margin space
            if not (x_2_0 >= 0 and x_2_0 + w < self.img_size and
                    y_2_0 >= 0 and y_2_0 + h < self.img_size):
                continue
            img = cv2.imread(fpath)
            img = cv2.resize(img, (self.img_size, self.img_size))  # tmp, remove later
            img_box, _, _ = utils.img_scale_squarify(img[int(pole_top-margin_h): int(pole_bottom+margin_h), 
                                                         int(pole_left-margin_w): int(pole_right+margin_w), :], 
                                                     self.box_size)
            # load x (images)
            batch_x[load_num] = img_box
            # load y (heatmaps)
            for j, index in enumerate(joints):
                x_2, y_2 = self.annots[h5_path_2][frame, 2*index: 2*index+2]
                if h > w:
                    x_2 = (x_2-x_2_0) * self.box_size/h + self.box_size//2 - (w*self.box_size/h)//2
                    y_2 = (y_2-y_2_0) * self.box_size/h
                else:
                    x_2 = (x_2-x_2_0) * self.box_size/w
                    y_2 = (y_2-y_2_0) * self.box_size/w + self.box_size//2 - (h*self.box_size/w)//2
                x_2, y_2 = x_2/self.hm_factor, y_2/self.hm_factor
                batch_y[load_num, ..., j] = self.gen_heatmap(self.heatmap_size, self.heatmap_size, x_2, y_2)
                # x_u3, y_u3, z_u3 = self.annots[h5_path_univ3][frame, 3 * index:3 * index + 3] / self.hm_factor
                # batch_y[i, ..., j + len(joints)] =
                # batch_y[i, ..., j + len(joints)*2] =
                # batch_y[i, ..., j + len(joints)*3] =
            
            load_num += 1

        return batch_x, batch_y

    @staticmethod
    def parse_path(fpath, train=True):
        if train:
            Subject = 'S'
            fname_re = 'frame_[0-9]*.jpg'
        else:
            Subject = 'TS'
            fname_re = 'img_[0-9]*.jpg'
        S = re.search(r'{}[0-9]*[\\|/]'.format(Subject), fpath).group()[1:-1]
        Seq = re.search(r'Seq[0-9]*[\\|/]', fpath).group()[3:-1]
        video = re.search(r'video_[0-9]*[\\|/]', fpath).group()[6:-1] if train else None
        frame = re.search(fname_re, fpath).group()
        frame = re.split(r'[_|.]', frame)[1]
        frame = int(frame) - 1
        return S, Seq, video, frame

    @staticmethod
    def gen_heatmap(height, width, center_x, center_y, sigma=1):
        heatmap = np.zeros((height, width), dtype=np.float32)
        th = 4.6052
        delta = math.sqrt(th * 2)
        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))
        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[y, x] = math.exp(-exp)
        return heatmap


if __name__ == '__main__':
    import time
    m = Mpi_Inf_3dhp(r'E:\Datasets\mpi_inf_3dhp')
    start = time.time()
    imgs, heatmaps = m.load_data(1)
    print('loading time: %.3fs' % (time.time() - start))

    img = imgs[0]
    heatmap = heatmaps[0, ..., 0]  # head
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    overlay = img * 0.5 + cv2.resize(heatmap_bgr, (368, 368)) * 255 * 0.5

    cv2.imshow('image', img)
    cv2.imshow('heatmap', cv2.resize(heatmap, (368, 368)))
    cv2.imshow('overlay', overlay.astype(np.uint8))

    cv2.waitKey()
    cv2.destroyAllWindows()
