#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import re
import cv2
import math
import h5py
import numpy as np
import pandas as pd


class Mpi_Inf_3dhp:
    # all joint names in mpi_inf_3dhp
    all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',  # 5
                       'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',  # 12
                       'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',  # 17
                       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',  # 23
                       'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']
    # joint indexes that being used in vnect model
    vnect_ids = [i - 1 for i in [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7, 18, 13, 28, 23]]
    # frame size of the processed mpi_inf_3dhp dataset
    img_size = 368

    def __init__(self, bpath, subjects=None, train_set=True, heatmap_size=46):
        # select training set or test set of the dataset
        self.train_set = train_set
        # heatmap size in the model
        self.heatmap_size = heatmap_size
        self.hm_factor = self.img_size / self.heatmap_size
        # load data index path
        self.list_path = os.path.join(bpath, 'train.txt' if self.train_set else 'test.txt')  # frame list
        self.annot_path = os.path.join(bpath, 'annots.h5')  # annotations
        # select subjects
        subjects = [1, 2, 3, 4, 5, 6, 7, 8] if subjects is None else subjects
        self.subjects = ['S%d' % n for n in subjects]
        # load frame path list
        self.df = pd.read_csv(self.list_path, sep=' ', header=None)  # load path data
        self.df = self.df.loc[self.df[0].isin(self.subjects), 1].sample(frac=1)  # select and suffle
        # frame annotation
        self.annots = h5py.File(self.annot_path, 'r')

    def load_data(self, batch_size, heatmap=True, joints=None):
        joints = self.vnect_ids if joints is None else joints
        batch = self.df.sample(batch_size, replace=True)
        # Parameter `replace`: bool, default False. Sample with or without replacement.
        ## init x(image)
        batch_x = np.zeros([batch_size, self.img_size, self.img_size, 3], dtype=np.uint8)
        ## init y(annotation)
        batch_y = np.zeros([batch_size, self.heatmap_size, self.heatmap_size, len(joints) * 4],
                           dtype=np.float32) if heatmap else np.zeros([batch_size, 5, len(joints)], np.float32)
        for i, fpath in zip(range(len(batch)), batch):
            # print(fpath)
            # load x
            img = cv2.imread(fpath)
            batch_x[i] = img
            # load y
            S, Seq, video, frame = self._parse_path(fpath, self.train_set)
            h5_path_2 = '/S{0}/Seq{1}/annot2/video_{2}'.format(S, Seq, video)
            h5_path_univ3 = '/S{0}/Seq{1}/univ_annot3/video_{2}'.format(S, Seq, video)
            for j, index in zip(range(len(joints)), joints):
                x_2, y_2 = self.annots[h5_path_2][frame, 2 * index:2 * index + 2] / self.hm_factor
                x_u3, y_u3, z_u3 = self.annots[h5_path_univ3][frame, 3 * index:3 * index + 3] / self.hm_factor
                if heatmap:
                    batch_y[i, ..., j] = self._gen_heatmap([self.heatmap_size, self.heatmap_size], [x_2, y_2])
                    # batch_y[i, ..., j + len(joints)] =
                    # batch_y[i, ..., j + len(joints)*2] =
                    # batch_y[i, ..., j + len(joints)*3] =
                else:
                    batch_y[i, :, j] = [x_2, y_2, x_u3, y_u3, z_u3]
        return batch_x, batch_y

    @staticmethod
    def _parse_path(fpath, train=True):
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
    def _gen_heatmap(img_shape, center, sigma=3):
        img_height, img_width = img_shape
        heatmap = np.zeros((img_height, img_width), dtype=np.float32)
        center_x, center_y = center
        th = 4.6052
        delta = math.sqrt(th * 2)
        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))
        x1 = int(min(img_width, center_x + delta * sigma))
        y1 = int(min(img_height, center_y + delta * sigma))
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[y][x] = np.clip(heatmap[y][x], math.exp(-exp), 1.0)
        return heatmap


if __name__ == '__main__':
    import time
    m = Mpi_Inf_3dhp(r'E:\Datasets\mpi_inf_3dhp')
    start = time.time()
    imgs, heatmaps = m.load_data(200)
    print(time.time()-start)
    img = cv2.resize(imgs[0], (46, 46), interpolation=cv2.INTER_LANCZOS4)
    heatmap = cv2.cvtColor(heatmaps[0, ..., 0], cv2.COLOR_GRAY2BGR)
    dst = img * 0.5 + heatmap * 255 * 0.5
    cv2.imshow('a', dst.astype(np.uint8))
    # cv2.imshow('b', heatmaps[0, ..., 0])
    cv2.waitKey()
    cv2.destroyAllWindows()
