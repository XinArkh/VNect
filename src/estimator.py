#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import cv2
import time
import math
import numpy as np
import tensorflow as tf
from OneEuroFilter import OneEuroFilter


class VNectEstimator:
    # the side length of the bounding box
    _box_size = 368
    # this factor indicates that the input box size is 8 times the side length of the output heatmaps
    _hm_factor = 8
    # number of the joints to be detected
    _joints_num = 21

    def __init__(self, T=False):
        print('Initializing VnectEstimator...')
        # whether apply transposed matrix (when camera is flipped)
        self.T = T
        # the ratio factors to scale the input image crops, no more than 1.0
        self.scales = [1]  # or [1, 0.7] to be consistent with the author when training
        # initialize one euro filters for all the joints
        config_2d = {
            'freq': 12,
            'mincutoff': 1.7,
            'beta': 0.3,
            'dcutoff': 1.0
        }
        config_3d = {
            'freq': 12,
            'mincutoff': 0.8,
            'beta': 0.4,
            'dcutoff': 1.0
        }
        self.filter_2d = [(OneEuroFilter(**config_2d), OneEuroFilter(**config_2d)) for _ in range(self._joints_num)]
        self.filter_3d = [(OneEuroFilter(**config_3d), OneEuroFilter(**config_3d), OneEuroFilter(**config_3d))
                          for _ in range(self._joints_num)]
        # load pretrained VNect model
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('../models/tf_model/vnect_tf.meta' if os.getcwd().endswith('src') else
                                           './models/tf_model/vnect_tf.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('../models/tf_model/'if os.getcwd().endswith('src') else
                                                            './models/tf_model/'))
        graph = tf.get_default_graph()
        self.input_crops = graph.get_tensor_by_name('Placeholder:0')
        self.heatmap = graph.get_tensor_by_name('split_2:0')
        self.x_heatmap = graph.get_tensor_by_name('split_2:1')
        self.y_heatmap = graph.get_tensor_by_name('split_2:2')
        self.z_heatmap = graph.get_tensor_by_name('split_2:3')
        print('Initialization done.')

    def __call__(self, img_input):
        t = time.time()
        img_input = img_input.T if self.T else img_input
        img_batch = self._gen_input_batch(img_input, self._box_size, self.scales)
        # inference
        hm, xm, ym, zm = self.sess.run([self.heatmap, self.x_heatmap, self.y_heatmap, self.z_heatmap],
                                       {self.input_crops: img_batch})
        # average scale outputs
        hm_size = self._box_size // self._hm_factor
        hm_avg = np.zeros((hm_size, hm_size, self._joints_num))
        xm_avg = np.zeros((hm_size, hm_size, self._joints_num))
        ym_avg = np.zeros((hm_size, hm_size, self._joints_num))
        zm_avg = np.zeros((hm_size, hm_size, self._joints_num))
        for i in range(len(self.scales)):
            rescale = 1.0 / self.scales[i]
            scaled_hm = img_scale(hm[i, :, :, :], rescale)
            scaled_x_hm = img_scale(xm[i, :, :, :], rescale)
            scaled_y_hm = img_scale(ym[i, :, :, :], rescale)
            scaled_z_hm = img_scale(zm[i, :, :, :], rescale)
            mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
            hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            xm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            ym_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            zm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
        hm_avg /= len(self.scales)
        xm_avg /= len(self.scales)
        ym_avg /= len(self.scales)
        zm_avg /= len(self.scales)
        joints_2d = extract_2d_joints_from_heatmap(hm_avg, self._box_size, self._hm_factor)
        joints_3d = extract_3d_joints_from_heatmap(joints_2d, xm_avg, ym_avg, zm_avg, self._box_size, self._hm_factor)
        joints_2d, joints_3d = self._joint_filter(joints_2d, joints_3d)
        if self.T:
            joints_2d = joints_2d[:, ::-1]
            joints_3d = joints_3d[:, [1, 0, 2]]
        print('FPS: {:>2.2f}'.format(1 / (time.time() - t)))
        return joints_2d, joints_3d

    @staticmethod
    def _gen_input_batch(img_input, box_size, scales):
        # any input image --> sqrared input image acceptable for the model
        img_square = img_scale_squareify(img_input, box_size)
        # generate multi-scale input batch
        input_batch = []
        for scale in scales:
            img = img_scale_padding(img_square, scale) if scale < 1 else img_square
            input_batch.append(img)
        # input image range: [0, 255) --> [-0.4, 0.6)
        input_batch = np.asarray(input_batch, dtype=np.float32) / 255 - 0.4
        return input_batch

    def _joint_filter(self, joints_2d, joints_3d):
        for i in range(self._joints_num):
            joints_2d[i, 0] = self.filter_2d[i][0](joints_2d[i, 0], time.time())
            joints_2d[i, 1] = self.filter_2d[i][1](joints_2d[i, 1], time.time())

            joints_3d[i, 0] = self.filter_3d[i][0](joints_3d[i, 0], time.time())
            joints_3d[i, 1] = self.filter_3d[i][1](joints_3d[i, 1], time.time())
            joints_3d[i, 2] = self.filter_3d[i][2](joints_3d[i, 2], time.time())
        return joints_2d, joints_3d


def img_scale(img, scale):
    """
    scale the input image by a same scale factor in both x and y directions
    """
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def img_padding(img, box_size, offset, padNum=0):
    """
    pad the image in left and right sides averagely to fill the box size

    padNum: the number to be filled (0-->(0, 0, 0)==black; 128-->(128, 128, 128)==grey)
    """
    h, w = img.shape[:2]
    assert h == box_size, 'height of the image not equal to box size'
    assert w < box_size, 'width of the image not smaller than box size'

    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * padNum
    img_padded[:, box_size // 2 - math.ceil(img.shape[1] / 2):box_size // 2 + math.ceil(img.shape[1] / 2) - offset,
               :] = img

    return img_padded


def img_scale_squareify(img, box_size):
    """
    scale and squareify the image to get a square image with standard box size

    img: BGR image
    boxsize: the length of the square area
    """
    h, w = img.shape[:2]
    scale = box_size / h
    img_scaled = img_scale(img, scale)

    if img_scaled.shape[1] < box_size:
        offset = img_scaled.shape[1] % 2
        img_cropped = img_padding(img_scaled, box_size, offset)
    else:
        img_cropped = img_scaled[:, img_scaled.shape[1] // 2 - box_size // 2:img_scaled.shape[1] // 2 + box_size // 2,
                                 :]

    assert img_cropped.shape == (box_size, box_size, 3)
    return img_cropped


def img_scale_padding(img, scale, padNum=0):
    """
    scale and pad the image

    scale: no bigger than 1.0
    """
    assert img.shape[0] == img.shape[1]
    box_size = img.shape[0]
    img_scaled = img_scale(img, scale)
    pad_h = (box_size - img_scaled.shape[0]) // 2
    pad_w = (box_size - img_scaled.shape[1]) // 2
    pad_h_offset = (box_size - img_scaled.shape[0]) % 2
    pad_w_offset = (box_size - img_scaled.shape[1]) % 2
    img_scaled_padded = np.pad(img_scaled, ((pad_w, pad_w + pad_w_offset), (pad_h, pad_h + pad_h_offset), (0, 0)),
                               mode='constant', constant_values=padNum)

    return img_scaled_padded


def extract_2d_joints_from_heatmap(heatmap, box_size, hm_factor):
    """
    rescale the heatmap to CNN input size, then record the coordinates of each joints

    joints_2d: a joints_num x 2 array, each row contains [row, column] coordinates of the corresponding joint
    """
    assert heatmap.shape[0] == heatmap.shape[1]
    heatmap_scaled = img_scale(heatmap, hm_factor)

    joints_2d = np.zeros((heatmap_scaled.shape[2], 2), dtype=np.int16)
    for joint_num in range(heatmap_scaled.shape[2]):
        joint_coord = np.unravel_index(np.argmax(heatmap_scaled[:, :, joint_num]), (box_size, box_size))
        joints_2d[joint_num, :] = joint_coord

    return joints_2d


def extract_3d_joints_from_heatmap(joints_2d, x_hm, y_hm, z_hm, box_size, hm_factor):
    """
    obtain the 3D coordinates of each joint from its 2D coordinates

    joints_3d: a joints_num x 3 array, each row contains [x, y, z] coordinates of the corresponding joint

    Notation:
    x direction: left --> right
    y direction: up --> down
    z direction: forawrd --> backward
    """
    scaler = 100
    joints_3d = np.zeros((x_hm.shape[2], 3), dtype=np.float32)

    for joint_num in range(x_hm.shape[2]):
        coord_2d_h, coord_2d_w = joints_2d[joint_num][:]

        joint_x = x_hm[max(int(coord_2d_h / hm_factor), 1), max(int(coord_2d_w / hm_factor), 1), joint_num] * scaler
        joint_y = y_hm[max(int(coord_2d_h / hm_factor), 1), max(int(coord_2d_w / hm_factor), 1), joint_num] * scaler
        joint_z = z_hm[max(int(coord_2d_h / hm_factor), 1), max(int(coord_2d_w / hm_factor), 1), joint_num] * scaler
        joints_3d[joint_num, :] = [joint_x, joint_y, joint_z]

    # Subtract the root location to normalize the data
    joints_3d -= joints_3d[14, :]

    return joints_3d


if __name__ == '__main__':
    estimator = VNectEstimator()
    joints_2d, joints_3d = estimator(cv2.imread('../pic/test_pic.jpg'))
    print('\njoints_2d\n', joints_2d, '\n\njoints_3d\n', joints_3d)
