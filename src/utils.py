#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


def img_scale(img, scale):
    """
    Resize a image by s scaler in both x and y directions.

    :param img: input image
    :param scale: scale  factor, new image side length / raw image side length
    :return: the scaled image
    """
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def hm_local_interp_bilinear(src, scale, center, area_size=10):
    """
    Heatmap interpolation using a local bilinear method.
    Reference website: https://zhuanlan.zhihu.com/p/49832048

    :param src: input heatmap
    :param scale: scale factor, new image side length / raw image side length
    :param center: coordinate of the local center in the heatmap, [row, column]
    :param area_size: side length of local area in the interpolated heatmap
    :return: the destination heatmap with local area being interpolated
    """
    src_h, src_w = src.shape[:]
    dst_h, dst_w = [s * scale for s in src.shape[:]]
    y, x = [c * scale for c in center]
    dst = np.zeros((dst_h, dst_w))
    for dst_y in range(max(y - area_size // 2, 0), min(y + int(np.ceil(area_size / 2)), dst_h)):
        for dst_x in range(max(x - area_size // 2, 0), min(x + int(np.ceil(area_size / 2)), dst_w)):
            # pixel alignment
            # src_x = dst_x / 8
            # src_y = dst_y / 8
            # center alignment
            src_x = (dst_x + 0.5) / scale - 0.5
            src_y = (dst_y + 0.5) / scale - 0.5
            src_x_0 = int(src_x)
            src_y_0 = int(src_y)
            src_x_1 = min(src_x_0 + 1, src_w - 1)
            src_y_1 = min(src_y_0 + 1, src_h - 1)

            value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0] + (src_x - src_x_0) * src[src_y_0, src_x_1]
            value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0] + (src_x - src_x_0) * src[src_y_1, src_x_1]
            dst[dst_y, dst_x] = (src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1
    return dst


def hm_pt_interp_bilinear(src, scale, point):
    """
    Determine the value of one desired point by bilinear interpolation.

    :param src: input heatmap
    :param scale: scale factor, input box side length / heatmap side length
    :param point: position of the desired point in input box, [row, column]
    :return: the value of the desired point
    """
    src_h, src_w = src.shape[:]
    dst_y, dst_x = point
    src_x = (dst_x + 0.5) / scale - 0.5
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x_0 = int(src_x)
    src_y_0 = int(src_y)
    src_x_1 = min(src_x_0 + 1, src_w - 1)
    src_y_1 = min(src_y_0 + 1, src_h - 1)

    value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0] + (src_x - src_x_0) * src[src_y_0, src_x_1]
    value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0] + (src_x - src_x_0) * src[src_y_1, src_x_1]
    dst_val = (src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1
    return dst_val


def img_padding(img, box_size, color='black'):
    """
    Given the input image and side length of the box, put the image into the center of the box.

    :param img: the input color image, whose longer side is equal to box size
    :param box_size: the side length of the square box
    :param color: indicating the padding area color
    :return: the padded image
    """
    h, w = img.shape[:2]
    offset_x, offset_y = 0, 0
    if color == 'black':
        pad_color = [0, 0, 0]
    elif color == 'grey':
        pad_color = [128, 128, 128]
    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    if h > w:
        offset_x = box_size // 2 - w // 2
        img_padded[:, offset_x: box_size // 2 + int(np.ceil(w / 2)), :] = img
    else:  # h <= w
        offset_y = box_size // 2 - h // 2
        img_padded[offset_y: box_size // 2 + int(np.ceil(h / 2)), :, :] = img
    return img_padded, [offset_x, offset_y]


def img_scale_squarify(img, box_size):
    """
    To scale and squarify the input image into a square box with fixed size.

    :param img: the input color image
    :param box_size: the length of the square box
    :return: box image, scaler and offsets
    """
    h, w = img.shape[:2]
    scaler = box_size / max(h, w)
    img_scaled = img_scale(img, scaler)
    img_padded, [offset_x, offset_y] = img_padding(img_scaled, box_size)
    assert img_padded.shape == (box_size, box_size, 3), 'padded image shape invalid'
    return img_padded, scaler, [offset_x, offset_y]


def img_scale_padding(img, scaler, box_size, color='black'):
    """
    For a box image, scale down it and then pad the former area.

    :param img: the input box image
    :param scaler: scale factor, new image side length / raw image side length, < 1
    :param box_size: side length of the square box
    :param color: the padding area color
    """
    img_scaled = img_scale(img, scaler)
    if color == 'black':
        pad_color = (0, 0, 0)
    elif color == 'grey':
        pad_color = (128, 128, 128)
    pad_h = (box_size - img_scaled.shape[0]) // 2
    pad_w = (box_size - img_scaled.shape[1]) // 2
    pad_h_offset = (box_size - img_scaled.shape[0]) % 2
    pad_w_offset = (box_size - img_scaled.shape[1]) % 2
    img_scale_padded = np.pad(img_scaled,
                              ((pad_w, pad_w + pad_w_offset),
                               (pad_h, pad_h + pad_h_offset),
                               (0, 0)),
                              mode='constant',
                              constant_values=(
                                  (pad_color[0], pad_color[0]),
                                  (pad_color[1], pad_color[1]),
                                  (pad_color[2], pad_color[2])))
    return img_scale_padded


def extract_2d_joints(heatmaps, box_size, hm_factor):
    """
    Rescale the heatmap to input box size, then extract the coordinates for every joint.

    :param heatmaps: the input heatmaps
    :param box_size: the side length of the input box
    :param hm_factor: heatmap factor, indicating box size / heatmap size
    :return: a 2D array with [joints_num, 2], each row of which means [row, column] coordinates of corresponding joint
    """
    joints_2d = np.zeros((heatmaps.shape[2], 2))
    for joint_num in range(heatmaps.shape[2]):
        # joint_coord_1 = np.unravel_index(np.argmax(heatmaps[:, :, joint_num]),
        #                                  (box_size // hm_factor, box_size // hm_factor))
        # heatmap_scaled = hm_local_interp_bilinear(heatmaps[:, :, joint_num], hm_factor, joint_coord_1)
        # joint_coord_2 = np.unravel_index(np.argmax(heatmap_scaled), (box_size, box_size))
        # joints_2d[joint_num, :] = joint_coord_2
        heatmap_scaled = cv2.resize(heatmaps[:, :, joint_num], (0, 0), 
                                    fx=hm_factor, fy=hm_factor, 
                                    interpolation=cv2.INTER_LINEAR)
        joint_coord = np.unravel_index(np.argmax(heatmap_scaled), 
                                       (box_size, box_size))
        joints_2d[joint_num, :] = joint_coord
    return joints_2d


def extract_3d_joints(joints_2d, x_hm, y_hm, z_hm, hm_factor):
    """
    Extract 3D coordinates of each joint according to its 2D coordinates.

    :param joints_2d: 2D array with [joints_num, 2], containing 2D coordinates the joints
    :param x_hm: x coordinate heatmaps
    :param y_hm: y coordinate heatmaps
    :param z_hm: z coordinate heatmaps
    :param hm_factor: heatmap factor, indicating box size / heatmap size
    :return: a 3D array with [joints_num, 3], each row of which contains [x, y, z] coordinates of corresponding joint

    Notation:
    x direction: left --> right
    y direction: up --> down
    z direction: nearer --> farther
    """
    scaler = 100  # scaler=100 -> mm unit; scaler=10 -> cm unit
    joints_3d = np.zeros((x_hm.shape[2], 3), dtype=np.float32)
    for joint_num in range(x_hm.shape[2]):
        # coord_2d_h, coord_2d_w = joints_2d[joint_num][:]
        # coord_3d_h = coord_2d_h
        # coord_3d_w = coord_2d_w
        # x_hm_scaled = img_scale(x_hm, hm_factor)
        # y_hm_scaled = img_scale(y_hm, hm_factor)
        # z_hm_scaled = img_scale(z_hm, hm_factor)
        # joint_x = x_hm_scaled[coord_3d_h, coord_3d_w, joint_num] * scaler
        # joint_y = y_hm_scaled[coord_3d_h, coord_3d_w, joint_num] * scaler
        # joint_z = z_hm_scaled[coord_3d_h, coord_3d_w, joint_num] * scaler
        y_2d, x_2d = joints_2d[joint_num][:]
        joint_x = hm_pt_interp_bilinear(x_hm[:, :, joint_num], 
                                        hm_factor,
                                        (y_2d, x_2d)) * scaler
        joint_y = hm_pt_interp_bilinear(y_hm[:, :, joint_num], 
                                        hm_factor,
                                        (y_2d, x_2d)) * scaler
        joint_z = hm_pt_interp_bilinear(z_hm[:, :, joint_num], 
                                        hm_factor,
                                        (y_2d, x_2d)) * scaler
        joints_3d[joint_num, :] = [joint_x, joint_y, joint_z]
    # Subtract the root location to normalize the data
    joints_3d -= joints_3d[14, :]
    return joints_3d


def draw_limbs_2d(img, joints_2d, limb_parents, rect):
    # draw skeleton
    for limb_num in range(len(limb_parents)):
        x1 = joints_2d[limb_num, 0]
        y1 = joints_2d[limb_num, 1]
        x2 = joints_2d[limb_parents[limb_num], 0]
        y2 = joints_2d[limb_parents[limb_num], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        # here round() returns float type, so use int() to convert it to integer type
        polygon = cv2.ellipse2Poly((int(round((y1+y2)/2)), int(round((x1+x2)/2))),
                                   (int(length/2), 3),
                                   int(deg),
                                   0, 360, 1)
        img = cv2.fillConvexPoly(img, polygon, color=(49, 22, 122))
        # draw rectangle
        x, y, w, h = rect
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, (60, 66, 207), 4)

    return img


def draw_limbs_3d(joints_3d, joint_parents):
    fig = plt.figure()
    ax_3d = plt.axes(projection='3d')
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
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[joint_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[joint_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[joint_parents[i], 2]]
        ax_3d.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
    plt.ion()
    plt.show()


class PoseAnimation3d:
    def __init__(self, ax, joint_parents):
        self.joint_parents = joint_parents
        self.ax = ax
        self.ax.view_init(-90, -90)
        self.ax.set_xlim(-500, 500)
        self.ax.set_ylim(-500, 500)
        self.ax.set_zlim(-500, 500)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        white_color = (1.0, 1.0, 1.0, 0.0)
        self.ax.w_xaxis.set_pane_color(white_color)
        self.ax.w_yaxis.set_pane_color(white_color)
        self.ax.w_xaxis.line.set_color(white_color)
        self.ax.w_yaxis.line.set_color(white_color)
        self.ax.w_zaxis.line.set_color(white_color)
        self.skeletons = [self.ax.plot([], [], [], '-', linewidth=3)[0] for _ in range(21)]

    def ani_init(self):
        for skeleton in self.skeletons:
            skeleton.set_data([], [])
            skeleton.set_3d_properties([])
        return self.skeletons

    def __call__(self, joints_3d):
        for i, skeleton in enumerate(self.skeletons):
            x_pair = [joints_3d[i, 0], joints_3d[self.joint_parents[i], 0]]
            y_pair = [joints_3d[i, 1], joints_3d[self.joint_parents[i], 1]]
            z_pair = [joints_3d[i, 2], joints_3d[self.joint_parents[i], 2]]
            skeleton.set_data(x_pair, y_pair)
            skeleton.set_3d_properties(z_pair)
        return self.skeletons


def plot_3d_init(joint_parents, joints_iter_gen):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ani_update = PoseAnimation3d(ax, joint_parents)
    global ani
    ani = FuncAnimation(fig, ani_update, frames=joints_iter_gen, init_func=ani_update.ani_init, interval=20, blit=True)
    plt.ion()
    plt.show()


def plot_3d(q_start3d, q_joints, joint_parents):
    q_start3d.get()

    def joints_iter_gen_inner():
        while 1:
            yield q_joints.get(True)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ani_update = PoseAnimation3d(ax, joint_parents)
    global ani
    ani = FuncAnimation(fig, ani_update, frames=joints_iter_gen_inner, init_func=ani_update.ani_init, interval=15,
                        blit=True)
    plt.show()


def gen_heatmap(img_shape, center, sigma=3):
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
            # heatmap[y][x] = np.clip(heatmap[y][x], math.exp(-exp), 1.0)
            heatmap[y, x] = math.exp(-exp)
    return heatmap
