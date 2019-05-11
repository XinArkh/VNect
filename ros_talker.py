#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import time
import roslibpy
import numpy as np
from src.OneEuroFilter import OneEuroFilter
from Baxter2Yumi import Space


def cal_angle(v1, v2):
    cos_a = vector_dot_product(v1, v2) / (vector_norm(v1) * vector_norm(v2))
    return np.arccos(cos_a)


def vector_cross_product(v1, v2):
    return np.cross(v1, v2)


def vector_dot_product(v1, v2):
    return np.dot(v1, v2)


def vector_norm(v):
    return np.linalg.norm(v)


class RosTalker:
    def __init__(self, host, port=9090, yumi=False):
        self.host = host
        self.port = port
        self.yumi = yumi

        # yumi transfer
        if self.yumi:
            self.Trans = Space()

        # filter
        config_filter = {
            'freq': 120,
            'mincutoff': 0.5,
            'beta': 0.3,
            'dcutoff': 1.0
        }
        self.filter_angles = [OneEuroFilter(**config_filter) for _ in range(4)]

        # ros connection
        self.client = roslibpy.Ros(host=self.host, port=self.port)
        self.talker = roslibpy.Topic(self.client, '/chatter', 'std_msgs/Float64MultiArray')
        self.client.run()
        count = 0
        while not self.client.is_connected:
            time.sleep(1)
            count += 1
            assert count < 5, 'ros connection overtime'

    def __call__(self, joints_3d):
        angles = list(self.joints2angles(joints_3d))
        for i, a, f in zip(np.arange(len(angles)), angles, self.filter_angles):
            angles[i] = f(a, time.time())
        s0, s1, e0, e1 = angles
        if not self.yumi:
            print('%5.2f | %5.2f | %5.2f | %5.2f *' %(np.rad2deg(s0), np.rad2deg(s1), np.rad2deg(e0), np.rad2deg(e1)))
            self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
                                                             'data_offset': 0},
                                                  'data': [s0.astype(np.float64), s1.astype(np.float64),
                                                           e0.astype(np.float64), e1.astype(np.float64)]}))
        else:
            x, y, z = self.Trans.mapping([s0, s1, e0, e1]) / 1000
            print('%5.2f | %5.2f | %5.2f' % (x, y, z))
            self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
                                                             'data_offset': 0},
                                                  'data': [x.astype(np.float64), y.astype(np.float64),
                                                           z.astype(np.float64)]}))
        print('send message')

    @staticmethod
    def joints2angles(joints_3d):
        shoulder_l = joints_3d[5]
        shoulder_r = joints_3d[2]
        elbow_r = joints_3d[3]
        wrist_r = joints_3d[4]
        head = joints_3d[0]
        pelvis = joints_3d[14]

        s_2_e = elbow_r - shoulder_r
        e_2_w = wrist_r - elbow_r

        aux_v1 = shoulder_l - shoulder_r
        aux_v2 = pelvis - head  # down direction
        aux_v3 = vector_cross_product(s_2_e, aux_v2)
        aux_v4 = vector_cross_product(s_2_e, e_2_w)

        s0 = np.pi / 4 - cal_angle(aux_v1, aux_v3)
        s1 = np.pi / 2 - cal_angle(aux_v2, s_2_e)
        e0 = cal_angle(aux_v3, aux_v4)
        e1 = cal_angle(s_2_e, e_2_w)
        print('%5.2f | %5.2f | %5.2f | %5.2f' % (np.rad2deg(s0), np.rad2deg(s1), np.rad2deg(e0), np.rad2deg(e1)))

        return s0, s1, e0, e1
