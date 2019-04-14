#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import time
import roslibpy
import numpy as np


class RosTalker:
    def __init__(self, host, port=9090):
        self.host = host
        self.port = port
        # self.client = roslibpy.Ros(host=self.host, port=self.port)
        # self.talker = roslibpy.Topic(self.client, '/chatter', 'std_msgs/Float64MultiArray')
        # self.client.run()
        # count = 0
        # while not self.client.is_connected:
        #     time.sleep(1)
        #     count += 1
        #     assert count < 5, 'ros connection overtime'

    def __call__(self, joints_3d):
        shoulder_l = joints_3d[5]
        shoulder_r = joints_3d[2]
        elbow_r = joints_3d[3]
        wrist_r = joints_3d[4]
        # neck = joints_3d[1]
        # pelvis = joints_3d[14]
        down_vector = [0, 1, 0]
        clavicle_vector = shoulder_l - shoulder_r
        s_2_e = elbow_r - shoulder_r
        e_2_w = wrist_r - elbow_r
        normal_s0 = np.cross(clavicle_vector, down_vector)
        s0 = np.pi / 4 - self.cal_angles(normal_s0, np.cross(s_2_e, down_vector))
        s1 = np.pi / 2 - self.cal_angles(s_2_e, down_vector)
        e1 = self.cal_angles(s_2_e, e_2_w)
        print(np.rad2deg(s0), np.rad2deg(s1), np.rad2deg(e1), shoulder_r, elbow_r, wrist_r)
        # self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
        #                                                  'data_offset': 0},
        #                                       'data': [s0.astype(np.float64), s1.astype(np.float64),
        #                                                e1.astype(np.float64)]}))
        print('send message')

    @staticmethod
    def cal_angles(v1, v2):
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cos_a) if cos_a >= 0 else np.pi / 2 + np.arccos(-cos_a)
