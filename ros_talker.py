#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import time
import roslibpy
import numpy as np


class RosTalker:
    def __init__(self, host, port=9090, queue=3):
        self.host = host
        self.port = port
        self.queue = queue
        self.sum = [0, 0, 0]
        self.count = 0
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
        head = joints_3d[0]
        pelvis = joints_3d[14]
        down_vector = pelvis - head
        clavicle_vector = shoulder_l - shoulder_r
        s_2_e = elbow_r - shoulder_r
        e_2_w = wrist_r - elbow_r
        # normal_s0 = np.cross(clavicle_vector, down_vector)
        s0 = np.pi / 4 - self.cal_angles(np.cross(s_2_e, down_vector), clavicle_vector)
        s1 = np.pi / 2 - self.cal_angles(down_vector, s_2_e)
        e1 = self.cal_angles(s_2_e, e_2_w)
        self.sum[0] += s0
        self.sum[1] += s1
        self.sum[2] += e1
        self.count += 1
        # if self.count == self.queue:
        #     self.count = 0
        #     s0, s1, e1 = np.array(self.sum) / self.queue
        #     self.sum = [0, 0, 0]
        #     _, _, e1 = np.array(self.sum)
        print(np.rad2deg(s0), np.rad2deg(s1), np.rad2deg(e1), s_2_e, e_2_w)
        #     self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
        #                                                      'data_offset': 0},
        #                                           'data': [s0.astype(np.float64), s1.astype(np.float64),
        #                                                    e1.astype(np.float64)]}))
        #     print('send message')

    @staticmethod
    def cal_angles(v1, v2):
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cos_a)
