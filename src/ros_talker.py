#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import sys
sys.path.extend([os.path.dirname(os.path.abspath(__file__))])
import time
import roslibpy
import numpy as np
from Baxter2Yumi import Space


class RosTalker:
    def __init__(self, host, port=9090, yumi=False):
        self.host = host
        self.port = port
        self.yumi = yumi

        # yumi transfer
        if self.yumi:
            self.Trans = Space()

        # ros connection
        self.client = roslibpy.Ros(host=self.host, port=self.port)
        self.talker = roslibpy.Topic(self.client, '/chatter', 'std_msgs/Float64MultiArray')
        self.client.run()
        count = 0
        while not self.client.is_connected:
            time.sleep(1)
            count += 1
            assert count < 5, 'ros connection overtime'

    def send(self, angles):
        s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r = angles
        if not self.yumi:
            self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
                                                             'data_offset': 0},
                                                  'data': [s0_l.astype(np.float64), s1_l.astype(np.float64),
                                                           e0_l.astype(np.float64), e1_l.astype(np.float64),
                                                           s0_r.astype(np.float64), s1_r.astype(np.float64),
                                                           e0_r.astype(np.float64), e1_r.astype(np.float64)]}))
        else:
            x, y, z = self.Trans.mapping([s0_r, s1_r, e0_r, e1_r]) / 1000
            print('%5.2f | %5.2f | %5.2f' % (x, y, z))
            self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
                                                             'data_offset': 0},
                                                  'data': [x.astype(np.float64), y.astype(np.float64),
                                                           z.astype(np.float64)]}))
        print('Sending message to ros master...')
