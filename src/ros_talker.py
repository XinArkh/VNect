#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import time
import roslibpy
import numpy as np
from OneEuroFilter import OneEuroFilter
from Baxter2Yumi import Space


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

    def __call__(self, angles):
        s0, s1, e0, e1 = angles[4:]
        if not self.yumi:
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
        print('sending message to ros master...')
