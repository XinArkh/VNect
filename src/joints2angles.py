import time
import numpy as np
from OneEuroFilter import OneEuroFilter


def cal_angle(v1, v2):
    cos_a = vector_dot_product(v1, v2) / (vector_norm(v1) * vector_norm(v2))
    return np.arccos(cos_a)


def vector_cross_product(v1, v2):
    return np.cross(v1, v2)


def vector_dot_product(v1, v2):
    return np.dot(v1, v2)


def vector_norm(v):
    return np.linalg.norm(v)


class joints2angles:
    def __init__(self, filter=True):
        self.filter = filter

        if self.filter:
            # filter configuration
            config_filter = {
                'freq': 120,
                'mincutoff': 0.5,
                'beta': 0.3,
                'dcutoff': 1.0
            }
            self.filter_angles = [OneEuroFilter(**config_filter) for _ in range(8)]

    def __call__(self, joints_3d):
        angles = list(self.joints2angles(joints_3d))

        if self.filter:
            for i, angle in enumerate(angles):
                angles[i] = self.filter_angles[i](angle, time.time())

        s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r = angles
        print('%5.2f | %5.2f | %5.2f | %5.2f | %5.2f | %5.2f | %5.2f | %5.2f' %
              (s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r))

        angles = [s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r]

        return angles

    @staticmethod
    def joints2angles(joints_3d):
        # left arm
        shoulder_l = joints_3d[5]
        elbow_l = joints_3d[6]
        wrist_l = joints_3d[7]

        s_2_e_l = elbow_l - shoulder_l
        e_2_w_l = wrist_l - elbow_l

        # right arm
        shoulder_r = joints_3d[2]
        elbow_r = joints_3d[3]
        wrist_r = joints_3d[4]

        s_2_e_r = elbow_r - shoulder_r
        e_2_w_r = wrist_r - elbow_r

        # auxiliary lines
        head = joints_3d[0]
        pelvis = joints_3d[14]

        aux_v1_l = shoulder_r - shoulder_l
        aux_v1_r = -aux_v1_l
        aux_v2 = pelvis - head  # down direction
        aux_v3_l = vector_cross_product(s_2_e_l, aux_v2)
        aux_v3_r = vector_cross_product(s_2_e_r, aux_v2)
        aux_v4_l = vector_cross_product(s_2_e_l, e_2_w_l)
        aux_v4_r = vector_cross_product(s_2_e_r, e_2_w_r)

        # calculate angles
        # left arm
        s0_l = np.pi * 3 / 4 - cal_angle(aux_v1_l, aux_v3_l)
        s1_l = np.pi / 2 - cal_angle(aux_v2, s_2_e_l)
        e0_l = -cal_angle(aux_v3_l, aux_v4_l)
        e1_l = cal_angle(s_2_e_l, e_2_w_l)

        # right arm
        s0_r = np.pi / 4 - cal_angle(aux_v1_r, aux_v3_r)
        s1_r = np.pi / 2 - cal_angle(aux_v2, s_2_e_r)
        e0_r = cal_angle(aux_v3_r, aux_v4_r)
        e1_r = cal_angle(s_2_e_r, e_2_w_r)

        # return s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r
        return np.round(np.rad2deg([s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r])).astype(np.int32)
