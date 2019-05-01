import numpy as np
from numpy import sin, cos

'''
在test.mapping()中输入一个列表，分别是四个角度，得到的返回值 position 即为机械臂的目标值
'''


class Space:
    def __init__(self):
        self.pi = np.pi
        # arm
        self.aa = np.array([0, 0, 0, 260, 0, 0, 0, 100])
        self.alphaa = np.array([0, -0.5, 0.5, 0, 0.5, 0.5, 0.5, 0]) * self.pi
        self.da = np.array([0, 0, 0, 0, 0, 280, 0, 0])
        self.theta0a = np.array([0, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0]) * self.pi
        self.theta_min_arm = np.array([-90, -180, -180, -10, -90, -90, -15, 0]) / 180 * self.pi
        self.theta_range_arm = np.array([180, 230, 260, 155, 180, 160, 55, 0]) / 180 * self.pi
        self.kmax = np.array([0.71204307, 0.96456569, 0.76982129])
        self.kmin = np.array([1.14989032, 0.79122682, 0.83789371])
        self.cabb = np.array([7.57962147, -8.68547614, 259.01196291])
        self.carm = np.array([-1.05283406, -23.66622694, 1.34800121])

        # self.aa = np.array([0, 0, 260, 0, 0, 0, 100])
        # self.alphaa = np.array([-0.5, 0.5, 0, 0.5, 0.5, 0.5, 0]) * self.pi
        # self.da = np.array([0, 0, 0, 0, 280, 0, 0])
        # self.theta0a = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0]) * self.pi
        # self.theta_min_arm = np.array([-90, -180, -180, -10, -90, -90, -15]) / 180 * self.pi
        # self.theta_range_arm = np.array([180, 230, 260, 155, 180, 160, 55]) / 180 * self.pi
        # self.kmax = [0.71204307, 0.96456569, 0.76982129]
        # self.kmin = [1.14989032, 0.79122682, 0.83789371]
        # map

    def TransMatArm(self, theta):
        theta = theta + self.theta0a
        TMatHand = np.eye(4)
        for i in range(8):
            T = np.array(
                [[cos(theta[i]), -sin(theta[i]), 0, self.aa[i]],
                 [sin(theta[i]) * cos(self.alphaa[i]), cos(theta[i]) * cos(self.alphaa[i]), -sin(self.alphaa[i]),
                  -sin(self.alphaa[i]) * self.da[i]],
                 [sin(theta[i]) * sin(self.alphaa[i]), cos(theta[i]) * sin(self.alphaa[i]), cos(self.alphaa[i]),
                  cos(self.alphaa[i]) * self.da[i]],
                 [0, 0, 0, 1]])
            TMatHand = TMatHand.dot(T)

        return TMatHand

    def mapping(self, theta):
        theta[1:3] = theta[1:3][::-1]
        theta = np.append(theta, [0, 0, 0, 0])
        position = self.TransMatArm(theta)[0:3, 3]
        # print(position)
        position = np.array([-position[0], position[1], -position[2]]) - self.carm
        for j in range(3):
            if position[j] >= 0:
                position[j] *= self.kmax[j]
            else:
                position[j] *= self.kmin[j]
        position += self.cabb

        return position


if __name__ == '__main__':
    test = Space()
    # position = test.mapping([0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi])
    position = test.mapping([0, 0.5*np.pi, 0, 0])
    print(position)
