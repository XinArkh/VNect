#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utils
from OneEuroFilter import OneEuroFilter


class VnectEstimator:
    def __init__(self, video=None, T=False):
        print('Initializing...')

        # the input camera serial number of the PC (int), or PATH to input video (str)
        self.video = 0 if not video else video
        # whether apply transposed matrix
        self.T = T

        ## hyper-parameters ##
        # the side length of the bounding box
        self.box_size = 368
        # this factor indicates that the input box size is 8 times the side length of the output heatmaps
        self.hm_factor = 8
        # number of the joints to be detected
        self.joints_num = 21
        # the ratio factors to scale the input image crops, no more than 1.0
        self.scales = [1]  # or [1, 0.7] to be consistent with the author
        # parent joint indexes of each joint (for plotting the skeleton lines)
        self.joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

        ## one euro filter ##
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
        self.filter_2d = [(OneEuroFilter(**config_2d), OneEuroFilter(**config_2d)) for _ in range(self.joints_num)]
        self.filter_3d = [(OneEuroFilter(**config_3d), OneEuroFilter(**config_3d), OneEuroFilter(**config_3d))
                          for _ in range(self.joints_num)]

        ## flags ##
        # flag for determining whether the left mouse button is clicked
        self._clicked = False

        ## place holders ##
        self.rect = None
        self.frame_square = None
        self.input_batch = None
        self.joints_2d = None
        self.joints_3d = None

        # VNect model
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('./models/tf_model/vnect_tf.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./models/tf_model/'))
        graph = tf.get_default_graph()
        self.input_crops = graph.get_tensor_by_name('Placeholder:0')
        self.heatmap = graph.get_tensor_by_name('split_2:0')
        self.x_heatmap = graph.get_tensor_by_name('split_2:1')
        self.y_heatmap = graph.get_tensor_by_name('split_2:2')
        self.z_heatmap = graph.get_tensor_by_name('split_2:3')

        # catch the video stream
        self.cameraCapture = cv2.VideoCapture(self.video)
        assert self.cameraCapture.isOpened(), 'Video stream not opened: %s' % self.video

        # frame width and height
        self.W = int(self.cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.T:
            self.W, self.H = self.H, self.W

        # 3D joints visualization
        self.fig = plt.figure()
        self.ax_3d = plt.axes(projection='3d')
        plt.ion()
        self.ax_3d.clear()

        print('Initialization done.')

    def BB_init(self):
        # use HOG method to initialize bounding box
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self._box_init_window_name = 'Bounding Box Initialization'
        cv2.namedWindow(self._box_init_window_name)
        cv2.setMouseCallback(self._box_init_window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        """
        attain mouse clicking message
        """
        if event == cv2.EVENT_LBUTTONUP:
            self._clicked = True

    def _draw_BB_rect(self, img, rect):
        """
        draw bounding box in the BB initialization window, and record current rect (x, y, w, h)
        """
        x, y, w, h = rect
        offset_w = int(0.4/2 * self.W)
        offset_h = int(0.2/2 * self.H)
        pt1 = (np.max([x - offset_w, 0]), np.max([y - offset_h, 0]))
        pt2 = (np.min([x + w + offset_w, self.W]), np.min([y + h + offset_h, self.H]))
        # print(pt1, pt2)
        cv2.rectangle(img, pt1, pt2, (28, 76, 242), 4)
        self.rect = [np.max([x - offset_w, 0]),  # x
                     np.max([y - offset_h, 0]),  # y
                     np.min([x + w + offset_w, self.W]) - np.max([x - offset_w, 0]),  # w
                     np.min([y + h + offset_h, self.H]) - np.max([y - offset_h, 0])]  # h

    def _create_input_batch(self):
        """
        create multi-scale input batch
        input image range: [0, 255) --> [-0.4, 0.6)
        """
        self.input_batch = []
        for scale in self.scales:
            img = utils.img_scale_padding(self.frame_square, scale) if scale < 1 else self.frame_square
            self.input_batch.append(img)

        self.input_batch = np.asarray(self.input_batch, dtype=np.float32) / 255 - 0.4

    def _run_net(self):
        # inference
        hm, xm, ym, zm = self.sess.run([self.heatmap, self.x_heatmap, self.y_heatmap, self.z_heatmap],
                                       {self.input_crops: self.input_batch})

        # average scale outputs
        hm_size = self.box_size // self.hm_factor
        hm_avg = np.zeros((hm_size, hm_size, self.joints_num))
        xm_avg = np.zeros((hm_size, hm_size, self.joints_num))
        ym_avg = np.zeros((hm_size, hm_size, self.joints_num))
        zm_avg = np.zeros((hm_size, hm_size, self.joints_num))
        for i in range(len(self.scales)):
            rescale = 1.0 / self.scales[i]
            scaled_hm = utils.img_scale(hm[i, :, :, :], rescale)
            scaled_x_hm = utils.img_scale(xm[i, :, :, :], rescale)
            scaled_y_hm = utils.img_scale(ym[i, :, :, :], rescale)
            scaled_z_hm = utils.img_scale(zm[i, :, :, :], rescale)
            mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
            hm_avg += scaled_hm[mid[0] - hm_size//2: mid[0] + hm_size//2, mid[1] - hm_size//2: mid[1] + hm_size//2, :]
            xm_avg += scaled_x_hm[mid[0] - hm_size//2: mid[0] + hm_size//2, mid[1] - hm_size//2: mid[1] + hm_size//2, :]
            ym_avg += scaled_y_hm[mid[0] - hm_size//2: mid[0] + hm_size//2, mid[1] - hm_size//2: mid[1] + hm_size//2, :]
            zm_avg += scaled_z_hm[mid[0] - hm_size//2: mid[0] + hm_size//2, mid[1] - hm_size//2: mid[1] + hm_size//2, :]

        hm_avg /= len(self.scales)
        xm_avg /= len(self.scales)
        ym_avg /= len(self.scales)
        zm_avg /= len(self.scales)

        self.joints_2d = utils.extract_2d_joints_from_heatmap(hm_avg, self.box_size, self.hm_factor)
        self.joints_3d = utils.extract_3d_joints_from_heatmap(self.joints_2d, xm_avg, ym_avg, zm_avg, self.box_size,
                                                              self.hm_factor)
        # print(self.joints_2d, '\n', self.joints_3d)

    def _joint_coord_filter(self):
        for i in range(self.joints_num):
            self.joints_2d[i, 0] = self.filter_2d[i][0](self.joints_2d[i, 0], time.time())
            self.joints_2d[i, 1] = self.filter_2d[i][1](self.joints_2d[i, 1], time.time())

            self.joints_3d[i, 0] = self.filter_3d[i][0](self.joints_3d[i, 0], time.time())
            self.joints_3d[i, 1] = self.filter_3d[i][1](self.joints_3d[i, 1], time.time())
            self.joints_3d[i, 2] = self.filter_3d[i][2](self.joints_3d[i, 2], time.time())

            # TODO: 监控并剔除跳变点

    def _imshow_3d(self):
        self.ax_3d.clear()
        self.ax_3d.view_init(-90, -90)
        self.ax_3d.set_xlim(-500, 500)
        self.ax_3d.set_ylim(-500, 500)
        self.ax_3d.set_zlim(-500, 500)
        self.ax_3d.set_xticks([])
        self.ax_3d.set_yticks([])
        self.ax_3d.set_zticks([])
        white = (1.0, 1.0, 1.0, 0.0)
        self.ax_3d.w_xaxis.set_pane_color(white)
        self.ax_3d.w_yaxis.set_pane_color(white)
        self.ax_3d.w_xaxis.line.set_color(white)
        self.ax_3d.w_yaxis.line.set_color(white)
        self.ax_3d.w_zaxis.line.set_color(white)
        utils.draw_limbs_3d(self.ax_3d, self.joints_3d, self.joint_parents)

        # the following line is unnecessary with matplotlib 3.0.0, but ought to be activated
        # under matplotlib 3.0.2 (other versions not tested)
        # plt.pause(0.00001)

    def run(self):
        # initial BB by HOG detection
        self.BB_init()
        success, frame = self.cameraCapture.read(); frame = frame.T if self.T else frame
        while success and cv2.waitKey(1) == -1:
            found, w = self.hog.detectMultiScale(frame)
            if len(found) > 0:
                self._draw_BB_rect(frame, found[np.argmax(w)])
            scale = 400 / self.H
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            cv2.imshow(self._box_init_window_name, frame)

            if self._clicked:
                self._clicked = False
                cv2.destroyWindow(self._box_init_window_name)
                break

            success, frame = self.cameraCapture.read(); frame = frame.T if self.T else frame

        x, y, w, h = self.rect
        plt.show()

        # main loop
        success, frame = self.cameraCapture.read(); frame = frame.T if self.T else frame
        while success and cv2.waitKey(1) == -1:
            t = time.time()

            # crop bounding box from the raw frame
            frame_cropped = frame[y:y+h, x:x+w, :]
            # crop --> one sqrare input img for CNN
            self.frame_square = utils.img_scale_squareify(frame_cropped, self.box_size)
            # one sqrare input img --> a batch of sqrare input imgs
            self._create_input_batch()
            # sqrare input img batch --CNN net--> 2d and 3d skeleton joint coordinates
            self._run_net()
            # filter to smooth the joint coordinate results
            self._joint_coord_filter()

            ## plot ##
            # 2d plotting
            self.frame_square = utils.draw_limbs_2d(self.frame_square, self.joints_2d, self.joint_parents)
            cv2.imshow('2D Prediction', self.frame_square)
            # 3d plotting
            self._imshow_3d()

            print('FPS: {:>2.2f}'.format(1 / (time.time() - t)))
            success, frame = self.cameraCapture.read(); frame = frame.T if self.T else frame

        self._exit()

    def _exit(self):
        try:
            self.cameraCapture.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)
            raise


if __name__ == '__main__':
    estimator = VnectEstimator('./pic/test_video.mp4')
	# estimator = VnectEstimator(0)
    estimator.run()
