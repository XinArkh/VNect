#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import utils


class VnectEstimator:
    def __init__(self, video=0, box_size=368, hm_factor=8, joints_num=21, scales=(1.0, 0.7)):
        print('Initializing...')

        # the side length of the bounding box
        self.box_size = box_size
        # this factor indicates that the input box size is 8 times the side length of the output heatmaps
        self.hm_factor = hm_factor
        # number of the joints to be detected
        self.joints_num = joints_num
        # to scale the input bounding box with different ratio, no more than 1.0
        self.scales = scales
        # Limb parents of each joint
        self.limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

        self._clicked = False
        self._first_frame = True

        # use HOG method to initialize bounding box
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.rect = None

        # initialize the CNN model
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('./models/tf_model/vnect_tf.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./models/tf_model/'))
        graph = tf.get_default_graph()
        self.input_crops = graph.get_tensor_by_name('Placeholder:0')
        self.heatmap = graph.get_tensor_by_name('split_2:0')
        self.x_heatmap = graph.get_tensor_by_name('split_2:1')
        self.y_heatmap = graph.get_tensor_by_name('split_2:2')
        self.z_heatmap = graph.get_tensor_by_name('split_2:3')

        self.frame_square = None
        self.input_batch = None

        self.joints_2d = np.zeros((self.joints_num, 2), dtype=np.int32)
        self.joints_2d_prior = np.zeros((self.joints_num, 2), dtype=np.int32)
        self.joints_3d = np.zeros((self.joints_num, 3), dtype=np.float32)
        self.joints_3d_prior = np.zeros((self.joints_num, 3), dtype=np.float32)

        self.joints_filter_threshold = 100
        self.joints_filter_m = 0.8

        try:
            self.cameraCapture = cv2.VideoCapture(video)
        except Exception as e:
            print(e)
            raise

        self.W = int(self.cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._box_init_window = 'box init'
        cv2.namedWindow(self._box_init_window)
        cv2.setMouseCallback(self._box_init_window, self._on_mouse)

        # 3D joints visualization
        self.fig = plt.figure()
        self.ax_3d = plt.axes(projection='3d')
        plt.ion()
        self.ax_3d.clear()

        print('Initializing done.')

    def _on_mouse(self, event, x, y, flags, param):
        """
        attain mouse clicking message
        """
        if event == cv2.EVENT_LBUTTONUP:
            self._clicked = True

    def _frame_to_batch(self):
        """
        create multi-scale input batch
        input image range: [0, 255) --> [-0.4, 0.6)
        """
        self.input_batch = []
        for scale in self.scales:
            img = utils.img_scale_padding(self.frame_square, scale)
            self.input_batch.append(img)

        self.input_batch = np.asarray(self.input_batch, dtype=np.float32) / 255 - 0.4

    def _draw_rect(self, img, rect):
        x, y, w, h = rect
        offset = 70
        cv2.rectangle(img, (np.max([x - offset, 0]), y), (np.min([x + w + offset, self.W]), y + h), (32, 105, 221), 4)
        self.rect = [np.max([x - offset, 0]), y, np.min([x + w + offset, self.W]) - np.max([x - offset, 0]), h]

    def _run_benchmark(self):
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

        self.joints_2d = utils.extract_2d_joints_from_heatmap(hm_avg, self.box_size)
        self.joints_3d = utils.extract_3d_joints_from_heatmap(self.joints_2d, xm_avg, ym_avg, zm_avg, self.box_size,
                                                              self.hm_factor)
        # print(self.joints_2d)

    def _joints_filter(self):
        if np.any(self.joints_2d_prior):
            for i in range(self.joints_num):
                if self.joints_filter_threshold < np.sqrt(np.sum((self.joints_2d[i, :]-self.joints_2d_prior[i, :])**2)):
                    self.joints_2d[i, :] = self.joints_filter_m * self.joints_2d_prior[i, :] + (
                            1-self.joints_filter_m) * self.joints_2d[i, :]
                    self.joints_3d[i, :] = self.joints_filter_m * self.joints_3d_prior[i, :] + (
                            1-self.joints_filter_m) * self.joints_3d[i, :]

        self.joints_2d_prior = self.joints_2d
        self.joints_3d_prior = self.joints_3d

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
        utils.draw_limbs_3d(self.ax_3d, self.joints_3d, self.limb_parents)

        # plt.pause(0.00001)  # this line is unnecessary under matplotlib 3.0.0, but ought to be activated
                              # under matplotlib 3.0.2 (other versions not tested)

    def run(self):
        start = False
        success, frame_raw = self.cameraCapture.read()
        while success and cv2.waitKey(1) == -1:
            t = time.time()
            frame = frame_raw.copy()

            if self._first_frame and self._clicked:
                self._first_frame = False
                self._clicked = False
                start = True
                cv2.destroyWindow(self._box_init_window)
                plt.show()
                x, y, w, h = self.rect
                continue

            if not start:
                found, w = self.hog.detectMultiScale(frame)
                if len(found) > 0:
                    self._draw_rect(frame, found[np.argmax(w)])
                scale = 400 / self.H
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                cv2.imshow(self._box_init_window, frame)
                success, frame_raw = self.cameraCapture.read()
                continue

            frame_cropped = frame[y:y+h, x:x+w, :]
            self.frame_square = utils.read_square_image(frame_cropped, self.box_size)
            self._frame_to_batch()  # generate input batch
            self._run_benchmark()  # generate 2d and 3d joint coordinates
            self._joints_filter()  # smooth the joint coordinate results
            self.frame_square = utils.draw_limbs_2d(self.frame_square, self.joints_2d, self.limb_parents)
            cv2.imshow('2D results', self.frame_square)
            self._imshow_3d()

            print('FPS: {:>2.2f}'.format(1 / (time.time() - t)))
            success, frame_raw = self.cameraCapture.read()

        self._exit()

    def _exit(self):
        try:
            self.cameraCapture.release()
            # self.videoWriter1.release()
            # self.videoWriter2.release()
            cv2.destroyAllWindows()
            # self.position.close()
        except Exception as e:
            print(e)
            raise


if __name__ == '__main__':
    estimator = VnectEstimator('./test_src/test_video.mp4')
    estimator.run()
