#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import cv2
import numpy as np
import tensorflow as tf
import utils


box_size = 368
hm_factor = 8
joints_num = 21
scales = [1.0, 0.7]
limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./models/tf_model/vnect_tf.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./models/tf_model/'))

    graph = tf.get_default_graph()
    input_batch = graph.get_tensor_by_name('Placeholder:0')
    heatmap = graph.get_tensor_by_name('split_2:0')
    x_heatmap = graph.get_tensor_by_name('split_2:1')
    y_heatmap = graph.get_tensor_by_name('split_2:2')
    z_heatmap = graph.get_tensor_by_name('split_2:3')

    img = cv2.imread('./test_src/test_pic.jpg')
    img_square = utils.img_scale_squareify(img, box_size)
    img_square = img_square[np.newaxis, ...]

    hm, xm, ym, zm = sess.run([heatmap, x_heatmap, y_heatmap, z_heatmap], {input_batch: img_square/255-0.4})

    joints_2d = utils.extract_2d_joints_from_heatmap(hm[0, ...], box_size, hm_factor)

    for i in range(21):
        if i == 0:
            himg = hm[0, :, :, i]
            ximg = xm[0, :, :, i]
            yimg = ym[0, :, :, i]
            zimg = zm[0, :, :, i]
        else:
            tmp = hm[0, :, :, i]
            himg = np.hstack([himg, tmp])
            tmp = xm[0, :, :, i]
            ximg = np.hstack([ximg, tmp])
            tmp = ym[0, :, :, i]
            yimg = np.hstack([yimg, tmp])
            tmp = zm[0, :, :, i]
            zimg = np.hstack([zimg, tmp])

    all_hm = np.vstack([himg, ximg, yimg, zimg])
    cv2.imshow('all heatmaps', all_hm)

    img_res2d = utils.draw_limbs_2d(img_square[0, ...], joints_2d, limb_parents)
    cv2.imshow('2D results', img_res2d)

    cv2.waitKey()
    cv2.destroyAllWindows()
