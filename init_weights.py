#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Run this code to build and save tensorflow model with corresponding weight values for VNect
"""


import os
import sys
sys.path.extend([os.path.join(os.path.abspath(__file__), 'src')])
import tensorflow as tf
from src.caffe2pkl import caffe2pkl
from src.vnect_model import VNect


def init_tf_weights(pfile, spath, model):
    # configurations
    PARAMSFILE = pfile
    SAVERPATH = spath

    if not tf.gfile.Exists(SAVERPATH):
        tf.gfile.MakeDirs(SAVERPATH)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        model.load_weights(sess, PARAMSFILE)
        saver.save(sess, os.path.join(SAVERPATH, 'vnect_tf'))


# caffe model basepath
caffe_bpath = './models/caffe_model'
# caffe model files
prototxt_name = 'vnect_net.prototxt'
caffemodel_name = 'vnect_model.caffemodel'
# pickle file name
pkl_name = 'params.pkl'
pkl_file = os.path.join(caffe_bpath, pkl_name)
# tensorflow model path
tf_save_path = './models/tf_model'

if not os.path.exists(pkl_file):
    caffe2pkl(caffe_bpath, prototxt_name, caffemodel_name, pkl_name)

model = VNect()
init_tf_weights(pkl_file, tf_save_path, model)
