import os
from init import *
from src.vnect_model import VNect


# caffe model path
caffe_path = './models/caffe_model'
prototxt_name = 'vnect_net.prototxt'
caffemodel_name = 'vnect_model.caffemodel'
# pickle file path
pkl_file = './models/caffe_model/params.pkl'
# tensorflow model path
tf_save_path = './models/tf_model'

if not os.path.exists(pkl_file):
    caffe2pkl(caffe_path, prototxt_name, caffemodel_name)

model = VNect()
load_tf_weights(pkl_file, tf_save_path, model)
