#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import time
import numpy as np
import tensorflow as tf
from src.vnect_model import VNect
from src.mpi_inf_3dhp import Mpi_Inf_3dhp


batch_size = 32

data = Mpi_Inf_3dhp(r'E:\Datasets\mpi_inf_3dhp')

model = VNect()

# sess = tf.Session()
# saver0 = tf.train.import_meta_graph('./models/tf_model/vnect_tf.meta')
# saver0.restore(sess, tf.train.latest_checkpoint('./models/tf_model/'))

graph = tf.get_default_graph()
input_crops = graph.get_tensor_by_name('Placeholder:0')
heatmap = graph.get_tensor_by_name('split_2:0')
# x_heatmap = graph.get_tensor_by_name('split_2:1')
# y_heatmap = graph.get_tensor_by_name('split_2:2')
# z_heatmap = graph.get_tensor_by_name('split_2:3')

labels = tf.placeholder(dtype=np.float32, shape=(None, 46, 46, 21))

loss = tf.losses.mean_squared_error(labels, heatmap, scope='loss') / batch_size
# loss = tf.reduce_sum(tf.nn.l2_loss(tf.concat(labels, axis=0) - tf.concat(heatmap, axis=0)))*2/(46*46*21)/32
# loss = tf.reduce_mean(tf.nn.l2_loss(labels * heatmap))

o_learning_rate = 1e-3
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(o_learning_rate, global_step, 10000, 0.96, staircase=True)
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step, name='optimizer')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Start the main loop...')
    step = 0
    while True:
        step += 1
        start_time = time.time()
        imgs, heatmaps = data.load_data(batch_size)
        _, loss_value = sess.run([optimizer, loss],
                                 feed_dict={input_crops: np.asarray(imgs, dtype=np.float32) / 255 - 0.4,
                                            labels: heatmaps[..., :21]})
        duration = time.time() - start_time
        if step % 10 == 0:
            # 保存当前训练状态
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            format_str = 'step %d, loss = %.5f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

            # 保存当前模型
            saver.save(sess, 'models/trained/vnect_tf', global_step=step)

            # exit()
