# dependencies (python3): pip3 install opencv-python, matplotlib, tensorflow
# run by:
# python3 test_inference.py

import tensorflow as tf
import numpy as np
import argparse
import Nets
import os
import sys
import time
import cv2
from matplotlib import pyplot as plt
from Data_utils import data_reader,weights_utils

def read_in_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # If image is grayscale, will have 3 channels now
    return image

left_img_batch = tf.placeholder(tf.float32,shape=[1,480,752,3], name='left_input')
right_img_batch = tf.placeholder(tf.float32, shape=[1,480,752,3], name='right_input')

with tf.variable_scope('model'):
    net_args = {}
    net_args['left_img'] = left_img_batch
    net_args['right_img'] = right_img_batch
    net_args['split_layers'] = [None]
    net_args['sequence'] = True
    net_args['train_portion'] = 'BEGIN'
    net_args['bulkhead'] = False
    stereo_net = Nets.get_stereo_net('MADNet', net_args)
    prediction = stereo_net.get_disparities()[-1]

sess = tf.Session()

#init stuff
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

#restore disparity inference weights
path_to_weights = '/home/thomas/master_thesis_code/deep_stereo/deep_stereo/MADNet_model/kitti/weights.ckpt'
var_to_restore = weights_utils.get_var_to_restore_list(path_to_weights, [])
assert(len(var_to_restore)>0)
restorer = tf.train.Saver(var_list=var_to_restore)
restorer.restore(sess, path_to_weights)
print('Disparity Net Restored?: {}, number of restored variables: {}'.format(True,len(var_to_restore)))

left_image = read_in_image('/home/thomas/data/stereoTuner_stereoSGBM/marvin/left_test_undist_rect_sky.PNG')
right_image = read_in_image('/home/thomas/data/stereoTuner_stereoSGBM/marvin/right_test_undist_rect_sky.PNG')


start = time.time()
number_of_runs = 100
for k in range(0,number_of_runs):
    inputs={
            left_img_batch: np.expand_dims(left_image, axis=0),
            right_img_batch: np.expand_dims(right_image, axis=0),
    }

    disparity = sess.run(prediction,feed_dict=inputs)

time_elapsed = (time.time() - start) / number_of_runs
print('time elapsed: %.6f' % time_elapsed)

disparity_to_show = disparity[0,:,:,0]

plt.imshow(disparity_to_show)
plt.show()

