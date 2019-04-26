# dependencies (python3): pip3 install opencv-python, matplotlib, tensorflow==1.9
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

parser = argparse.ArgumentParser(prog="inference deep stereo")
parser.add_argument('--path_to_original_model_ckpt', type=str, default='/home/thomas/master_thesis_code/deep_stereo/deep_stereo/MADNet_model/kitti/weights.ckpt')
parser.add_argument('--test_image_left', type=str, default='/home/thomas/data/stereoTuner_stereoSGBM/marvin/left_test_undist_rect_sky.PNG')
parser.add_argument('--test_image_right', type=str, default='/home/thomas/data/stereoTuner_stereoSGBM/marvin/right_test_undist_rect_sky.PNG')
parser.add_argument('--path_where_to_store_the_model', type=str, default='.')
parser.add_argument('--save_the_model', type=bool, default=False)
parser.add_argument('--subsample_image_by_2', type=bool, default=False)
parser.add_argument('--measure_inference_time_by_10_runs', type=bool, default=False)
args = parser.parse_args()

print('\nIMPORTANT: If you want to save a graph with this script which will be used to run the model in C++. \
Make sure you use the same python version to run this script as you will use in C++ to run the model. \
E.g. use tensorflow 1.9 for both!\n')

if (args.subsample_image_by_2):
    image_size = [240, 376]
else:
    image_size = [480, 752]

input_layer_left = 'left_input_image'
input_layer_right = 'right_input_image'
left_img_batch = tf.placeholder(tf.float32,shape=[1,image_size[0],image_size[1],3], name=input_layer_left)
right_img_batch = tf.placeholder(tf.float32, shape=[1,image_size[0],image_size[1],3], name=input_layer_right)

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

print('\ninput tensor names are: ', input_layer_left, ' and ', input_layer_right, '. Output tensor is: ', prediction, '\n')
sess = tf.Session()

#init stuff
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

#restore disparity inference weights
path_to_weights = args.path_to_original_model_ckpt
var_to_restore = weights_utils.get_var_to_restore_list(path_to_weights, [])
assert(len(var_to_restore)>0)
restorer = tf.train.Saver(var_list=var_to_restore)
restorer.restore(sess, path_to_weights)
print('Disparity Net Restored?: {}, number of restored variables: {}'.format(True,len(var_to_restore)))

left_image = read_in_image(args.test_image_left)
right_image = read_in_image(args.test_image_right)

if args.subsample_image_by_2:
    left_image = left_image[0:-1:2, 0:-1:2, :]
    right_image = right_image[0:-1:2, 0:-1:2, :]

assert(image_size[0] == left_image.shape[0])
assert(image_size[1] == left_image.shape[1])

start = time.time()
if args.measure_inference_time_by_10_runs:
    number_of_runs = 10
else:
    number_of_runs = 1
for k in range(0,number_of_runs):
    inputs={
            left_img_batch: np.expand_dims(left_image, axis=0),
            right_img_batch: np.expand_dims(right_image, axis=0),
    }
    disparity = sess.run(prediction,feed_dict=inputs)

time_elapsed = (time.time() - start) / number_of_runs
print('time elapsed: %.6f' % time_elapsed)

print('CLOSE THE SHOWN IMAGE TO PROCEED TO THE MODEL AND GRAPH SAVING')
disparity_to_show = disparity[0,:,:,0]
plt.imshow(disparity_to_show)
plt.show()

#save the model
if args.save_the_model:
    print('will save the model here: ', args.path_where_to_store_the_model)
    if not os.path.isdir(args.path_where_to_store_the_model):
        print('given locatino does not exist!')
        sys.exit(0)
    saver = tf.train.Saver()
    if args.subsample_image_by_2:
        prefix = 'subsampled_'
    else:
        prefix = ''
    saver.save(sess, os.path.join(args.path_where_to_store_the_model, prefix + 'proper_named_model.ckpt'))
    tf.train.write_graph(sess.graph.as_graph_def(), args.path_where_to_store_the_model, prefix + 'graph.pb')
    print('model and graph saved!')
    print('DONE')