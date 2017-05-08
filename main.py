import tensorflow as tf
import numpy as np
import os
import shutil
import time
import random
from options import trainOptions

from layers import *

# from tensorflow.examples.tutorials.mnist import input_data
# from scipy.misc import imsave
# from PIL import Image

# print(opt[0].num_iter)

class VAE:
	def initialize(self):
		self.opt = trainOptions().parse()
		self.batch_size = opt.batch_size
		self.img_width = opt.img_width
		self.img_height = opt.img_height
		self.img_depth = opt.img_depth
		self.z_size = opt.z_size
		self.img_size = self.img_depth*self.img_height*self.img_width



	def encoder(input_x, name="encoder"):
		
		with tf.variable_scope(name) as scope:

			o_c1 = general_conv2d(input_x, )

			weight = tf.get_variable('weight',[self.z_size, self.img_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
			bias = tf.get_variable('bias',[self.z_size], initializer=tf.constant_initializer(0.0))


	def train(self):

		input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_size])