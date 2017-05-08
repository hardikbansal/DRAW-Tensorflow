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


class VAE:
	def initialize(self):
		self.opt = trainOptions().parse()
		self.batch_size = opt.batch_size
		self.img_width = opt.img_width
		self.img_height = opt.img_height
		self.img_depth = opt.img_depth
		self.z_size = opt.z_size
		self.img_size = self.img_depth*self.img_height*self.img_width
		self.nef = opt.nef



	def encoder(self,input_x, name="encoder"):
		
		with tf.variable_scope(name) as scope:

			o_c1 = general_conv2d(input_x, self.nef, 5, 5, 2, 2, padding="SAME", name="c1", do_norm=False)
			o_c2 = general_conv2d(o_c1, self.nef*2, 5, 5, 2, 2, padding="SAME", name="c1", do_norm=False)

			shape_c = tf.shape(o_c2)
			size_h = shape_h[1]*shape_h[2]*shape_h[3]

			h = tf.reshape(o_c2,[self.batch_size, size_h])

			mean = linear1d(h, size_h, self.z_size, name="mean")
			stddev = linear1d(h, size_h, self.z_size, name="stddev")

			return mean, stddev


	def train(self):

		input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_depth])