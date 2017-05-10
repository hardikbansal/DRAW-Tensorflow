import tensorflow as tf
import numpy as np
import os
import shutil
import time
import random
from options import trainOptions

from layers import *

from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
from PIL import Image


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

		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



	def encoder(self, input_x, name="encoder"):
		
		with tf.variable_scope(name) as scope:

			o_c1 = general_conv2d(input_x, self.nef, 5, 5, 2, 2, padding="SAME", name="c1", do_norm=False)
			o_c2 = general_conv2d(o_c1, self.nef*2, 5, 5, 2, 2, padding="SAME", name="c1", do_norm=False)

			shape_c = tf.shape(o_c2)
			size_h = shape_h[1]*shape_h[2]*shape_h[3]

			h = tf.reshape(o_c2,[self.batch_size, size_h])

			mean = linear1d(h, size_h, self.z_size, name="mean")
			stddev = linear1d(h, size_h, self.z_size, name="stddev")

			return mean, stddev


	def decoder(self, input_z, name="decoder"):

		with tf.variable_scope(name) as scope:

			o_l = linear1d(input_z, self.z_size, 7*7*nef*2, name="revlin")

			o_h = tf.nn.relu(tf.reshape(o_l, [self.batch_size, 7, 7, nef*2))
			o_d1 = general_deconv2d(o_h, nef, 5, 5, 2, 2, padding="SAME", name="d1", do_norm=False)
			o_d2 = general_deconv2d(o_d1, 1, 5, 5, 2, 2, padding="SAME", name="d1", do_norm=False, do_relu=False)

			return tf.nn.sigmoid(o_d2)

	def generation_loss(self, input_img, output_img, loss_type='diff'):

		if (loss_type == 'diff'):
			return tf.reduce_sum(tf.squared_difference(input_img, output_img),1)
		elif (loss_type == 'log_diff'):
			epsilon = 1e-8
			return tf.reduce_sum(input_img*tf.log(output_img+epsilon) + (1 - input_img)*tf.log(epsilon + 1 - output_img),1) 

	def setup(self):

		self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_depth])

		mean_z, std_z = self.encoder(self.input_x, "encoder")

		#Now we need to extract a vector from N(mean_z, std_z)

		z_sample = tf.random_normal([self.batch_size, self.z_size], 0 , 1)
		z_sample = z_sample*std_z + mean_z

		gen_x_temp = self.decoder(z_sample, "decoder")

		self.gen_x = tf.reshape(gen_x_temp,[self.batch_size, self.img_width, self.img_height, self.img_depth])

		model_vars = tf.trainable_variables()
		self.encoder_variables = [var for var in model_vars if 'encoder' in var.name]
		self.decoder_variables = [var for var in model_vars if 'decoder' in var.name]

		# Loss Function

		self.gen_loss = self.generation_loss(self.input_x, self.gen_x)
		self.latent_loss = tf.reduce_sum(tf.square(mean_z) + tf.square(std_z) - tf.log(tf.square(std_z)),1)

		self.vae_loss = self.gen_loss + self.latent_loss

		optimizer = tf.train.AdamOptimizer(0.001)

		self.loss_optimizer = optimizer.minimize()
		
		#Printing the model variables

		for vars in  model_vars:
			print(vars.name)

	def train(self):
		
		self.setup()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		# Train

		with tf.Session() as sess:

			sess.run(init)

			for epochs in range(0,self.max_epoch):
				for itr in range(0,int(self.n_samples/self.batch_size)):
					batch = self.mnist.next_bacth(self.batch_size)
					imgs = batch[0]
					labels = batch[1]

					mean, stddev = sess.run(,feed_dict={input_x:imgs})
