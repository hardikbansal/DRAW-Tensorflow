import tensorflow as tf
import numpy as np
import os
import shutil
import time
import random
import sys

from layers import *
from ops import *

from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
from PIL import Image
from options import trainOptions


class VAE():
	def initialize(self):
		opt = trainOptions().parse()[0]
		self.batch_size = opt.batch_size
		self.img_width = opt.img_width
		self.img_height = opt.img_height
		self.img_depth = opt.img_depth
		self.z_size = opt.z_size
		self.img_size = self.img_depth*self.img_height*self.img_width
		self.nef = opt.nef
		self.max_epoch = opt.max_epoch
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
		self.to_test = opt.test

		self.n_samples = self.mnist.train.num_examples

		self.tensorboard_dir = "./output/vae/tensorboard"
		self.check_dir = "./output/vae/checkpoints"
		self.images_dir = "./output/vae/imgs"

		self.setup_done = False





	def encoder(self, input_x, name="encoder"):

		with tf.variable_scope(name) as scope:

			o_c1 = general_conv2d(input_x, self.nef, 5, 5, 2, 2, padding="SAME", name="c1", do_norm=False)
			o_c2 = general_conv2d(o_c1, self.nef*2, 5, 5, 2, 2, padding="SAME", name="c2", do_norm=False)

			shape_c = o_c2.get_shape().as_list()
			size_h = shape_c[1]*shape_c[2]*shape_c[3]


			h = tf.reshape(o_c2,[self.batch_size, size_h])

			mean = linear1d(h, size_h, self.z_size, name="mean")
			stddev = linear1d(h, size_h, self.z_size, name="stddev")

			return mean, stddev


	def decoder(self, input_z, name="decoder"):

		with tf.variable_scope(name) as scope:

			o_l = linear1d(input_z, self.z_size, 7*7*self.nef*2, name="revlin")

			o_h = tf.nn.relu(tf.reshape(o_l, [self.batch_size, 7, 7, self.nef*2]))
			o_d1 = general_deconv2d(o_h, self.nef, 5, 5, 2, 2, padding="SAME", name="d1", do_norm=False)
			o_d2 = general_deconv2d(o_d1, 1, 5, 5, 2, 2, padding="SAME", name="d2", do_norm=False, do_relu=False)

			return tf.nn.sigmoid(o_d2)

	def generation_loss(self, input_img, output_img, loss_type='log_diff'):

		if (loss_type == 'diff'):
			return tf.reduce_sum(tf.squared_difference(input_img, output_img))
		elif (loss_type == 'log_diff'):
			epsilon = 1e-8
			return -tf.reduce_sum(input_img*tf.log(output_img+epsilon) + (1 - input_img)*tf.log(epsilon + 1 - output_img),[1, 2])

	def setup(self):


		with tf.variable_scope("Model") as scope:

			self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_depth])

			self.input_z = tf.placeholder(tf.float32, [self.batch_size, self.z_size]) # For testing

			self.mean_z, self.std_z = self.encoder(self.input_x, "encoder")

			#Now we need to extract a vector from N(mean_z, std_z)

			self.z_sample = tf.random_normal([self.batch_size, self.z_size], 0 , 1, dtype=tf.float32)
			self.z_sample = self.z_sample*self.std_z + self.mean_z

			self.gen_x = self.decoder(self.z_sample, "decoder")

			scope.reuse_variables()

			self.output_x = self.decoder(self.input_z,"decoder")

		model_vars = tf.trainable_variables()

		# Loss Function

		self.gen_loss = self.generation_loss(self.input_x, self.gen_x)
		self.latent_loss = 0.5*tf.reduce_sum(tf.square(self.mean_z) + tf.square(self.std_z) - tf.log(tf.square(self.std_z)) - 1,1)

		self.vae_loss = tf.reduce_mean(self.gen_loss + self.latent_loss)

		optimizer = tf.train.AdamOptimizer(0.001)

		self.loss_optimizer = optimizer.minimize(self.vae_loss)

		vae_loss_summ = tf.summary.scalar("vae_loss", self.vae_loss)

		self.summary_op = tf.summary.merge_all()

		#Printing the model variables

		for vars in  model_vars:
			print(vars.name)

	def train(self):

		#Setting up the model and graph
		self.setup()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		if not os.path.exists(self.images_dir+"/train/"):
			os.makedirs(self.images_dir+"/train/")
		if not os.path.exists(self.check_dir):
			os.makedirs(self.check_dir)

		# Train

		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)

			test_imgs = self.mnist.train.next_batch(self.batch_size)[0]
			test_imgs = test_imgs.reshape((self.batch_size,28,28,1))

			for epoch in range(0,self.max_epoch):

				for itr in range(0,int(self.n_samples/self.batch_size)):
					batch = self.mnist.train.next_batch(self.batch_size)
					imgs = batch[0]
					labels = batch[1]

					imgs = imgs.reshape((self.batch_size,28,28,1))

					print('In the iteration '+str(itr)+" of epoch"+str(epoch))

					_, summary_str = sess.run([self.loss_optimizer,self.summary_op],feed_dict={self.input_x:imgs})

					writer.add_summary(summary_str,epoch*int(self.n_samples/self.batch_size) + itr)

				# After each epoch things

				saver.save(sess,os.path.join(self.check_dir,"vae"),global_step=epoch)

				out_img_test = sess.run(self.gen_x,feed_dict={self.input_x:test_imgs})

				imsave(self.images_dir+"/train/epoch_"+str(epoch)+".jpg", flat_batch(out_img_test,self.batch_size,10,10))

			writer.add_graph(sess.graph)

	def test(self):

		if not os.path.exists(self.images_dir+"/test/"):
			os.makedirs(self.images_dir+"/test/")


		if not self.setup_done:
			self.setup()
			self.setup_done=True

		saver = tf.train.Saver()



		with tf.Session() as sess:

			chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
			saver.restore(sess,chkpt_fname)


			z_sample = np.random.normal(0, 1, [self.batch_size, self.z_size])

			gen_x_temp = sess.run(self.output_x,feed_dict={self.input_z:z_sample})

			imsave(self.images_dir+"/test/output.jpg", flat_batch(gen_x_temp,self.batch_size,10,10))

	def get_stats(self):

		if not os.path.exists(self.images_dir+"/test/"):
			os.makedirs(self.images_dir+"/test/")

		if not self.setup_done:
			self.setup()
			self.setup_done=True

		saver = tf.train.Saver()

		with tf.Session() as sess:

			chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
			saver.restore(sess,chkpt_fname)

			batch = self.mnist.test.next_batch(self.batch_size)

			imgs = batch[0]
			labels = batch[1]

			imgs = imgs.reshape((self.batch_size,28,28,1))

			z_sample_test, out_img_test = sess.run([self.latent_loss, self.gen_x],feed_dict={self.input_x:imgs})

			imsave(self.images_dir+"/test/input_sample.jpg", flat_batch(imgs,self.batch_size,10,10))
			imsave(self.images_dir+"/test/output_sample.jpg", flat_batch(out_img_test,self.batch_size,10,10))

			# mean_z_sample_test = np.mean(z_sample_test,axis=1)
			# std_z_sample_test = np.std(z_sample_test,axis=1)
			#
			# print(mean_z_sample_test, std_z_sample_test)
			print(z_sample_test)







def main():
	model = VAE()
	model.initialize()
	if(model.to_test == True):
		model.test()
		model.get_stats()
	else:
		model.train()

if __name__ == "__main__":
	main()
