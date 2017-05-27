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


class Draw():
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
		self.steps = opt.steps
		self.enc_size = opt.enc_size
		self.dec_size = opt.dec_size
		self.filter_size = 5

		self.n_samples = self.mnist.train.num_examples

		self.tensorboard_dir = "./output/draw_attn/tensorboard"
		self.check_dir = "./output/draw_attn/checkpoints"
		self.images_dir = "./output/draw_attn/imgs"


	def create_filters(self, g_x, g_y, delta, sigma_squared, filter_size):

		eps=1e-8

		temp_1 = tf.stack([tf.stack([tf.range(filter_size, dtype=tf.float32) - filter_size/2.0 - 1/2.0]*self.img_width)]*self.batch_size) + tf.reshape(g_x,[self.batch_size, 1, 1])
		temp_2 = tf.stack([tf.stack([tf.range(filter_size, dtype=tf.float32) - filter_size/2.0 - 1/2.0]*self.img_height)]*self.batch_size) + tf.reshape(g_y,[self.batch_size, 1, 1])

		temp_3 = tf.stack([tf.transpose(tf.stack([tf.range(self.img_width, dtype=tf.float32)]*self.filter_size))]*self.batch_size)
		temp_4 = tf.stack([tf.transpose(tf.stack([tf.range(self.img_height, dtype=tf.float32)]*self.filter_size))]*self.batch_size)

		F_x = tf.exp(-1*tf.square((temp_1 - temp_3))/(2*tf.reshape(sigma_squared,[self.batch_size, 1, 1])))
		F_y = tf.exp(-1*tf.square((temp_2 - temp_4))/(2*tf.reshape(sigma_squared,[self.batch_size, 1, 1])))

		F_x = F_x/tf.maximum(tf.reduce_sum(F_x, 1, keep_dims=True), eps)
		F_y = F_y/tf.maximum(tf.reduce_sum(F_y, 1, keep_dims=True), eps)


		return F_x, F_y

	def filterbank(self, g_x, g_y, delta, sigma_squared, filter_size):

		eps=1e-8

		temp_1 = tf.reshape(tf.range(filter_size, dtype=tf.float32) - filter_size/2.0 - 1/2.0,[1, 1 ,filter_size])
		temp_2 = tf.reshape(tf.range(self.img_width, dtype=tf.float32),[1, self.img_width ,1])

		mat_1 = temp_1*tf.reshape(delta, [self.batch_size, 1, 1]) + tf.reshape(g_x, [self.batch_size, 1, 1])
		mat_2 = temp_1*tf.reshape(delta, [self.batch_size, 1, 1]) + tf.reshape(g_y, [self.batch_size, 1, 1])

		F_x = tf.exp(-1.0*tf.square(temp_2 - mat_1)/2*tf.reshape(sigma_squared,[self.batch_size, 1, 1]))
		F_y = tf.exp(-1.0*tf.square(temp_2 - mat_2)/2*tf.reshape(sigma_squared,[self.batch_size, 1, 1]))

		F_x = F_x/tf.maximum(tf.reduce_sum(F_x, 1, keep_dims=True), eps)
		F_y = F_y/tf.maximum(tf.reduce_sum(F_y, 1, keep_dims=True), eps)

		return F_x, F_y


	def downsample(self, F_x, F_y, img):

		img_temp = tf.reshape(img, [self.batch_size, self.img_width, self.img_height])
		F_y_temp = tf.transpose(F_y, [0, 2, 1])

		return tf.reshape(tf.matmul(F_y_temp,tf.matmul(img_temp,F_x)),[self.batch_size, self.filter_size*self.filter_size])

	def upsample(self, F_x, F_y, img):

		img_temp = tf.reshape(img, [self.batch_size, self.filter_size, self.filter_size])
		F_x_temp = tf.transpose(F_x, [0, 2, 1])

		return tf.reshape(tf.matmul(F_y,tf.matmul(img_temp,F_x_temp)),[self.batch_size, self.img_width*self.img_height])


	def read(self, input_x, input_x_hat, input_h, name="read"):
		with tf.variable_scope(name) as scope:

			# Getting 5 features out of input_h

			g_x_hat = linear1d(input_h, self.dec_size, 1, name="g_x_hat")
			g_y_hat = linear1d(input_h, self.dec_size, 1, name="g_y_hat")
			sigma_squared = tf.exp(linear1d(input_h, self.dec_size, 1, name="sigma_squared"))
			delta = tf.exp(linear1d(input_h, self.dec_size, 1, name="delta"))
			gamma = tf.exp(linear1d(input_h, self.dec_size, 1, name="gamma"))

			g_x = (self.img_width + 1)/2*(g_x_hat+1)
			g_y = (self.img_height + 1)/2*(g_y_hat+1)

			# Getting the filters for the Downsampling

			filter_x, filter_y = self.filterbank(g_x, g_y, delta, sigma_squared, self.filter_size)

			r_temp_1 = self.downsample(filter_x, filter_y, input_x)
			r_temp_2 = self.downsample(filter_x, filter_y, input_x_hat)

			return tf.concat((r_temp_1, r_temp_2),1)


	def write(self, input_h, name="write"):
		with tf.variable_scope(name) as scope:

			g_x_hat = linear1d(input_h, self.dec_size, 1, name="g_x_hat")
			g_y_hat = linear1d(input_h, self.dec_size, 1, name="g_y_hat")
			sigma_squared = tf.exp(linear1d(input_h, self.dec_size, 1, name="sigma_squared"))
			delta = tf.exp(linear1d(input_h, self.dec_size, 1, name="delta"))
			gamma = tf.exp(linear1d(input_h, self.dec_size, 1, name="gamma"))

			g_x = (self.img_width + 1)/2*(g_x_hat+1)
			g_y = (self.img_height + 1)/2*(g_y_hat+1)

			filter_x, filter_y = self.filterbank(g_x, g_y, delta, sigma_squared, self.filter_size)

			img_temp = linear1d(input_h, self.dec_size, self.filter_size*self.filter_size, name="linear")

			r_temp = self.upsample(filter_x, filter_y, img_temp)

			return r_temp




	def encoder(self, input_x, enc_state, name="encoder"):
		with tf.variable_scope(name) as scope:

			return self.LSTM_enc(input_x, enc_state)

	def decoder(self, input_z, dec_state, name="decoder"):
		with tf.variable_scope(name) as scope:

			return self.LSTM_dec(input_z, dec_state)



	def linear(self, input_h, name="linear"):
		with tf.variable_scope(name) as scope:

			mean = linear1d(input_h, self.enc_size, self.z_size, name="mean")
			stddev = linear1d(input_h, self.enc_size, self.z_size, name="stddev")
			return mean, tf.exp(stddev)


	def sampler(self, mean, stddev, name="sampler"):
		with tf.variable_scope(name) as scope:

			z = tf.random_normal([self.batch_size, self.z_size], 0 , 1, dtype=tf.float32)
			return z*stddev + mean


	def generation_loss(self, input_img, output_img, loss_type='log_diff'):

		if (loss_type == 'diff'):
			return tf.reduce_sum(tf.squared_difference(input_img, output_img),1)
		elif (loss_type == 'log_diff'):
			epsilon = 1e-8
			return -tf.reduce_sum(input_img*tf.log(output_img+epsilon) + (1 - input_img)*tf.log(epsilon + 1 - output_img),1)

	def latent_loss(self, mean_z, std_z) :

		loss = [0]*self.steps

		for i in range(0, self.steps):
			loss[i] = 0.5*tf.reduce_sum(tf.square(mean_z[i]) + tf.square(std_z[i]) - tf.log(tf.square(std_z[i])) - 1,1)

		return tf.add_n(loss)




	def model_setup(self):


		with tf.variable_scope("Model") as scope:

			self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_size])

			# For testing
			self.input_z = tf.placeholder(tf.float32, [self.batch_size, self.z_size])

			self.LSTM_enc = tf.contrib.rnn.LSTMCell(self.enc_size, state_is_tuple=True)
			self.LSTM_dec = tf.contrib.rnn.LSTMCell(self.dec_size, state_is_tuple=True)

			#Loop to train the Model

			self.gen_x = tf.zeros([self.batch_size, self.img_size])
			enc_state = self.LSTM_enc.zero_state(self.batch_size, tf.float32)
			dec_state = self.LSTM_dec.zero_state(self.batch_size, tf.float32)
			h_dec = tf.zeros([self.batch_size, self.dec_size])

			self.mean_z = [0]*self.steps
			self.std_z = [0]*self.steps
			self.check_field = [0]*self.steps

			#T steps for traning and training

			for t in range(0, self.steps):

				x_hat = self.input_x - tf.nn.sigmoid(self.gen_x)
				r = self.read(self.input_x,x_hat,h_dec)
				self.check_field[t] = r
				h_enc, enc_state = self.encoder(tf.concat((r,h_dec),1), enc_state)
				self.mean_z[t], self.std_z[t] = self.linear(h_enc)
				z = self.sampler(self.mean_z[t], self.std_z[t])
				h_dec, dec_state = self.decoder(z, dec_state)
				self.gen_x = self.gen_x + self.write(h_dec)

				scope.reuse_variables()

		self.model_vars = tf.trainable_variables()

		for var in self.model_vars: print(var.name, var.get_shape())

		# sys.exit()



	def loss_setup(self):

		self.images_loss = self.generation_loss(self.input_x, tf.nn.sigmoid(self.gen_x))
		self.lat_loss = self.latent_loss(self.mean_z, self.std_z)

		self.images_loss_mean = tf.reduce_mean(self.images_loss)
		self.lat_loss_mean = tf.reduce_mean(self.lat_loss)

		self.draw_loss = self.images_loss_mean + self.lat_loss_mean

		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
		grads = optimizer.compute_gradients(self.draw_loss)
		for i,(g,v) in enumerate(grads):
			if g is not None:
				grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
		self.loss_optimizer=optimizer.apply_gradients(grads)
		# self.loss_optimizer = optimizer.minimize(self.draw_loss)

		self.images_loss_summ = tf.summary.scalar("images_loss", self.images_loss_mean)
		self.draw_loss_summ = tf.summary.scalar("draw_loss", self.draw_loss)
		self.lat_loss_summ = tf.summary.scalar("images_loss", self.lat_loss_mean)

		self.merged_summ = tf.summary.merge_all()



	def train(self):

		#Setting up the model and graph

		print("In the training function")
		self.model_setup()

		self.loss_setup()



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
			test_imgs = test_imgs.reshape((self.batch_size,28*28*1))

			for epoch in range(0,self.max_epoch):

				for itr in range(0,int(self.n_samples/self.batch_size)):

					# print(time.time())
					batch = self.mnist.train.next_batch(self.batch_size)
					imgs = batch[0]
					labels = batch[1]

					imgs = imgs.reshape((self.batch_size,28*28*1))

					# print("Image is equal to ",imgs[0])
					_, summary_str, img_loss_temp, lat_loss_temp, check_field = sess.run([self.loss_optimizer, self.merged_summ, self.images_loss_mean, self.lat_loss_mean, self.check_field],feed_dict={self.input_x:imgs})


					# print("check_field is: " + str(check_field[0][0]))
					print('In the iteration '+str(itr)+" of epoch "+str(epoch)+" with image loss of "+str(img_loss_temp)+ " and lat loss of "+ str(lat_loss_temp))

					writer.add_summary(summary_str,epoch*int(self.n_samples/self.batch_size) + itr)

				# After each epoch things

				out_img_test = sess.run(self.gen_x,feed_dict={self.input_x:test_imgs})
				out_img_test = np.reshape(out_img_test,[self.batch_size, self.img_width, self.img_height, self.img_depth])
				out_img_test = sigmoid(out_img_test)

				imsave(self.images_dir+"/train/epoch_"+str(epoch)+".jpg", flat_batch(out_img_test,self.batch_size,10,10))

				saver.save(sess,os.path.join(self.check_dir,"draw"),global_step=epoch)

			writer.add_graph(sess.graph)

	def test(self):

		if not os.path.exists(self.images_dir+"/test/"):
			os.makedirs(self.images_dir+"/test/")

		self.model_setup()

		saver = tf.train.Saver()



		with tf.Session() as sess:

			chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
			saver.restore(sess,chkpt_fname)


			z_sample = np.random.normal(0, 1, [self.batch_size, self.z_size])

			gen_x_temp = sess.run(self.output_x,feed_dict={self.input_z:z_sample})

			imsave(self.images_dir+"/test/output_draw.jpg", flat_batch(gen_x_temp,self.batch_size,10,10))




def main():

	model = Draw()
	model.initialize()

	if(model.to_test == True):
		model.test()
	else:
		model.train()

if __name__ == "__main__":
	main()
