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
from tqdm import tqdm

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
		self.load_checkpoint = False

		self.n_samples = self.mnist.train.num_examples

		self.tensorboard_dir = "./output/draw/tensorboard"
		self.check_dir = "./output/draw/checkpoints"
		self.images_dir = "./output/draw/imgs"




	def read(self, input_x, input_x_hat, input_h, name="read"):
		with tf.variable_scope(name) as scope:
			r_temp = tf.concat((input_x, input_x_hat),1)
			return tf.concat((r_temp, input_h),1)


	def write(self, input_h, name="write"):
		with tf.variable_scope(name) as scope:
			return linear1d(input_h, self.dec_size, self.img_size, name="linear")


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


	def same_sample(self, num_tensor, tensor_size):

		elem = tf.random_normal([tensor_size],0,1,dtype=tf.float32)
		list_tensor = [elem] * num_tensor
		return tf.stack(list_tensor)




	def model_setup(self):


		with tf.variable_scope("Model") as scope:

			self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.img_size])

			# For testing

			self.LSTM_enc = tf.contrib.rnn.LSTMCell(self.enc_size, state_is_tuple=True)
			self.LSTM_dec = tf.contrib.rnn.LSTMCell(self.dec_size, state_is_tuple=True)

			#Loop to train the Model

			self.gen_x = tf.zeros([self.batch_size, self.img_size])
			enc_state = self.LSTM_enc.zero_state(self.batch_size, tf.float32)
			dec_state = self.LSTM_dec.zero_state(self.batch_size, tf.float32)
			h_dec = tf.zeros([self.batch_size, self.dec_size])

			self.mean_z = [0]*self.steps
			self.std_z = [0]*self.steps

			#T steps for traning and training

			for t in range(0, self.steps):

				x_hat = self.input_x - tf.nn.sigmoid(self.gen_x)
				r = self.read(self.input_x,x_hat,h_dec)
				h_enc, enc_state = self.encoder(r, enc_state)
				self.mean_z[t], self.std_z[t] = self.linear(h_enc)
				z = self.sampler(self.mean_z[t], self.std_z[t])
				h_dec, dec_state = self.decoder(z, dec_state)
				self.gen_x = self.gen_x + self.write(h_dec)
				scope.reuse_variables()


			self.gen_x = tf.nn.sigmoid(self.gen_x)

			# T steps for generating after training


			if(self.to_test):

				self.gen_x_gen = tf.zeros([self.batch_size, self.img_size])
				self.images_output = [0]*self.steps
				dec_state_gen = self.LSTM_dec.zero_state(self.batch_size, tf.float32)


				for t in range(0, self.steps):

					# if t > 2:
					# 	z_gen = self.same_sample(self.batch_size, self.z_size)
					# else :
					# 	z_gen = tf.random_normal([self.batch_size, self.z_size], 0 , 1, dtype=tf.float32)

					z_gen = tf.random_normal([self.batch_size, self.z_size], 0 , 1, dtype=tf.float32)

					h_dec_gen, dec_state_gen = self.decoder(z_gen, dec_state_gen)
					self.gen_x_gen = self.gen_x_gen + self.write(h_dec_gen)
					self.images_output[t] = tf.sigmoid(self.gen_x_gen)

					scope.reuse_variables()

		self.model_vars = tf.trainable_variables()

		# for var in self.model_vars: print(var.name, var.get_shape())



	def loss_setup(self):

		self.images_loss = self.generation_loss(self.input_x, self.gen_x)
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
		self.lat_loss_summ = tf.summary.scalar("latent_loss", self.lat_loss_mean)

		self.merged_summ = tf.summary.merge_all()



	def train(self):

		#Setting up the model and graph
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

			if self.load_checkpoint:
				chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
				saver.restore(sess,chkpt_fname)

			for epoch in tqdm(range(0,self.max_epoch),"Epoch    "):

				for itr in tqdm(range(0,int(self.n_samples/self.batch_size)),"Iteration"):

					batch = self.mnist.train.next_batch(self.batch_size)
					imgs = batch[0]
					labels = batch[1]

					imgs = imgs.reshape((self.batch_size,28*28*1))


					_, summary_str = sess.run([self.loss_optimizer, self.merged_summ],feed_dict={self.input_x:imgs})
					# print('In the iteration '+str(itr)+" of epoch"+str(epoch))

					writer.add_summary(summary_str,epoch*int(self.n_samples/self.batch_size) + itr)

				# After each epoch things

				out_img_test = sess.run(self.gen_x,feed_dict={self.input_x:test_imgs})
				out_img_test = np.reshape(out_img_test,[self.batch_size, self.img_width, self.img_height, self.img_depth])

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


			z_sample = np.random.normal(0, 1, [self.batch_size, self.steps, self.z_size])

			gen_x_temp = sess.run(self.images_output)

			for t in range(self.steps):
				gen_x_temp[t] = np.reshape(gen_x_temp[t],[self.batch_size, self.img_width, self.img_height, self.img_depth])
				gen_x_temp[t] = flat_batch(gen_x_temp[t],self.batch_size,10,10)
				imsave(self.images_dir+"/test/output_draw_" + str(t) + ".jpg", gen_x_temp[t])


def main():

	model = Draw()
	model.initialize()

	if(model.to_test):
		model.test()
	else:
		model.train()

if __name__ == "__main__":
	main()
