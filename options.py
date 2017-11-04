import optparse
import os

class trainOptions():
	def __init__(self):
		self.parser = optparse.OptionParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
		self.parser.add_option('--batch_size', type='int', default=100, dest='batch_size')
		self.parser.add_option('--img_width', type='int', default=28, dest='img_width')
		self.parser.add_option('--img_height', type='int', default=28, dest='img_height')
		self.parser.add_option('--img_depth', type='int', default=1, dest='img_depth')
		self.parser.add_option('--z_size', type='int', default=10, dest='z_size')
		self.parser.add_option('--nef', type='int', default=16, dest='nef')
		self.parser.add_option('--max_epoch', type='int', default=20, dest='max_epoch')
		self.parser.add_option('--n_samples', type='int', default=50000, dest='n_samples')
		self.parser.add_option('--test', action="store_true", default=False, dest="test")
		self.parser.add_option('--steps', type='int', default=10, dest='steps')
		self.parser.add_option('--enc_size', type='int', default=256, dest='enc_size')
		self.parser.add_option('--dec_size', type='int', default=256, dest='dec_size')
		self.parser.add_option('--model', type='string', default="draw_attn", dest='model')
		self.parser.add_option('--dataset', type='string', default="mnist", dest='dataset')

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		return self.opt
