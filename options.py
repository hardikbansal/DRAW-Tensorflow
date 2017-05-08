import optparse
import os

class trainOptions():
	def __init__(self):
		self.parser = optparse.OptionParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
		self.parser.add_option('--batch_size', type='int', default=10, dest='batch_size')

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		return self.opt