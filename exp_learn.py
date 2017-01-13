import logging
from Queue import Queue
import random
import threading
import time
from pprint import pprint

import client.generals as generals
from viewer import GeneralsViewer

import numpy
import common
import tensorflow

class ExperienceLearner:
	def __init__(self):
		self.model = common.build_model()
		self.graph = tensorflow.get_default_graph()

	def learn_once(self):
		print "Loading experiences..."
		states, predictions = common.load_experiences()

		print "Fitting..."
		nb_epoch = 8
		with self.graph.as_default():
			self.model.fit(states, predictions, nb_epoch=nb_epoch, batch_size=10)

		print "Saving..."
		common.save_model(self.model)

learner = ExperienceLearner()
while True:
	learner.learn_once()
