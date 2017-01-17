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
		states, moves = common.load_experiences()
		predictions = numpy.zeros((len(states), common.MOVE_COUNT))

		print "Re-computing Q values..."
		for idx, state in enumerate(states):
			move = moves[idx]
			oldReward = move['reward']
			predicted = self.model.predict(numpy.array([state]))[0]
			moveDir = move['pidx']
			# Compute max of predictions
			maxMove = numpy.max(predicted)
			predicted[moveDir] = maxMove * common.DISCOUNT_FACTOR + oldReward
			predictions[idx] = predicted

		print "Fitting..."
		nb_epoch = 8
		with self.graph.as_default():
			self.model.fit(numpy.array(states), numpy.array(predictions), nb_epoch=nb_epoch, batch_size=30, shuffle=True)

		print "Saving..."
		common.save_model(self.model)

learner = ExperienceLearner()
while True:
	learner.learn_once()
