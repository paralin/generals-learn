import json
import numpy
import os
import random
import pickle

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D, BatchNormalization
from keras.optimizers import RMSprop, SGD, Nadam
from keras.metrics import fbeta_score

# Directions
DIRECTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
DIRECTION_VEC = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Map size, squared
MAP_SIZE = 20
# Size of learning grid edge
LEARN_MAP_SIZE = int(MAP_SIZE*0.5)
DISCOUNT_FACTOR = 0.8
WIN_REWARD = 100.0

# Given map size, and directions, we can compute possible total number of moves.
TILE_COUNT = MAP_SIZE*MAP_SIZE
MOVE_COUNT = len(DIRECTIONS) * 2.0

def load_experiences():
	experiences = []
	moves = []

	# list files
	if not os.path.exists('./experiences'):
		return experiences, moves

	files = os.listdir('./experiences')
	if len(files) == 0:
		return experiences, moves
	random.shuffle(files)

	i = 0
	for filen in files:
		with open('./experiences/'+filen, 'rb') as f:
			arn = pickle.load(f)
			experiences.extend(arn[0])
			moves.extend(arn[1])

		if i == 1:
			break
		i += 1

	return experiences, moves

filePath = 'weights.h5'
def build_model():
	model = Sequential()

	# model.add(BatchNormalization(mode=0, axis=1, ))
	model.add(Flatten(input_shape=(2, LEARN_MAP_SIZE, LEARN_MAP_SIZE)))
	model.add(Dense(164))
	model.add(Activation('relu'))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dense(MOVE_COUNT))
	model.add(Activation('linear'))

	model.compile(Nadam(), 'mse',
				  metrics=['accuracy'] )

	print model.summary()

	if os.path.exists(filePath):
		model.load_weights(filePath)

	return model

def save_model(model):
	model.save_weights(filePath)

# Return (row, col)
def idx_to_coord(idx):
	return (idx / MAP_SIZE, idx % MAP_SIZE)
