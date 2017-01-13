import json
import numpy
import os
import random
import pickle

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import AtrousConvolution2D, MaxPooling2D, LocallyConnected2D
from keras.optimizers import sgd
from keras.metrics import fbeta_score

# Directions
DIRECTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
DIRECTION_VEC = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Map size, squared
MAP_SIZE = 18

# Given map size, and directions, we can compute possible total number of moves.
TILE_COUNT = MAP_SIZE*MAP_SIZE
MOVE_COUNT = TILE_COUNT*len(DIRECTIONS)

# Given a move ID, determine move.
def build_move(id):
	# Direction
	direction = int(id % len(DIRECTIONS))
	# Tile
	tile_id = int(id / len(DIRECTIONS))
	# X coord
	x = int(tile_id % MAP_SIZE)
	y = int(tile_id / MAP_SIZE)

	(dx, dy) = DIRECTION_VEC[direction]

	return {
			'id': id,
			'x': x,
			'y': y,
			'dx': dx,
			'dy': dy,
			'tile_id': tile_id,
			'direction': direction,
	}

# Given a tile and direction, determine move id
def build_move_id(tile, direction):
	return len(DIRECTIONS) * tile + direction

def load_data(filenames):
	print "Loading files..."
	data = []
	for filename in filenames:
		print " --> "+str(filename)
		fdata = []

		with open(filename) as f:
			fdata = json.load(f)
		data.extend(fdata)

	# Array of [ [state, moveid] ]
	# state is [type, faction, population] in a 2d array.
	input_data = numpy.zeros((len(data), MAP_SIZE, MAP_SIZE, 3))
	output_data = numpy.zeros((len(data), 2))

	print "Transforming data..."
	# Take each frame, transform to 1d array state, move id output
	for turnid, state in enumerate(data):
		# Ignore the last state
		if turnid == (len(data)-3):
			break

		# Ignore frames with no move.
		if not ('cells' in state and 'move' in state):
			continue

		# Build 3xMAP_SIZExMAP_SIZE array
		cells = input_data[turnid]
		for tileid, cell in enumerate(state['cells']):
			tx = int(tileid % MAP_SIZE)
			ty = int(tileid / MAP_SIZE)

			cellState = cells[ty][tx]
			cellState[0] = cell['type'] if 'type' in cell else 0
			cellState[1] = cell['faction'] if 'faction' in cell else 0
			cellState[2] = cell['population'] if 'population' in cell else 0

			if cellState[1] > 0:
				cellState[1] = 1

		move = state['move']
		movefrom = move['from'] if 'from' in move else 0
		movedir = move['dir'] if 'dir' in move else 0

		moveid = build_move_id(movefrom, movedir)
		output_data[turnid][0] = int(moveid)
		pop_diff = int(data[turnid+1]['production'] - state['production'])
		# print pop_diff
		output_data[turnid][1] = pop_diff
		# print output_data[turnid]

	# print training_data
	return input_data, output_data

def load_experiences():
	experiences = []
	predictions = []

	# list files
	if not os.path.exists('./experiences'):
		return numpy.array(experiences), numpy.array(predictions)

	files = os.listdir('./experiences')
	if len(files) == 0:
		return numpy.array(experiences), numpy.array(predictions)
	random.shuffle(files)

	i = 0
	for filen in files:
		with open('./experiences/'+filen, 'rb') as f:
			arn = pickle.load(f)
			experiences.extend(arn[0])
			predictions.extend(arn[1])

		if i == 3:
			break
		i += 1

	return numpy.array(experiences), numpy.array(predictions)

filePath = 'weights.h5'
def build_model():
	model = Sequential()

	model.add(Flatten(input_shape=(MAP_SIZE, MAP_SIZE, 3)))
	model.add(Dense(MOVE_COUNT*1.5, activation='relu'))
	model.add(Dense(MOVE_COUNT*1.2, activation='tanh'))
	model.add(Dense(MOVE_COUNT, activation='linear'))

	model.compile(sgd(lr=.2), "mse")


	#model.add(AtrousConvolution2D(64, 3, 3, atrous_rate=(2,2), border_mode='valid', input_shape=(MAP_SIZE, MAP_SIZE, 3)))
	#model.add(Activation('relu'))
	#model.add(AtrousConvolution2D(64, 2, 2, atrous_rate=(1,1)))
	#model.add(Activation('relu'))
	#model.add(Flatten())
	# model.add(Dense(MOVE_COUNT*1.2))
	#model.add(Activation('relu'))
	#model.add(Dense(MOVE_COUNT))
	#model.add(Activation('relu'))

	#model.compile(Adagrad(), loss="mean_absolute_error", metrics=['accuracy'])

	print model.summary()

	if os.path.exists(filePath):
		model.load_weights(filePath)

	return model

def save_model(model):
	model.save_weights(filePath)
