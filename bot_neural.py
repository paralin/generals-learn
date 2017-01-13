'''
	@ Harris Christiansen (Harris@HarrisChristiansen.com)
	January 2016
	Generals.io Automated Client - https://github.com/harrischristiansen/generals-bot
	Bot_blob: Creates a blob of troops.
'''

import os
import logging
import random
import threading
import time
import datetime

import numpy
import common
import tensorflow
import pickle

import client.generals as generals
from viewer import GeneralsViewer

BOT_NAME = 'GayFierri'
MAP_SIZE = common.MAP_SIZE

# Show all logging
logging.basicConfig(level=logging.DEBUG)

class GeneralsBot(object):
	def __init__(self, start_game=False):
		print "Building model..."
		self.model = common.build_model()
		self.graph = tensorflow.get_default_graph()
		self.model_mtx = threading.Lock()
		self.data_mtx = threading.Lock()

		self._update = None
		self._last_predictions = None

		# Start Game Loop
		if start_game:
			_create_thread(self._start_game_loop)

		# Start Game Viewer
		self._viewer = GeneralsViewer()
		self._viewer.mainViewerLoop()

	def _start_learn_loop(self):
		while self._running:
			time.sleep(5)
			if not self._running:
				break
			print "Applying weights..."
			with self.model_mtx:
				if os.path.exists(common.filePath):
					self.model.load_weights(common.filePath)

		if len(self._last_states) > 20:
			print "Dumping experiences..."
			if not os.path.exists('./experiences'):
				os.mkdir('./experiences')
			timestr = time.strftime("%Y%m%d-%H%M%S")
			with open('./experiences/'+timestr+'.exp', 'wb') as f:
				pickle.dump([self._last_states, self._last_predictions], f)

		os._exit(0)

	def _start_game_loop(self):
		# Create Game
		# self._game = generals.Generals(BOT_NAME, BOT_NAME, '1v1')
		self._game = generals.Generals(BOT_NAME, BOT_NAME, 'private', gameid='HyI4d3_rl') # PRIVATE

		# Start Game Update Loop
		self._running = True
		_create_thread(self._start_update_loop)

		# Start learner
		_create_thread(self._start_learn_loop)

		while (self._running):
			msg = input('Send Msg:')
			print("Sending MSG: " + msg)
			# TODO: Send msg

	######################### Handle Updates From Server #########################

	def _start_update_loop(self):
		firstUpdate = True

		for update in self._game.get_updates():
			self._set_update(update)

			if (not self._running):
				return

			if (firstUpdate):
				_create_thread(self._start_moves)
				firstUpdate = False

			# Update GeneralsViewer Grid
			if '_viewer' in dir(self):
				self._viewer.updateGrid(self._update)

	def _set_update(self, update):
		self._update = update
		if (update['complete']):
			print("!!!! Game Complete. Result = " + str(update['result']) + " !!!!")
			reward = 800.0
			if not update['result']:
				reward = -reward
			lpn = len(self._last_predictions)
			resultn = min(lpn-1, 10)
			for i in range(1, resultn):
				lmove = self._last_chosen_moves[lpn-i]
				self._last_predictions[lpn-i][lmove] += int((i/resultn)*reward)

			self._running = False
			return

		self._pi = update['player_index']
		self._opponent_position = None
		self._rows = update['rows']
		self._cols = update['cols']

	def _print_scores(self):
		scores = sorted(self._update['scores'], key=lambda general: general['total'], reverse=True) # Sort Scores
		lands = sorted(self._update['lands'], reverse=True)
		armies = sorted(self._update['armies'], reverse=True)

		print(" -------- Scores --------")
		for score in scores:
			pos_lands = lands.index(score['tiles'])
			pos_armies = armies.index(score['total'])

			if (score['i'] == self._pi):
				print("SELF: ")
			print('Land: %d (%4d), Army: %d (%4d) / %d' % (pos_lands+1, score['tiles'], pos_armies+1, score['total'], len(scores)))

	######################### Thread: Make Moves #########################

	def _start_moves(self):
		self._last_prediction = False
		self._turn_count = 0
		while (self._running and not self._update['complete']):
			self._turn_start = datetime.datetime.utcnow()
			self._make_move()
			self._turn_end = datetime.datetime.utcnow()
			turn_delta = self._turn_end - self._turn_start
			till_next = 600.0 - (turn_delta.microseconds / 1000.0)
			if till_next > 0.0:
				time.sleep(till_next/1000.0)
			self._turn_count += 1

	def _make_move(self):
		# Build state array
		input_data = numpy.zeros((MAP_SIZE, MAP_SIZE, 3))
		for row in range(0, MAP_SIZE):
			for col in range(0, MAP_SIZE):
				rowd = input_data[row][col]

				if row >= self._rows or col >= self._cols:
					rowd[0] = 2
					rowd[1] = 2
					rowd[2] = 0
					continue

				tile_faction = int(self._update['tile_grid'][row][col] != self._pi)

				# Determine type, faction, and population.
				tile_type = 0.0
				if self._update['tile_grid'][row][col] == generals.FOG:
					tile_type = 4.0
					tile_faction = 2.0
				elif (row, col) in self._update['cities']:
					tile_type = 1.0
				elif self._update['tile_grid'][row][col] == generals.MOUNTAIN:
					tile_type = 2.0
				elif (row, col) in self._update['generals']:
					tile_type = 3.0
				rowd[0] = tile_type
				rowd[1] = tile_faction
				rowd[2] = int(self._update['army_grid'][row][col])

		# Make predictions
		predictions = []
		with self.model_mtx:
			with self.graph.as_default():
				predictions = self.model.predict(numpy.array([input_data]))[0]

		# Build [moveid, q]
		moves = numpy.array([[idx, qval] for idx, qval in enumerate(predictions)])

		# Iterate from the end of the array backwards.
		# Find a move that works.
		possible_moves = []
		for move in moves:
			amove = common.build_move(move[0])
			oy = int(amove['y'])
			ox = int(amove['x'])

			if oy >= self._rows or ox >= self._cols:
				if predictions[move[0]] > 0:
					predictions[move[0]] = 0
				continue

			if not self._validPosition(ox, oy):
				if predictions[move[0]] > 0:
					predictions[move[0]] = 0
				continue

			source_tile = self._update['tile_grid'][oy][ox]
			source_army = self._update['army_grid'][oy][ox]

			if source_tile != self._pi or source_army < 2:
				if predictions[move[0]] > 0:
					predictions[move[0]] = 0
				continue

			nx = ox+int(amove['dx'])
			ny = oy+int(amove['dy'])
			if nx == ox and ny == oy:
				if predictions[move[0]] > 0:
					predictions[move[0]] = 0
				continue
			if not self._validPosition(nx, ny):
				if predictions[move[0]] > 0:
					predictions[move[0]] = 0
				continue
			dest_tile = self._update['tile_grid'][ny][nx]
			dest_army = self._update['army_grid'][ny][nx]

			# Don't attack a city unless we're sure
			if ((nx, ny) in self._update['cities'] or (nx, ny) in self._update['generals']) and dest_army >= source_army and dest_tile != self._pi:
				if predictions[move[0]] > -1.0:
					predictions[move[0]] = -1.0
				continue
			possible_moves.append([amove, predictions[move[0]]])

		if len(possible_moves) == 0:
			return

		numpy.argsort(possible_moves, axis=1)

		# amovei = possible_moves[len(possible_moves)-random.randint(0, min(len(possible_moves)-1, 2))-1]
		amovei = possible_moves[len(possible_moves)-random.randint(1, 2)]
		amove = amovei[0]
		chosen_move = int(amove['id'])

		ox = int(amove['x'])
		oy = int(amove['y'])
		nx = ox+int(amove['dx'])
		ny = oy+int(amove['dy'])
		print amove
		print "Moving", ox, oy, nx, ny
		self._place_move(oy, ox, ny, nx, False)

		# Update prediction with actual outcome
		with self.data_mtx:
			currProd = self._sum_production()
			if self._last_prediction:
				prodChange = currProd - self._last_production
				if prodChange >= 0:
					prodChange += 1
				stackSizeChange = self._last_stack_change
				if stackSizeChange > 1:
					prodChange += 1
				print "Production:", currProd, "Score:", prodChange
				self._last_predictions[len(self._last_predictions)-1][int(self._last_chosen_move)] = prodChange+stackSizeChange

			self._last_chosen_move = chosen_move

			if self._last_predictions == None:
				self._last_predictions = numpy.array([predictions])
				self._last_states = numpy.array([input_data])
				self._last_chosen_moves = numpy.array([self._last_chosen_move])
			else:
				self._last_predictions = numpy.append(self._last_predictions, numpy.array([predictions]), axis=0)
				self._last_states = numpy.append(self._last_states, numpy.array([input_data]), axis=0)
				self._last_chosen_moves = numpy.append(self._last_chosen_moves, numpy.array([self._last_chosen_move]))
			self._last_production = currProd
			self._last_prediction = True
		return

	def _place_move(self, y1, x1, y2, x2, move_half=False):
		last_stack_size = max(self._update['army_grid'][y1][x1], self._update['army_grid'][y2][x2])
		new_stack_size = 0
		self._game.move(y1, x1, y2, x2, move_half)

		# Calculate Remaining Army
		army_remaining = 1
		took_enemy = False
		captured_size = 1
		if move_half:
			army_remaining = self._update['army_grid'][y1][x1] / 2

		if (self._update['tile_grid'][y2][x2] == self._pi): # Owned By Self
			new_stack_size = self._update['army_grid'][y1][x1] + self._update['army_grid'][y2][x2] - army_remaining
		else:
			took_enemy = True
			captured_size = self._update['army_grid'][y2][x2]
			new_stack_size = self._update['army_grid'][y1][x1] - self._update['army_grid'][y2][x2] - army_remaining

		if took_enemy:
			self._last_stack_change = captured_size
		else:
			self._last_stack_change = new_stack_size-last_stack_size
			self._last_stack_change = int(self._last_stack_change * 0.2)

	def _validPosition(self, x, y):
		if not ('tile_grid' in self._update):
			return False
		return 0 <= y < self._rows and 0 <= x < self._cols and self._update['tile_grid'][y][x] != generals.MOUNTAIN

	def _sum_production(self):
		total = 0
		for x in range(self._cols):
			for y in range(self._rows):
				tile = self._update['tile_grid'][y][x]
				pos = (y,x)
				if tile != self._pi:
					continue
				if pos in self._update['generals']:
					total += 500
				elif pos in self._update['cities']:
					total += 100
				else:
					total += 50
		return total

######################### Global Helpers #########################

def _create_thread(f):
	t = threading.Thread(target=f)
	t.daemon = True
	t.start()

def _shuffle(seq):
	shuffled = list(seq)
	random.shuffle(shuffled)
	return iter(shuffled)

######################### Main #########################

# Start Bot
GeneralsBot(True)
