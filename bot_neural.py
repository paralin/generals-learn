import os
import logging
import random
import threading
import time
import datetime
import numpy
import common
import world
import tensorflow
import pickle

import client.generals as generals
from viewer import GeneralsViewer

BOT_NAME_N = numpy.random.randint(0, 2)
BOT_NAME = str(BOT_NAME_N)+'Fiery'+str(BOT_NAME_N+1)+"Me"
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
		self._dirty = True
		self._last_moves = []

		# Start Game Loop
		if start_game:
			_create_thread(self._start_game_loop)

		# Start Game Viewer
		self._viewer = GeneralsViewer()
		self._viewer.mainViewerLoop()

	def _start_learn_loop(self):
		n = 0
		while self._running:
			time.sleep(0.1)
			if not self._running:
				break
			n += 1
			if n < 150:
				continue
			n = 0
			print "Applying weights..."
			try:
				with self.model_mtx:
					if os.path.exists(common.filePath):
						self.model.load_weights(common.filePath)
			except:
				pass

		if len(self._last_moves) > 20:
			print "Simplifying experiences..."
			exper = []
			preds = []

			for last_move in self._last_moves:
				if 'reward' not in last_move:
					continue
				exper.append(last_move['input'])
				isTerminal = 'terminal' in last_move
				preds.append({
					'pred': last_move['pred'],
					'reward': last_move['reward'],
					'pidx': last_move['pidx'],
					'terminal': isTerminal,
					})

			print "Dumping experiences..."
			if not os.path.exists('./experiences'):
				os.mkdir('./experiences')
			timestr = time.strftime("%Y%m%d-%H%M%S")
			with open('./experiences/'+timestr+'.exp', 'wb') as f:
				pickle.dump([exper, preds], f)
		else:
			print "Less than 20 moves, not saving:", len(self._last_moves)

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

		updates = []
		try:
			updates = self._game.get_updates()
		except ex:
			print ex
			os.exit(1)

		for update in updates:
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
		self._dirty = True
		if (update['complete']):
			print("!!!! Game Complete. Result = " + str(update['result']) + " !!!!")
			lm = self._last_moves[len(self._last_moves)-1]
			lm['terminal'] = True
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
		self._last_moves = []
		self._turn_count = 0
		while (self._running and not self._update['complete']):
			#self._turn_start = datetime.datetime.utcnow()
			time.sleep(0.1)
			if not self._dirty:
				continue
			self._dirty = False
			if self._update['complete']:
				break
			self._make_move()
			#self._turn_end = datetime.datetime.utcnow()
			#turn_delta = self._turn_end - self._turn_start
			#till_next = 600.0 - (turn_delta.microseconds / 1000.0)
			#if till_next > 0.0:
			#	time.sleep(till_next/1000.0)
			self._turn_count += 1

	def _make_move(self):
		owned_tiles = world.build_game_view(self._update)
		if len(owned_tiles) == 0:
			return
		input_data = []

		for tile in owned_tiles:
			input_data.append(numpy.array(tile['view']))
		input_data = numpy.array(input_data)

		# Make predictions
		predictions = []
		with self.model_mtx:
			with self.graph.as_default():
				predictions = self.model.predict(input_data)

		# Best moves
		best_moves = []
		all_moves = []

		for idx, owned_tile in enumerate(owned_tiles):
			pred = predictions[idx]
			row = owned_tile['row']
			col = owned_tile['col']
			for direction, reward in enumerate(predictions[idx]):
				adir = direction
				is_50 = (direction / len(common.DIRECTIONS)) > 0
				if is_50:
					direction = direction - len(common.DIRECTIONS)

				dx, dy = common.DIRECTION_VEC[direction]
				nrow, ncol = (row - dy, col - dx)
				if nrow < 0 or nrow >= self._rows:
					continue
				if ncol < 0 or ncol >= self._cols:
					continue
				tile_typ = self._update['tile_grid'][nrow][ncol]
				if tile_typ in [generals.MOUNTAIN, generals.FOG, generals.OBSTACLE]:
					continue

				move = {
						'is_50': is_50,
						'x': col,
						'y': row,
						'nx': ncol,
						'ny': nrow,
						'tile': owned_tile,
						'pred': pred,
						'pidx': adir,
						'input': input_data[idx],
						}
				best_moves.append([reward, len(all_moves)])
				all_moves.append(move)

		if len(best_moves) == 0:
			return

		move_count = len(best_moves)
		best_moves = numpy.sort(best_moves, axis=0)
		best_move_idx = numpy.random.choice(min(move_count, 2))
		best_move = all_moves[int(best_moves[int(best_move_idx)][1])]

		ox = best_move['x']
		oy = best_move['y']
		nx = best_move['nx']
		ny = best_move['ny']
		is_50 = best_move['is_50']
		print "Moving", ox, oy, nx, ny, is_50
		self._place_move(oy, ox, ny, nx, is_50)

		# Update prediction with actual outcome
		with self.data_mtx:
			currProd = self._sum_production()
			if len(self._last_moves) > 0:
				lmov = self._last_moves[len(self._last_moves)-1]
				prodChange = currProd - self._last_production
				print "Production:", currProd, "Reward:", prodChange, "dSS:", self._last_stack_change
				if prodChange > 0:
					prodChange += min(self._last_stack_change, 3)
					print "Adjusted production:", prodChange

				print "Prediction:"
				print lmov['pred']

				lmov['pred'][lmov['pidx']] = prodChange + common.DISCOUNT_FACTOR * lmov['pred'][lmov['pidx']]
				lmov['reward'] = prodChange

			self._last_production = currProd
			self._last_moves.append(best_move)
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
				elif pos in self._update['cities']:
					total += 50
				else:
					total += 10
		return total

######################### Global Helpers #########################

def _create_thread(f):
	t = threading.Thread(target=f)
	t.daemon = True
	t.start()

######################### Main #########################

# Start Bot
GeneralsBot(True)
