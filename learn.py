import common
import os
import random
import numpy
import bot_neural

jsonFilenames = os.listdir('./matches_json')
jsonPaths = ['./matches_json/{0}'.format(i) for i in jsonFilenames]
training_data, output_data = common.load_data(jsonPaths)
nb_classes = common.MOVE_COUNT

def do_training():
	model = common.build_model()
	batch_size = 320

	running = True
	while running:
		# Select a random 32 states from the set.
		sample = numpy.random.randint(0, len(training_data), batch_size)

		# Build arrays of states and expected results
		states = numpy.array([training_data[i] for i in sample])

		# Results
		expected = numpy.array([output_data[i] for i in sample])

		# For each state, predict the outcome of each move.
		print "Performing predictions..."
		predictions = model.predict(states)
		print predictions.shape

		print "Applying reality..."
		for idx, expec in enumerate(expected):
			predictions[idx][expec[0]] = expec[1]

		for idx, predset in enumerate(predictions):
			state = states[idx]
			for moveid, prediction in enumerate(predset):
				# Check if the move is possible
				move = common.build_move(moveid)
				y = move['y']
				dy = move['dy']
				x = move['x']
				dx = move['dx']

				if y+dy >= len(state) or x+dx >= len(state[0]):
					if predset[moveid] > 0:
						predset[moveid] = 0
					continue

				target_tile = state[y+dy][x+dx]
				source_tile = state[y][x]
				# We cannot move over mountains, or from tiles we don't own.
				if target_tile[0] == 2 or source_tile[0] != 0 or source_tile[1] < 2:
					if predset[moveid] > 0:
						predset[moveid] = 0
					continue
				if target_tile[1] != 0 and target_tile[2] > source_tile[2]:
					# we will lose our tile if we do this, so predict a loss.
					if predset[moveid] > -1:
						predset[moveid] = -1
					continue

				target_pop = target_tile[2]
				curr_pop = source_tile[2]
				if target_tile[1] != 0:
					predset[moveid] = -1*(target_pop-curr_pop)
				else:
					predset[moveid] = (target_pop+curr_pop-1)

				target_type = target_tile[0]
				source_type = source_tile[0]

				if target_type == 1:
					predset[moveid] += 50
				elif target_type == 2:
					predset[moveid] = 0
					continue
				elif target_type == 3:
					predset[moveid] += 200

		print "Training..."
		model.fit(states, predictions, batch_size=5, nb_epoch=2)

		print "Evaluating..."

		score = model.evaluate(states, predictions, verbose=0)
		print('Test score:', score[0])
		print('Test accuracy:', score[1])
		model.save_weights('weights.5g')

if __name__ == "__main__":
	bot = bot_neural.GeneralsBot(False)
	# do_training()
