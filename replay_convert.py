import numpy
import common
import sys
import json
import world
import time
import bot_neural

if len(sys.argv) != 2:
	print "Usage: replay_convert.py path_to_match"
	sys.exit(1)

match_path = sys.argv[1]

print "Loading", match_path
data = []
with open(match_path, 'rb') as f:
	data = json.load(f)

agent = bot_neural.GeneralsBot(False)

firstUpdate = True
skipped = 0
agent._running = True
for turn, step in enumerate(data):
	if not ('move' in step):
		skipped += 1
		continue

	if not firstUpdate:
		nlm = len(agent._last_moves)
		move = step['move']
		fromIdx = move['from'] if 'from' in move else 0
		dirIdx = move['dir'] if 'dir' in move else 0
		(x, y) = common.idx_to_coord(fromIdx)
		(dx, dy) = common.DIRECTION_VEC[dirIdx]
		(nx, ny) = (x + dx, y + dy)
		olm = agent._last_moves[nlm-1]
		agent._last_moves[nlm-1] = {
				'is_50': 'is50' in step and step['is50'],
				'x': x,
				'y': y,
				'nx': nx,
				'ny': ny,
				'tile': olm['tile'],
				'pred': olm['pred'],
				'pidx': dirIdx,
				'input': olm['input'],
		}
	update = world.replay_to_update(step)
	update['turn'] = turn - skipped
	agent._next_ready = False
	agent._set_update(update)
	if firstUpdate:
		bot_neural._create_thread(agent._start_moves)
		bot_neural._create_thread(agent._start_learn_loop)
		firstUpdate = False
	#if '_viewer' in dir(agent):
		#agent._viewer.updateGrid(update)
	while not agent._next_ready:
		time.sleep(0.01)

agent._set_update({'complete': True, 'result': True})
time.sleep(2)
