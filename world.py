import common
import numpy
import client.generals as generals
import math

MAP_SIZE = common.MAP_SIZE

# Convert a replay step to an update
def replay_to_update(step):
	update = {}
	cells = step['cells']
	map_size = int(math.sqrt(len(cells)))
	update['rows'] = map_size
	update['cols'] = map_size
	update['player_index'] = 0
	update['complete'] = False
	tile_grid = update['tile_grid'] = numpy.zeros((map_size, map_size))
	army_grid = update['army_grid'] = numpy.zeros((map_size, map_size))
	generalsa = update['generals'] = []
	cities = update['cities'] = []
	for idx, cell in enumerate(cells):
		row = int(idx / map_size)
		col = int(idx % map_size)

		faction = cell['faction'] if 'faction' in cell else 0
		ctype = cell['type'] if 'type' in cell else 0
		population = cell['population'] if 'population' in cell else 0

		army_grid[row][col] = population

		pos_tup = (row,col)
		tile_typ = 0
		if ctype is 3:
			generalsa.append(pos_tup)
		elif ctype is 1:
			cities.append(pos_tup)
		elif ctype is 2:
			tile_typ = generals.MOUNTAIN
		elif ctype is 4:
			tile_typ = generals.FOG

		if tile_typ != 0:
			tile_grid[row][col] = tile_typ
		else:
			tile_grid[row][col] = 0 if faction is 1 else 1
	return update

# Build the neural net game view
def build_game_view(update):
	if not ('rows' in update):
		return None

	world_rows = update['rows']
	world_cols = update['cols']

	# Build an array of tiles we own.
	owned_tiles = []
	for row in range(0, MAP_SIZE):
		if row >= world_rows:
			continue

		for col in range(0, MAP_SIZE):
			if col >= world_cols:
				continue

			# Skip if we don't own this tile, or it's not actionable.
			tile_army = int(update['army_grid'][row][col])
			if update['tile_grid'][row][col] != update['player_index'] or tile_army < 2:
				continue

			# Build state array. Represent enemies as negative, type of tile in second channel.
			input_data = numpy.zeros((2, common.LEARN_MAP_SIZE, common.LEARN_MAP_SIZE))

			# Build the data object
			owned_tiles.append({
				'row': row,
				'col': col,
				'view': input_data,
			})

			for ncoli in range(0, common.LEARN_MAP_SIZE):
				ncol = col - int(common.LEARN_MAP_SIZE / 2) + ncoli
				if ncol < 0 or ncol >= len(update['tile_grid'][0]):
					continue

				for nrowi in range(0, common.LEARN_MAP_SIZE):
					tile_pop = 0
					tile_typ = generals.MOUNTAIN

					nrow = row - int(common.LEARN_MAP_SIZE / 2) + nrowi
					if not (nrow < 0 or nrow >= len(update['tile_grid'])):
						tile_typ = update['tile_grid'][nrow][ncol]

					# Determine type, faction, and population.
					if tile_typ in [generals.MOUNTAIN, generals.FOG, generals.OBSTACLE]:
						tile_typ = 0
						tile_pop = 0
					elif tile_typ == update['player_index']:
						if (nrow, ncol) in update['generals']:
							tile_typ = 1.0
						else:
							tile_typ = 0.5
					else:
						tile_typ = 0.25

					input_data[0][nrowi][ncoli] = tile_pop
					input_data[1][nrowi][ncoli] = tile_typ
	return owned_tiles
