import common
import numpy
import client.generals as generals

MAP_SIZE = common.MAP_SIZE

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
