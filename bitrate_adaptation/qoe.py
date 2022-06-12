import numpy as np 
import math
def cal_max_in_bitrate(config, out_bitrate, bandwidth):
	window_width = config["window_width"]
	window_height = config["window_height"]
	video_width = config["video_width"]
	video_height = config["video_height"]
	width_tile_number = config["width_tile_number"]
	height_tile_number = config["height_tile_number"]
	tile_width = video_width / width_tile_number
	tile_height = video_height / height_tile_number
	width_window_tile_number = math.ceil(window_width / tile_width)
	height_window_tile_number = math.ceil(window_height / tile_height)
	in_bitrate = bandwidth * width_tile_number * height_tile_number - out_bitrate * (width_tile_number * height_tile_number - width_window_tile_number * height_window_tile_number) / float(width_window_tile_number * height_window_tile_number) 
	return in_bitrate

def cal_chunk_size(config, bitrate, chunk_period):
	o_b = bitrate[0]
	i_b = bitrate[1]
	window_width = config["window_width"]
	window_height = config["window_height"]
	video_width = config["video_width"]
	video_height = config["video_height"]
	width_tile_number = config["width_tile_number"]
	height_tile_number = config["height_tile_number"]
	tile_width = video_width / width_tile_number
	tile_height = video_height / height_tile_number
	width_window_tile_number = math.ceil(window_width / tile_width)
	height_window_tile_number = math.ceil(window_height / tile_height)
	chunk_size = (i_b * width_window_tile_number * height_window_tile_number + o_b * (width_tile_number * height_tile_number - width_window_tile_number * height_window_tile_number)) / float(width_tile_number * height_tile_number) * chunk_period
	return chunk_size
	
def generate_bitrate(config, bitrate, viewport):
	o_b = bitrate[0]
	i_b = bitrate[1]
	window_width = config["window_width"]
	window_height = config["window_height"]
	video_width = config["video_width"]
	video_height = config["video_height"]
	width_tile_number = config["width_tile_number"]
	height_tile_number = config["height_tile_number"]
	tile_width = video_width / width_tile_number
	tile_height = video_height / height_tile_number
	width_window_tile_number = math.ceil(window_width / tile_width)
	height_window_tile_number = math.ceil(window_height / tile_height)
	viewport_x, viewport_y = viewport
	chunk_bitrates = np.ones((width_tile_number, height_tile_number))
	chunk_bitrates.fill(o_b)
	for w in range(width_window_tile_number):
		for h in range(height_window_tile_number):
			new_w = int(viewport_x - int(width_window_tile_number / 2) + w)
			new_h = int(viewport_y - int(height_window_tile_number / 2) + h)
			if new_w < 0:
				new_w += width_tile_number
			elif new_w >= width_tile_number - 1:
				new_w -= width_tile_number
			if new_h < 0:
				new_h += height_tile_number
			elif new_h >= height_tile_number - 1:
				new_h -= height_tile_number
			chunk_bitrates[new_w][new_h] = i_b
	return chunk_bitrates
	
class QoEModel:
	def __init__(self, weight_1, weight_2, weight_3):
		self.weight_1 = weight_1
		self.weight_2 = weight_2
		self.weight_3 = weight_3
		self.qoe_1 = 0.0
		self.qoe_2 = 0.0
		self.qoe_3 = 0.0
		self.window_quality = 0.0
		self.rebuffer_time = 0.0
		self.window_quality_variance = 0.0
		self.prev_window_quality = None
		self.prev_rebuffer_time = 0.0
		self.count = 0

	def cal_qoe(self, chunk_bitrates, config, viewport_gt, rebuffer_time):
		window_width = config["window_width"]
		window_height = config["window_height"]
		video_width = config["video_width"]
		video_height = config["video_height"]
		width_tile_number = config["width_tile_number"]
		height_tile_number = config["height_tile_number"]
		tile_width = video_width / width_tile_number
		tile_height = video_height / height_tile_number
		width_window_tile_number = math.ceil(window_width / tile_width)
		height_window_tile_number = math.ceil(window_height / tile_height)
		viewport_x, viewport_y = viewport_gt
		tile_bitrates = []
		for w in range(width_window_tile_number):
			for h in range(height_window_tile_number):
				new_w = viewport_x - int(width_window_tile_number / 2) + w
				new_h = viewport_y - int(height_window_tile_number / 2) + h
				if new_w < 0:
					new_w += width_tile_number
				elif new_w >= width_tile_number - 1:
					new_w -= width_tile_number
				if new_h < 0:
					new_h += height_tile_number
				elif new_h >= height_tile_number - 1:
					new_h -= height_tile_number
				tile_bitrate = chunk_bitrates[new_w][new_h]
				tile_bitrates.append(tile_bitrate)
		# tile_bitrates = np.array(tile_bitrates)
		window_quality = sum(tile_bitrates) / len(tile_bitrates)
		# print(tile_bitrates, window_quality)
		window_quality_variance = abs(window_quality - self.prev_window_quality) if self.prev_window_quality is not None else 0.0
		rebuffer_time_variance = rebuffer_time - self.prev_rebuffer_time 
		self.prev_window_quality = window_quality
		self.prev_rebuffer_time = rebuffer_time
		self.qoe_1 = (self.qoe_1 * self.count + window_quality) / (self.count + 1)
		self.qoe_2 = rebuffer_time
		self.qoe_3 = (self.qoe_3 * self.count + window_quality_variance) / (self.count + 1)
		self.count += 1
		qoe = self.weight_1 * self.qoe_1 - self.weight_2 * self.qoe_2 - self.weight_3 * self.qoe_3
		tmp_qoe = self.weight_1 * window_quality - self.weight_2 * rebuffer_time_variance - self.weight_3 * window_quality_variance
		return qoe, tmp_qoe, window_quality

if __name__ == "__main__":
	q = QoEModel(1, 1, 1)
	config= {}
	config["window_width"] = 30
	config["window_height"] = 60
	config["video_width"] = 50
	config["video_height"] = 100
	config["width_tile_number"] = 5
	config["height_tile_number"] = 5
	chunk_bitrates = [[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15],
                   [16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25]]
	viewport_gt = (2, 3)
	print(q.cal_qoe(chunk_bitrates, config, (2, 3), 10))
	print(q.cal_qoe(chunk_bitrates, config, (3, 4), 11))
	print(q.cal_qoe(chunk_bitrates, config, (3, 0), 11))

