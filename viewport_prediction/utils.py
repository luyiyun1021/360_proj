
import header
DISCRETE_WIDTH_INTERVAL = [1.0/header.width_tile_number * (i+1) for i in range(header.width_tile_number)]
DISCRETE_HEIGHT_INTERVAL = [1.0/header.height_tile_number * (i+1) for i in range(header.height_tile_number)]

def find_nearest(val, key):
    if key == "w":
        min_val = abs(DISCRETE_WIDTH_INTERVAL[0] - val)
        min_index = 0
        for i in range(1, len(DISCRETE_WIDTH_INTERVAL)):
            curr_val = abs(DISCRETE_WIDTH_INTERVAL[i] - val)
            if curr_val < min_val:
                min_val = curr_val
                min_index = i
        return DISCRETE_WIDTH_INTERVAL[min_index]
    elif key == "h":
        min_val = abs(DISCRETE_HEIGHT_INTERVAL[0] - val)
        min_index = 0
        for i in range(1, len(DISCRETE_HEIGHT_INTERVAL)):
            curr_val = abs(DISCRETE_HEIGHT_INTERVAL[i] - val)
            if curr_val < min_val:
                min_val = curr_val
                min_index = i
        return DISCRETE_HEIGHT_INTERVAL[min_index]
    else:
        raise ValueError('Invalid key! Should be "w" or "h"')

def simplify_data(data):
    ret = []
    timestamp_set = set()
    for item in data:
        timestamp = int(item[0])
        if len(timestamp_set) == 0 and timestamp != 0: continue
        point = (find_nearest(item[1][0], "w"), find_nearest(item[1][1], "h"))
        if timestamp not in timestamp_set:
            timestamp_set.add(timestamp)
            ret.append([timestamp, point])
    return ret