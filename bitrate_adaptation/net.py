### Bandwidth trace 
# Belgium 4G/LTE bandwidth logs (bonus)
# wget http://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip
# unzip logs_all.zip -d belgium

CHUNK_PERIOD = 1
MAX_BUFFER = 10
from util import PriorityQueue
from random import *
class Env:
    def __init__(self, trace_path):
        self.buffer_remaining = 0.0
        self.queue = PriorityQueue()
        self.history = PriorityQueue()
        self.last_update_time = 0.0
        self.rebuffer_time = 0.0
        self.bandwidth_trace = {}
        self.init_bandwidth_trace(trace_path)

    def init_bandwidth_trace(self, trace_path):
        timestamp = 0
        with open(trace_path, 'r') as f:
            lines = f.readlines()     
            for line in lines:
                bandwidth = int(line.split()[4]) / (1024 * 1024)
                if bandwidth == 0: continue
                self.bandwidth_trace[timestamp] = bandwidth
                timestamp += 1

    def debug(self):
        print("==========DEBUG INFO===========")
        print("buffer_remaining: {}, \nrebuffer_time: {}, \nqueue: {}".format(self.buffer_remaining, self.rebuffer_time, self.queue.heap ))
        print("===============================")
    
    def update(self, curr_time):
        interval = curr_time - self.last_update_time
        while not self.queue.isEmpty() and self.queue.heap[0][2][2] < curr_time:
            self.queue.pop()
            self.buffer_remaining += CHUNK_PERIOD
        self.rebuffer_time += max(0, interval - self.buffer_remaining)
        self.buffer_remaining = min(MAX_BUFFER, max(0, self.buffer_remaining - interval))
        self.last_update_time = curr_time
        return self.buffer_remaining
    
    def push_video_chunk(self, chunk_size):
        finish_time = 0.0
        curr_time = 0.0 if self.history.isEmpty() else self.history.heap[0][2][2]
        if self.history.isEmpty():
            finish_time = self.cal_finish_time(chunk_size, curr_time)
            self.history.push((chunk_size, curr_time, finish_time), -finish_time)
            self.queue.push((chunk_size, curr_time, finish_time), finish_time)
        else:
            finish_time = self.cal_finish_time(chunk_size, self.history.heap[0][2][2])
            self.history.push((chunk_size, curr_time, finish_time), -finish_time)
            self.queue.push((chunk_size, curr_time, finish_time), finish_time)
        return finish_time

    
    def cal_finish_time(self, chunk_size, start_time):
        time = start_time
        while chunk_size > 0:
            curr_bw = self.bandwidth_trace[int(time) % len(self.bandwidth_trace)]
            time_left = 1 - (time - int(time))
            possible_size = time_left * curr_bw
            if possible_size < chunk_size:
                chunk_size -= possible_size
                time = int(time) + 1
            else:
                end_time = time + chunk_size / curr_bw
                return end_time
    
    def reset(self):
        self.buffer_remaining = 0.0
        self.queue = PriorityQueue()
        self.history = PriorityQueue()
        self.last_update_time = 0.0
        self.rebuffer_time = 0.0

    
    def bandwidth(self, time=None):
        if time is None: 
            return self.bandwidth_trace
        else: 
            return self.bandwidth_trace[int(time) % len(self.bandwidth_trace)]




if __name__ == "__main__":
    trace_path = "../belgium/report_bicycle_0001.log"
    env = Env(trace_path)
    # print(env.bandwidth_trace)
    # print(env.bandwidth(1))
    # print(env.bandwidth(100))
    # print(env.bandwidth(1000))
    # curr_t = 0.0
    # prev_t = 0.0
    # for i in range(15):
    #     print("-----------------------time: {}------------------------".format(i + 1))
    #     while curr_t <= i + 1:
    #         chunk_size = 5 * random()
    #         curr_t = env.push_video_chunk(chunk_size)
    #         print("Video chunk size {}MB will finish at time {}s".format(chunk_size, curr_t))
    #     env.update(i + 1)
    #     env.debug()

    curr_t = 0.0
    while curr_t < 15:
        print("-----------------------time: {}------------------------".format(curr_t))
        chunk_size = 5 * random()
        curr_t = env.push_video_chunk(chunk_size)
        print("Video chunk size {}MB will finish at time {}s".format(chunk_size, curr_t))
        env.update(curr_t)
        env.debug()
    #print(env.bandwidth_trace)