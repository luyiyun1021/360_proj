import sys
sys.path.append("/home/sjtu/yylu/VE490/360_proj/viewport_prediction") 
sys.path.append("/home/sjtu/yylu/VE490/360_proj") 
sys.path.append("/home/sjtu/yylu/VE490/360_proj/bandwidth_prediction")
import torch
from viewport_prediction.module.MetaViewportLSTM import MetaViewportLSTM
from bandwidth_prediction.module.bandwidth_lstm import BandwidthLSTM
import os
import pickle
from viewport_prediction.cluster import get_cluster
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np


WIDTH_TILE_NUMBER = 8
HEIGHT_TILE_NUMBER = 8
DISCRETE_WIDTH_INTERVAL = [1.0 / WIDTH_TILE_NUMBER * (i+1) for i in range(WIDTH_TILE_NUMBER)]
DISCRETE_HEIGHT_INTERVAL = [1.0 / HEIGHT_TILE_NUMBER * (i+1) for i in range(HEIGHT_TILE_NUMBER)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USER_NUM = 48
VIEWPORT_DATASET = "../Viewport/ds2"
BANDWIDTH_DATASET = "../belgium"
VIEWPORT_TEST_PATH = os.path.join(VIEWPORT_DATASET, "test")
VIEWPORT_TRAIN_PATH = os.path.join(VIEWPORT_DATASET, "train")
VIEWPORT_VALID_PATH = os.path.join(VIEWPORT_DATASET, "val")
CLUSTER_FILE_PATH = "../feature_vector.pkl"
CLUSTER_NUM = 4
TASK_NUM = 6
BATCH_SIZE = 1
TOPIC_NUM = 9
VIEWPORT_MODEL_NAME = "meta_viewport_lstm_without_task1"
BANDWIDTH_MODEL_NAME = "bandwidth_lstm"
BANDWIDTH_SCALE_FACTOR = 3


VIDEO_CONFIG = {"window_width": 17.0,
          "window_height": 34.0,
          "video_width": 50.0,
          "video_height": 100.0,
          "width_tile_number": WIDTH_TILE_NUMBER,
          "height_tile_number": HEIGHT_TILE_NUMBER
          }

### Viewport related
class MyViewportDataset(DataLoader):
    def __init__(self, input, gt, filename):
        self.input = input
        self.gt = gt
        self.filename = filename

    def __getitem__(self, index):
        return (self.input[index], self.gt[index]), self.filename

    def __len__(self):
        return len(self.input)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def find_nearest(val, key):
    if key == "w":
        min_val = abs(DISCRETE_WIDTH_INTERVAL[0] - val)
        min_index = 0
        for i in range(1, len(DISCRETE_WIDTH_INTERVAL)):
            curr_val = abs(DISCRETE_WIDTH_INTERVAL[i] - val)
            if curr_val <= min_val:
                min_val = curr_val
                min_index = i
        return DISCRETE_WIDTH_INTERVAL[min_index]
    elif key == "h":
        min_val = abs(DISCRETE_HEIGHT_INTERVAL[0] - val)
        min_index = 0
        for i in range(1, len(DISCRETE_HEIGHT_INTERVAL)):
            curr_val = abs(DISCRETE_HEIGHT_INTERVAL[i] - val)
            if curr_val <= min_val:
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

def complete_viewport_dataset(data):
    first_data = data[0][0]
    for i in reversed(range(1, len(first_data[0]))):
        x = first_data[:, :i, :]
        y = first_data[:, i, :]
        data.insert(0, (x, y))


def create_viewport_dataset(dataset, look_back=None):
    train = []
    gt = []
    limit = 10
    if look_back is not None:
        for i in range(len(dataset) - look_back):
            train.append([item[1] for item in dataset[i : i + look_back]])
            gt.append(dataset[i + look_back][1])
    return np.asarray(train), np.asarray(gt) 

def get_viewport_files(path):
    file_group = [[] for i in range(CLUSTER_NUM)]
    filenames = os.listdir(path)
    filenames.sort()
    # cluster(viewport_train_path, cluster_file_path)
    _, user_cluster, _ = get_cluster(CLUSTER_NUM, CLUSTER_FILE_PATH)
    for filename in filenames:
        user_name = filename.split("_")[-1]
        for i in range(CLUSTER_NUM):
            if user_name in user_cluster[i]:
                file_group[i].append(filename)
                break
    return file_group, user_cluster
        

def create_viewport_dataloaders(filenames, path, processed_viewport_data, cluster_id):
    for filename in tqdm(filenames):
        file_path=os.path.join(path, filename)
        with open(file_path, 'rb' ) as f:
            data = pickle.load(f)
            data = simplify_data(data)
            # print(filename, data[-1])
            input, gt = create_viewport_dataset(data, 10)
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
            user_name = filename.split("_")[-1]
            topic_name = filename.split("_")[-2]
            for i in range(len(input)):
                processed_viewport_data[cluster_id][topic_name][user_name].append((input[i: i+1], gt[i: i+1]))  
            complete_viewport_dataset(processed_viewport_data[cluster_id][topic_name][user_name])

def create_processed_viewport_data_and_models(path):
    lstms = []
    hiddens = []
    test_file_group, user_cluster = get_viewport_files(path)
    processed_viewport_data = {}
    for i in range(TASK_NUM - 2):
        processed_viewport_data[i] = {}    
        for j in range(TOPIC_NUM):
            topic_name = "topic" + str(j)
            processed_viewport_data[i][topic_name] = {}
            for k in range(USER_NUM):
                user_name = "user" + str(k + 1)
                if user_name in user_cluster[i]:
                    processed_viewport_data[i][topic_name][user_name] = []
    for i in range(TASK_NUM - 2):
        create_viewport_dataloaders(test_file_group[i], path, processed_viewport_data, i)           
    return processed_viewport_data, user_cluster

def eval_viewport_correctness(x, gt):
    x_tmp = []
    gt_tmp = []
    diff = []
    for i in range(x.shape[0]):
        x_tmp.append((find_nearest(x[i][0], "w"), find_nearest(x[i][1], "h")))
        gt_tmp.append((find_nearest(gt[i][0], "w"), find_nearest(gt[i][1], "h")))
        x_tmp[i] = (int(WIDTH_TILE_NUMBER * x_tmp[i][0]), int(WIDTH_TILE_NUMBER * x_tmp[i][1]))
        gt_tmp[i] = (int(HEIGHT_TILE_NUMBER * gt_tmp[i][0]), int(HEIGHT_TILE_NUMBER * gt_tmp[i][1]))
        diff_x = x_tmp[i][0] - gt_tmp[i][0]
        diff_y = x_tmp[i][1] - gt_tmp[i][1]
        diff.append(diff_x * diff_x + diff_y * diff_y)
    return x_tmp, gt_tmp, diff

def test_one_viewport_data(model, data, task_index, hidden):
    hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
    b_x = data[0].transpose(0, 1)
    b_x = b_x.type(torch.FloatTensor).to(DEVICE) 
    b_y = data[1].type(torch.FloatTensor).to(DEVICE) 
    with torch.no_grad():
        prediction,  hidden = model(b_x, (hidden, task_index)) 
    x, gt, diff = eval_viewport_correctness(prediction[-1, :, :], b_y)
    # for i in range(len(x)):
    #     print("x: {}, gt: {}, diff: {}".format(x[i], gt[i], diff[i]))
    tmp_res = prediction[-1, :, :]
    tmp_res = torch.tensor([find_nearest(tmp_res[0][0], "w"), find_nearest(tmp_res[0][1], "h")])
    return (x, gt, diff), (tmp_res, b_y), hidden

def predict_next_few_seconds_viewport(model, data, task_index, hidden, ret):
    tmp_data = deepcopy(data)
    data_input = tmp_data[0][0]
    for i in range(len(tmp_data)):
        tile_predict_result, predict_result, hidden = test_one_viewport_data(model, [data_input, tmp_data[i][1]], task_index, hidden)
        if i == 0: ret_hidden = hidden
        for i in range(len(tile_predict_result[0])):
            ret.append((tile_predict_result[0][i], tile_predict_result[1][i], tile_predict_result[2][i]))
        new_input = predict_result[0]
        new_input = new_input.unsqueeze(0).unsqueeze(0)
        data_input = torch.cat((data_input, new_input), 1)
        data_input = data_input[:, 1:, :]
    return ret_hidden

### Bandwidth related
class MyBwDataset(DataLoader):
    def __init__(self, input, gt):
        self.input = input
        self.gt = gt

    def __getitem__(self, index):
        return self.input[index], self.gt[index]

    def __len__(self):
        return len(self.input)

def create_bw_dataset(dataset, look_back=None):
    train = []
    gt = []
    limit = 10
    if look_back is not None:
        for i in range(len(dataset) - look_back):
            train.append([item for item in dataset[i : i + look_back]])
            gt.append(dataset[i + look_back])
    return np.asarray(train), np.asarray(gt) 

def complete_bw_dataset(data):
    first_data = data[0][0]
    for i in reversed(range(1, len(first_data[0]))):
        x = first_data[:, :i]
        y = first_data[:, i]
        data.insert(0, (x, y))

def get_bw_files(path):
    filenames = os.listdir(path)
    filenames.sort()
    return filenames
        

def create_bw_dataloaders(loaders, filenames, path):
    for filename in tqdm(filenames):
        file_path=os.path.join(path, filename)
        with open(file_path, 'r' ) as f:
            lines = f.readlines()
            data = []
            for line in lines:
                bandwidth = int(line.split()[4]) / (1024 * 1024)
                if bandwidth == 0: continue
                data.append(bandwidth)
            input, gt = create_bw_dataset(data, 10)
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
            loader = DataLoader(MyBwDataset(input, gt), batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=4, drop_last=True)
            loaders.append(loader)
    return loaders

def create_processed_bw_data_and_models():
    load_path = "../models/" + BANDWIDTH_MODEL_NAME + ".pt"
    lstm = BandwidthLSTM().to(DEVICE)
    lstm.load_state_dict(torch.load(load_path)["model"])
    hidden = lstm.init_hidden(BATCH_SIZE)
    processed_data = []
    file_groups = get_bw_files(BANDWIDTH_DATASET)
    test_file_group = [file_groups[-1]]
    test_loaders = []
    create_bw_dataloaders(test_loaders, test_file_group, BANDWIDTH_DATASET)
    for test_loader in test_loaders:
        for step, data in enumerate(test_loader):
            processed_data.append(data)
        complete_bw_dataset(processed_data)
    return lstm, hidden, processed_data

def test_one_bw_data(model, data, hidden):
    hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
    b_x = data[0].transpose(0, 1)
    b_x = b_x.unsqueeze(2)
    b_x = b_x.type(torch.FloatTensor).to(DEVICE) 
    b_y = data[1].type(torch.FloatTensor).to(DEVICE) 
    with torch.no_grad():
        prediction,  hidden = model(b_x, hidden) 
    return (prediction[-1, :, :], b_y), hidden

def predict_next_few_seconds_bw(model, data, hidden, ret):
    tmp_data = deepcopy(data)
    data_input = tmp_data[0][0]
    for i in range(len(tmp_data)):
        predict_result, hidden = test_one_bw_data(model, [data_input, tmp_data[i][1]], hidden)
        if i == 0: ret_hidden = hidden
        for i in range(len(predict_result[0])):
            ret.append((float(predict_result[0][i].cpu().numpy()) * BANDWIDTH_SCALE_FACTOR, float(predict_result[1][i].cpu().numpy()) * BANDWIDTH_SCALE_FACTOR))
        new_input = predict_result[0].cpu()
        data_input = torch.cat((data_input, new_input), 1)
        data_input = data_input[:, 1:]
    return ret_hidden

if __name__ == "__main__":
    ### test viewport
    model_name = "meta_viewport_lstm"
    lstms = []
    hiddens = []
    test_loaders = [[] for i in range(CLUSTER_NUM)]
    test_file_group, _ = get_viewport_files(VIEWPORT_TEST_PATH)
    length = 1
    

    for i in range(CLUSTER_NUM):
        test_loaders[i] = create_viewport_dataloaders(test_loaders[i], test_file_group[i], VIEWPORT_TEST_PATH)
    for i in range(TASK_NUM - 3):
        lstms.append(MetaViewportLSTM(task_num=TASK_NUM).to(DEVICE))
        load_path = "../models/"+ model_name + "_" + str(i) + ".pt"
        lstms[i].load_state_dict(torch.load(load_path)["model"])
        hiddens.append(lstms[i].init_hidden(BATCH_SIZE))
    for i in range(TASK_NUM - 3):
        for test_loader in test_loaders[i]:
            ret = []
            hiddens[i] = lstms[i].init_hidden(BATCH_SIZE)
            loaderit = iter(test_loader)
            _, file_name = loaderit.next()
            file_name = file_name[0]
            user_name = file_name.split("_")[-1]
            topic_name = file_name.split("_")[-2]
            print("testing: {} {}".format(topic_name, user_name))
            processed_data = []
            for step, (data, _)  in enumerate(test_loader):
                processed_data.append(data)
            complete_viewport_dataset(processed_data)
            for item in processed_data:
                print(item)
            for id in range(len(processed_data)):
                ret = []
                hiddens[i] = predict_next_few_seconds_viewport(lstms[i], processed_data[id : id + length], i, hiddens[i], ret)
                id += length
                for item in ret:
                    print(item)
                print("==============================")
    
    ### test bandwidth
    # length = 2
    # load_path = "../models/bandwidth_lstm.pt"
    # lstm = BandwidthLSTM().to(DEVICE)
    # lstm.load_state_dict(torch.load(load_path)["model"])
    # hidden = lstm.init_hidden(BATCH_SIZE)
    # processed_data = []
    # file_groups = get_bw_files(BANDWIDTH_DATASET)
    # test_file_group = [file_groups[-1]]
    # test_loaders = []
    # create_bw_dataloaders(test_loaders, test_file_group, BANDWIDTH_DATASET)
    # for test_loader in test_loaders:
    #     ret = []
    #     for step, data in enumerate(test_loader):
    #         processed_data.append(data)
    #     complete_bw_dataset(processed_data)
    #     for item in processed_data:
    #         print(item)
    #     for id in range(len(processed_data)):
    #         ret = []
    #         hidden = predict_next_few_seconds_bw(lstm, processed_data[id : id + length], hidden, ret)
    #         for item in ret:
    #             print(item, abs(item[1] - item[0]))
    #         print("============================")


        

                

                
import heapq
class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
