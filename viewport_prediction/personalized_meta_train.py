import pickle

from numpy.core.numeric import Inf
import header
import numpy as np
from copy import deepcopy
from utils import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataloader
import logging
from time import strftime
from matplotlib import pyplot as plt
import os
from module.MetaViewportLSTM import MetaViewportLSTM
from tqdm import tqdm
import argparse
from cluster import cluster, get_cluster


#LOAD_PATH = "../models/meta_viewport_lstm.pt"
DISCRETE_WIDTH_INTERVAL = [1.0/header.width_tile_number * (i+1) for i in range(header.width_tile_number)]
DISCRETE_HEIGHT_INTERVAL = [1.0/header.height_tile_number * (i+1) for i in range(header.height_tile_number)]
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False, help='Train')
parser.add_argument('--test', type=bool, default=False, help='Test')
parser.add_argument('--ftrain', type=bool, default=False, help='Fast Train')
parser.add_argument('--ftest', type=bool, default=False, help='Fast Test')
parser.add_argument('--task', type=int, default=0, help='Task index')
parser.add_argument('--epoch', type=int, default=10, help='Epoch')
parser.add_argument('--lr', type=float, default=0.01, help='lr')
parser.add_argument('--load', type=bool, default=False, help='Load saved model')
parser.add_argument('--model_name', type=str, default="meta_viewport_lstm", help='Model name')
args = parser.parse_args()

BATCH_SIZE = 128
EPOCH = args.epoch

class MyDataset(DataLoader):
    def __init__(self, input, gt):
        self.input = input
        self.gt = gt

    def __getitem__(self, index):
        return self.input[index], self.gt[index]

    def __len__(self):
        return len(self.input)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def create_dataset(dataset, look_back=None):
    train = []
    gt = []
    limit = 10
    if look_back is not None:
        for i in range(len(dataset) - look_back):
            train.append([item[1] for item in dataset[i : i + look_back]])
            gt.append(dataset[i + look_back][1])
    else:
        history = []
        #count = 0
        for i in range(len(dataset) - 1):
            history.append(dataset[i])
            train.append(deepcopy(history))
            gt.append(dataset[i + 1])
            #count += 1
            #if count >= 10: break
    return np.asarray(train), np.asarray(gt) 







#logger
global print
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
try:
    os.makedirs("logger", exist_ok=True)
except TypeError:
    raise Exception("Direction not create!")
logger.addHandler(logging.FileHandler(strftime('logger/{}_%m-%d-%Y-%H:%M.log'.format("meta")), 'a'))
print = logger.info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
viewport_dataset_path = header.viewport_dataset
viewport_train_path = os.path.join(viewport_dataset_path, "train")
viewport_val_path = os.path.join(viewport_dataset_path, "val")
viewport_test_path = os.path.join(viewport_dataset_path, "test")
topic_dict = header.topic_info_dict

cluster_file_path = header.cluster_file_path
cluster_num = header.cluster_num
task_num = cluster_num + 2

train = None
train_gt = None
train_loaders = [[] for i in range(cluster_num)]
val_loaders = [[] for i in range(cluster_num)]
test_loaders = [[] for i in range(cluster_num)]

def get_files(path):
    file_group = [[] for i in range(cluster_num)]
    filenames = os.listdir(path)
    filenames.sort()
    # cluster(viewport_train_path, cluster_file_path)
    _, user_cluster, _ = get_cluster(cluster_num, cluster_file_path)
    for filename in filenames:
        user_name = filename.split("_")[-1]
        for i in range(cluster_num):
            if user_name in user_cluster[i]:
                file_group[i].append(filename)
                break
    return file_group
        

def create_dataloaders(loaders, filenames, path):
    for filename in tqdm(filenames):
        file_path=os.path.join(path, filename)
        with open(file_path, 'rb' ) as f:
            data = pickle.load(f)
            data = simplify_data(data)
            input, gt = create_dataset(data, 10)
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
            loader = DataLoader(MyDataset(input, gt), batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=4, drop_last=True)
            loaders.append(loader)
    return loaders


lstm = MetaViewportLSTM(task_num=task_num).to(device)
if args.load:
    load_path = "../models/" + args.model_name +".pt"
    lstm.load_state_dict(torch.load(load_path)["model"])
    saved_loss = torch.load(load_path)["valid_loss"]
    print("saved_loss: {}".format(saved_loss))
print(next(lstm.lstm.parameters()).device)
print(lstm)


if args.train or args.ftrain:
    print("Loading training dataset...")
    train_file_group = get_files(viewport_train_path)
    for i in range(cluster_num):
        train_loaders[i] = create_dataloaders(train_loaders[i], train_file_group[i], viewport_train_path)
    print("Loading validation dataset...")
    val_file_group = get_files(viewport_val_path)
    for i in range(cluster_num):
        val_loaders[i] = create_dataloaders(val_loaders[i], val_file_group[i], viewport_val_path)
if args.test or args.ftest:
    print("Loading test dataset...")
    test_file_group = get_files(viewport_test_path)
    for i in range(cluster_num):
        test_loaders[i] = create_dataloaders(test_loaders[i], test_file_group[i], viewport_test_path)

# 训练+验证

def train(train_loaders, val_loaders, save_path, task_index, min_valid_loss):
    optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH/2, gamma=0.3)
    loss_func = nn.MSELoss(size_average=True, reduce=True)
    train_loss = []
    valid_loss = []
    # min_valid_loss = np.inf
    for i in range(EPOCH):
        total_train_loss = []    
        lstm.train()
        hidden = lstm.init_hidden(BATCH_SIZE)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        print("Epoch: {}, training...".format(i))
        cnt = 0
        for train_loader in tqdm(train_loaders):
            if cnt >= 100: break
            cnt += 1
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.transpose(0, 1)
                b_x = b_x.type(torch.FloatTensor).to(device)
                b_y = b_y.type(torch.FloatTensor).to(device)
                prediction,  hidden = lstm(b_x, (hidden, task_index))
                loss = loss_func(prediction[-1, :, :], b_y)
                hidden = repackage_hidden(hidden)  
                optimizer.zero_grad()
                loss.backward(retain_graph=True)                         
                optimizer.step()                        
                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss)) 
    
        
        total_valid_loss = [] 
        lstm.eval()
        print("Epoch: {}, validation...".format(i))
        correct = 0
        total = 0   
        hidden = lstm.init_hidden(BATCH_SIZE)
        for valid_loader in tqdm(val_loaders):
            hidden = (hidden[0].to(device), hidden[1].to(device))
            for step, (b_x, b_y) in enumerate(valid_loader):
                b_x = b_x.transpose(0, 1)
                b_x = b_x.type(torch.FloatTensor).to(device) 
                b_y = b_y.type(torch.FloatTensor).to(device) 
                with torch.no_grad():
                    prediction,  hidden = lstm(b_x, (hidden, task_index)) 
                c, t = eval_correctness(prediction[-1, :, :], b_y)
                loss = loss_func(prediction[-1, :, :], b_y)     
                total_valid_loss.append(loss.item()) 
                correct += c
                total += t
        correctness = correct / total * 100 
        # valid_loss.append(np.mean(total_valid_loss))      
        valid_loss.append(correctness)        
        if (valid_loss[-1] > min_valid_loss[task_index]):      
            min_valid_loss[task_index] = valid_loss[-1]
            torch.save({'epoch': i, 'model': lstm.state_dict(), 'train_loss': train_loss,
                    'valid_loss': min_valid_loss}, save_path) # 保存字典对象，里面'model'的value是模型
    #         torch.save(optimizer, './LSTM.optim')     # 保存优化器      
            
            
        # 编写日志
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, correctness: {:0.6f}, '
                        'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                        train_loss[-1],
                                                                        valid_loss[-1],
                                                                        correctness,
                                                                        min_valid_loss[task_index],
                                                                        optimizer.param_groups[0]['lr'])
        logger.info(log_string)    # 打印日志
        scheduler.step()
    return min_valid_loss

def eval_correctness(x, gt):
    correct = 0
    total = 0
    for i in range(x.shape[0]):
        a = (find_nearest(x[i][0], "w"), find_nearest(x[i][1], "h"))
        b = (find_nearest(gt[i][0], "w"), find_nearest(gt[i][1], "h"))
        if a == b: correct += 1
        total += 1
    return correct, total
    
def ftrain(train_loaders, val_loaders, save_path, task_index, min_valid_loss):
    optimizer = torch.optim.Adam(lstm.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH / 2, gamma=0.5)
    loss_func = nn.MSELoss(size_average=True, reduce=True)
    train_loss = []
    valid_loss = []
    # min_valid_loss = np.inf
    for i in range(EPOCH):
        total_train_loss = []    
        lstm.train()
        hidden = lstm.init_hidden(BATCH_SIZE)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        print("Epoch: {}, training...".format(i))
        cnt = 0
        for train_loader in tqdm(train_loaders):
            if cnt >= 100: break
            cnt += 1
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.transpose(0, 1)
                b_x = b_x.type(torch.FloatTensor).to(device)
                b_y = b_y.type(torch.FloatTensor).to(device)
                prediction,  hidden = lstm(b_x, (hidden, task_index))
                loss = loss_func(prediction[-1, :, :], b_y)
                hidden = repackage_hidden(hidden)  
                optimizer.zero_grad()
                loss.backward(retain_graph=True)                         
                optimizer.step()                        
                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss)) 
    
        
        total_valid_loss = [] 
        lstm.eval()
        print("Epoch: {}, validation...".format(i))
        correct = 0
        total = 0   
        hidden = lstm.init_hidden(BATCH_SIZE)
        for valid_loader in tqdm(val_loaders):
            hidden = (hidden[0].to(device), hidden[1].to(device))
            for step, (b_x, b_y) in enumerate(valid_loader):
                b_x = b_x.transpose(0, 1)
                b_x = b_x.type(torch.FloatTensor).to(device) 
                b_y = b_y.type(torch.FloatTensor).to(device) 
                with torch.no_grad():
                    prediction,  hidden = lstm(b_x, (hidden, task_index)) 
                c, t = eval_correctness(prediction[-1, :, :], b_y)
                loss = loss_func(prediction[-1, :, :], b_y)     
                total_valid_loss.append(loss.item()) 
                correct += c
                total += t
        correctness = correct / total * 100 
        # valid_loss.append(np.mean(total_valid_loss))      
        valid_loss.append(correctness)        
        if (valid_loss[-1] > min_valid_loss[task_index]):      
            torch.save({'epoch': i, 'model': lstm.state_dict(), 'train_loss': train_loss,
                    'valid_loss': min_valid_loss}, save_path) # 保存字典对象，里面'model'的value是模型
    #         torch.save(optimizer, './LSTM.optim')     # 保存优化器      
            min_valid_loss[task_index] = valid_loss[-1]
            
        # 编写日志
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, correctness: {:0.6f}, '
                        'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                        train_loss[-1],
                                                                        valid_loss[-1],
                                                                        correctness,
                                                                        min_valid_loss[task_index],
                                                                        optimizer.param_groups[0]['lr'])
        logger.info(log_string)    # 打印日志
        scheduler.step()
    # return min_valid_loss

def test(test_loaders, load_path, task_index):
    lstm = MetaViewportLSTM(task_num=task_num).to(device)
    lstm.load_state_dict(torch.load(load_path)["model"])
    print("Testing...")
    correct = 0
    total = 0
    hidden = lstm.init_hidden(BATCH_SIZE)
    for test_loader in tqdm(test_loaders):
        hidden = (hidden[0].to(device), hidden[1].to(device))
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.transpose(0, 1)
            b_x = b_x.type(torch.FloatTensor).to(device) 
            b_y = b_y.type(torch.FloatTensor).to(device) 
            with torch.no_grad():
                prediction,  hidden = lstm(b_x, (hidden, task_index)) 
            c, t = eval_correctness(prediction[-1, :, :], b_y)
            correct += c
            total += t
    correctness = correct / total * 100
    print("Test correctness: {:.4f}%".format(correctness))

if args.ftrain:
    min_loss = [-np.inf for i in range(cluster_num)]
    task_index = args.task
    # min_loss = saved_loss
    print("fast training task {}...".format(task_index))
    save_path = "../models/" + args.model_name + "_" + str(task_index) + ".pt"             
    ftrain(train_loaders[task_index], test_loaders[task_index], save_path, task_index, min_loss)

if args.train:
    min_loss = [-np.inf for i in range(cluster_num)]
    for j in range(10):
        print("training task 0...")
        save_path = "../models/" + args.model_name +".pt"           
        train(train_loaders[0], val_loaders[0], save_path, 0, min_loss)

        print("training task 1...")
        save_path = "../models/" + args.model_name +".pt"           
        train(train_loaders[1], val_loaders[1], save_path, 1, min_loss)

        print("training task 2...")
        save_path = "../models/" + args.model_name +".pt"             
        train(train_loaders[2], val_loaders[2], save_path, 2, min_loss)

        print("training task 3...")
        save_path = "../models/" + args.model_name +".pt"            
        train(train_loaders[3], val_loaders[3], save_path, 3, min_loss)

        print("Epoch {} ends. Correctness: {}".format(j, min_loss))
        

if args.ftest:
    task_index = args.task
    print("testing task {}...".format(task_index))
    load_path  = "../models/" + args.model_name + "_" + str(task_index) + ".pt"   
    print("After Fast Training")    
    test(test_loaders[task_index], load_path, task_index) 

elif args.test:
    print("testing task 0...")
    load_path = "../models/" + args.model_name +".pt"  
    print("After clustering")    
    test(test_loaders[0], load_path, 0) 

    print("testing task 1...")
    load_path = "../models/" + args.model_name +".pt"  
    print("After clustering")    
    test(test_loaders[1], load_path, 1) 

    print("testing task 2...")
    load_path = "../models/" + args.model_name +".pt"  
    print("After clustering")    
    test(test_loaders[2], load_path, 2)   

    print("testing task 3...")
    load_path = "../models/" + args.model_name +".pt"  
    print("After clustering")    
    test(test_loaders[3], load_path, 3)              
