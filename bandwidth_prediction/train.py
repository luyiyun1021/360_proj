import pickle

from numpy.core.numeric import Inf
import numpy as np
from copy import deepcopy
import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataloader
import logging
from time import strftime
from matplotlib import pyplot as plt
import os
from module.bandwidth_lstm import BandwidthLSTM
from tqdm import tqdm
import argparse

BATCH_SIZE = 1
EPOCH = 50
LOAD_PATH = "../models/bandwidth_lstm2.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "../belgium"
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False, help='Train')
parser.add_argument('--test', type=bool, default=False, help='Test')
args = parser.parse_args()

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
            train.append([item for item in dataset[i : i + look_back]])
            gt.append(dataset[i + look_back])
    return np.asarray(train), np.asarray(gt) 

#logger
global print
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
try:
    os.makedirs("logger", exist_ok=True)
except TypeError:
    raise Exception("Direction not create!")
logger.addHandler(logging.FileHandler(strftime('logger/{}_%m-%d-%Y-%H:%M.log'.format("bandwidth_estimate")), 'a'))
print = logger.info



def get_files(path):
    filenames = os.listdir(path)
    filenames.sort()
    return filenames
        

def create_dataloaders(loaders, filenames, path):
    for filename in tqdm(filenames):
        file_path=os.path.join(path, filename)
        with open(file_path, 'r' ) as f:
            lines = f.readlines()
            data = []
            for line in lines:
                bandwidth = int(line.split()[4]) / (1024 * 1024)
                if bandwidth == 0: continue
                data.append(bandwidth)
            input, gt = create_dataset(data, 10)
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
            loader = DataLoader(MyDataset(input, gt), batch_size=BATCH_SIZE, shuffle=False, sampler=None, num_workers=4, drop_last=True)
            loaders.append(loader)
    return loaders

train_loaders = []
val_loaders = []
test_loaders = []
file_groups = get_files(DATASET_PATH)
train_file_group = file_groups[:-2]
val_file_group = [file_groups[-2]]
test_file_group = [file_groups[-1]]
print("Loading training dataset...")
create_dataloaders(train_loaders, train_file_group, DATASET_PATH)
print("Loading validation dataset...")
create_dataloaders(val_loaders, val_file_group, DATASET_PATH)
print("Loading test dataset...")
create_dataloaders(test_loaders, test_file_group, DATASET_PATH)

# 训练+验证
def train(train_loaders, val_loaders, save_path):
    lstm = BandwidthLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH / 2, gamma=0.3)
    loss_func = nn.MSELoss(size_average=True, reduce=True)
    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf
    for i in range(EPOCH):
        total_train_loss = []    
        lstm.train()
        print("Epoch: {}, training...".format(i))
        for train_loader in tqdm(train_loaders):
            hidden = lstm.init_hidden(BATCH_SIZE)
            hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.transpose(0, 1)
                b_x = b_x.unsqueeze(2)
                b_x = b_x.type(torch.FloatTensor).to(DEVICE)
                b_y = b_y.type(torch.FloatTensor).to(DEVICE)
                prediction,  hidden = lstm(b_x, hidden)
                loss = loss_func(prediction[-1, :, :], b_y)
                hidden = repackage_hidden(hidden)  
                optimizer.zero_grad()                   
                loss.backward()                         
                optimizer.step()                        
                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss)) 
    
        
        total_valid_loss = [] 
        lstm.eval()
        print("Epoch: {}, validation...".format(i))
        for valid_loader in tqdm(val_loaders):
            hidden = lstm.init_hidden(BATCH_SIZE)
            hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
            for step, (b_x, b_y) in enumerate(valid_loader):
                b_x = b_x.transpose(0, 1)
                b_x = b_x.unsqueeze(2)
                b_x = b_x.type(torch.FloatTensor).to(DEVICE) 
                b_y = b_y.type(torch.FloatTensor).to(DEVICE) 
                with torch.no_grad():
                    prediction,  hidden = lstm(b_x, hidden) 
                loss = loss_func(prediction[-1, :, :], b_y)     
                total_valid_loss.append(loss.item()) 
        valid_loss.append(np.mean(total_valid_loss))      
        if (valid_loss[-1] < min_valid_loss):      
            torch.save({'epoch': i, 'model': lstm.state_dict(), 'train_loss': train_loss,
                    'valid_loss': valid_loss}, save_path) # 保存字典对象，里面'model'的value是模型
    #         torch.save(optimizer, './LSTM.optim')     # 保存优化器      
            min_valid_loss = valid_loss[-1]
            
        # 编写日志
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                        'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                        train_loss[-1],
                                                                        valid_loss[-1],
                                                                        min_valid_loss,
                                                                        optimizer.param_groups[0]['lr'])
        logger.info(log_string)    # 打印日志
        scheduler.step()
    

def test(test_loaders, load_path):
    lstm = BandwidthLSTM().to(DEVICE)
    loss_func = nn.MSELoss(size_average=True, reduce=True)
    lstm.load_state_dict(torch.load(load_path)["model"])
    print("Testing...")
    total_loss = []
    x = []
    y = []
    for test_loader in tqdm(test_loaders):
        hidden = lstm.init_hidden(BATCH_SIZE)
        hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.transpose(0, 1)
            b_x = b_x.unsqueeze(2)
            b_x = b_x.type(torch.FloatTensor).to(DEVICE) 
            b_y = b_y.type(torch.FloatTensor).to(DEVICE) 
            with torch.no_grad():
                prediction,  hidden = lstm(b_x, hidden) 
            loss = loss_func(prediction[-1, :, :], b_y)     
            total_loss.append(loss.item()) 
            x.extend([item for item in prediction[-1, :, 0].cpu().data.numpy()])
            y.extend([item for item in b_y.cpu().data.numpy()])
        mean_loss = np.mean(total_loss)
    print("Test loss: {:.4f}".format(mean_loss))
    plt.figure(dpi=600)
    plt.plot(x, 'r', label='prediction', lw=1, alpha=0.8)
    plt.plot(y, 'b', label='real', lw=1, alpha=0.8)
    plt.legend(loc='best')
    plt.savefig('./bandwidth_estimate.jpg')

if args.train:
    save_path = LOAD_PATH          
    train(train_loaders, val_loaders, save_path)

if args.test:
    load_path = LOAD_PATH
    test(test_loaders, load_path)           
