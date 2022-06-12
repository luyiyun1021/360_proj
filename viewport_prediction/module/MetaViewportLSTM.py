import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.MetaLSTM import MetaLSTM
B_FEATURE_NUM = 2
B_HIDDEN_SIZE = 8
B_NUM_LAYERS = 2
B_HYPER_HIDDEN_SIZE = 16
B_HYPER_EMBEDDING_SIZE = 16
B_TASK_NUM = 5
class MetaViewportLSTM(nn.Module):
    def __init__(self,
                 input_size=B_FEATURE_NUM,
                 hidden_size=B_HIDDEN_SIZE,
                 hyper_hidden_size = B_HYPER_HIDDEN_SIZE,
                 hyper_embedding_size = B_HYPER_EMBEDDING_SIZE,
                 num_layers=B_NUM_LAYERS,
                 task_num=B_TASK_NUM,
                 gpu=True):

        super(MetaViewportLSTM, self).__init__() 
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.num_layers = num_layers 
        self.task_num = task_num
        self.lstm = MetaLSTM(input_size=self.input_size, hidden_size=self.hidden_size, hyper_hidden_size=self.hyper_hidden_size,
        hyper_embedding_size=self.hyper_embedding_size, num_layers=self.num_layers, task_num=self.task_num, gpu=gpu)
        #self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size, B_FEATURE_NUM) 
     

    def init_hidden(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.hidden_size),
        torch.randn(self.num_layers, batch_size, self.hidden_size))


    def forward(self, input, hidden=None):
        x = input
        x, hidden = self.lstm(x, hidden) # input of shape (seq_len, batch, input_size)
        x = self.fc(x)
        return x, hidden # output of shape (seq_len, batch, input_size), (num_layers, batch, hidden_size)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    model = MetaViewportLSTM(input_size=2,
                             hidden_size=5,
                             hyper_hidden_size=10,
                             hyper_embedding_size=15,
                             num_layers=1,
                             task_num=5)
    loss_func = nn.MSELoss(size_average=True, reduce=True)
    device = "cuda:0"
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    h = model.init_hidden(4)
    h = (h[0].to(device), h[1].to(device))
    input = torch.rand(6, 4, 2).to(device)
    out_gt = torch.rand(4, 2).to(device)
    for i in range(5):
        print("round {}: ".format(i))
        input = input.transpose(0, 1)
        input = input.transpose(1, 0)
        input = input.type(torch.FloatTensor).to(device)
        out, h = model(input, (h, i))
        #print(input.shape)
        #print(out.shape)
        #print(h[0].shape)
        #print(h[1].shape)
        loss = loss_func(out[-1, :, :], out_gt)
        h = repackage_hidden(h) 
        optimizer.zero_grad()                
        loss.backward(retain_graph=True)   
        print(loss.item())
        optimizer.step()
    '''
    h = model.init_hidden(3)
    input = torch.rand(4, 3, 1)
    a, h = model(input, h)
    print(a, h)
    '''
