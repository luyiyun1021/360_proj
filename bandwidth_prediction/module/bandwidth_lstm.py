import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
B_FEATURE_NUM = 1
B_HIDDEN_SIZE = 16
B_NUM_LAYERS = 2
class BandwidthLSTM(nn.Module):
    def __init__(self,
                 input_size=B_FEATURE_NUM,
                 hidden_size=B_HIDDEN_SIZE,
                 num_layers=B_NUM_LAYERS):

        super(BandwidthLSTM, self).__init__() 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size, B_FEATURE_NUM) 
     

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


    def forward(self, input, hidden=None):
        x = input
        x, hidden = self.lstm(x, hidden) # input of shape (seq_len, batch, input_size)
        x = self.fc(x)
        return x, hidden # output of shape (seq_len, batch, 1)

if __name__ == "__main__":
    model = BandwidthLSTM(input_size=2,
                         hidden_size=5,
                         num_layers=3)
    h = model.init_hidden(4)
    input = torch.rand(6, 4, 2)
    print(input.shape)
    out, h = model(input, h)
    print(out.shape)
    print(h[0].shape)
    print(h[1].shape)
    '''
    h = model.init_hidden(3)
    input = torch.rand(4, 3, 1)
    a, h = model(input, h)
    print(a, h)
    '''
