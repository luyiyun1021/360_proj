import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(2, 2)
 
    def forward(self,x):
        return self.linear(x)