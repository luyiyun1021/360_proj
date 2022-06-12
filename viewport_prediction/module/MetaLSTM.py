# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/25 21:19
'''
import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.autograd import Variable
import torch.nn.functional as F

class MetaRNNCellBase(Module):
    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}, {hyper_hidden_size}, {hyper_embedding_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'bias_hyper' in self.__dict__ and self.bias is not True:
            s += ', bias_hyper={bias_hyper}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class MetaRNNCell(MetaRNNCellBase):
    def __init__(self, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, bias=True, bias_hyper=True):
        super(MetaRNNCell, self).__init__()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class MetaLSTMCell(MetaRNNCellBase):
    def __init__(self, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, bias=True, bias_hyper=True, grad_clip=None, task_num=1):
        super(MetaLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.grad_clip = grad_clip
        self.dropout = torch.nn.Dropout(0.5)

        # nn
        '''
        self.iH = []
        self.HH = []
        for i in range(task_num):
            self.iH.append(nn.Linear(input_size, 4 * hidden_size))
            self.HH.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.ih = nn.Linear(input_size + hidden_size, 4 * hyper_hidden_size)
        self.hh = nn.Linear(hyper_hidden_size, 4 * hyper_hidden_size)
        self.hzi = nn.Linear(hyper_hidden_size, hyper_embedding_size)
        self.hzH = nn.Linear(hyper_hidden_size, hyper_embedding_size)
        self.hzb = nn.Linear(hyper_hidden_size, hyper_embedding_size)
        self.dziH = nn.Linear(hyper_embedding_size, 4 * hidden_size)
        self.dzHH = nn.Linear(hyper_embedding_size, 4 * hidden_size)
        self.bzH = nn.Linear(hyper_embedding_size, 4 * hidden_size)
        '''

        # F
        self.weight_iH = Parameter(torch.Tensor(task_num, 4 * hidden_size, input_size))
        self.weight_HH = Parameter(torch.Tensor(task_num, 4 * hidden_size, hidden_size))

        self.weight_ih = Parameter(torch.Tensor(4 * hyper_hidden_size, input_size + hidden_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hyper_hidden_size, hyper_hidden_size))

        self.weight_hzi = Parameter(torch.Tensor(hyper_embedding_size, hyper_hidden_size))
        self.weight_hzH = Parameter(torch.Tensor(hyper_embedding_size, hyper_hidden_size))
        self.weight_hzb = Parameter(torch.Tensor(hyper_embedding_size, hyper_hidden_size))
        self.weight_dziH = Parameter(torch.Tensor(4 * hidden_size, hyper_embedding_size))
        self.weight_dzHH = Parameter(torch.Tensor(4 * hidden_size, hyper_embedding_size))
        self.weight_bzH = Parameter(torch.Tensor(4 * hidden_size, hyper_embedding_size))
        if bias:
            self.bias_i = Parameter(torch.Tensor(hyper_embedding_size))
            self.bias_H = Parameter(torch.Tensor(hyper_embedding_size))
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        if bias_hyper:
            self.bias_hyper = Parameter(torch.Tensor(4 * hyper_hidden_size))
        else:
            self.register_parameter('bias_hyper', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        task_index = state[1]
        state = state[0]
        main_state = state[0]
        meta_state = state[1]
        main_h = main_state[0]
        main_c = main_state[1]
        meta_h = meta_state[0]
        meta_c = meta_state[1]
        
        # F
        # solution to inplace operation in F.linear (A^T)
        tmp_weight_hh = self.weight_hh.clone()
        tmp_weight_ih = self.weight_ih.clone()
        tmp_weight_hzi = self.weight_hzi.clone()
        tmp_weight_hzH = self.weight_hzH.clone()
        tmp_weight_hzb = self.weight_hzb.clone()
        tmp_weight_dziH = self.weight_dziH.clone()
        tmp_weight_iH = self.weight_iH[task_index].clone()
        tmp_weight_dzHH = self.weight_dzHH.clone()
        tmp_weight_HH = self.weight_HH[task_index].clone()
        tmp_weight_bzH = self.weight_bzH.clone()


        meta_pre = F.linear(torch.cat((input, main_h), 1), tmp_weight_ih) + F.linear(meta_h, tmp_weight_hh) + self.bias_hyper

        meta_i = F.sigmoid(meta_pre[:, : self.hyper_hidden_size])
        meta_f = F.sigmoid(meta_pre[:, self.hyper_hidden_size: self.hyper_hidden_size * 2])
        meta_g = F.tanh(meta_pre[:, self.hyper_hidden_size * 2: self.hyper_hidden_size * 3])
        meta_o = F.sigmoid(meta_pre[:, self.hyper_hidden_size * 3: ])
        meta_c = meta_f * meta_c + meta_i * meta_g
        meta_h = meta_o * F.tanh(meta_c)

        zi = F.linear(meta_h, tmp_weight_hzi) + self.bias_i
        zH = F.linear(meta_h, tmp_weight_hzH) + self.bias_H
        zb = F.linear(meta_h, tmp_weight_hzb)

        pre = F.linear(zi, tmp_weight_dziH) * F.linear(input, tmp_weight_iH) + F.linear(zH, tmp_weight_dzHH) * F.linear(main_h, tmp_weight_HH) + F.linear(zb, tmp_weight_bzH) + self.bias

        main_i = F.sigmoid(pre[:, : self.hidden_size])
        main_f = F.sigmoid(pre[:, self.hidden_size: self.hidden_size * 2])
        main_g = F.tanh(pre[:, self.hidden_size * 2: self.hidden_size * 3])
        main_o = F.sigmoid(pre[:, self.hidden_size * 3:])
        main_c = main_f * main_c + main_i * self.dropout(main_g)
        main_h = main_o * F.tanh(main_c)
        

        ### nn
        '''
        meta_pre = self.ih(torch.cat((input, main_h), 1)) + self.hh(meta_h) + self.bias_hyper

        meta_i = F.sigmoid(meta_pre[:, : self.hyper_hidden_size])
        meta_f = F.sigmoid(meta_pre[:, self.hyper_hidden_size: self.hyper_hidden_size * 2])
        meta_g = F.tanh(meta_pre[:, self.hyper_hidden_size * 2: self.hyper_hidden_size * 3])
        meta_o = F.sigmoid(meta_pre[:, self.hyper_hidden_size * 3: ])
        meta_c = meta_f * meta_c + meta_i * meta_g
        meta_h = meta_o * F.tanh(meta_c)

        zi = self.hzi(meta_h) + self.bias_i
        zH = self.hzH(meta_h) + self.bias_H
        zb = self.hzb(meta_h)

        pre = self.dziH(zi) * self.iH[task_index](input) + self.dzHH(zH) * self.HH[task_index](main_h) + self.bzH(zb) + self.bias

        main_i = F.sigmoid(pre[:, : self.hidden_size])
        main_f = F.sigmoid(pre[:, self.hidden_size: self.hidden_size * 2])
        main_g = F.tanh(pre[:, self.hidden_size * 2: self.hidden_size * 3])
        main_o = F.sigmoid(pre[:, self.hidden_size * 3:])
        main_c = main_f * main_c + main_i * self.dropout(main_g)
        main_h = main_o * F.tanh(main_c)
        '''
        return ((main_h, main_c), (meta_h, meta_c))

class MetaRNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, num_layers, bias=True, bias_hyper=True, gpu=False, bidirectional=False, task_num=1):
        super(MetaRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.num_layers = num_layers
        self.bias = bias
        self.bias_hyper = bias_hyper
        self.gpu = gpu
        self.bidirectional=bidirectional

        mode2cell = {'MetaRNN': MetaRNNCell,
                     'MetaLSTM': MetaLSTMCell}

        Cell = mode2cell[mode]

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'hyper_hidden_size': hyper_hidden_size,
                  'hyper_embedding_size': hyper_embedding_size,
                  'bias': bias,
                  'bias_hyper': bias_hyper, 
                  'task_num':task_num}
        self.cell0 = Cell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'] = hidden_size
            cell = Cell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

        self.states = None

    def _initial_states(self, inputSize):
        main_zeros = Variable(torch.zeros(inputSize, self.hidden_size))
        meta_zeros = Variable(torch.zeros(inputSize, self.hyper_hidden_size))
        if self.gpu:
            main_zeros = main_zeros.cuda()
            meta_zeros = meta_zeros.cuda()
        zeros = (main_zeros, meta_zeros)
        if self.mode == 'MetaLSTM':
            states = [((main_zeros, main_zeros), (meta_zeros, meta_zeros)), ] * self.num_layers
        else:
            states = [zeros] * self.num_layers
        return states

    def forward(self, input, hidden=None):
        if self.states == None:
            self.states = self._initial_states(input.size(1))
        outputs = []
        time_steps = input.size(0)

        '''
        if length is None:
            length = Variable(torch.LongTensor([time_steps] * input.size(1)))
            if self.gpu:
                length = length.cuda()
        '''

        outputs_f = []
        outputs_h = []
        outputs_c = []
        task_index = 0
        if hidden is not None:
            task_index = hidden[1]
            hidden = hidden[0]
            for num in range(self.num_layers):
                (main_h, main_c), (meta_h, meta_c) = self.states[num]
                main_h = hidden[0][num]
                main_c = hidden[1][num]
                self.states[num] = ((main_h, main_c), (meta_h, meta_c))

        for num in range(self.num_layers):
            for t in range(time_steps):
                x = input[t, :, :]
                (main_h, main_c), (meta_h, meta_c) = getattr(self, 'cell{}'.format(num))(x, (self.states[num], task_index))
                '''
                mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                mask_main_c = (t < length).float().unsqueeze(1).expand_as(main_c)
                mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                mask_meta_c = (t < length).float().unsqueeze(1).expand_as(meta_c)
                main_h = main_h * mask_main_h + states[0][0][0] * (1 - mask_main_h)
                main_c = main_c * mask_main_c + states[0][0][1] * (1 - mask_main_c)
                meta_h = meta_h * mask_meta_h + states[0][1][0] * (1 - mask_meta_h)
                meta_c = meta_c * mask_meta_c + states[0][1][1] * (1 - mask_meta_c)
                '''
                self.states[num] = ((main_h, main_c),(meta_h, meta_c))
                outputs_f.append(main_h)
                
            outputs_h.append(main_h)
            outputs_c.append(main_c)
            input = torch.stack(outputs_f)
            outputs_f = []

        output = input, (torch.stack(outputs_h), torch.stack(outputs_c))
        return output

class MetaRNN(MetaRNNBase):
    def __init__(self, *args, **kwargs):
        super(MetaRNN, self).__init__('MetaRNN', *args, **kwargs)

class MetaLSTM(MetaRNNBase):
    def __init__(self, *args, **kwargs):
        super(MetaLSTM, self).__init__('MetaLSTM', *args, **kwargs)
