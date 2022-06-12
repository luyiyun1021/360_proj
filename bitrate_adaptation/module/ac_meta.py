import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_SPACE = 5 * 2
INPUT_SIZE = 4 * 2 + 1 + 2 + 1 # s + r + a + t
HIDDEN_SIZE = 48
NUM_LAYERS = 1
class MetaAC(torch.nn.Module):
    def __init__(self, action_space=ACTION_SPACE, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(MetaAC, self).__init__()
        self.action_space = action_space
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ac_fc_1 = nn.Linear(1, 128)
        self.ac_fc_2 = nn.Linear(1, 128)
        self.ac_fc_3 = nn.Linear(1, 128)
        self.ac_fc_4 = nn.Linear(1, 128)
        self.ac_conv1 = nn.Conv1d(1, 128, 1) # L_out = 2 - (2-1) -1 + 1 = 1
        self.ac_conv2 = nn.Conv1d(1, 128, 1)
        self.ac_conv3 = nn.Conv1d(1, 128, 1) # L_out = 2 - (2-1) -1 + 1 = 1
        self.incoming_size =  10 * 128
        self.lstm = nn.LSTM(input_size=self.incoming_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.lstm_fc = nn.Linear(self.hidden_size, self.incoming_size) 

        self.a_fc1 = nn.Linear(in_features=self.incoming_size, out_features=128)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.a_fc3 = nn.Linear(in_features=128, out_features=self.action_space)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

        self.c_fc1 = nn.Linear(in_features=self.incoming_size, out_features=128)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.c_fc3 = nn.Linear(in_features=128, out_features=1)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, inputs):
        states = inputs[0]
        reward = inputs[1]
        action = inputs[2]
        time = inputs[3]
        hidden = inputs[4]
        time_batch = states[:, 0:1, -1] 
        buffer_batch = states[:, 1:2, -1]
        viewport_batch = states[:, 2:3, :2]
        bandwidth_batch = states[:, 3:4, :2]
        batch_size = states.shape[0]

        x_1 = F.relu(self.ac_fc_1(time_batch))
        x_2 = F.relu(self.ac_fc_2(buffer_batch))
        x_3 = F.relu(self.ac_conv1(viewport_batch))
        x_4 = F.relu(self.ac_conv2(bandwidth_batch))
        x_5 = F.relu(self.ac_fc_3(reward))
        x_6 = F.relu(self.ac_conv3(action))
        x_7 = F.relu(self.ac_fc_4(time))

        x_1 = x_1.contiguous().view(-1, self.num_flat_features(x_1))
        x_2 = x_2.contiguous().view(-1, self.num_flat_features(x_2))
        x_3 = x_3.contiguous().view(-1, self.num_flat_features(x_3))
        x_4 = x_4.contiguous().view(-1, self.num_flat_features(x_4))
        x_5 = x_5.contiguous().view(-1, self.num_flat_features(x_5))
        x_6 = x_6.contiguous().view(-1, self.num_flat_features(x_6))
        x_7 = x_7.contiguous().view(-1, self.num_flat_features(x_7))
        x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7], 1)
        lstm_input = x.view(-1, batch_size, self.incoming_size)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        lstm_out = F.relu(self.lstm_fc(lstm_out)).view(batch_size, -1, self.incoming_size).view(batch_size, -1)

        a_x = F.relu(self.a_fc1(lstm_out))
        a_x = self.a_fc3(a_x)
        a_x = a_x.view(-1, int(self.action_space / 5), 5)
        a_x = F.softmax(a_x, dim=2)
        actor = a_x.view(-1, a_x.shape[1] * a_x.shape[2])

        c_x = F.relu(self.c_fc1(x))
        critic = self.c_fc3(c_x)

        return actor, critic, hidden

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

if __name__ == "__main__":
    bs = 4
    model = MetaAC()
    states = torch.rand(4, 4, 2)
    reward = torch.rand(4, 1, 1)
    action = torch.rand(4, 1, 2)
    time = torch.rand(4, 1, 1)
    hidden = model.init_hidden(bs)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(10):
        optimizer.zero_grad()
        actor, critic, hidden = model((states, reward, action, time, hidden))
        hidden = (hidden[0].detach(), hidden[1].detach())
        print(actor.shape, critic.shape, hidden[0].shape)
        loss = (0.5 - actor).mean() + (0.1 - critic).mean()
        loss.backward()
        optimizer.step()
