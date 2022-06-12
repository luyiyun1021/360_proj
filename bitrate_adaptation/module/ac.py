import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(torch.nn.Module):
    def __init__(self, action_space=5*2):
        super(Actor, self).__init__()
        self.action_space = action_space

        self.actor_fc_1 = nn.Linear(1, 128)
        self.actor_fc_2 = nn.Linear(1, 128)
        # self.actor_fc_3 = nn.Linear(2, 128)
        # self.actor_fc_4 = nn.Linear(2, 128)
        self.actor_conv1 = nn.Conv1d(1, 128, 1) # L_out = 2 - (2-1) -1 + 1 = 1
        self.actor_conv2 = nn.Conv1d(1, 128, 1)
        #===================Hide layer=========================
        incoming_size =  6 * 128 

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=128)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=128, out_features=self.action_space)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        time_batch = inputs[:, 0:1, -1] 
        buffer_batch = inputs[:, 1:2, -1]
        viewport_batch = inputs[:, 2:3, :2]
        bandwidth_batch = inputs[:, 3:4, :2]

        x_1 = F.relu(self.actor_fc_1(time_batch))
        x_2 = F.relu(self.actor_fc_2(buffer_batch))
        x_3 = F.relu(self.actor_conv1(viewport_batch))
        x_4 = F.relu(self.actor_conv2(bandwidth_batch))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))

        x = torch.cat([x_1, x_2, x_3, x_4], 1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        x = x.view(-1, int(self.action_space / 5), 5)
        x = F.softmax(x, dim=2)
        actor = x.view(-1, x.shape[1] * x.shape[2])
        return actor

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.critic_fc_1 = nn.Linear(1, 128)
        self.critic_fc_2 = nn.Linear(1, 128)
        # self.critic_fc_3 = nn.Linear(2, 128)
        # self.critic_fc_4 = nn.Linear(2, 128)
        self.critic_conv1 = nn.Conv1d(1, 128, 1) # L_out = 2 - (2-1) -1 + 1 = 1
        self.critic_conv2 = nn.Conv1d(1, 128, 1)

        #===================Hide layer=========================
        incoming_size =  6 * 128 

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=128)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        time_batch = inputs[:, 0:1, -1] 
        buffer_batch = inputs[:, 1:2, -1]
        viewport_batch = inputs[:, 2:3, :2]
        bandwidth_batch = inputs[:, 3:4, :2]

        x_1 = F.relu(self.critic_fc_1(time_batch))
        x_2 = F.relu(self.critic_fc_2(buffer_batch))
        x_3 = F.relu(self.critic_conv1(viewport_batch))
        x_4 = F.relu(self.critic_conv2(bandwidth_batch))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))

        x = torch.cat([x_1, x_2, x_3, x_4], 1)
        x = F.relu(self.fc1(x))
        critic = self.fc3(x)
        return critic

    def num_flat_features(self,x):
        size=x.size()[1:] # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features