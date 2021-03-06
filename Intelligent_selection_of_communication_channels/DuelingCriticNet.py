import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingCriticNet(nn.Module):
    def __init__(self, in_dim, action_size):
        super(DuelingCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 10, kernel_size = 2, stride = 1, padding = 0)
        self.pool = nn.AvgPool2d((2,1))
        # self.fc1 = nn.Linear(10*1*1, 6)
        # self.fc1 = nn.Linear(10*1*1, 30)
        # self.fc2 = nn.Linear(30, 6)
        self.value = nn.Linear(10*1*1, 1)
        self.advantage = nn.Linear(10*1*1, action_size)

    def forward(self,state):
        # print('state size : ',state.size())
        x = self.pool(F.relu(self.conv1(state)))
        # print('size : ', x.size())
        x = x.view(-1, 10*1*1)
        # x = F.relu(self.fc1(x))

        value = self.value(x)
        advantage = self.advantage(x)
        # out = self.fc1(x)
        # out = self.fc2(x)
        Q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return Q