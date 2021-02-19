import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ActorCriticCNN(nn.Module):
    '''
        input: [None, 3, 64, 64]; output: [None, 1024] -> [None, 512];
    '''
    def __init__(self, input_dims, n_actions):
        super(ActorCriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 32, kernel_size = 8, stride = 4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)       # 64 -> 128 / 64 -> 64

        self.fc1 = nn.Linear(64*4*4, 512)

        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*4*4)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

class ActorCriticCNN_2(nn.Module):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 4, 84, 84]; output: [None, 2592] -> [None, 256];
    '''
    def __init__(self, input_dims, n_actions):
        super(ActorCriticCNN_2, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)

        self.fc1 = nn.Linear(None, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self,s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = x.view(-1, None)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

class ActorCriticCNN_3(nn.Module):
    ''' DQN Nature 2015 paper
        input: [None, 4, 84, 84]; output: [None, 3136] -> [None, 512];
    '''
    def __init__(self, input_dims, n_actions):
        super(ActorCriticCNN_3, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(None, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self,s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, None)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

class ActorCriticCNN_4(nn.Module):
    ''' Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 2, 120, 160]; output: [None, 1280] -> [None, 256];
    '''
    def __init__(self, input_dims, n_actions):
        super(ActorCriticCNN_4, self).__init__()
        self.conv1 = nn.Conv2d(input_dims, 8, kernel_size = 5, stride = 4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2)

        self.fc1 = nn.Linear(None, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self,s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, None)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out