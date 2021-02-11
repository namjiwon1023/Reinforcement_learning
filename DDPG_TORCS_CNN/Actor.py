import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_size, 8, kernel_size = 4, stride = 2, padding = 2)
        self.b1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size = 4, stride = 1, padding = 2)
        self.b2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size = 4, stride = 1, padding = 2)
        self.b3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(39200, 2048)
        self.fc2 = nn.Linear(2048, 512)

        self.steering = nn.Linear(512, 1)
        self.steering.weight.data.normal_(0,1e-4)

        self.acceleration = nn.Linear(512, 1)
        self.acceleration.weight.data.normal_(0,1e-4)

        self.brake = nn.Linear(512, 1)
        self.brake.weight.data.normal_(0,1e-4)

    def forward(self, s):
        x = F.relu(self.b1(self.conv1(s)))
        x = F.relu(self.b2(self.conv2(x)))
        x = F.relu(self.b3(self.conv3(x)))
        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        steering_out = torch.tanh(self.steering(x))
        acceleration_out = torch.sigmoid(self.acceleration(x))
        brake_out = torch.sigmoid(self.brake(x))

        out = torch.cat((steering_out, acceleration_out, brake_out), 1)
        return out

