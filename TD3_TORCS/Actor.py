import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, hidden_layer_1 = 300, hidden_layer_2 = 600):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)

        self.steering = nn.Linear(hidden_layer_2, 1)
        self.steering.weight.data.normal_(0,1e-4)

        self.acceleration = nn.Linear(hidden_layer_2, 1)
        self.acceleration.weight.data.normal_(0,1e-4)

        self.brake = nn.Linear(hidden_layer_2, 1)
        self.brake.weight.data.normal_(0,1e-4)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        steering_out = torch.tanh(self.steering(x))
        acceleration_out = torch.sigmoid(self.acceleration(x))
        brake_out = torch.sigmoid(self.brake(x))

        out = torch.cat((steering_out, acceleration_out, brake_out), 1)
        return out

