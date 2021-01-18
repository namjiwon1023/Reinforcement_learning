import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def initialize_uniformly(layer = nn.Linear, init_w = 3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, state_size, hidden_layer_1 = 300, hidden_layer_2 = 600):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)

        self.steering_mu_layer = nn.Linear(600, 1)
        self.steering_log_std_layer = nn.Linear(600, 1)

        initialize_uniformly(self.steering_mu_layer)
        initialize_uniformly(self.steering_log_std_layer)

        self.acceleration_mu_layer = nn.Linear(600, 1)
        self.acceleration_log_std_layer = nn.Linear(600, 1)

        initialize_uniformly(self.acceleration_mu_layer)
        initialize_uniformly(self.acceleration_log_std_layer)

        self.brake_mu_layer = nn.Linear(600, 1)
        self.brake_log_std_layer = nn.Linear(600, 1)

        initialize_uniformly(self.brake_mu_layer)
        initialize_uniformly(self.brake_log_std_layer)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        steering_mu = torch.tanh(self.steering_mu_layer(x))
        steering_log_std = F.softplus(self.steering_log_std_layer(x))
        steering_std = torch.exp(steering_log_std)


        acceleration_mu = torch.sigmoid(self.acceleration_mu_layer(x))
        acceleration_log_std = F.softplus(self.acceleration_log_std_layer(x))
        acceleration_std = torch.exp(acceleration_log_std)


        brake_mu = torch.sigmoid(self.brake_mu_layer(x))
        brake_log_std = F.softplus(self.brake_log_std_layer(x))
        brake_std = torch.exp(brake_log_std)

        steering_dist = Normal(steering_mu, steering_std)
        steering_action = steering_dist.sample()

        acceleration_dist = Normal(acceleration_mu, acceleration_std)
        acceleration_action = acceleration_dist.sample()

        brake_dist = Normal(brake_mu, brake_std)
        brake_action = brake_dist.sample()


        action_out = torch.cat((steering_action, acceleration_action, brake_action), 1)
        # dist_out = [steering_dist, acceleration_dist, brake_dist]
        # dist_out = torch.cat((steering_dist, acceleration_dist, brake_dist), 1)
        return action_out,steering_dist,acceleration_dist,brake_dist

