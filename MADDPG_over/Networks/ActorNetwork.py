import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super(Actor, self).__init__()
        self.args = args
        self.actor = nn.Sequential(
                                nn.Linear(obs_dim, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, act_dim)
                                )
        self.reset_parameters(self.actor)
        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.to(args.device)

    def forward(self, x):
        x = self.actor(x)
        action = T.tanh(x)
        return action

    def reset_parameters(self, layers, std=1.0, bias_const=1e-6):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def act(self, obs, current_step, noise=True):
        if current_step < self.args.train_start:
            choose_action = np.random.uniform(-1, 1, [5])
        else:
            action = self.forward(obs).detach().cpu.numpy()
            if noise:
                noise = np.random.normal(0, self.args.exploration_noise, size = action.shape)
                choose_action = action + noise
                choose_action = np.clip(choose_action, -1, 1)
            else:
                choose_action = action
        return choose_action
