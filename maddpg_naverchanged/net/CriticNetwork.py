import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

        self._reset_parameters(self.critic)
        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.to(args.device)

    def forward(self, state, action):
        state = T.cat(state, dim=1)
        action = T.cat(action, dim=1)
        x = T.cat([state, action], dim=1)
        value = self.critic(x)
        return value

    def _reset_parameters(self, layers, std=1.0, bias_const=1e-6):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)