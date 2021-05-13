import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(args.obs_shape[agent_id], 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, args.action_shape[agent_id]))

        self._reset_parameters(self.actor)
        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.to(args.device)

    def forward(self, x):
        x = self.actor(x)
        actions = T.tanh(x)
        return actions

    def _reset_parameters(self, layers, std=1.0, bias_const=1e-6):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)