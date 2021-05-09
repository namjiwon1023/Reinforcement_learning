import torch as T
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
                                nn.Linear(args.obs_shape, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.action_shape)
                                )
        self.to(args.device)

    def forward(self, x):
        x = self.actor(x)
        action = T.tanh(x)
        return action