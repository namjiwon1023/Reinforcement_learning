import torch as T
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, sum_obs, sum_act, args):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                                nn.Linear(sum_obs + sum_act, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, 1)
                                )
        self.to(args.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        value = self.critic(x)
        return value