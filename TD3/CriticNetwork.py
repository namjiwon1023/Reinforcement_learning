import torch as T
import torch.nn as nn
import torch.optim as optim
import os

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, alpha, checkpoint_path='/home/nam/Reinforcement_learning/TD3'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.dirPath = os.path.join(self.checkpoint_path, 'torch_TD3_critic')

        self.critic1 = nn.Sequential(nn.Linear(n_states + n_actions, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))

        self.critic2 = nn.Sequential(nn.Linear(n_states + n_actions, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat([state, action], 1)
        Q1 = self.critic1(cat)
        return Q1

    def get_double_Q(self, state, action):
        cat = T.cat([state, action], 1)
        Q1 = self.critic1(cat)
        Q2 = self.critic2(cat)
        return Q1, Q2

    def save_models(self):
        T.save(self.state_dict(), self.dirPath)
        T.save(self.optimizer.state_dict(), self.dirPath + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.dirPath))
        self.optimizer.load_state_dict(T.load(self.dirPath + '_optimizer'))