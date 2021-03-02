import torch as T
import torch.nn as nn
import torch.optim as optim
import os

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, alpha,
        checkpoint_path='/home/nam/Reinforcement_learning/TD3'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.dirPath = os.path.join(self.checkpoint_path, 'torch_TD3_actor')

        self.actor = nn.Sequential(nn.Linear(n_states, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, n_actions)
                                    )

        self.optimizer = optim.Adam(self.paramters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        action = self.fc1(state)
        out = torch.tanh(action)
        return 2 * action

    def save_models(self):
        T.save(self.state_dict(), self.dirPath)
        T.save(self.optimizer.state_dict(), self.dirPath + '_optimizer')

    def load_model(self):
        self.load_state_dict(T.load(self.dirPath))
        self.optimizer.load_state_dict(T.load(self.dirPath + '_optimizer'))
