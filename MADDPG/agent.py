import numpy as np
import torch as T
import os
from MADDPG import MADDPG

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def choose_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = T.as_tensor(o, dtype=T.float32, device=self.device).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            u = pi.detach().cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape) # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)