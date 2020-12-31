import copy
import random
from gym_torcs import TorcsEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Actor import Actor
from Critic import Critic
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer

EPISODE_COUNT = 2e3
MAX_STEPS = 1e5
EXPLORE = 100000.

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size)
        self.transition = list()

        self.actor_eval = Actor(self.state_size).to(self.device)
        self.actor_eval.apply(init_weights)
        self.actor_target = Actor(self.state_size).to(self.device)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.actor_target.eval()

        self.critic_eval = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.critic_lr)

        self.critic_loss_func = nn.MSELoss()

        self.noise_steering = OUNoise(size = 1, mu = 0.0, theta = 0.6, sigma = 0.30)
        self.noise_acceleration = OUNoise(size = 1, mu = 0.5, theta = 1.0, sigma = 0.10)
        self.noise_brake = OUNoise(size = 1, mu = -0.1, theta = 1.0, sigma = 0.05)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 1e-4)
            m.bias.data.fill_(0.0)

    def learn(self):
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(self.device)
        next_state = torch.FloatTensor(samples['next_obs']).to(self.device)
        action = torch.LongTensor(samples['act']).reshape(-1,1).to(self.device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(self.device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(self.device)

        # Critic Network Update
        mask = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        target_values = reward + self.gamma * next_value * mask

        eval_values = self.critic_eval(state, action)
        critic_loss = self.critic_loss_func(eval_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Network Update
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._target_soft_update()

    def _target_soft_update(self):
        tau = self.tau

        for target_params , eval_params in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

        for target_params , eval_params in zip(
            self.critic_target.parameters(), self.critic_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

if __name__ == "__main__":
    params = {
                'memory_size' : 1e5,
                'batch_size' : 32,
                'state_size' : 29,
                'action_size' : 3,
                'gamma' : 0.99,
                'tau' : 1e-3,
                'vision' : False,
                'actor_lr' : 1e-4,
                'critic_lr' : 1e-3,
                'epsilon' = 1,
        }
    agent = Agent(**params)
    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)

    for e in range(EPISODE_COUNT):
        if np.mod(e, 3) == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        done = False

        for i in range(MAX_STEPS):
            agent.epsilon -= 1.0 / EXPLORE
            a = agent.actor_eval(torch.FloatTensor(s).to(agent.device)).detach().cpu().numpy()

            noise_s = max(agent.epsilon, 0) * agent.noise_steering
            noise_a = max(agent.epsilon, 0) * agent.noise_acceleration
            noise_b = max(agent.epsilon, 0) * agent.noise_brake

            a_n[0][0] = a[0][0] + noise_s
            a_n[0][1] = a[0][1] + noise_a
            a_n[0][2] = a[0][2] + noise_b

            ob, r, done, info = env.step(a_n[0])

            s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            agent.transition = [s, a_n[0], r, s_, done]
            agent.memory.store(*agent.transition)

            agent.learn()

            total_reward += r
            s = s_

            if done :
                print("---Episode ", e , "  Action:", a_n, "  Reward:", total_reward, "  Loss:", agent.critic_loss')
                break