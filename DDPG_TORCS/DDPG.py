import copy
import random

from gym_torcs import TorcsEnv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

import matplotlib.pyplot as plt

from Actor import Actor
from Critic import Critic
from OU import OU
from ReplayBuffer import ReplayBuffer

EPISODE_COUNT = int(2e3)
MAX_STEPS = int(1e5)
EXPLORE = 100000.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # memory setting
        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size)
        self.transition = list()

        # Actor Network setting(eval , target)
        self.actor_eval = Actor(self.state_size).to(device)
        self.actor_eval.apply(init_weights)

        self.actor_target = Actor(self.state_size).to(device)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.actor_target.eval()

        # Critic Network setting(eval , target)
        self.critic_eval = Critic(self.state_size, self.action_size).to(device)

        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.critic_target.eval()

        # Optimization Setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.critic_lr)

        # Loss Function
        self.critic_loss_func = nn.MSELoss()

        self.critic_L = 0.
        self.actor_L = 0.

        # Load Neural Network Parameters
        self.dirPath = './load_state/DDPG_epsilon_'
        self.Actor_dirPath = './load_state/DDPG_actor_'
        self.Critic_dirPath = './load_state/DDPG_critic_'

        if self.load_model :
            self.actor_eval.load_state_dict(torch.load(self.Actor_dirPath + str(self.load_episode) + '.h5'))
            self.critic_eval.load_state_dict(torch.load(self.Critic_dirPath + str(self.load_episode) + '.h5'))
            with open(self.dirPath + str(self.load_episode) + '.pkl','rb') as outfile:
                param = pickle.load(outfile)
                if param is not None:
                    self.epsilon = param.get('epsilon')
                else:
                    pass

    def learn(self):
        # Random Samples
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.FloatTensor(samples['act']).reshape(-1,3).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        # Critic Network Update
        mask = (1 - done).to(device)
        next_action = self.actor_target(next_state).to(device)
        next_value = self.critic_target(next_state, next_action).to(device)
        target_values = (reward + self.gamma * next_value * mask).to(device)

        eval_values = self.critic_eval(state, action).to(device)
        critic_loss = self.critic_loss_func(eval_values, target_values).to(device)
        self.critic_L = critic_loss.detach().cpu().numpy()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Network Update
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).to(device).mean()
        self.actor_L = actor_loss.detach().cpu().numpy()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update -> Target Network Parameters
        # θQ‘←τθQ+(1−τ)θQ‘
        # θμ‘←τθμ+(1−τ)θμ‘
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

    def _plot(self, frame_idx, scores, actor_losses, critic_losses,):
        def subplot(loc, title, values):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.savefig('./torcs_al_cl_r.jpg')
        plt.show()


if __name__ == "__main__":

    # Setting Neural Network Parameters
    params = {
                'memory_size' : int(1e5),
                'batch_size' : 32,
                'state_size' : 29,
                'action_size' : 3,
                'gamma' : 0.95,
                'tau' : 1e-3,
                'vision' : False,
                'actor_lr' : 1e-4,
                'critic_lr' : 1e-3,
                'epsilon' : 1,
                'load_model' : True,
                'load_episode' : 890,
                'step' : 0,
                'train' : True,
        }

    agent = Agent(**params)

    # Ornstein-Uhlenbeck Process
    OU = OU()

    # Environment Setting
    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)
    param_dictionary = dict()

    actor_losses = []
    critic_losses = []
    scores = []

    # Train Process
    for e in range(agent.load_episode, EPISODE_COUNT):

        if e % 3 == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        score = 0.

        # Store Neural Network Parameters
        if e % 10 == 0:
            torch.save(agent.actor_eval.state_dict(), agent.Actor_dirPath + str(e) + '.h5')
            torch.save(agent.critic_eval.state_dict(), agent.Critic_dirPath + str(e) + '.h5')
            with open(agent.dirPath + str(e) + '.pkl' , 'wb') as outfile:
                if agent.epsilon is not None:
                    pickle.dump(param_dictionary, outfile)
                else:
                    pass

        for i in range(MAX_STEPS):

            loss = 0
            agent.epsilon -= 1.0 / EXPLORE
            noise = np.zeros([1, agent.action_size])
            a_n = np.zeros([1, agent.action_size])

            # Choose Action
            a = agent.actor_eval(torch.unsqueeze(torch.FloatTensor(s),0).to(device)).detach().cpu().numpy()

            # Setting Noise Functions
            if agent.train is True:
                noise[0][0] = max(agent.epsilon, 0) * OU.function(a[0][0], 0.0, 0.60, 0.30)
                noise[0][1] = max(agent.epsilon, 0) * OU.function(a[0][1], 0.5, 1.00, 0.10)
                noise[0][2] = max(agent.epsilon, 0) * OU.function(a[0][2], -0.1, 1.00, 0.05)

                if random.random() <= 0.1:
                    print("apply the brake")
                    noise[0][2] = max(agent.epsilon, 0) * OU.function(a[0][2], 0.2, 1.00, 0.10)
            else:
                pass

            a_n[0][0] = a[0][0] + noise[0][0]
            a_n[0][1] = a[0][1] + noise[0][1]
            a_n[0][2] = a[0][2] + noise[0][2]

            ob, r, done, _ = env.step(a_n[0])

            s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            agent.transition = [s, a_n[0], r, s_, done]
            agent.memory.store(*agent.transition)

            agent.learn()

            loss = agent.critic_L
            score += r
            s = s_

            print('Episode : {} Step : {}  Action : {} Reward : {} Loss : {}'.format(e, agent.step , a_n, r, loss))
            agent.step += 1

            if done :

                scores.append(score)
                actor_losses.append(agent.actor_L)
                critic_losses.append(agent.critic_L)

                param_keys = ['epsilon']
                param_value = [agent.epsilon]
                param_dictionary = dict(zip(param_keys,param_value))

                break

        if e == EPISODE_COUNT:
            agent._plot(agent.step, scores, actor_losses, critic_losses,)

        print('|============================================================================================|')
        print('|=========================================  Result  =========================================|')
        print('|                                     Total_Step : {}  '.format(agent.step))
        print('|                      Episode : {} Total_Reward : {} '.format(e, score))
        print('|============================================================================================|')

    env.end()
    print('Finish.')
