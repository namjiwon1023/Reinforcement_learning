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


EPISODE_COUNT = int(2e3)
MAX_STEPS = int(1e5)
EXPLORE = 100000.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

init_w = 3e-3
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, -init_w, init_w)
        m.bias.data.fill_(0.0)

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.transition = list()

        # Actor Network setting(eval , target)
        self.actor = Actor(self.state_size).to(device)
        self.actor.apply(init_weights)

        # Critic Network setting(eval , target)
        self.critic = Critic(self.state_size).to(device)

        # Optimization Setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Loss Function
        self.critic_loss_func = nn.MSELoss()

        self.critic_L = 0.
        self.actor_L = 0.

        # Load Neural Network Parameters
        self.dirPath = './load_state/DDPG_epsilon_'
        self.Actor_dirPath = './load_state/DDPG_actor_'
        self.Critic_dirPath = './load_state/DDPG_critic_'

        if self.load_model :
            self.actor.load_state_dict(torch.load(self.Actor_dirPath + str(self.load_episode) + '.h5'))
            self.critic.load_state_dict(torch.load(self.Critic_dirPath + str(self.load_episode) + '.h5'))
            with open(self.dirPath + str(self.load_episode) + '.pkl','rb') as outfile:
                param = pickle.load(outfile)
                if param is not None:
                    self.epsilon = param.get('epsilon')
                else:
                    pass

    def learn(self):
        state, log_prob, next_state, reward, done = self.transition

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)

        mask = 1 - done
        pred_value = self.critic(state)
        targ_value = reward + self.gamma * self.critic(next_state) * mask
        value_loss = self.critic_loss_func(pred_value, targ_value.detach())

        self.critic_L = value_loss.detach().cpu().numpy()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor Network Update
        advantage = (targ_value - pred_value).detach()
        policy_loss = -advantage * log_prob
        policy_loss += self.entropy_weight * -log_prob

        self.actor_L = policy_loss.detach().cpu().numpy()

        self.actor_optimizer.zero_grad()
        # policy_loss.backward()
        policy_loss.sum().backward()
        self.actor_optimizer.step()

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
                'state_size' : 29,
                'action_size' : 3,
                'gamma' : 0.95,
                'vision' : False,
                'actor_lr' : 1e-4,
                'critic_lr' : 1e-3,
                'entropy_weight' : 1e-2,
                'epsilon' : 1,
                'load_model' : False,
                'load_episode' : 0,
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
            torch.save(agent.actor.state_dict(), agent.Actor_dirPath + str(e) + '.h5')
            torch.save(agent.critic.state_dict(), agent.Critic_dirPath + str(e) + '.h5')
            np.savetxt("./test.txt",scores, delimiter=",")
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
            log_prob = torch.zeros([1, agent.action_size]).to(device)
            # Choose Action
            a, steering_dist, acceleration_dist, brake_dist = agent.actor(torch.unsqueeze(torch.FloatTensor(s),0).to(device))
            a = a.detach().cpu().numpy()
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
            log_prob[0][0] = steering_dist.log_prob(a_n[0][0]).sum(dim=-1)
            log_prob[0][1] = acceleration_dist.log_prob(a_n[0][1]).sum(dim=-1)
            log_prob[0][2] = brake_dist.log_prob(a_n[0][2]).sum(dim=-1)

            agent.transition = [s, log_prob[0], s_, r, done]

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

        if e == 1999:
            agent._plot(agent.step, scores, actor_losses, critic_losses,)
            np.savetxt("./scores.txt",scores, delimiter=",")

        print('|============================================================================================|')
        print('|=========================================  Result  =========================================|')
        print('|                                     Total_Step : {}  '.format(agent.step))
        print('|                      Episode : {} Total_Reward : {} '.format(e, score))
        print('|============================================================================================|')

    env.end()
    print('Finish.')
