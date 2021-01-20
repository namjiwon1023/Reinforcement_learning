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
from GaussianNoise import GaussianNoise
from ReplayBuffer import ReplayBuffer

EPISODE_COUNT = 2000
MAX_STEPS = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # GaussianNoise
        self.GN = GaussianNoise()
        self.train_noise = GN(self..action_size, self.train_noise, self.train_noise)
        self.target_noise = GN(self.action_size, self.target_noise, self.target_noise)

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
        self.critic_eval_1 = Critic(self.state_size, self.action_size).to(device)

        self.critic_target_1 = Critic(self.state_size, self.action_size).to(device)
        self.critic_target_1.load_state_dict(self.critic_eval_1.state_dict())
        self.critic_target_1.eval()

        # Critic Network setting(eval , target)
        self.critic_eval_2 = Critic(self.state_size, self.action_size).to(device)

        self.critic_target_2 = Critic(self.state_size, self.action_size).to(device)
        self.critic_target_2.load_state_dict(self.critic_eval_2.state_dict())
        self.critic_target_2.eval()

        critic_eval_parameters = list(self.critic_eval_1.parameters()) + list(self.critic_eval_2.parameters())

        # Optimization Setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(critic_eval_parameters, lr=self.critic_lr)

        # Loss Function
        self.critic_loss_func = nn.MSELoss()

        self.critic_L = 0.
        self.actor_L = 0.

        # Load Neural Network Parameters
        self.dirPath = './load_state/TD3_epsilon_'
        self.Actor_dirPath = './load_state/TD3_actor_'
        self.Critic_dirPath = './load_state/TD3_critic_'

        if self.load_model :
            self.actor_eval.load_state_dict(torch.load(self.Actor_dirPath + str(self.load_episode) + '.h5'))
            self.critic_eval_1.load_state_dict(torch.load(self.Critic_dirPath + '_1' + str(self.load_episode) + '.h5'))
            self.critic_eval_2.load_state_dict(torch.load(self.Critic_dirPath + '_2' + str(self.load_episode) + '.h5'))

    def learn(self):
        # Random Samples
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.FloatTensor(samples['act']).reshape(-1,3).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)
        mask = (1 - done).to(device)

        noise = torch.FloatTensor(self.target_noise.sample()).to(device)
        clipped_noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)

        # Critic Network Update
        next_action[0][0] = (self.actor_target(next_state)[0][0] + clipped_noise[0]).clamp(-1.0, 1.0)
        next_action[0][1] = (self.actor_target(next_state)[0][1] + clipped_noise[1]).clamp(0.0, 1.0)
        next_action[0][2] = (self.actor_target(next_state)[0][2] + clipped_noise[2]).clamp(0.0, 1.0)


        # min
        next_value_1 = self.critic_target_1(next_state, next_action).to(device)
        next_value_2 = self.critic_target_2(next_state, next_action).to(device)
        next_value = torch.min(next_value_1, next_value_2)

        target_values = (reward + self.gamma * next_value * mask).to(device)
        target_values = target_values.detach()

        eval_values_1 = self.critic_eval_1(state, action).to(device)
        eval_values_2 = self.critic_eval_2(state, action).to(device)
        critic_loss_1 = self.critic_loss_func(eval_values_1, target_values).to(device)
        critic_loss_2 = self.critic_loss_func(eval_values_2, target_values).to(device)

        critic_loss = critic_loss_1 + critic_loss_2
        self.critic_L = critic_loss.detach().cpu().numpy()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Network Update
        if self.total_episode % self.actor_update_time == 0:
            # Actor Network Update
            actor_loss = -self.critic_eval_1(state, self.actor_eval(state)).to(device).mean()
            self.actor_L = actor_loss.detach().cpu().numpy()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update -> Target Network Parameters
            # θQ‘←τθQ+(1−τ)θQ‘
            # θμ‘←τθμ+(1−τ)θμ‘
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

    def _target_soft_update(self):
        tau = self.tau

        for target_params , eval_params in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

        for target_params , eval_params in zip(
            self.critic_target_1.parameters(), self.critic_eval_1.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

        for target_params , eval_params in zip(
            self.critic_target_2.parameters(), self.critic_eval_2.parameters()
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
                'memory_size' : 100000,
                'batch_size' : 64,
                'state_size' : 29,
                'action_size' : 3,
                'gamma' : 0.96,
                'tau' : 5e-3,
                'train_noise' : 0.1,
                'target_noise' : 0.2,
                'target_noise_clip' : 0.5,
                'actor_update_time' : 2,
                'vision' : False,
                'actor_lr' : 3e-4,
                'critic_lr' : 1e-3,
                'load_model' : False,
                'load_episode' : 0,
                'total_episode' : 0,
                'step' : 0,
        }

    agent = Agent(**params)

    # Environment Setting
    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)
    param_dictionary = dict()

    actor_losses = []
    critic_losses = []
    scores = []

    # Train Process
    for e in range(agent.load_episode, EPISODE_COUNT):

        agent.total_episode = e

        if e % 4 == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        score = 0.

        # Store Neural Network Parameters
        if e % 10 == 0:
            torch.save(agent.actor_eval.state_dict(), agent.Actor_dirPath + str(e) + '.h5')
            torch.save(agent.critic_eval_1.state_dict(), agent.Critic_dirPath + '_1' + str(e) + '.h5')
            torch.save(agent.critic_eval_2.state_dict(), agent.Critic_dirPath + '_2' + str(e) + '.h5')
            np.savetxt("./test.txt",scores, delimiter=",")

        for i in range(MAX_STEPS):

            loss = 0
            a_n = np.zeros([1, agent.action_size])
            noise = np.zeros([1, agent.action_size])
            # Choose Action
            a = agent.actor_eval(torch.unsqueeze(torch.FloatTensor(s),0).to(device)).detach().cpu().numpy()

            # Setting Noise Functions
            noise[0][0] = agent.train_noise.sample()[0]
            noise[0][1] = agent.train_noise.sample()[1]
            noise[0][2] = agent.train_noise.sample()[2]

            a_n[0][0] = np.clip(a[0][0] + noise[0][0], -1.0, 1.0)
            a_n[0][1] = np.clip(a[0][1] + noise[0][1], 0.0, 1.0)
            a_n[0][2] = np.clip(a[0][2] + noise[0][2], 0.0, 1.0)

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

        if e == (EPISODE_COUNT - 1):
            agent._plot(agent.step, scores, actor_losses, critic_losses,)
            np.savetxt("./scores.txt",scores, delimiter=",")

        print('|============================================================================================|')
        print('|=========================================  Result  =========================================|')
        print('|                                     Total_Step : {}  '.format(agent.step))
        print('|                      Episode : {} Total_Reward : {} '.format(e, score))
        print('|============================================================================================|')

    env.end()
    print('Finish.')
