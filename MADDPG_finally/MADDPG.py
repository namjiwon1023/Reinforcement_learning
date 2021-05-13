# -*- coding:utf8 -*-
import torch as T
import torch.optim as optim
import os
import numpy as np
import random
import copy

from ActorCriticNetwork import Actor, Critic
from utils import  _layer_norm

class MADDPG:
    '''# Because the obs and act dimensions of different agents may be different, so the neural network is different, and agent_id is needed to distinguish'''
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # Create the Actor-Critic Network
        self.actor = Actor(args, agent_id)
        self.actor.apply(_layer_norm)

        self.critic = Critic(args)
        self.critic.apply(_layer_norm)

        # build up the actor target network
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # build up the critic target network
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for q in self.critic_target.parameters():
            q.requires_grad = False

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # load model
        if os.path.exists(self.model_path + '/actor_params.pth'):
            self.actor.load_state_dict(T.load(self.model_path + '/actor_params.pth'))
            self.critic.load_state_dict(T.load(self.model_path + '/critic_params.pth'))
            print('Agent {} successfully loaded actor network: {}'.format(self.agent_id, self.model_path + '/actor_params.pth'))
            print('Agent {} successfully loaded critic network: {}'.format(self.agent_id, self.model_path + '/critic_params.pth'))

    # soft update
    def _soft_update_target_network(self):
        with T.no_grad():
            for t_p, l_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_p.data.copy_(self.args.tau * l_p.data + (1 - self.args.tau) * t_p.data)

            for t_p, l_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_p.data.copy_(self.args.tau * l_p.data + (1 - self.args.tau) * t_p.data)

    # update the Actor-Critic Networks
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = T.tensor(transitions[key], dtype=T.float32, device=self.args.device)

        r = transitions['r_%d' % self.agent_id] # You only need your own reward during training
        done = transitions['done_%d' % self.agent_id]

        o, u, o_next = [], [], [] # Used to store the various items in each agent's experience

        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with T.no_grad():
            # Get the action corresponding to the next state
            index = 0
            for agent_id in range(self.args.n_agents):
                # If the current agent_id is the same as the updated id, the current agent will be used directly
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target(o_next[agent_id]))
                else:
                    # Because the incoming other_agents is one less than the total number, maybe some agent in the middle is the current agent and cannot be traversed to select actions
                    u_next.append(other_agents[index].policy.actor_target(o_next[agent_id]))
                    index += 1

            q_next = self.critic_target(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next * (1-done)).detach()


        q_value = self.critic(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # actor loss
        # Reselect the action of the current agent in the joint action, and the actions of other agents remain unchanged
        u[self.agent_id] = self.actor(o[self.agent_id])
        actor_loss = -self.critic(o, u).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)

        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)

        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        T.save(self.actor.state_dict(), model_path + '/' + num + '_actor_params.pth')
        T.save(self.critic.state_dict(),  model_path + '/' + num + '_critic_params.pth')