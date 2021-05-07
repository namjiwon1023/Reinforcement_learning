import torch as T
import os
import numpy as np
import random
import copy

from Network.ActorNetwork import Actor
from Network.CriticNetwork import Critic

class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.actor = Actor(args, agent_id)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic = Critic(args)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for q in self.critic_target.parameters():
            q.requires_grad = False

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/actor_params.pth'):
            self.actor.load_state_dict(T.load(self.model_path + '/actor_params.pth'))
            self.critic.load_state_dict(T.load(self.model_path + '/critic_params.pth'))
            print('Agent {} successfully loaded actor network: {}'.format(self.agent_id, self.model_path + '/actor_params.pth'))
            print('Agent {} successfully loaded critic network: {}'.format(self.agent_id, self.model_path + '/critic_params.pth'))

    def _soft_update_target_network(self):
        tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)
            for t_p, l_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = T.as_tensor(transitions[key], dtype=T.float32, device=self.args.device)

        r = transitions['r_%d' % self.agent_id]
        done = transitions['done_%d' % self.agent_id]
        mask = 1 - done

        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        u_next = []
        with T.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target(o_next[agent_id]))
                else:
                    u_next.append(other_agents[index].policy.actor_target(o_next[agent_id]))
                    index += 1

            q_next = self.critic_target(o_next, u_next)
            target_q = r + self.args.gamma * q_next * mask


        q_value = self.critic(o, u)
        critic_loss = (target_q - q_value) ** 2
        critic_loss = critic_loss.mean()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        u_new = copy.deepcopy(u)
        u_new[self.agent_id] = self.actor(o[self.agent_id])

        actor_loss = -self.critic(o, u_new).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

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