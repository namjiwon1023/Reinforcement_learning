import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from ReplayBuffer import Buffer
import os
import matplotlib.pyplot as plt

from MADDPG import MADDPG
from arguments import get_args
from utils import make_env, _random_seed

# Single Agent
class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def choose_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            with T.no_grad():
                inputs = T.as_tensor(o, dtype=T.float32, device=self.args.device).unsqueeze(0) # Improve the dimensionality of state
                pi = self.policy.actor_network(inputs).squeeze(0) # Reduce the dimensionality of the action
                u = pi.detach().cpu().numpy()

                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape) # gaussian noise
                u += noise # explore
                u = np.clip(u, -self.args.high_action, self.args.high_action) # Clip the action value
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env

        self.agents = self._init_agents()

        self.buffer = Buffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # make a lot of agent
    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    # train function
    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                self.env.render()

            u = []
            actions = []
            for agent_id, agent in enumerate(self.agents):
                action = agent.choose_action(s[agent_id], self.noise, self.epsilon)
                u.append(action)
                actions.append(action)
            # good agent random action settings
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next

            # learning start
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)

            # Predict Data Drawing
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

            # Gaussian noise discount factor
            self.noise = max(0.05, self.noise - 0.0000005)

            # epsilon discount factor
            self.epsilon = max(0.05, self.noise - 0.0000005)

            # save total reward
            np.save(self.save_path + '/returns.pkl', returns)

    # evaluate function
    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    action = agent.choose_action(s[agent_id], 0, 0)
                    actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes

if __name__ == '__main__':
    # random seed set
    _random_seed(123)
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()