# -*- coding:utf8 -*-
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def _layer_norm(layer, std=1.0, bias_const=1e-6):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')   # mode : fan_in or fan_out , nonlinearity : leaky_relu or relu

def _random_seed(seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)

def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + '.py').Scenario()

    # create world
    world = scenario.make_world()

    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    args.n_players = env.n # The number of all players including enemies
    args.n_agents = env.n - args.num_adversaries # The number of players that need to be controlled, although the enemy can also be controlled, but if both parties learn, different algorithms are required
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)] # Each dimension represents the obs dimension of the agent

    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents] # Each dimension represents the act dimension of the agent

    '''Define the highest and lowest value of the action'''
    args.high_action = 1
    args.low_action = -1

    return env, args