# -*- coding:utf8 -*-
import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser("Multi Agent Reinforcement Learning Parameters Settings")

    # GPU
    parser.add_argument("--device", default=device, help="Using GPU or CPU")

    # Network neuron settings
    parser.add_argument("--n-hiddens", type=int, default=64, help="hidden layers neuron")

    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="Multi Agent Environment Name")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length (steps)")
    parser.add_argument("--time-steps", type=int, default=2000000, help="training total steps")

    # A map can have at most env.n agents, the user can define min (env.n,num-adversaries) enemies, and the rest are good agents
    # Example : simple_tag : 1 good agent and 3 bed agents
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    # Core training parameters
    # Actor-Critic learning Rate
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")

    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")

    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")

    # Memory Setting
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")

    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")

    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")

    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
