# -*- coding:utf8 -*-
import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    parser.add_argument("--use-cuda", type=bool, default=True, help="Using GPU or CPU")
    parser.add_argument("--device", default=device, help="Using GPU or CPU")
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden layer units")

    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--total-steps", type=int, default=2000000, help="number of time steps")

    parser.add_argument("--value-iter", type=int, default=1, help="learning rate of actor")
    parser.add_argument("--policy-iter", type=int, default=1, help="learning rate of actor")

    parser.add_argument("--actor-lr", type=float, default=1e-2, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-2, help="learning rate of critic")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="exploration noise")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=1e-2, help="parameter for updating the target network")
    parser.add_argument("--memory-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--train-start", type=int, default=2000, help="training start step")

    parser.add_argument("--checkpoint", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    return parser.parse_args()
