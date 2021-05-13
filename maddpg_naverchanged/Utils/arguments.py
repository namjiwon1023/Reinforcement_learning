import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
def get_args():
    parser = argparse.ArgumentParser("Multi Agent Reinforcement Learning MADDPG")

    parser.add_argument("--device", default=device, help="Use CPU or GPU")

    parser.add_argument("--benchmark", type=bool, default=False, help="whether you want to produce benchmarking data")
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=int(3e7), help="number of time steps")

    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    parser.add_argument("--actor-lr", type=float, default=1e-2, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-2, help="learning rate of critic")

    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")

    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=1e-2, help="parameter for updating the target network")
    parser.add_argument("--memory-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()
