import numpy as np
import matplotlib.pyplot as plt
import torch as T
import random

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def _random_seed(self, seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True
    self.seed = seed
    T.manual_seed(self.seed)
    np.random.seed(self.seed)
    random.seed(self.seed)