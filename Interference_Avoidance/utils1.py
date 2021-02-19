import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve_1(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[:])
    plt.plot(x, running_avg)
    plt.title('Running average of scores')
    plt.savefig(figure_file)