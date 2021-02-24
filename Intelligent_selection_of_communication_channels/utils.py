import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    plt.figure()
    plt.title("Avg_scores")
    plt.grid(True)
    plt.xlabel("Episode")
    plt.xlim(0, x)
    plt.ylabel("Total reward")
    plt.ylim(0, 4)
    plt.plot(scores, "b-", linewidth=2.0, label="Avg_scores")
    plt.legend(loc="upper left", shadow=True)
    plt.savefig(figure_file)
    plt.show()