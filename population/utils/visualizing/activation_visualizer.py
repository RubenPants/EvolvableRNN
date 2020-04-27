"""
activations.py

Visualize each of the activation functions.
"""
import argparse
import os

import matplotlib.pyplot as plt

from population.utils.network_util.activations import *


def visualize_function(f, name: str, save: bool = True):
    """Visualize the given function."""
    x = []
    y = []
    for i in range(-350, 350):
        x.append(i / 100)
        y.append(f(i / 100))
    
    # Plot the distance function
    plt.figure(figsize=(8,2.2))
    plt.axvline(color='k')
    plt.axhline(color='k')
    plt.plot(x, y, linewidth=4)
    
    # Shared labels across all functions
    plt.title(f'{name} activation function')
    plt.xlabel("Input-value")
    plt.ylabel("Output-value")
    plt.tight_layout()
    plt.grid(axis='both')
    if save: plt.savefig(f'population/utils/visualizing/images/{name}.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir('../../../')
    
    for s, f in [('sigmoid', sigmoid), ('tanh', np.tanh)]:
        visualize_function(
                f=f,
                name=s,
                save=args.save
        )
