"""
fitness_functions.py

Create visualizations for each of the fitness functions.
"""
import argparse
import os

import matplotlib.pyplot as plt
from numpy import clip

from config import GameConfig
from utils.dictionary import *


def distance(save: bool = True):
    """Create an image for the distance-fitness."""
    cfg = GameConfig()
    diagonal = 10
    
    # Define the function
    def get_score(d, reached=False):
        """Get a score for the given distance."""
        return 1 if reached else clip((1 - (d - cfg.target_reached) / diagonal) ** 2, a_min=0, a_max=1)
    
    # Create the figure
    x = []
    y = []
    for i in range(0, round(diagonal * 100)):
        x.append(i / 100)
        y.append(get_score(i / 100))
    
    plt.figure(figsize=(5, 2))
    
    # Plot the distance function
    plt.plot(x, y, color=COLORS[D_NEAT], label='distance-based score')
    plt.axvspan(0, cfg.target_reached, color='#97e08d', label='target reached')
    
    # Beautify the plot
    plt.title('Fitness in function of distance to target')
    plt.xlabel("Distance to target")
    plt.xticks([i * 2 for i in range(round(diagonal / 2) + 1)])
    plt.xlim(0, 10)
    plt.ylabel("Fitness")
    plt.ylim(0)
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.tight_layout()
    plt.grid()
    if save:
        plt.savefig('population/utils/visualizing/images/distance_fitness.png', bbox_inches='tight', pad_inches=0.02)
        plt.savefig('population/utils/visualizing/images/distance_fitness.eps',
                    format='eps',
                    bbox_inches='tight',
                    pad_inches=0.02)
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save', type=bool, default=True)
    
    # Fitness functions
    parser.add_argument('--distance', type=bool, default=True)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir('../../../')
    
    if args.distance:
        distance(
                save=args.save
        )
