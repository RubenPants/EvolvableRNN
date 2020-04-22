"""
fitness_functions.py

Create visualizations for each of the fitness functions.
"""
import argparse
import os
from math import sqrt

import matplotlib.pyplot as plt
from numpy import clip

from config import GameConfig


def distance(save: bool = True):
    """Create an image for the distance-fitness."""
    cfg = GameConfig()
    diagonal = sqrt(cfg.x_axis ** 2 + cfg.y_axis ** 2)
    
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
    
    plt.figure()
    
    # Plot the distance function
    plt.plot(x, y, 'b', label='distance-based score')
    plt.axvspan(0, cfg.target_reached, alpha=0.5, color='green', label='target reached')
    
    # Beautify the plot
    plt.title('Fitness in function of distance to target')
    plt.xlabel("Distance to target")
    plt.xticks([i * 2 for i in range(round(diagonal / 2) + 1)])
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='x')
    if save: plt.savefig('population/utils/visualizing/images/distance_fitness.png')
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
