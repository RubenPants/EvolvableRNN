"""
specie_visualizer.py

Visualization of the distance between the current species.
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import zeros

from population.population import Population
from population.utils.population_util.species import GenomeDistanceCache
from utils.myutils import get_subfolder


def main(pop: Population, show: bool = True):
    """
    Visualize the elites of the given population. Each generation, the average fitness of the three stored elites is
    taken.

    :param pop: Population object
    :param show: Show the result
    """
    # Determine the distance between the representatives
    cache = GenomeDistanceCache(config=pop.config.genome)
    temp = [(s.key, s.representative) for s in pop.species.species.values()]
    species, representatives = zip(*sorted(temp, key=lambda x: x[0]))
    distances = zeros((len(representatives),) * 2)  # Square matrix
    for row, r1 in enumerate(representatives):
        for col, r2 in enumerate(representatives):
            distances[row, col] = cache(r1, r2)
    
    # Create the figure
    ax = sns.heatmap(distances, linewidth=0.5, annot=True, xticklabels=species, yticklabels=species)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(f"Distance between elite representatives at generation {pop.generation}")
    plt.xlabel("specie")
    plt.ylabel("specie")
    plt.tick_params(labelbottom='on', labeltop='on', labelleft='on', labelright='on')
    plt.tight_layout()
    
    # Save the result
    f = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/", 'images')
    f = get_subfolder(f, 'species')
    plt.savefig(f'{f}distances_gen_{pop.generation}')
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    os.chdir("../../../")
    
    population = Population(
            name='test',
            folder_name='test',
    )
    main(population)
