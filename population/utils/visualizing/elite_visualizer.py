"""
elite_visualizer.py

Display a graph of the elites for each generation.
"""
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from population.population import Population
from population.utils.cache.genome_distance import GenomeDistanceCache
from population.utils.visualizing.averaging_functions import EMA, Forward, SMA
from utils.myutils import get_subfolder


def elite_fitness(pop: Population, window: int = 5, show: bool = True):
    """
    Visualize the elites of the given population. Each generation, the average fitness of the three stored elites is
    taken.

    :param pop: Population object
    :param window: Window-size used in the function
    :param show: Show the result
    """
    f = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/", 'images')
    f = get_subfolder(f, 'elites')
    for func in [Forward, SMA, EMA]:
        # Fetch name based on used function
        name = f'{"EMA_" if func == EMA else "SMA_" if func == SMA else ""}gen_{pop.generation}'
        
        # Load in the relevant data
        history = sorted(pop.best_fitness.items(), key=lambda x: x[0])
        generations, fitness = zip(*history)
        
        # Create the figure
        ax = plt.figure().gca()
        plt.plot(generations, func(fitness, window))
        if func == SMA:
            plt.title(f"Elite fitness in population: {pop}\nSimple Moving Average (window={window})")
        elif func == EMA:
            plt.title(f"Elite fitness in population: {pop}\nExponential Moving Average (window={window})")
        else:
            plt.title(f"Elite fitness in population: {pop}")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
        if max(fitness) <= 1:
            plt.yticks([i / 10 for i in range(11)])  # Fitness expressed in range of 0..1 (hops of 0.1)
        plt.grid(axis='y')
        plt.tight_layout()
        
        # Save the result
        plt.savefig(f'{f}{name}')
        if show:
            plt.show()
        plt.close()


def elite_architecture(pop: Population, show: bool = False, del_cache: bool = True):
    """
    Visualize each architectural change in the population's elites.
    
    :param pop: Population object
    :param show: Show the result
    :param del_cache: Remove intermediate architecture-results when images are created
    """
    # Initialize the architecture-list with the first genome
    distance = GenomeDistanceCache(config=pop.config.genome)
    new_architectures = [(0, pop.best_genome_hist[0][1])]  # Only interested in genome itself
    for gen in range(1, pop.generation):
        gen_genome = pop.best_genome_hist[gen][1]
        if distance(gen_genome, new_architectures[-1][1]) > 2:  # Take only the more significant changes into account
            new_architectures.append((gen, gen_genome))
    new_architectures.append((pop.generation - 1, pop.best_genome_hist[pop.generation - 1][1]))
    
    # Create the architectures of the unique genomes
    for _, g in new_architectures:
        pop.visualize_genome(
                debug=False,  # Keep the networks simple
                genome=g,
                show=False,
        )
    
    # Combine in one figure
    hor = min(len(new_architectures), 5)
    vert = max((len(new_architectures) - 1) // 5 + 1, 1)
    plt.figure(figsize=(5 * hor, 5 * vert))
    plt.tight_layout()
    f_images = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/",
                             "images")
    f_architectures = get_subfolder(f_images, "architectures")
    for i, (gen, g) in enumerate(new_architectures):
        plt.subplot(vert, hor, i + 1)
        img = mpimg.imread(f'{f_architectures}genome_{g.key}.png')
        plt.imshow(img)
        plt.title(f'Generation {gen}')
        plt.axis('off')
    
    # Save the result
    f_elites = get_subfolder(f_images, "elites")
    plt.savefig(f'{f_elites}/architecture_timeline.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    if del_cache:
        for (_, g) in new_architectures:
            path = f'{f_architectures}genome_{g.key}.png'
            if os.path.exists(path): os.remove(path)


def main(pop: Population, window: int = 5, show: bool = False):
    """Create visualizations for the elites of each generation."""
    elite_fitness(
            pop=pop,
            window=window,
            show=show,
    )
    
    elite_architecture(
            pop=pop,
            show=show,
    )


if __name__ == '__main__':
    os.chdir("../../../")
    
    population = Population(
            name='test',
            folder_name='test',
    )
    
    main(
            pop=population,
            show=True,
    )
