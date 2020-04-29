"""
specie_visualizer.py

Visualization of a specie's elites.
"""
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from population.population import Population
from population.utils.visualizing.averaging_functions import EMA, SMA
from utils.myutils import get_subfolder


def specie_elite_fitness(pop: Population, func, window: int = 5, show: bool = True):
    """
    Visualize the elites of the given population. Each generation, the average fitness of the three stored elites is
    taken.

    :param pop: Population object
    :param func: Function used to flatten the curve
    :param window: Window-size used in the function
    :param show: Show the result
    """
    # Fetch name based on used function
    name = f'elites{"_EMA" if func == EMA else "_SMA" if func == SMA else ""}_gen_{pop.generation}'
    
    ax = plt.figure(figsize=(20, 10)).gca()
    max_gen = 0
    max_fitness = 0
    for specie_id, specie in pop.species_hist.items():
        # Fetch specie-data
        history = sorted(specie.items(), key=lambda x: x[0])
        if len(history) < window: continue
        generations, elite_fitness = zip(*history)
        assert len(elite_fitness) == len(generations)
        max_fitness = max(max_fitness, max(elite_fitness))
        
        # Update max_gen
        if generations[-1] > max_gen: max_gen = generations[-1]
        
        # Plot the specie
        plt.plot(generations, func(elite_fitness, window), label=f'specie {specie_id}')
    
    # Additional plot attributes
    if max_gen == 0: return
    if func == SMA:
        plt.title(f"Specie fitness in population: {pop}\nSimple Moving Average (window={window})")
    elif func == EMA:
        plt.title(f"Specie fitness in population: {pop}\nExponential Moving Average (window={window})")
    else:
        plt.title(f"Specie fitness in population: {pop}")
    plt.xlabel("generation")
    plt.ylabel("fitness of specie's elites")
    box = ax.get_positions()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    if max_fitness <= 1:
        plt.yticks([i / 10 for i in range(11)])  # Fitness expressed in range of 0..1 (hops of 0.1)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save the result
    plt.savefig(f"population{'_backup' if pop.use_backup else ''}/"
                f"storage/"
                f"{pop.folder_name}/"
                f"{pop}/"
                f"images/"
                f"species/"
                f"{name}")
    if show:
        plt.show()
    plt.close()


def specie_representatives(pop: Population, show: bool = True, del_cache: bool = True):
    """Show for each of the current species their representative's architecture."""
    species = pop.species.species
    elite_id = dict()
    for sid, s in sorted(species.items()):
        elite_id[sid] = s.representative.key
        pop.visualize_genome(
                debug=False,  # Keep the networks simple
                genome=s.representative,
                show=False,
        )
    
    hor = min(len(elite_id), 5)
    vert = max(len(elite_id) // 5 + 1, 1)
    plt.figure(figsize=(5 * hor, 5 * vert))
    plt.tight_layout()
    path = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/", "images")
    path_architectures = get_subfolder(path, "architectures")
    for i, (sid, eid) in enumerate(elite_id.items()):
        plt.subplot(vert, hor, i + 1)
        img = mpimg.imread(f'{path_architectures}genome_{eid}.png')
        plt.imshow(img)
        plt.title(f'Specie {sid}')
        plt.axis('off')
    
    # Save the result
    path_species = get_subfolder(path, 'species')
    plt.savefig(f'{path_species}representatives_gen{pop.generation}.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    if del_cache:
        for eid in elite_id.values():
            path = f'{path_architectures}genome_{eid}.png'
            if os.path.exists(path): os.remove(path)


def main(pop: Population, window: int = 5, show: bool = True):
    """Display for each specie its elites' performance, and their architecture (different images)."""
   # TODO: Commented out since not much of valuable insights
    """
    for f in [Forward, SMA, EMA]:
        specie_elite_fitness(
                pop=pop,
                func=f,
                window=window,
                show=show,
        )
    """
    
    specie_representatives(
            pop=pop,
            show=show,
    )


if __name__ == '__main__':
    os.chdir("../../../")
    
    population = Population(
            name='test',
            folder_name='test',
    )
    
    specie_representatives(
            pop=population,
    )
    
    # main(
    #         pop=population,
    # )
