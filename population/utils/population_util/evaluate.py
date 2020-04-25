"""
evaluate.py

Evaluate a population throughout its lifetime.
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from main import evaluate, get_folder, get_game_ids
from population.population import Population
from utils.dictionary import *
from utils.myutils import get_subfolder, load_dict


# ------------------------------------------------------> MAIN <------------------------------------------------------ #

def evaluate_generations(name, experiment_id, folder=None, hops: int = 10, unused_cpu: int = 2):
    """
    Evaluate the population across its lifetime. At each generation, the ten best genomes are evaluated together with
    the elite genome of the past five generations.

    :param name: Name of the population
    :param experiment_id: Experiment for which the population is trained (and now will be evaluated)
    :param folder: Population-folder (~experiment level)
    :param hops: Number of generations between each saved population
    :param unused_cpu: Number of CPU cores not used
    """
    # Fetch population and evaluation games
    folder = folder if folder else get_folder(experiment_id)
    pop = Population(
            name=name,
            folder_name=folder,
            log_print=False,
            use_backup=True,
    )
    _, game_ids_eval = get_game_ids(experiment_id=experiment_id)
    
    # Perform the evaluations
    max_gen = pop.generation
    for gen in tqdm(range(0, max_gen + 1, hops)):
        # Load in the current generation
        pop.load(gen=gen)
        
        # Collect the used genomes
        if gen > 5:
            genomes = sorted([g for g in pop.population.values()],
                             key=lambda x: x.fitness if x.fitness else 0,
                             reverse=True)[:10]
            for i in range(1, 6):
                keys = [g.key for g in genomes]
                g = copy.deepcopy(pop.best_genome_hist[gen - i][1])  # Copy since chance of mutation
                while g.key in keys:  # Already added to genomes, update keys
                    g.key += 1
                genomes.append(g)
        else:
            # No history yet, use only the ten most fit genomes from the current generation
            genomes = sorted([g for g in pop.population.values()],
                             key=lambda x: x.fitness if x.fitness else 0,
                             reverse=True)[:15]
        
        # Evaluate the selected genomes
        evaluate(
                population=pop,
                game_config=pop.config,
                games=game_ids_eval,
                genomes=genomes,
                unused_cpu=unused_cpu,
                overwrite=True,
        )


# -------------------------------------------------> VISUALISATIONS <------------------------------------------------- #

def visualize_generations(name, experiment_id, folder=None, hops: int = 10):
    """Visualize the result of the 'evaluate_generations' script. Should only be used for debugging!"""
    # Fetch population and evaluation games
    folder = folder if folder else get_folder(experiment_id)
    pop = Population(
            name=name,
            folder_name=folder,
            log_print=False,
            use_backup=True,
    )
    
    # Parse the results
    fitness_dict = dict()
    finished_dict = dict()
    score_dict = dict()
    distance_dict = dict()
    time_dict = dict()
    max_gen = pop.generation
    for gen in tqdm(range(0, max_gen + 1, hops)):
        results: dict = load_dict(f"population{'_backup' if pop.use_backup else ''}/"
                                  f"storage/"
                                  f"{pop.folder_name}/"
                                  f"{pop}/"
                                  f"evaluation/"
                                  f"{gen:05d}/"
                                  f"results")
        
        # Fitness
        fitness = [results[k][D_FITNESS] for k in results.keys()]
        fitness_dict[gen] = fitness
        
        # Finished
        finished = [results[k][D_FINISHED] / 100 for k in results.keys()]  # Normalize to percentage (0..1)
        finished_dict[gen] = finished
        
        # Score
        score = [results[k][D_SCORE_AVG] for k in results.keys()]
        score_dict[gen] = score
        
        # Distance
        distance = [results[k][D_DISTANCE_AVG] for k in results.keys()]
        distance_dict[gen] = distance
        
        # Time
        time = [results[k][D_TIME_AVG] for k in results.keys()]
        time_dict[gen] = time
    
    # Create visualizations for each of the results
    sf = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/"
                       f"storage/"
                       f"{pop.folder_name}/"
                       f"{pop}/"
                       f"images/", 'evaluation')
    plot_population(fitness_dict,
                    ylabel="Fitness score",
                    title="Fitness (higher is better)",
                    save_path=f'{sf}fitness_group.png')
    plot_elite(fitness_dict,
               f=max,
               ylabel="Fitness score",
               title="Fitness of population's elite (higher is better)",
               save_path=f'{sf}fitness_elite.png')
    plot_population(finished_dict,
                    ylabel="Percentage finished (%)",
                    title="Percentage finished (higher is better)",
                    save_path=f'{sf}finished_group.png')
    plot_elite(finished_dict,
               f=max,
               ylabel="Percentage finished (%)",
               title="Percentage finished by population's elite (higher is better)",
               save_path=f'{sf}finished_elite.png')
    plot_population(score_dict,
                    ylabel="Score",
                    title="Final scores (higher is better)",
                    save_path=f'{sf}score_group.png')
    plot_elite(score_dict,
               f=max,
               ylabel="Score",
               title="Final score of population's elite (higher is better)",
               save_path=f'{sf}score_elite.png')
    plot_population(distance_dict,
                    ylabel="Distance (m)",
                    title="Distance from target (lower is better)",
                    save_path=f'{sf}distance_group.png')
    plot_elite(distance_dict,
               f=min,
               ylabel="Distance (m)",
               title="Distance from target for population's elite (lower is better)",
               save_path=f'{sf}distance_elite.png')
    plot_population(time_dict,
                    ylabel="Time (s)",
                    title="Time needed to reach target (lower is better)",
                    save_path=f'{sf}time_group.png')
    plot_elite(time_dict,
               f=min,
               ylabel="Time (s)",
               title="Time needed to reach target for population's elite (lower is better)",
               save_path=f'{sf}time_elite.png')


def plot_population(d: dict, ylabel: str, title: str, save_path: str):
    """Create a plot of the given dictionary. Each value of d consists of a list of length 3 (min, avg, max)."""
    # Parse the values
    keys = sorted(d.keys())
    data = np.zeros((len(d[keys[0]]), len(keys)))
    for i, k in enumerate(keys):
        data[:, i] = d[k]
    
    # Create the plot
    plt.figure(figsize=(len(keys) / 4, 3))
    plt.boxplot(data, labels=[str(k) for k in keys], whis=[0, 100])
    plt.title(title)
    plt.xticks(rotation=90)
    plt.xlabel("Generations")
    plt.ylabel(ylabel)
    plt.ylim(0, max(np.max(data) * 1.05, 1.01))
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_elite(d: dict, f, ylabel: str, title: str, save_path: str):
    """Create a plot of the given dictionary. Each value of d consists of a list of length 3 (min, avg, max)."""
    # Parse the values
    keys = sorted(d.keys())
    elite_data = [f(d[k]) for k in keys]
    
    # Create the plot
    plt.figure(figsize=(len(keys) / 2, 3))
    plt.plot(keys, elite_data)
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel(ylabel)
    plt.ylim(0, max(elite_data) * 1.05)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    os.chdir("../../..")
    evaluate_generations(
            name="v1",
            experiment_id=1,
            folder="test",
            hops=10,  # TODO
            unused_cpu=2,
    )
