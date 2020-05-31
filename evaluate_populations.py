"""
evaluate_populations.py

Evaluate all the populations across their generations and compare each of them against each other.

Note: Evaluation is only done on backed-up populations.
"""
import argparse
from collections import Counter
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from main import get_folder
from population.population import Population
from population.utils.population_util.evaluate import evaluate_generations as eval_gen
from utils.dictionary import *
from utils.myutils import get_subfolder, load_dict, update_dict

HOPS = 10


def evaluate_generations(experiment_id: int,
                         pop_folder: str,
                         folder: str = None,
                         hops: int = HOPS,
                         max_v: int = 50,
                         unused_cpu: int = 2):
    """Evaluate all the populations' generations in a given folder of a given experiment."""
    if pop_folder[-1] != '/': pop_folder += '/'
    # for v in range(31, max_v + 1):
    for v in [51,52]:
        print(f"\n===> EVALUATING POPULATION {pop_folder}v{v} <===")
        eval_gen(
                name=f"{pop_folder}v{v}",
                experiment_id=experiment_id,
                folder=folder,
                hops=hops,
                unused_cpu=unused_cpu,
        )


def evaluate_populations(folder: str, pop_folder: str, max_v: int = 50):
    """
    Evaluate the various populations against each other. Note that it is assumed that 'evaluate_generations' has ran
    first.
    """
    if folder[-1] != '/': folder += '/'
    if pop_folder[-1] != '/': pop_folder += '/'
    
    # Load in dummy population
    print(f"\n===> COMBINING POPULATION RESULTS OF FOLDER {folder}{pop_folder} <===")
    pop = Population(
            name=f'{pop_folder}v1',
            folder_name=folder,
            log_print=False,
            use_backup=True,
    )
    max_gen = pop.generation
    
    # Parse the results
    fitness_dict = dict()
    finished_dict = dict()
    score_dict = dict()
    distance_dict = dict()
    time_dict = dict()
    for g in range(0, max_gen + 1, HOPS):
        fitness_dict[g] = []
        finished_dict[g] = []
        score_dict[g] = []
        distance_dict[g] = []
        time_dict[g] = []
        for v in range(1, max_v + 1):
            results: dict = load_dict(f"population_backup/storage/{folder}{pop_folder}v{v}/evaluation/{g:05d}/results")
            fitness_dict[g].append(max([results[k][D_FITNESS] for k in results.keys()]))
            finished_dict[g].append(max([results[k][D_FINISHED] / 100 for k in results.keys()]))
            score_dict[g].append(max([results[k][D_SCORE_AVG] for k in results.keys()]))
            distance_dict[g].append(min([results[k][D_DISTANCE_AVG] for k in results.keys()]))
            time_dict[g].append(min([results[k][D_TIME_AVG] for k in results.keys()]))
    
    # Save received data in evaluation subfolder of the population folder
    path = get_subfolder(f'population_backup/storage/{folder}{pop_folder}', 'evaluation')
    update_dict(f'{path}fitness', fitness_dict, overwrite=True)
    update_dict(f'{path}finished', finished_dict, overwrite=True)
    update_dict(f'{path}score', score_dict, overwrite=True)
    update_dict(f'{path}distance', distance_dict, overwrite=True)
    update_dict(f'{path}time', time_dict, overwrite=True)
    
    # Visualize the data
    path_images = get_subfolder(path, 'images')
    plot_result(d=fitness_dict,
                ylabel="fitness",
                title="Average fitness",
                save_path=f'{path_images}fitness.png')
    plot_result(d=finished_dict,
                ylabel="finished ratio",
                title="Averaged finished ratio",
                save_path=f'{path_images}finished.png')
    plot_result(d=score_dict,
                ylabel="score",
                title="Average score",
                save_path=f'{path_images}score.png')
    plot_result(d=distance_dict,
                ylabel="distance (m)",
                title="Average final distance to target",
                save_path=f'{path_images}distance.png')
    plot_result(d=time_dict,
                ylabel="time (s)",
                title="Average simulation time",
                save_path=f'{path_images}time.png')


def combine_all_populations(folder: str,
                            max_v: int = None,
                            neat: bool = False,
                            neat_gru: bool = False,
                            neat_lstm: bool = False,
                            neat_sru: bool = False,
                            neat_sru_s: bool = False,
                            ):
    """Combine the scores for all of the populations in a given folder."""
    # Collect all the populations
    populations = []
    if neat: populations.append(D_NEAT)
    if neat_gru: populations.append(D_NEAT_GRU)
    if neat_lstm: populations.append(D_NEAT_LSTM)
    if neat_sru: populations.append(D_NEAT_SRU)
    if neat_sru_s: populations.append(D_NEAT_SRU_S)
    if len(populations) == 0: return
    
    # Collect all the measure options
    # OPTIONS = ['distance', 'finished', 'fitness', 'score', 'time', 'training']
    OPTIONS = ['fitness']
    
    # Go over all possibilities
    print(f"\n===> COMBINING POPULATIONS OF FOLDER {folder} <===")
    path = f"population_backup/storage/{folder}/"
    path_images = get_subfolder(path, 'images')
    for option in OPTIONS:
        plt.figure(figsize=(8, 2.5))
        max_data = 0
        max_gen = 0
        for pop in populations:
            # Load the dictionary
            d = load_dict(f"{path}{pop}/evaluation/{option}")
            size = len(list(d.values())[0])
            if max_v: assert size == max_v
            
            # Prepare the data containers
            q1 = []
            q2 = []  # Median
            q3 = []
            idx_q1 = int(round(1 / 4 * size))
            idx_q2 = int(round(2 / 4 * size))
            idx_q3 = int(round(3 / 4 * size))
            
            # Loop over each iteration
            x = sorted([int(k) for k in d.keys()])
            for g in x:
                if g > max_gen: max_gen = g
                lst = sorted(d[str(g)])  # Sort values from low to high
                q1.append(lst[idx_q1])
                q2.append(lst[idx_q2])
                q3.append(lst[idx_q3])
            
            # Plot the results
            plt.plot(x, q1, color=COLORS[pop], linestyle=":", linewidth=.5)
            plt.plot(x, q3, color=COLORS[pop], linestyle=":", linewidth=.5)
            plt.plot(x, q2, color=COLORS[pop], linestyle="-", linewidth=2, label=pop)
            plt.fill_between(x, q1, q3, color=COLORS[pop], alpha=0.2)
            
            # Update the max-counter
            if max(q3) > max_data: max_data = max(q3)
        
        # Finalize the figure
        leg = plt.legend(loc='upper center',
                         bbox_to_anchor=(0.5, 1.25),
                         fancybox=True,
                         fontsize=10,
                         ncol=len(populations))
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        # plt.xticks([i * 100 for i in range(11)])  # TODO
        plt.xlabel("generation")
        plt.xlim(0, max_gen)
        # plt.yticks([i for i in range(7)])  # TODO
        plt.ylabel(option)
        plt.ylim(0, max(max_data * 1.05, 1.05))
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{path_images}comb_{option}.png", bbox_inches='tight', pad_inches=0.02)
        # plt.savefig(f"{path_images}comb_{option}.eps", format="eps", bbox_inches='tight', pad_inches=0.02)
        # plt.show()
        plt.close()


def evaluate_training(experiment_id: int, pop_folder: str, folder: str = None, max_v: int = 50):
    """Evaluate the fitness of a population's elite each training generation."""
    if pop_folder[-1] != '/': pop_folder += '/'
    folder = folder if folder else get_folder(experiment_id)
    if folder[-1] != '/': folder += '/'
    
    # Get dummy population
    pop = Population(
            name=f"{pop_folder}v1",
            folder_name=folder,
            log_print=False,
            use_backup=True,
    )
    max_gen = pop.generation
    
    # Initialize data container
    training_fitness = dict()
    for g in range(0, max_gen + 1, HOPS):
        training_fitness[g] = []
    
    # Pull the training scores
    print(f"\n===> PULLING TRAINING FITNESS OF THE {pop_folder} POPULATIONS <===")
    pbar = tqdm(range(int(max_v * (max_gen / HOPS + 1))))
    for v in range(1, max_v + 1):
        name = f"{pop_folder}v{v}"
        pop = Population(
                name=name,
                folder_name=folder,
                log_print=False,
                use_backup=True,
        )
        
        # Perform the evaluations
        max_gen = pop.generation
        for gen in range(0, max_gen + 1, HOPS):
            if not pop.load(gen=gen):
                raise Exception(f"Population {name} is not trained for generation {gen}")
            training_fitness[gen].append(pop.best_genome.fitness if pop.best_genome else 0)
            pbar.update()
    pbar.close()
    
    # Plot the result
    path = get_subfolder(f'population_backup/storage/{folder}{pop_folder}', 'evaluation')
    update_dict(f'{path}training', training_fitness, overwrite=True)
    path_images = get_subfolder(path, 'images')
    plot_result(d=training_fitness,
                ylabel="fitness",
                title="Average training fitness",
                save_path=f'{path_images}training.png')


def plot_result(d: dict, ylabel: str, title: str, save_path: str):
    """Create a plot of the given dictionary. Each value of d consists of a list of length 3 (min, avg, max)."""
    # Parse the values
    keys = sorted(d.keys())
    data = np.zeros((len(d[keys[0]]), len(keys)))
    for i, k in enumerate(keys):
        data[:, i] = d[k]
    
    # Create the plot
    plt.figure(figsize=(12, 2.5))
    plt.boxplot(data, labels=[str(k) if k % 20 == 0 else '' for k in keys], whis=[0, 100])
    plt.xticks(rotation=90)
    plt.xlabel("generations")
    plt.ylabel(ylabel)
    plt.ylim(0, max(np.max(data) * 1.05, 1.05))
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    plt.close()


def plot_distribution(folder: str,
                      neat: bool = False,
                      neat_gru: bool = False,
                      neat_lstm: bool = False,
                      neat_sru: bool = False,
                      neat_sru_s: bool = False,
                      gen: int = 500,
                      ):
    """
    Plot the one-dimensional distribution of all of the populations on each of the evaluation measures for the requested
     generation. It is assumed that the evaluation-data has already been collected.
    """
    # Collect all the populations
    populations = []
    if neat: populations.append(D_NEAT)
    if neat_gru: populations.append(D_NEAT_GRU)
    if neat_lstm: populations.append(D_NEAT_LSTM)
    if neat_sru: populations.append(D_NEAT_SRU)
    if neat_sru_s: populations.append(D_NEAT_SRU_S)
    if len(populations) == 0: return
    
    # Collect all the measure options
    OPTIONS = ['distance', 'finished', 'fitness', 'score', 'time', 'training']
    
    # Go over all possibilities
    print(f"\n===> CREATING POPULATION DISTRIBUTIONS <===")
    path = f"population_backup/storage/{folder}/"
    path_images = get_subfolder(path, 'images')
    for option in OPTIONS:
        plt.figure(figsize=(10, 2.5))
        min_val = float("inf")
        max_val = -float("inf")
        for pop in populations:
            d = load_dict(f"{path}{pop}/evaluation/{option}")
            dist = d[str(gen)]
            if min(dist) < min_val: min_val = min(dist)
            if max(dist) > max_val: max_val = max(dist)
            
            # Remove outliers first
            dist = sorted(dist)
            q1 = min(dist[int(round(1 / 4 * len(dist)))], dist[int(round(3 / 4 * len(dist)))])
            q3 = max(dist[int(round(1 / 4 * len(dist)))], dist[int(round(3 / 4 * len(dist)))])
            iqr = q3 - q1
            
            for i in range(len(dist) - 1, -1, -1):
                if (dist[i] < (q1 - 1.5 * iqr)) or (dist[i] > (q3 + 1.5 * iqr)): del dist[i]
            sns.distplot(dist,
                         hist=False,
                         kde=True,
                         norm_hist=True,
                         bins=100,
                         color=COLORS[pop],
                         kde_kws={'linewidth': 2},
                         label=pop,
                         )
        plt.xlim(min_val, max_val)
        # plt.title(f"Probability density across populations for '{option}' at generation {gen}")
        plt.xlabel(option)
        # plt.yticks([])
        plt.ylabel('probability density')
        leg = plt.legend(loc='upper center',
                         bbox_to_anchor=(0.5, 1.2),
                         fancybox=True,
                         fontsize=8,
                         ncol=len(populations))
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        plt.tight_layout()
        plt.savefig(f"{path_images}dist_{option}.png", bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f"{path_images}dist_{option}.eps", format='eps', bbox_inches='tight', pad_inches=0.02)
        # plt.show()
        plt.close()


def compute_complexity(folder: str,
                       neat: bool = False,
                       neat_gru: bool = False,
                       neat_lstm: bool = False,
                       neat_sru: bool = False,
                       neat_sru_s: bool = False,
                       gen: int = 500,
                       max_v: int = 50,
                       ):
    """Compute the complexity of the populations' elites."""
    # Collect all the populations
    populations = []
    if neat: populations.append(D_NEAT)
    if neat_gru: populations.append(D_NEAT_GRU)
    if neat_lstm: populations.append(D_NEAT_LSTM)
    if neat_sru: populations.append(D_NEAT_SRU)
    if neat_sru_s: populations.append(D_NEAT_SRU_S)
    if len(populations) == 0: return
    
    # Go over all possibilities
    print(f"\n===> COMPUTING POPULATION'S ELITE COMPLEXITY <===")
    path = f"population_backup/storage/{folder}/"
    genes_dict = dict()
    for pop in populations:
        path_eval = get_subfolder(f"{path}{pop}/", 'evaluation')
        complexity = Counter()
        genes = Counter()
        genes_detailed = dict()
        for v in range(1, max_v + 1):
            population = Population(
                    name=f'{pop}/v{v}',
                    folder_name=folder,
                    use_backup=True,
            )
            if population.generation == 0: raise Exception(f"Population {pop}/v{v} loaded incorrectly")
            if population.generation != gen: population.load(gen=gen)
            s = population.best_genome.size()
            complexity[str(s)] += 1
            c = str(s[0] + s[1])
            genes[c] += 1
            if c in genes_detailed:
                genes_detailed[c].append(v)
            else:
                genes_detailed[c] = [v]
        
        # Store results at populations themselves
        update_dict(f'{path_eval}complexity_topology', complexity, overwrite=True)
        update_dict(f'{path_eval}complexity_genes', genes, overwrite=True)
        update_dict(f'{path_eval}complexity_genes_detailed', genes_detailed, overwrite=True)
        
        # Update global dictionary
        keys = list(genes.keys())
        for k in keys:
            genes[int(k)] = genes[k]
            del genes[k]
        genes_dict[pop] = list(sorted(genes.items()))
    
    plt.figure(figsize=(10, 2.5))
    max_x = max([max([a for a, _ in genes_dict[pop]]) for pop in populations])
    min_x = min([min([a for a, _ in genes_dict[pop]]) for pop in populations])
    for idx, pop in enumerate(populations):
        keys = [a for a, _ in genes_dict[pop]]
        for x in range(max_x):
            if x not in keys: genes_dict[pop].append((x, 0))
        x, y = zip(*genes_dict[pop])
        width = 0.8 / len(populations)
        plt.bar(x=np.asarray(x) - 0.4 + width / 2 + idx * width,
                height=y,
                width=width,
                linewidth=2,
                label=pop,
                color=COLORS[pop])
    
    # Beautify the plot
    plt.xlim(min_x - .5, max_x + .5)
    plt.xticks([i for i in range(min_x, max_x + 1)])
    leg = plt.legend(loc='upper center',
                     bbox_to_anchor=(0.5, 1.18),
                     fancybox=True,
                     fontsize=10,
                     ncol=len(populations))
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.xlabel("complexity expressed in #genes")
    plt.ylabel("#elites")
    plt.savefig(f"population_backup/storage/{folder}/images/complexity.png", bbox_inches='tight', pad_inches=0.02)
    plt.savefig(f"population_backup/storage/{folder}/images/complexity.eps",
                format='eps',
                bbox_inches='tight',
                pad_inches=0.02)
    # plt.show()
    plt.close()
    
    # Also create a violin plot of the distribution if only two populations
    if len(populations) == 2:
        max_x = 0
        min_x = float('inf')
        df = pd.DataFrame()
        palette = []
        for idx, pop in enumerate(populations):
            values = []
            for a, b in genes_dict[pop]:
                for _ in range(b):
                    values.append(a)
            
            # Remove outliers
            values = sorted(values)
            q1 = min(values[int(round(1 / 4 * len(values)))], values[int(round(3 / 4 * len(values)))])
            q3 = max(values[int(round(1 / 4 * len(values)))], values[int(round(3 / 4 * len(values)))])
            iqr = q3 - q1
            
            for i in range(len(values) - 1, -1, -1):
                if (values[i] < (q1 - 1.5 * iqr)) or (values[i] > (q3 + 1.5 * iqr)): del values[i]
            if min(values) < min_x: min_x = min(values)
            if max(values) > max_x: max_x = max(values)
            df = df.append(pd.DataFrame({'complexity': values, 'y': 'ignore', 'pop': pop}))
            palette.append(COLORS[pop])
        
        # Create the plot
        plt.figure(figsize=(10, 2.5))
        sns.violinplot(data=df,
                       x="complexity", y="y", hue="pop",
                       palette=palette, split=True,
                       inner="quartile")
        plt.xlim(min_x - .5, max_x + .5)
        plt.xticks([i for i in range(min_x, max_x + 1)])
        plt.xlabel("complexity expressed in #genes")
        plt.yticks([])
        plt.ylabel('elite genome density')
        leg = plt.legend(loc='upper center',
                         bbox_to_anchor=(0.5, 1.25),
                         fancybox=True,
                         fontsize=10,
                         ncol=len(populations))
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        plt.tight_layout()
        plt.savefig(f"population_backup/storage/{folder}/images/complexity_violin.png",
                    bbox_inches='tight',
                    pad_inches=0.02)
        plt.savefig(f"population_backup/storage/{folder}/images/complexity_violin.eps",
                    format='eps',
                    bbox_inches='tight',
                    pad_inches=0.02)
        plt.show()
        plt.close()


def correctness_check(folder: str,
                      neat: bool = False,
                      neat_gru: bool = False,
                      neat_lstm: bool = False,
                      neat_sru: bool = False,
                      neat_sru_s: bool = False,
                      max_v: int = 50,
                      max_gen: int = 500,
                      gen_hops: int = 10,
                      ):
    """Test if all the files are present."""
    # Collect all the populations
    populations = []
    if neat: populations.append('NEAT')
    if neat_gru: populations.append('NEAT-GRU')
    if neat_lstm: populations.append('NEAT-LSTM')
    if neat_sru: populations.append('NEAT-SRU')
    if neat_sru_s: populations.append('NEAT-SRU-S')
    
    # Go over all possibilities
    path = f"population_backup/storage/{folder}/"
    pbar = tqdm(range(int(len(populations) * max_v * (max_gen / gen_hops + 1))), desc="Evaluating correctness")
    for pop in populations:
        for v in range(1, max_v + 1):
            for gen in range(0, max_gen + 1, gen_hops):
                files = glob(f"{path}{pop}/v{v}/generations/gen_{gen:05d}")
                # Load in the current generation
                if len(files) == 0:
                    raise Exception(f"Population {pop}/v{v} is not trained for generation {gen}")
                pbar.update()
    pbar.close()


# TODO: Usage of backed-up populations is assumed
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--evaluate_gen', type=int, default=1)
    parser.add_argument('--evaluate_pop', type=int, default=0)
    parser.add_argument('--combine_pop', type=int, default=0)  # Goes over all the populations types
    parser.add_argument('--evaluate_training', type=int, default=0)
    parser.add_argument('--plot_distribution', type=int, default=0)  # Goes over all the populations types
    parser.add_argument('--compute_topology', type=int, default=0)  # Goes over all the populations types
    parser.add_argument('--test_correctness', type=int, default=0)
    parser.add_argument('--experiment', type=int, default=8)
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--folder_pop', type=str, default='default')
    parser.add_argument('--hops', type=int, default=HOPS)
    parser.add_argument('--max_gen', type=int, default=100)
    parser.add_argument('--max_v', type=int, default=30)
    parser.add_argument('--unused_cpu', type=int, default=2)
    args = parser.parse_args()
    
    # Set parameters
    f = args.folder if args.folder else get_folder(args.experiment)
    
    # Execute the program
    if bool(args.test_correctness):
        correctness_check(folder=f,
                          neat=True,
                          neat_gru=True,
                          neat_lstm=False,
                          neat_sru=True,
                          neat_sru_s=False,
                          max_gen=args.max_gen,
                          max_v=args.max_v,
                          )  # Use the default parameters
    
    if bool(args.evaluate_gen):
        evaluate_generations(
                experiment_id=args.experiment,
                folder=f,
                hops=args.hops,
                pop_folder=args.folder_pop,
                max_v=args.max_v,
                unused_cpu=args.unused_cpu,
        )
    
    if bool(args.evaluate_pop):
        evaluate_populations(
                folder=f,
                pop_folder=args.folder_pop,
                max_v=args.max_v,
        )
    
    if bool(args.combine_pop):
        combine_all_populations(
                folder=f,
                max_v=args.max_v,
                neat=False,
                neat_gru=True,
                neat_lstm=False,
                neat_sru=True,
                neat_sru_s=False,
        )
    
    if bool(args.evaluate_training):
        evaluate_training(
                experiment_id=args.experiment,
                folder=f,
                pop_folder=args.folder_pop,
                max_v=args.max_v,
        )
    
    if bool(args.plot_distribution):
        plot_distribution(folder=f,
                          neat=False,
                          neat_gru=True,
                          neat_lstm=False,
                          neat_sru=True,
                          neat_sru_s=False,
                          )
    
    if bool(args.compute_topology):
        compute_complexity(folder=f,
                           neat=False,
                           neat_gru=True,
                           neat_lstm=False,
                           neat_sru=True,
                           neat_sru_s=False,
                           gen=args.max_gen,
                           max_v=args.max_v,
                           )
