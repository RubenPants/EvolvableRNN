"""
evaluate_populations.py

Evaluate all the populations across their generations and compare each of them against each other.

Note: Evaluation is only done on backed-up populations.
"""
import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from main import get_folder
from population.population import Population
from population.utils.population_util.evaluate import evaluate_generations as eval_gen
from utils.dictionary import *
from utils.myutils import get_subfolder, load_dict, update_dict

HOPS = 10


def evaluate_generations(experiment_id: int, pop_folder: str, folder: str = None, max_v: int = 50, unused_cpu: int = 2):
    """Evaluate all the populations' generations in a given folder of a given experiment."""
    if pop_folder[-1] != '/': pop_folder += '/'
    for v in range(1, max_v + 1):
        print(f"\n===> EVALUATING POPULATION {pop_folder}v{v} <===")
        eval_gen(
                name=f"{pop_folder}v{v}",
                experiment_id=experiment_id,
                folder=folder,
                hops=HOPS,
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


def plot_result(d: dict, ylabel: str, title: str, save_path: str):
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
    plt.xlabel("generations")
    plt.ylabel(ylabel)
    plt.ylim(0, max(np.max(data) * 1.05, 1.01))
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def correctness_check(folder: str = 'experiment1',
                      neat: bool = True,
                      neat_gru: bool = True,
                      neat_lstm: bool = True,
                      neat_sru: bool = True,
                      neat_sru_s: bool = True,
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
    pbar = tqdm(range(int(len(populations) * max_v * max_gen / gen_hops)), desc="Evaluating correctness")
    for pop in populations:
        for v in range(1, max_v + 1):
            for gen in range(0, max_gen + 1, gen_hops):
                f = glob(f"{path}{pop}/v{v}/generations/gen_{gen:05d}")
                # Load in the current generation
                if len(f) == 0:
                    raise Exception(f"Population {pop}/v{v} is not trained for generation {gen}")
                pbar.update()
    pbar.close()


# TODO: Usage of backed-up populations is assumed
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--evaluate_gen', type=int, default=1)
    parser.add_argument('--evaluate_pop', type=int, default=1)
    parser.add_argument('--test_correctness', type=bool, default=0)
    parser.add_argument('--experiment', type=int, default=2)
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--folder_pop', type=str, default='NEAT-LSTM')
    parser.add_argument('--max_v', type=int, default=1)
    parser.add_argument('--unused_cpu', type=int, default=2)
    args = parser.parse_args()
    
    # Set parameters
    f = args.folder if args.folder else get_folder(args.experiment)
    
    # Execute the program
    if bool(args.test_correctness):
        correctness_check(folder='experiment1')  # Use the default parameters
        correctness_check(folder='experiment2')  # Use the default parameters
    
    if bool(args.evaluate_gen):
        evaluate_generations(
                experiment_id=args.experiment,
                folder=f,
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
