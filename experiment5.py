"""
experiment5.py

Builds further on the gru_analysis.py file found in population/utils/visualization.
In here, it is checked if a stable NEAT-GRU genome can be developed by incrementally updating its hidden-state weights
via the gradient.

Each iteration, the four differences are considered:
    * Change in bias-value
    * Change in reset-gate weights
    * Change in update-gate weights
    * Change in hidden-state weights
The genome with the most improvement (with mutation in only one of the above categories) is "parent" of next generation.
"""
import copy
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from tqdm import tqdm

from environment.env_multi import get_multi_env
from main import get_game_ids
from population.population import Population
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.genome import Genome
from population.utils.population_util.fitness_functions import calc_pop_fitness
from population.utils.visualizing.genome_visualizer import draw_net
from population.utils.visualizing.monitor_genome_single_gru import main as monitor
from utils.myutils import get_subfolder


def main(topology_id: int,
         iterations: int,
         eval_interval: int,
         experiment_id: int,
         hops: float = 0.1,
         weight_range: float = 1,
         mutate_combine: bool = False,
         mutate_bias: bool = True,
         mutate_reset: bool = True,
         mutate_update: bool = True,
         mutate_candidate: bool = True,
         init_population_size: int = 1000,
         unused_cpu: int = 2):
    """
    Run the fifth experiment.
    
    :param topology_id: Chosen topology to investigate
    :param iterations: Number of training iterations performed
    :param eval_interval: After how much training iterations evaluation is performed
    :param experiment_id: ID of the experiment used to train and evaluate the population on
    :param hops: Hops between variable-configurations
    :param weight_range: Range of deviation for one's weights
    :param mutate_combine: Combine the best results of each mutation type
    :param mutate_bias: Mutate the GRU's bias values
    :param mutate_reset: Mutate the reset-gate's weights
    :param mutate_update: Mutate the update-gate's weights
    :param mutate_candidate: Mutate the candidate-state's weights
    :param init_population_size: Initial size of the randomized population
    :param unused_cpu: Number of CPU-cores not used
    """
    # Get the population
    name = f"experiment{experiment_id}_topology{topology_id}_hops{hops}_range{weight_range}"
    pop = Population(
            name=name,
            folder_name='experiment5',
            use_backup=False,
    )
    
    # Define the games specific to the experiment
    _, game_ids_eval = get_game_ids(experiment_id=experiment_id)
    
    # Set the genomes if population is new
    if pop.generation == 0:
        # Get the requested topology-type
        if topology_id == 1:
            topology = get_topology1
        elif topology_id == 2:
            topology = get_topology2
        else:
            raise Exception(f"Topology {topology_id} not supported.")
        
        # Initialize the population with a randomized population
        pop.population = dict()
        for gid in range(init_population_size):
            new_genome = topology(pop.config, random_init=True)
            new_genome.key = gid
            pop.population[gid] = new_genome
        
        # Perform an initial training
        train_population(pop=pop,
                         games=game_ids_eval,
                         unused_cpu=unused_cpu)
        pop.generation = 0  # Don't count initial training as a generation
        
        # Get the fittest genome
        best = None
        for g in pop.population.values():
            if best is None or g.fitness > best.fitness: best = g
        pop.best_genome = best
        visualize_best_genome(pop=pop)
        
        # Test the initial population
        test_population(pop=pop,
                        games=game_ids_eval)
        
        # Set the most fit genome as the starting-point
        set_population(pop=pop,
                       hops=hops,
                       weight_range=weight_range,
                       mutate_bias=mutate_bias,
                       mutate_reset=mutate_reset,
                       mutate_update=mutate_update,
                       mutate_candidate=mutate_candidate)
    
    # Evaluate the population
    for i in range(iterations):
        train_population(pop=pop,
                         games=game_ids_eval,
                         unused_cpu=unused_cpu)
        evaluate_fitness(pop=pop,
                         hops=hops,
                         weight_range=weight_range,
                         mutate_combine=mutate_combine,
                         mutate_bias=mutate_bias,
                         mutate_reset=mutate_reset,
                         mutate_update=mutate_update,
                         mutate_candidate=mutate_candidate)
        visualize_best_genome(pop=pop)
        if pop.generation % eval_interval == 0:
            test_population(pop=pop,
                            games=game_ids_eval)
        set_population(pop=pop,
                       hops=hops,
                       weight_range=weight_range,
                       mutate_bias=mutate_bias,
                       mutate_reset=mutate_reset,
                       mutate_update=mutate_update,
                       mutate_candidate=mutate_candidate)


def set_population(pop: Population,
                   hops: float,
                   weight_range: float,
                   mutate_bias: bool = False,
                   mutate_reset: bool = False,
                   mutate_update: bool = False,
                   mutate_candidate: bool = False,
                   ):
    """Set the given population as mutations of the given genome."""
    assert weight_range > hops
    r = int(weight_range / hops)
    
    # Re-initialize the weight around the provided genome
    pop.population = dict()
    genome_key = 0
    
    # Create genome-mutations
    if mutate_bias:
        for i in range(3):
            for a in range(-r, r + 1):
                new_genome = copy.deepcopy(pop.best_genome)
                new_genome.nodes[2].bias_h[i] = new_genome.nodes[2].bias_h[i] + a * hops
                new_genome.key = genome_key
                pop.population[genome_key] = new_genome
                genome_key += 1
    
    if mutate_reset:
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                new_genome = copy.deepcopy(pop.best_genome)
                new_genome.nodes[2].weight_xh_full[0, 0] = new_genome.nodes[2].weight_xh_full[0, 0] + a * hops
                new_genome.nodes[2].weight_hh[0, 0] = new_genome.nodes[2].weight_hh[0, 0] + b * hops
                new_genome.key = genome_key
                pop.population[genome_key] = new_genome
                genome_key += 1
    
    if mutate_update:
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                new_genome = copy.deepcopy(pop.best_genome)
                new_genome.nodes[2].weight_xh_full[1, 0] = new_genome.nodes[2].weight_xh_full[1, 0] + a * hops
                new_genome.nodes[2].weight_hh[1, 0] = new_genome.nodes[2].weight_hh[1, 0] + b * hops
                new_genome.key = genome_key
                pop.population[genome_key] = new_genome
                genome_key += 1
    
    if mutate_candidate:
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                new_genome = copy.deepcopy(pop.best_genome)
                new_genome.nodes[2].weight_xh_full[2, 0] = new_genome.nodes[2].weight_xh_full[2, 0] + a * hops
                new_genome.nodes[2].weight_hh[2, 0] = new_genome.nodes[2].weight_hh[2, 0] + b * hops
                new_genome.key = genome_key
                pop.population[genome_key] = new_genome
                genome_key += 1
    
    # Save the updated population
    pop.save()


def visualize_best_genome(pop: Population):
    """Visualize the population's fittest genome."""
    pop.log(f"Most fit genome: {pop.best_genome.key} with fitness: {round(pop.best_genome.fitness, 3)}")
    name = f"genome_{pop.best_genome.key}"
    sf = get_subfolder(f'population/storage/{pop.folder_name}/{pop}/', 'images')
    sf = get_subfolder(sf, f'gen{pop.generation:05d}')
    draw_net(config=pop.config.genome,
             genome=pop.best_genome,
             debug=True,
             filename=f'{sf}{name}',
             view=False)


def train_population(pop: Population, games: list, unused_cpu: int = 2):
    """Evaluate the given population on a training set."""
    multi_env = get_multi_env(pop=pop, game_config=pop.config)
    multi_env.set_games(games, noise=False)
    pool = mp.Pool(mp.cpu_count() - unused_cpu)
    manager = mp.Manager()
    return_dict = manager.dict()
    pbar = tqdm(total=len(pop.population), desc="Evaluating")
    
    def update(*_):
        pbar.update()
    
    for genome_id, genome in pop.population.items():
        pool.apply_async(func=multi_env.eval_genome, args=((genome_id, genome), return_dict), callback=update)
    pool.close()  # Close the pool
    pool.join()  # Postpone continuation until everything is finished
    pbar.close()
    
    # Calculate the fitness from the given return_dict
    fitness = calc_pop_fitness(
            fitness_cfg=pop.config.evaluation,
            game_cfg=pop.config.game,
            game_obs=return_dict,
            gen=pop.generation,
    )
    for i, genome in pop.population.items():
        genome.fitness = fitness[i]
    
    # Save the results
    pop.generation += 1
    pop.save()


def test_population(pop: Population, games: list):
    """Visualize the population's performance on the test-set."""
    monitor(
            debug=False,
            game_cfg=pop.config,
            game_id=games[0],
            genome=pop.best_genome,
            population=pop,
    )
    monitor(
            debug=False,
            game_cfg=pop.config,
            game_id=games[int(len(games) / 2)],  # Opposite side
            genome=pop.best_genome,
            population=pop,
    )
    # TODO: Move created image to correct generation (+remove monitor directory?)


def evaluate_fitness(pop: Population,
                     hops: float,
                     weight_range: float,
                     mutate_combine: bool = False,
                     mutate_bias: bool = True,
                     mutate_reset: bool = True,
                     mutate_update: bool = True,
                     mutate_candidate: bool = True):
    """Visualize the fitness-values of the population."""
    # Initialization
    r = int(weight_range / hops)
    dim = 2 * r + 1
    genome_key = 0
    
    # Enroll the previous best genome
    init_gru = pop.best_genome.nodes[2]
    best_bias_genome = None
    best_reset_genome = None
    best_update_genome = None
    best_candidate_genome = None
    
    # Create genome-mutations
    if mutate_bias:
        bias_result = np.zeros((3, dim))
        for i in range(3):
            for a in range(dim):
                g = pop.population[genome_key]
                bias_result[i, a] = g.fitness
                if best_bias_genome is None or g.fitness > best_bias_genome.fitness:
                    best_bias_genome = g
                genome_key += 1
        
        # Formalize the data
        points = [[x, y] for x in range(3) for y in range(dim)]
        points_normalized = [[p1, (p2 - r) * hops] for p1, p2 in points]
        values = [bias_result[p[0], p[1]] for p in points]
        grid_x, grid_y = np.mgrid[0:3:1, -r * hops:(r + 1) * hops:hops]
        
        # Create the figure
        plt.figure(figsize=(10, 2.5))  # Rather horizontal plot due to limited number of rows
        knn_data = griddata(points_normalized, values, (grid_x, grid_y), method='nearest')
        ax = sns.heatmap(knn_data,
                         annot=True,
                         fmt='.3g',
                         # vmin=0,
                         # vmax=1,
                         xticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         yticklabels=['r', 'z', 'h'],
                         cbar_kws={"pad": 0.02, "fraction": 0.05},
                         )
        ax.invert_yaxis()
        plt.title('Bias mutation')
        plt.xlabel(r'$\Delta bias_h$' + f' (init={list(init_gru.bias_h)!r})')
        plt.ylabel(f'bias components')
        plt.tight_layout()
        path = f"population/storage/{pop.folder_name}/{pop}/"
        path = get_subfolder(path, 'images')
        path = get_subfolder(path, f'gen{pop.generation:05d}')
        plt.savefig(f"{path}bias.png")
        plt.close()
    
    if mutate_reset:
        reset_result = np.zeros((dim, dim))
        for a in range(dim):
            for b in range(dim):
                g = pop.population[genome_key]
                reset_result[a, b] = g.fitness
                if best_reset_genome is None or g.fitness > best_reset_genome.fitness:
                    best_reset_genome = g
                genome_key += 1
        
        # Formalize the data
        points = [[x, y] for x in range(dim) for y in range(dim)]
        points_normalized = [[(p1 - r) * hops, (p2 - r) * hops] for p1, p2 in points]
        values = [reset_result[p[0], p[1]] for p in points]
        grid_x, grid_y = np.mgrid[-r * hops:(r + 1) * hops:hops, -r * hops:(r + 1) * hops:hops]
        
        # Create the figure
        plt.figure(figsize=(15, 15))  # Rather horizontal plot due to limited number of rows  TODO set back to (5, 5)
        knn_data = griddata(points_normalized, values, (grid_x, grid_y), method='nearest')
        ax = sns.heatmap(knn_data,
                         annot=True,
                         fmt='.3g',
                         # vmin=0,
                         # vmax=1,
                         xticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         yticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         cbar_kws={"pad": 0.02, "fraction": 0.05},
                         )
        ax.invert_yaxis()
        plt.title('Reset-gate mutation')
        plt.xlabel(r'$\Delta W_{hr}$' + f' (init={round(init_gru.weight_hh[0, 0], 3)})')
        plt.ylabel(r'$\Delta W_{xr}$' + f' (init={round(init_gru.weight_xh[0, 0], 3)})')
        plt.tight_layout()
        path = f"population/storage/{pop.folder_name}/{pop}/"
        path = get_subfolder(path, 'images')
        path = get_subfolder(path, f'gen{pop.generation:05d}')
        plt.savefig(f"{path}reset_gate.png")
        plt.close()
    
    if mutate_update:
        update_result = np.zeros((dim, dim))
        for a in range(dim):
            for b in range(dim):
                g = pop.population[genome_key]
                update_result[a, b] = g.fitness
                if best_update_genome is None or g.fitness > best_update_genome.fitness:
                    best_update_genome = g
                genome_key += 1
        
        # Formalize the data
        points = [[x, y] for x in range(dim) for y in range(dim)]
        points_normalized = [[(p1 - r) * hops, (p2 - r) * hops] for p1, p2 in points]
        values = [update_result[p[0], p[1]] for p in points]
        grid_x, grid_y = np.mgrid[-r * hops:(r + 1) * hops:hops, -r * hops:(r + 1) * hops:hops]
        
        # Create the figure
        plt.figure(figsize=(15, 15))  # Rather horizontal plot due to limited number of rows  TODO set back to (5, 5)
        knn_data = griddata(points_normalized, values, (grid_x, grid_y), method='nearest')
        ax = sns.heatmap(knn_data,
                         annot=True,
                         fmt='.3g',
                         # vmin=0,
                         # vmax=1,
                         xticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         yticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         cbar_kws={"pad": 0.02, "fraction": 0.05},
                         )
        ax.invert_yaxis()
        plt.title('Update-gate mutation')
        plt.xlabel(r'$\Delta W_{hz}$' + f' (init={round(init_gru.weight_hh[1, 0], 3)})')
        plt.ylabel(r'$\Delta W_{xz}$' + f' (init={round(init_gru.weight_xh[1, 0], 3)})')
        plt.tight_layout()
        path = f"population/storage/{pop.folder_name}/{pop}/"
        path = get_subfolder(path, 'images')
        path = get_subfolder(path, f'gen{pop.generation:05d}')
        plt.savefig(f"{path}update_gate.png")
        plt.close()
    
    if mutate_candidate:
        candidate_result = np.zeros((dim, dim))
        for a in range(dim):
            for b in range(dim):
                g = pop.population[genome_key]
                candidate_result[a, b] = g.fitness
                if best_candidate_genome is None or g.fitness > best_candidate_genome.fitness:
                    best_candidate_genome = g
                genome_key += 1
        
        # Formalize the data
        points = [[x, y] for x in range(dim) for y in range(dim)]
        points_normalized = [[(p1 - r) * hops, (p2 - r) * hops] for p1, p2 in points]
        values = [candidate_result[p[0], p[1]] for p in points]
        grid_x, grid_y = np.mgrid[-r * hops:(r + 1) * hops:hops, -r * hops:(r + 1) * hops:hops]
        
        # Create the figure
        plt.figure(figsize=(15, 15))  # Rather horizontal plot due to limited number of rows  TODO set back to (5, 5)
        knn_data = griddata(points_normalized, values, (grid_x, grid_y), method='nearest')
        ax = sns.heatmap(knn_data,
                         annot=True,
                         fmt='.3g',
                         # vmin=0,
                         # vmax=1,
                         xticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         yticklabels=[round((i - r) * hops, 2) for i in range(dim)],
                         cbar_kws={"pad": 0.02, "fraction": 0.05},
                         )
        ax.invert_yaxis()
        plt.title('Candidate-state mutation')
        plt.xlabel(r'$\Delta W_{hh}$' + f' (init={round(init_gru.weight_hh[2, 0], 3)})')
        plt.ylabel(r'$\Delta W_{xh}$' + f' (init={round(init_gru.weight_xh[2, 0], 3)})')
        plt.tight_layout()
        path = f"population/storage/{pop.folder_name}/{pop}/"
        path = get_subfolder(path, 'images')
        path = get_subfolder(path, f'gen{pop.generation:05d}')
        plt.savefig(f"{path}candidate_state.png")
        plt.close()
    
    # Set the most fit genome
    pop.best_genome.fitness = 0
    if mutate_bias and best_bias_genome.fitness > pop.best_genome.fitness:
        pop.best_genome = copy.deepcopy(best_bias_genome)
    if mutate_reset and best_reset_genome.fitness > pop.best_genome.fitness:
        pop.best_genome = copy.deepcopy(best_reset_genome)
    if mutate_update and best_update_genome.fitness > pop.best_genome.fitness:
        pop.best_genome = copy.deepcopy(best_update_genome)
    if mutate_candidate and best_candidate_genome.fitness > pop.best_genome.fitness:
        pop.best_genome = copy.deepcopy(best_candidate_genome)
    if mutate_combine:
        pop.best_genome.nodes[2].bias_h = best_bias_genome.nodes[2].bias_h
        pop.best_genome.nodes[2].weight_xh_full[0, 0] = best_reset_genome.nodes[2].weight_xh_full[0, 0]
        pop.best_genome.nodes[2].weight_xh_full[1, 0] = best_update_genome.nodes[2].weight_xh_full[1, 0]
        pop.best_genome.nodes[2].weight_xh_full[2, 0] = best_candidate_genome.nodes[2].weight_xh_full[2, 0]
        pop.best_genome.nodes[2].weight_hh[0, 0] = best_reset_genome.nodes[2].weight_hh[0, 0]
        pop.best_genome.nodes[2].weight_hh[1, 0] = best_update_genome.nodes[2].weight_hh[1, 0]
        pop.best_genome.nodes[2].weight_hh[2, 0] = best_candidate_genome.nodes[2].weight_hh[2, 0]


def get_topology1(cfg, random_init: bool = False):
    """
    Simple genome with only two connections:
        0   1
       /
      2
       \
       -1
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=0,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0  # Drive with 0.5 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0  # Drive with 0.5 actuation by default
    genome.nodes[2] = GruNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    if not random_init:
        genome.nodes[2].bias_h = np.zeros((3,))
        genome.nodes[2].weight_xh_full = np.zeros((3, 1))
        genome.nodes[2].weight_hh = np.zeros((3, 1))
    
    # Create the connections
    genome.connections = dict()
    
    # Input-GRU
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 1  # Simply forward distance
    genome.connections[key].enabled = True
    
    # GRU-Output
    key = (2, 0)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 3  # Increase magnitude to be a value between -3..3 (possible to steer left wheel)
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def get_topology2(cfg, random_init: bool = False):
    """
    Simple genome with only two connections:
        0   1
       /    |
      2    /
       \  /
       -1
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=0,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0  # Drive with 0.5 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0  # Drive with 0.5 actuation by default
    genome.nodes[2] = GruNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    if not random_init:
        genome.nodes[2].bias_h = np.zeros((3,))
        genome.nodes[2].weight_xh_full = np.zeros((3, 1))
        genome.nodes[2].weight_hh = np.zeros((3, 1))
    
    # Create the connections
    genome.connections = dict()
    
    # Input-GRU
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 1  # Simply forward distance
    genome.connections[key].enabled = True
    
    # GRU-Output
    key = (2, 0)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 3  # Increase magnitude to be a value between -3..3 (possible to steer left wheel)
    genome.connections[key].enabled = True
    
    # GRU-Output
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = -1
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


if __name__ == '__main__':
    main(
            topology_id=2,
            iterations=20,
            eval_interval=5,
            experiment_id=1,
            unused_cpu=0,
    )
