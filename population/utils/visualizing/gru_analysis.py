"""
gru_analysis.py

Analyse a single-GRU genome of the population.
"""
import argparse
import copy
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

from config import Config
from environment.env_multi import get_multi_env
from main import get_game_ids, trace_most_fit, visualize_genome
from population.population import Population
from population.utils.gene_util.gru import GruNodeGene
from population.utils.genome import Genome
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.myutils import get_subfolder, update_dict


def main(pop: Population,
         d: int = 10,
         range_width: int = 20,
         genome: Genome = None,
         cpu: int = 2,
         experiment_id: int = 1,
         overwrite: bool = False,
         ):
    """
    Analyse the given single-hidden single-GRU genome.
    
    Analysis is performed on following number of genomes: 3 * ((2 * range + 1) ** 2)
     * 10 =  1'323
     * 20 =  5'043
     * 50 = 30'603

    :param pop: Population on which the analysis is performed
    :param genome: The genome that is being mutated
    :param d: Divisor, indicating hops of 1/d
    :param range_width: Width of the parameter-range taken into account (note: true range is range_width/d)
    :param overwrite: Decide if re-evaluation
    :param experiment_id: Experiment-games used for evaluation
    :param cpu: Number of CPU cores not used for simulation
    """
    # Get genome if not defined
    if not genome: genome = pop.best_genome
    
    # Create a cache-population
    cfg = copy.deepcopy(pop.config)
    cfg.population.pop_size = 2  # Dummy candidates, to be removed
    cfg.update()
    
    # Create the populations
    name = f"genome{genome.key}_divisor{d}_range{range_width}"
    pop_cache = Population(
            name=name,
            folder_name='../../cache_populations',  # I do the hack hack
            config=cfg,
            overwrite=overwrite,
    )
    if len(pop_cache.population) == 2:
        create_genomes(genome=genome, pop=pop_cache, d=d, range_width=range_width)
    
    # Evaluate the populations
    if pop_cache.population[0].fitness is None:
        evaluate_population(
                pop=pop_cache,
                cfg=cfg,
                cpu=cpu,
                experiment_id=experiment_id,
        )
    
    # Evaluate the results - Create 3d array of the results
    visualize_score(pop=pop, pop_cache=pop_cache, genome=genome, d=d, range_width=range_width)


def create_genomes(genome: Genome, pop: Population, d: int, range_width: int):
    """Create mutations of the provided genome and inject these in the given population.."""
    pop.log(f"{pop.name} - Setting up genomes...")
    pbar = tqdm(range(((range_width + range_width + 1) ** 2) * 3), desc="Generating genomes")
    genome_key = 0
    
    gru_node_id = None
    for node_id, node in genome.nodes.items():
        if type(node) == GruNodeGene:
            gru_node_id = node_id
            break
    
    # Create the reset-mutated genomes
    for a in range(-range_width, range_width + 1):
        w_xh = np.asarray([[a / d], [0], [0]])
        for b in range(-range_width, range_width + 1):
            w_hh = np.asarray([[b / d], [0], [0]])
            
            # Add the specified genome to the population
            new_genome = copy.deepcopy(genome)
            new_genome.nodes[gru_node_id].weight_xh_full += w_xh
            new_genome.nodes[gru_node_id].weight_hh += w_hh
            new_genome.fitness = None
            new_genome.key = genome_key
            pop.population[genome_key] = new_genome
            genome_key += 1
            pbar.update()
    
    # Create the update-mutated genomes
    for a in range(-range_width, range_width + 1):
        w_xh = np.asarray([[0], [a / d], [0]])
        for b in range(-range_width, range_width + 1):
            w_hh = np.asarray([[0], [b / d], [0]])
            
            # Add the specified genome to the population
            new_genome = copy.deepcopy(genome)
            new_genome.nodes[gru_node_id].weight_xh_full += w_xh
            new_genome.nodes[gru_node_id].weight_hh += w_hh
            new_genome.fitness = None
            new_genome.key = genome_key
            pop.population[genome_key] = new_genome
            genome_key += 1
            pbar.update()
    
    # Create the candidate-mutated genomes
    for a in range(-range_width, range_width + 1):
        w_xh = np.asarray([[0], [0], [a / d]])
        for b in range(-range_width, range_width + 1):
            w_hh = np.asarray([[0], [0], [b / d]])
            
            # Add the specified genome to the population
            new_genome = copy.deepcopy(genome)
            new_genome.nodes[gru_node_id].weight_xh_full += w_xh
            new_genome.nodes[gru_node_id].weight_hh += w_hh
            new_genome.fitness = None
            new_genome.key = genome_key
            pop.population[genome_key] = new_genome
            genome_key += 1
            pbar.update()
    pbar.close()
    assert len(pop.population) == (((range_width - -range_width + 1) ** 2) * 3)
    pop.save()


def evaluate_population(pop: Population, cfg: Config, cpu: int, experiment_id: int):
    """Evaluate the given population."""
    pop.log(f"{pop.name} - Evaluating the population...")
    _, game_ids_eval = get_game_ids(experiment_id=experiment_id)
    multi_env = get_multi_env(pop=pop, game_config=cfg)
    multi_env.set_games(game_ids_eval, noise=False)
    pool = mp.Pool(mp.cpu_count() - cpu)
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
    pop.log(f"{pop.name} - Calculating fitness scores...")
    fitness = calc_pop_fitness(
            fitness_cfg=pop.config.evaluation,
            game_cfg=cfg.game,
            game_obs=return_dict,
            gen=pop.generation,
    )
    for i, genome in pop.population.items():
        genome.fitness = fitness[i]
    
    # Get the fittest genome
    best = None
    for g in pop.population.values():
        if best is None or g.fitness > best.fitness: best = g
    pop.best_genome = best
    
    # Save the results
    pop.save()
    
    # Visualize most fit genome
    visualize_genome(
            debug=True,
            genome=best,
            population=pop,
    )
    
    # Trace the most fit genome
    trace_most_fit(
            debug=False,
            games=game_ids_eval,
            genome=best,
            population=pop,
            unused_cpu=cpu,
    )


def visualize_score(pop: Population, pop_cache: Population, genome: Genome, d: int, range_width: int):
    """Visualize the score of the evaluated population."""
    pop_cache.log(f"{pop_cache.name} - Fetching fitness scores...")
    dim = (2 * range_width + 1)
    pbar = tqdm(range((dim ** 2) * 3), desc="Fetching fitness scores")
    genome_key = 0
    
    # Fetching scores of reset-mutations
    reset_scores = np.zeros((dim, dim))
    for a in range(dim):
        for b in range(dim):
            reset_scores[a, b] = pop_cache.population[genome_key].fitness
            genome_key += 1
            pbar.update()
    
    # Fetching scores of update-mutations
    update_scores = np.zeros((dim, dim))
    for a in range(dim):
        for b in range(dim):
            update_scores[a, b] = pop_cache.population[genome_key].fitness
            genome_key += 1
            pbar.update()
    
    # Fetching scores of candidate-mutations
    candidate_scores = np.zeros((dim, dim))
    for a in range(dim):
        for b in range(dim):
            candidate_scores[a, b] = pop_cache.population[genome_key].fitness
            genome_key += 1
            pbar.update()
    pbar.close()
    
    # Visualize the result
    pop_cache.log(f"{pop_cache.name} - Visualizing the result...")
    
    # GRU-node needed for labels
    genome.update_rnn_nodes(config=pop.config.genome)
    gru_node = None
    for node in genome.nodes.values():
        if type(node) == GruNodeGene:
            gru_node = node
    
    # Create the points and retrieve data for the plot
    points = [[x, y] for x in range(dim) for y in range(dim)]
    points_normalized = [[(p1 + -range_width) / d, (p2 + -range_width) / d] for p1, p2 in points]
    values_reset = [reset_scores[p[0], p[1]] for p in points]
    values_update = [update_scores[p[0], p[1]] for p in points]
    values_candidate = [candidate_scores[p[0], p[1]] for p in points]
    
    # K-nearest neighbours with hops of 0.01 is performed
    grid_x, grid_y = np.mgrid[-range_width / d:range_width / d:0.01, -range_width / d:range_width / d:0.01]
    
    # Perform k-NN
    data_reset = griddata(points_normalized, values_reset, (grid_x, grid_y), method='nearest')
    data_update = griddata(points_normalized, values_update, (grid_x, grid_y), method='nearest')
    data_candidate = griddata(points_normalized, values_candidate, (grid_x, grid_y), method='nearest')
    
    # Create the plots
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(data_reset.T,
               vmin=0,
               vmax=1,
               extent=(-range_width / d, range_width / d, -range_width / d, range_width / d),
               origin='lower')
    plt.title('Reset-gate mutation')
    plt.xlabel(r'$\Delta W_{hr}$' + f' (init={round(gru_node.weight_hh[0, 0], 3)})')
    plt.ylabel(r'$\Delta W_{xr}$' + f' (init={round(gru_node.weight_xh[0, 0], 3)})')
    plt.subplot(132)
    plt.imshow(data_update.T,
               vmin=0,
               vmax=1,
               extent=(-range_width / d, range_width / d, -range_width / d, range_width / d),
               origin='lower')
    plt.title('Update-gate mutation')
    plt.xlabel(r'$\Delta W_{hz}$' + f' (init={round(gru_node.weight_hh[1, 0], 3)})')
    plt.ylabel(r'$\Delta W_{xz}$' + f' (init={round(gru_node.weight_xh[1, 0], 3)})')
    plt.subplot(133)
    plt.imshow(data_candidate.T,
               vmin=0,
               vmax=1,
               extent=(-range_width / d, range_width / d, -range_width / d, range_width / d),
               origin='lower')
    plt.title('Candidate-state mutation')
    plt.xlabel(r'$\Delta W_{hh}$' + f' (init={round(gru_node.weight_hh[2, 0], 3)})')
    plt.ylabel(r'$\Delta W_{xh}$' + f' (init={round(gru_node.weight_xh[2, 0], 3)})')
    
    # Store the plot
    plt.tight_layout()
    path = f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/"
    path = get_subfolder(path, 'images')
    path = get_subfolder(path, 'gru_analysis')
    plt.savefig(f"{path}{genome.key}.png")
    plt.close()
    
    # Create overview
    pop_cache.log("Overview of results:")
    log = dict()
    
    # Overview: reset-mutation
    max_index_reset, max_value_reset, min_index_reset, min_value_reset = None, 0, None, 1
    for index, x in np.ndenumerate(reset_scores):
        if x < min_value_reset: min_index_reset, min_value_reset = index, x
        if x > max_value_reset: max_index_reset, max_value_reset = index, x
    pop_cache.log(f"\tReset-gate mutation:")
    pop_cache.log(f"\t > Maximum fitness: {round(max_value_reset, 2)} for index {max_index_reset!r}")
    pop_cache.log(f"\t > Average fitness: {round(np.average(reset_scores), 2)}")
    pop_cache.log(f"\t > Minimum fitness: {round(min_value_reset, 2)} for index {min_index_reset!r}")
    log['Reset-gate maximum fitness'] = f"{round(max_value_reset, 2)} for index {max_index_reset!r}"
    log['Reset-gate average fitness'] = f"{round(np.average(reset_scores), 2)}"
    log['Reset-gate minimum fitness'] = f"{round(min_value_reset, 2)} for index {min_index_reset!r}"
    
    # Overview: update-mutation
    max_index_update, max_value_update, min_index_update, min_value_update = None, 0, None, 1
    for index, x in np.ndenumerate(update_scores):
        if x < min_value_update: min_index_update, min_value_update = index, x
        if x > max_value_update: max_index_update, max_value_update = index, x
    pop_cache.log(f"\tUpdate-gate mutation:")
    pop_cache.log(f"\t > Maximum fitness: {round(max_value_update, 2)} for index {max_index_update!r}")
    pop_cache.log(f"\t > Average fitness: {round(np.average(update_scores), 2)}")
    pop_cache.log(f"\t > Minimum fitness: {round(min_value_update, 2)} for index {min_index_update!r}")
    log['Update-gate maximum fitness'] = f"{round(max_value_update, 2)} for index {max_index_update!r}"
    log['Update-gate average fitness'] = f"{round(np.average(update_scores), 2)}"
    log['Update-gate minimum fitness'] = f"{round(min_value_update, 2)} for index {min_index_update!r}"
    
    # Overview: candidate-mutation
    max_index_candidate, max_value_candidate, min_index_candidate, min_value_candidate = None, 0, None, 1
    for index, x in np.ndenumerate(candidate_scores):
        if x < min_value_candidate: min_index_candidate, min_value_candidate = index, x
        if x > max_value_candidate: max_index_candidate, max_value_candidate = index, x
    pop_cache.log(f"\tCandidate-state mutation:")
    pop_cache.log(f"\t > Maximum fitness: {round(max_value_candidate, 2)} for index {max_index_candidate!r}")
    pop_cache.log(f"\t > Average fitness: {round(np.average(candidate_scores), 2)}")
    pop_cache.log(f"\t > Minimum fitness: {round(min_value_candidate, 2)} for index {min_index_candidate!r}")
    log['Candidate-state maximum fitness'] = f"{round(max_value_candidate, 2)} for index {max_index_candidate!r}"
    log['Candidate-state average fitness'] = f"{round(np.average(candidate_scores), 2)}"
    log['Candidate-state minimum fitness'] = f"{round(min_value_candidate, 2)} for index {min_index_candidate!r}"
    update_dict(f'{path}{genome.key}.txt', log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--range', type=int, default=10)  # TODO: Change to 50
    parser.add_argument('--experiment', type=int, default=4)
    parser.add_argument('--unused_cpu', type=int, default=0)
    parser.add_argument('--overwrite', type=int, default=0)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("../../../")
    
    # Fetch the genome we want to copy
    population = Population(
            name='test',
            folder_name='test',
    )
    
    # Execute the algorithm
    main(
            pop=population,
            d=args.d,
            range_width=args.range,
            overwrite=bool(args.overwrite),
            experiment_id=args.experiment,
            cpu=args.unused_cpu,
    )
