"""
experiment6_2.py

Extension to experiment6 that uses the NEAT algorithm to evolve the network's weights.
"""
import argparse
import csv
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from glob import glob

import matplotlib.pyplot as plt
from six import iteritems, itervalues
from tqdm import tqdm

from experiment6 import enforce_topology, get_config as get_cfg, get_genome, get_genome_parameters, get_multi_env
from main import get_folder, get_game_ids
from population.population import Population
from population.utils.gene_util.gru import GruNodeGene
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.myutils import get_subfolder


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def main(topology_id: int,
         batch_size: int = 1000,
         train_batch: int = 5,
         unused_cpu: int = 2,
         save_pop: bool = False,
         use_backup: bool = False):
    """Run a population infinitely long and store all its good genomes."""
    # Get the CSV used to store the results in
    csv_path, csv_name = get_csv_path(topology_id, use_backup=use_backup)
    
    # Create the population
    name = csv_name if save_pop else 'dummy'
    cfg = get_config()
    folder = get_folder(experiment_id=6)
    pop = Population(
            name=name,
            config=cfg,
            folder_name=folder,
            use_backup=use_backup,
            overwrite=True,  # Every iteration, create a new population from scratch
    )
    
    # Replace the population's initial population with the requested topologies genomes
    for g_id in pop.population.keys():
        pop.population[g_id] = get_genome(topology_id, g_id=g_id, cfg=cfg)
    pop.species.speciate(config=pop.config,
                         population=pop.population,
                         generation=pop.generation,
                         logger=pop.log)
    pop.log(f"\n\n\n===> RUNNING EXPERIMENT 6 <===\n")
    
    # Set games and environment used for training and evaluation
    games_train, games_eval = get_game_ids(experiment_id=6)
    train_env = get_multi_env(config=cfg)
    eval_env = get_multi_env(config=cfg)
    eval_env.set_games(games_eval, noise=False)
    
    # Keep training and evolving the network until the complete CSV is filled
    added = 0
    try:
        while added < batch_size:
            t = time.localtime()
            pop.log(f"\n\n===> Selective genome creation at {added / batch_size}%, "
                    f"storing in csv '{csv_path.split('/')[-1]}' "
                    f"({t.tm_hour:02d}h-{t.tm_min:02d}m-{t.tm_sec:02d}s) <===")
            
            # Train the population
            pop.log("\n===> Training <===")
            for _ in tqdm(range(train_batch), desc="Training"):
                train_env.set_games(games_train, noise=True)
                genomes = list(iteritems(pop.population))
                
                # Initialize the evaluation-pool
                pool = mp.Pool(mp.cpu_count() - unused_cpu)
                manager = mp.Manager()
                return_dict = manager.dict()
                
                for genome in genomes:
                    pool.apply_async(func=train_env.eval_genome, args=(genome, return_dict))
                pool.close()  # Close the pool
                pool.join()  # Postpone continuation until everything is finished
                
                # Calculate the fitness from the given return_dict
                fitness = calc_pop_fitness(
                        fitness_cfg=pop.config.evaluation,
                        game_cfg=cfg.game,
                        game_obs=return_dict,
                        gen=pop.generation,
                )
                for i, genome in genomes:
                    genome.fitness = fitness[i]
                
                # Update the population's best_genome
                best = None
                for g in itervalues(pop.population):
                    if best is None or g.fitness > best.fitness: best = g
                genomes = sorted(pop.population.items(), key=lambda x: x[1].fitness, reverse=True)
                pop.best_fitness[pop.generation] = genomes[0][1].fitness
                pop.best_genome_hist[pop.generation] = genomes[0]
                pop.best_genome = best
                pop.log(f"Best training fitness: {best.fitness}")
                
                # Let population evolve
                pop.evolve()
                
                # Constraint each of the population's new genomes to the given topology
                for g in pop.population.values():
                    enforce_topology(g, topology_id=topology_id)
            
            # Save the population after training
            pop.save()
            
            # Evaluate the current population as was done in experiment6
            pop.log("\n===> EVALUATING <===")
            genomes = list(iteritems(pop.population))
            pool = mp.Pool(mp.cpu_count() - unused_cpu)
            manager = mp.Manager()
            return_dict = manager.dict()
            for genome in genomes:
                pool.apply_async(func=eval_env.eval_genome, args=(genome, return_dict))
            pool.close()  # Close the pool
            pool.join()  # Postpone continuation until everything is finished
            
            # Calculate the fitness from the given return_dict
            fitness = calc_pop_fitness(
                    fitness_cfg=cfg.evaluation,
                    game_cfg=cfg.game,
                    game_obs=return_dict,
            )
            best_fitness = 0
            for i, genome in genomes:
                genome.fitness = fitness[i]
                if fitness[i] > best_fitness: best_fitness = fitness[i]
            pop.log(f"Best evaluation fitness: {round(best_fitness, 3)}")
            
            # Write the result to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for _, g in genomes:
                    writer.writerow(get_genome_parameters(g, topology_id=topology_id))
            added += len(pop.population)
    except KeyboardInterrupt:
        # Remove the temporary CSV
        os.remove(csv_path)
        raise KeyboardInterrupt
    finally:
        # Remove the dummy population if it exists
        path = f"population{'_backup' if use_backup else ''}/storage/{pop.folder_name}/dummy/"
        if os.path.exists(path):
            shutil.rmtree(path)


def visualize_bar(topology_id: int, csv_id: int = None, rounding: int = 2, use_backup: bool = False):
    """Visualize a bar-plot of how many genomes obtained which fitness score"""
    fitness = []
    path_shared = get_subfolder(f"population{'_backup' if use_backup else ''}/storage/", "experiment6")
    path_data = get_subfolder(path_shared, "data_neat")
    path_images = get_subfolder(path_shared, 'images_neat')
    if csv_id is None: csv_id = len(glob(f"{path_data}*.csv")) - 1
    if csv_id < 0: raise Exception("No data yet created!")
    name = f"topology_{topology_id}_{csv_id}"
    csv_name = f"{path_data}{name}.csv"
    
    # Read in the scores
    total_size = 0
    with open(csv_name, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for row in reader:
            fitness.append(round(float(row[-1]), rounding))
            total_size += 1
    
    # Count the scores
    c = Counter()
    for f in fitness:
        c[f] += 1
    
    # Plot the result
    plt.figure(figsize=(10, 5))
    x, y = zip(*sorted(c.items()))
    i = 1
    while i <= max(y):
        plt.axhline(i, color="grey", linewidth=0.5)
        i *= 10
    plt.bar(x, y, width=1 / (10 ** rounding))
    plt.yscale('log')
    plt.title("Fitness-distribution of trained fixed-topology genome-space")
    plt.ylabel("Number of genomes")
    plt.xlabel("Fitness score")
    plt.savefig(f"{path_images}{name}.png")
    # plt.show()
    plt.close()


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #


def get_config():
    cfg = get_cfg()
    cfg.population.pop_size = 512
    cfg.population.compatibility_thr = .5  # Keep threshold low to enforce new species to be discovered
    cfg.population.specie_elitism = 0  # Do not keep any specie after it stagnated
    cfg.population.specie_stagnation = 10  # Keep a relative low stagnation threshold to make room for new species
    cfg.population.parent_selection = .1  # Low parent selection since large number of species used
    cfg.genome.node_add_prob = 0  # No topology mutations allowed
    cfg.genome.node_disable_prob = 0  # No topology mutations allowed
    cfg.genome.conn_add_prob = 0  # No topology mutations allowed
    cfg.genome.conn_disable_prob = 0  # No topology mutations allowed
    cfg.genome.enabled_mutate_rate = 0  # No topology mutations allowed
    cfg.update()
    return cfg


def get_csv_path(topology_id: int, use_backup: bool):
    """Get the genome-key based on CSV-file's length."""
    path = get_subfolder(f"population{'_backup' if use_backup else ''}/storage/", "experiment6")
    path = get_subfolder(path, "data_neat")
    n_files = len(glob(f"{path}*.csv"))
    csv_name = f"topology_{topology_id}_{n_files}"
    path = f"{path}{csv_name}.csv"
    
    # CSV exists, count number of rows
    if os.path.exists(path):
        return path, csv_name
    
    # CSV does not exist, create new
    else:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bias0', 'bias1', 'bias2',
                             'weight_xr', 'weight_xz', 'weight_xh',
                             'weight_hr', 'weight_hz', 'weight_hh',
                             'conn1', 'conn2',
                             'fitness'])
            return path, csv_name


def execution_test():
    """Test if the algorithm works properly."""
    cfg = get_config()
    g = get_genome(1, g_id=0, cfg=cfg)
    for _ in range(100):
        g.mutate(cfg.genome)
        enforce_topology(g, topology_id=1)
    print(g)
    assert g.nodes[0].bias == 2
    assert g.nodes[1].bias == 0
    assert g.connections[(-1, 2)].weight == 1
    assert type(g.nodes[2]) == GruNodeGene


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--evaluate', type=bool, default=True)  # Evaluate new genomes
    parser.add_argument('--topology_id', type=int, default=2)  # ID of the used topology
    parser.add_argument('--batch', type=int, default=10)  # Number of genomes evaluated per batch
    parser.add_argument('--unused_cpu', type=int, default=2)  # Number of CPU cores not used during evaluation
    parser.add_argument('--save_population', type=bool, default=True)  # Save the final population after finishing
    parser.add_argument('--visualize', type=bool, default=True)  # Visualize the current results
    parser.add_argument('--test', type=bool, default=False)  # Visualize the current results
    parser.add_argument('--use_backup', type=bool, default=False)  # Use the backup-data
    args = parser.parse_args()
    
    # Run the program
    if args.evaluate:
        main(
                topology_id=args.topology_id,
                save_pop=args.save_population,
                batch_size=args.batch,
                unused_cpu=args.unused_cpu,
                use_backup=args.use_backup,
        )
    if args.visualize:
        visualize_bar(
                topology_id=args.topology_id,
                use_backup=args.use_backup,
        )
    if args.test:
        execution_test()
