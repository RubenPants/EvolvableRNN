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
from numpy import mean
from scipy.stats import gmean
from six import iteritems, iterkeys, itervalues
from tqdm import tqdm

from config import Config
from configs.evaluation_config import EvaluationConfig
from experiment6 import enforce_topology, get_genome, get_genome_parameters, get_multi_env
from main import get_folder, get_game_ids
from population.population import Population
from population.utils.gene_util.gru import GruNodeGene
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.dictionary import *
from utils.myutils import get_subfolder

# Minimal ratio of evaluation games finished before added to the CSV
MIN_FINISHED = 1  # Finish 15/18 or more


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def main(topology_id: int,
         batch_size: int = 1000,
         train_batch: int = 3,
         min_finished: float = MIN_FINISHED,
         unused_cpu: int = 2,
         save_pop: bool = False,
         use_backup: bool = False):
    """Run a population infinitely long and store all its good genomes."""
    # Get the CSV used to store the results in
    csv_path, csv_name, added = get_csv_path(topology_id, use_backup=use_backup, batch_size=batch_size)
    
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
    try:
        while added < batch_size:
            t = time.localtime()
            pop.log(f"\n\n===> Selective genome creation at {added / batch_size * 100}%, "
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
            finished = calc_finished_ratio(
                    fitness_cfg=cfg.evaluation,
                    game_obs=return_dict,
            )
            best_finish = 0
            for i, genome in genomes:
                genome.fitness = finished[i]
                if finished[i] > best_finish: best_finish = finished[i]
            
            # Give evaluation overview of population
            pop.log(f"Best evaluation finish ratio: {round(best_finish, 2)}")
            sids = list(iterkeys(pop.species.species))
            sids.sort()
            msg = f"\nPopulation '{name}' has {len(pop.species.species):d} species:" \
                  f"\n\t specie    age    size    finished    stag " \
                  f"\n\t========  =====  ======  ==========  ======"
            pop.log(msg) if pop.log else print(msg)
            for sid in sids:
                s = pop.species.species[sid]
                a = pop.generation - s.created
                n = len(s.members)
                sf = [g.fitness for g in s.members.values() if g.fitness]
                f = "--" if len(sf) == 0 else f"{max(sf):.2f}"
                st = pop.generation - s.last_improved
                msg = f"\t{sid:^8}  {a:^5}  {n:^6}  {f:^10}  {st:^6}"
                pop.log(msg) if pop.log else print(msg)
            
            # Write the result to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for _, g in genomes:
                    # Only write the genomes that exceed the minimum 'finished ratio' threshold!
                    if g.fitness >= min_finished:
                        writer.writerow(get_genome_parameters(g, topology_id=topology_id))
                        added += 1
    finally:
        # Remove the dummy population if it exists
        path = f"population{'_backup' if use_backup else ''}/storage/{pop.folder_name}/dummy/"
        if os.path.exists(path):
            shutil.rmtree(path)


def visualize_bar(topology_id: int, csv_id: int = None, use_backup: bool = False):
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
            n_finished = int(round(float(row[-1]) * 18))
            fitness.append(n_finished)
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
    plt.bar(x, y, width=0.9)
    plt.xticks(x)
    plt.yscale('log')
    plt.title("Number of evaluation games finished of trained fixed-topology genome-space")
    plt.ylabel("Number of genomes")
    plt.xlabel("Number of evaluation games finished out of 18")
    plt.savefig(f"{path_images}{name}.png")
    # plt.show()
    plt.close()


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #

def calc_finished_ratio(fitness_cfg: EvaluationConfig, game_obs: dict):
    """Calculate the ratio of finished games for each genome."""
    # 1) Evaluate fitness for each of the games
    fitness = dict()
    for k, v in game_obs.items():  # Iterate over the candidates
        fitness[k] = [1 if o[D_DONE] else 0 for o in v]
    
    # 2) Combine the fitness-functions
    t = fitness_cfg.fitness_comb
    assert (t in [D_MIN, D_AVG, D_MAX, D_GMEAN])
    f = min if t == D_MIN else max if t == D_MAX else mean if D_AVG else gmean
    for k in fitness.keys(): fitness[k] = f(fitness[k])
    return fitness


def get_config():
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.evaluation.fitness = D_DISTANCE
    cfg.game.duration = 60  # 60 seconds should be just enough to reach each of the spawned targets
    cfg.genome.node_add_prob = 0  # No topology mutations allowed
    cfg.genome.node_disable_prob = 0  # No topology mutations allowed
    cfg.genome.conn_add_prob = 0  # No topology mutations allowed
    cfg.genome.conn_disable_prob = 0  # No topology mutations allowed
    cfg.genome.enabled_mutate_rate = 0  # No topology mutations allowed
    cfg.population.pop_size = 512
    cfg.population.compatibility_thr = .5  # Keep threshold low to enforce new species to be discovered
    cfg.population.specie_elitism = 0  # Do not keep any specie after it stagnated
    cfg.population.specie_stagnation = 10  # Keep a relative low stagnation threshold to make room for new species
    cfg.population.parent_selection = .05  # Low parent selection since large number of species used
    cfg.update()
    return cfg


def get_csv_path(topology_id: int, use_backup: bool, batch_size: int):
    """Get the genome-key based on CSV-file's length."""
    path = get_subfolder(f"population{'_backup' if use_backup else ''}/storage/", "experiment6")
    path = get_subfolder(path, "data_neat")
    n_files = len(glob(f"{path}*.csv"))
    if n_files > 0:
        csv_name = f"topology_{topology_id}_{n_files - 1}"
        path_temp = f"{path}{csv_name}.csv"
        
        # If CSV exists, check if not yet full
        if os.path.exists(path_temp):
            with open(path_temp, 'r') as f:
                rows = sum(1 for _ in f) - 1  # Do not count header
                if rows < batch_size: return path_temp, csv_name, rows
    
    # CSV does not yet exist, or is already full, create new CSV
    csv_name = f"topology_{topology_id}_{n_files}"
    path = f"{path}{csv_name}.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Construct the CSV's head, all genomes have the full GRU-parameter suite
        head = ['bias_r', 'bias_z', 'bias_h',
                'weight_xr', 'weight_xz', 'weight_xh',
                'weight_hr', 'weight_hz', 'weight_hh']
        if topology_id in [1]:
            head += ['conn1', 'conn2']
        elif topology_id in [2]:
            head += ['bias_rw', 'conn2']
        elif topology_id in [3]:
            head += ['bias_rw', 'conn0', 'conn1', 'conn2']
        head += ['fitness']
        writer.writerow(head)
        return path, csv_name, 0


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
    parser.add_argument('--min_finished', type=float, default=MIN_FINISHED)  # Minimal finish ratio before added to CSV
    parser.add_argument('--unused_cpu', type=int, default=2)  # Number of CPU cores not used during evaluation
    parser.add_argument('--save_population', type=bool, default=True)  # Save the final population after finishing
    parser.add_argument('--visualize', type=bool, default=True)  # Visualize the current results
    parser.add_argument('--test', type=bool, default=False)  # Visualize the current results
    parser.add_argument('--use_backup', type=bool, default=True)  # Use the backup-data
    args = parser.parse_args()
    
    # Run the program
    if args.evaluate:
        main(
                topology_id=args.topology_id,
                save_pop=args.save_population,
                batch_size=args.batch,
                min_finished=args.min_finished,
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
