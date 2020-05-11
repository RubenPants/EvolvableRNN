"""
experiment7.py

Evaluate a network of fixed topology on the games found in experiment3.
"""
import argparse
import multiprocessing as mp

from six import iteritems, itervalues

from config import Config
from experiment6 import enforce_topology, get_genome, get_multi_env
from main import get_folder, get_game_ids
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.dictionary import *


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def main(topology_id: int,
         version:int,
         iterations: int,
         batch: int = 10,
         unused_cpu: int = 2,
         use_backup: bool = False):
    # Create the population
    cfg = get_config()
    folder = get_folder(experiment_id=7)
    pop = Population(
            name=f'topology_{topology_id}/v{version}',
            config=cfg,
            folder_name=folder,
            use_backup=use_backup,
    )
    
    # Replace the population's initial population with the requested topologies genomes
    if pop.generation == 0:
        for g_id in pop.population.keys():
            pop.population[g_id] = get_genome(topology_id, g_id=g_id, cfg=cfg)
        pop.species.speciate(config=pop.config,
                             population=pop.population,
                             generation=pop.generation,
                             logger=pop.log)
    
    pop.log(f"\n\n\n===> RUNNING EXPERIMENT 7 <===\n")
    # Set games and environment used for training and evaluation
    games_train, games_eval = get_game_ids(experiment_id=7)
    train_env = get_multi_env(config=cfg)
    eval_env = get_multi_env(config=cfg)
    eval_env.set_games(games_eval, noise=False)
    
    # Train the population
    for _ in range(iterations):
        train_env.set_games(games_train, noise=True)
        
        # Prepare the generation's reporters for the generation
        pop.reporters.start_generation(gen=pop.generation, logger=pop.log)
        
        # Fetch the dictionary of genomes
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
        pop.reporters.post_evaluate(population=pop.population,
                                    species=pop.species,
                                    best_genome=best,
                                    logger=pop.log)
        
        # Update the population's best_genome
        genomes = sorted(pop.population.items(), key=lambda x: x[1].fitness, reverse=True)
        pop.best_fitness[pop.generation] = genomes[0][1].fitness
        pop.best_genome_hist[pop.generation] = genomes[0]
        pop.best_genome = best
        
        # Let population evolve
        pop.evolve()
        
        # Constraint each of the population's new genomes to the given topology
        for g in pop.population.values():
            enforce_topology(g, topology_id=topology_id)
        
        # End generation
        pop.reporters.end_generation(population=pop.population,
                                     name=str(pop),
                                     species_set=pop.species,
                                     logger=pop.log)
        
        # Save the population after training
        if pop.generation % batch == 0:
            pop.save()


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #


def get_config():
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.evaluation.fitness = D_DISTANCE_SCORE
    cfg.game.duration = 200  # 200 seconds, similar to experiment 3
    cfg.genome.node_add_prob = 0  # No topology mutations allowed
    cfg.genome.node_disable_prob = 0  # No topology mutations allowed
    cfg.genome.conn_add_prob = 0  # No topology mutations allowed
    cfg.genome.conn_disable_prob = 0  # No topology mutations allowed
    cfg.genome.enabled_mutate_rate = 0  # No topology mutations allowed
    cfg.population.pop_size = 512
    cfg.population.compatibility_thr = 1  # Keep threshold low to enforce new species to be discovered
    cfg.update()
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--topology_id', type=int, default=3)  # ID of the used topology
    parser.add_argument('--version', type=int, default=1)  # Version of the population
    parser.add_argument('--iterations', type=int, default=0)  # Number of training iterations
    parser.add_argument('--batch', type=int, default=10)  # Hops of saving during training
    parser.add_argument('--unused_cpu', type=int, default=2)  # Number of CPU cores not used during evaluation
    parser.add_argument('--use_backup', type=bool, default=False)  # Use the backup-data
    args = parser.parse_args()
    
    # Run the program
    main(
            topology_id=args.topology_id,
            version=args.version,
            iterations=args.iterations,
            batch=args.batch,
            unused_cpu=args.unused_cpu,
            use_backup=args.use_backup,
    )
