"""
experiment8.py

Similar to experiment7.py.

Evaluate a network of fixed topology on the games found in experiment3.

Supported populations:
 * default: No changes
 * gru_nr: Replaced GRU with GRU-NR
 * biased: Optimize by halving GRU-related connections' SS
"""
import argparse
import multiprocessing as mp

from six import iteritems, itervalues

from config import Config
from experiment6 import get_multi_env
from experiment7 import evolve, get_topology
from main import get_folder, get_game_ids
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.dictionary import D_DISTANCE_SCORE

P_DEFAULT = 'default'
P_BIASED = 'biased'
P_GRU_NR = 'gru_nr'
SUPPORTED = [P_DEFAULT, P_BIASED, P_GRU_NR]


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def main(pop_name: str,
         version: int,
         iterations: int = 100,
         unused_cpu: int = 2,
         use_backup: bool = False):
    # Check if valid population name
    if pop_name not in SUPPORTED: raise Exception(f"Population '{pop_name}' not supported!")
    # Create the population
    cfg = get_config()
    folder = get_folder(experiment_id=8)
    pop = Population(
            name=f'{pop_name}/v{version}',
            config=cfg,
            folder_name=folder,
            use_backup=use_backup,
    )
    
    # Replace the population's initial population with the requested topologies genomes
    if pop.generation == 0:
        for g_id in pop.population.keys(): pop.population[g_id] = get_topology(pop_name, gid=g_id, cfg=cfg)
        pop.species.speciate(config=pop.config,
                             population=pop.population,
                             generation=pop.generation,
                             logger=pop.log)
    
    pop.log(f"\n\n\n===> RUNNING EXPERIMENT 8 FOR POPULATION '{pop}' <===\n")
    # Set games and environment used for training and evaluation
    games_train, games_eval = get_game_ids(experiment_id=8)
    train_env = get_multi_env(config=cfg)
    eval_env = get_multi_env(config=cfg)
    eval_env.set_games(games_eval, noise=False)
    
    for iteration in range(iterations):
        # Train the population for a single iteration
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
        evolve(pop, pop_name)
        
        # End generation
        pop.reporters.end_generation(population=pop.population,
                                     name=str(pop),
                                     species_set=pop.species,
                                     logger=pop.log)
        
        # Save the population every ten generations
        if pop.generation % 10 == 0: pop.save()


def get_config():
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.evaluation.fitness = D_DISTANCE_SCORE
    cfg.game.duration = 200  # Experiment3 environments!
    cfg.population.compatibility_thr = 1.  # Keep threshold low since variation constrained for populations
    cfg.genome.node_add_prob = 0  # No topology mutations allowed
    cfg.genome.node_disable_prob = 0  # No topology mutations allowed
    cfg.genome.conn_add_prob = 0  # No topology mutations allowed
    cfg.genome.conn_disable_prob = 0  # No topology mutations allowed
    cfg.genome.enabled_mutate_rate = 0  # No topology mutations allowed
    cfg.update()
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pop_name', type=str)  # ID of the used topology
    parser.add_argument('--version', type=int, default=0)  # Version of the population
    parser.add_argument('--iterations', type=int, default=10)  # Number of training iterations  TODO: Change!
    parser.add_argument('--unused_cpu', type=int, default=2)  # Number of CPU cores not used during evaluation
    parser.add_argument('--use_backup', type=bool, default=False)  # Use the backup-data
    args = parser.parse_args()
    
    # Run the program
    main(
            pop_name=args.pop_name,
            version=args.version,
            iterations=args.iterations,
            unused_cpu=args.unused_cpu,
            use_backup=args.use_backup,
    )
