"""
experiment7.py

Evaluate a network of fixed topology on the games found in experiment3.

Supported populations:
 * default: No changes
 * connection: Optimize by halving GRU-related connections' SS
"""
import argparse
import multiprocessing as mp
from random import random

from numpy.random import normal, random as rand_arr
from six import iteritems, itervalues

from config import Config
from experiment6 import get_multi_env
from experiment6_2 import calc_finished_ratio, get_config
from main import get_folder, get_game_ids
from population.population import Population
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.gru_no_reset import GruNoResetNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.genome import Genome
from population.utils.population_util.fitness_functions import calc_pop_fitness

P_DEFAULT = 'default'
P_BIASED = 'biased'
P_GRU_NR = 'gru_nr'
SUPPORTED = [P_DEFAULT, P_BIASED, P_GRU_NR]


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def main(pop_name: str,
         version: int,
         unused_cpu: int = 2,
         use_backup: bool = False):
    # Check if valid population name
    if pop_name not in SUPPORTED: raise Exception(f"Population '{pop_name}' not supported!")
    # Create the population
    cfg = get_config()
    cfg.population.specie_elitism = 1
    folder = get_folder(experiment_id=7)
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
    
    pop.log(f"\n\n\n===> RUNNING EXPERIMENT 7 <===\n")
    # Set games and environment used for training and evaluation
    games_train, games_eval = get_game_ids(experiment_id=7)
    train_env = get_multi_env(config=cfg)
    eval_env = get_multi_env(config=cfg)
    eval_env.set_games(games_eval, noise=False)
    
    solution_found = False
    while not solution_found:
        # Train the population for a single iteration
        pop.log("\n===> TRAINING <===")
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
        
        # Test if evaluation finds a solution for the new generation, impossible if fitness < 0.7
        if pop.best_genome.fitness > 0.7 or pop.generation % 10 == 0:
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
            best = None
            for i, genome in genomes:
                genome.fitness = finished[i]
                if best is None or finished[i] > best.fitness: best = genome
            pop.log(f"Best genome:\n{best}\n{best.nodes[2]}")
            
            # Solution is found
            if best.fitness == 1:
                pop.best_genome = best
                pop.log(f"Solution found!")
                solution_found = True  # End the outer while-loop
        
        # Save the population with their evaluation results
        pop.save()


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #

def evolve(pop: Population, pop_name: str):
    """Evolve with the set constraints in mind."""
    # Create the next generation from the current generation
    pop.population = pop.reproduction.reproduce(
            config=pop.config,
            species=pop.species,
            generation=pop.generation,
            logger=pop.log,
    )
    
    # Constraint each of the population's new genomes to the given topology
    for g in pop.population.values(): enforce_topology(pop_name, genome=g)
    
    # Check for complete extinction
    if not pop.species.species:
        pop.reporters.complete_extinction(logger=pop.log)
        
        # If requested by the user, create a completely new population, otherwise raise an exception
        pop.population = pop.reproduction.create_new(config=pop.config,
                                                     num_genomes=pop.config.population.pop_size)
    
    # Divide the new population into species
    pop.species.speciate(config=pop.config,
                         population=pop.population,
                         generation=pop.generation,
                         logger=pop.log)
    
    # Add to each of the species its elites
    pop.update_species_fitness_hist()
    
    # Increment generation count
    pop.generation += 1


def get_topology(pop_name, gid: int, cfg: Config):
    """
    Create a uniformly and randomly sampled genome of fixed topology:
    Sigmoid with bias 1.5 --> Actuation default of 95,3%
      (key=0, bias=1.5)   (key=1, bias=?)
                     ____ /   /
                   /         /
                GRU         /
                |     _____/
                |   /
              (key=-1)
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=gid,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Setup the parameter-ranges
    conn_range = cfg.genome.weight_max_value - cfg.genome.weight_min_value
    bias_range = cfg.genome.bias_max_value - cfg.genome.bias_min_value
    rnn_range = cfg.genome.rnn_max_value - cfg.genome.rnn_min_value
    
    # Create the output nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    if pop_name in [P_BIASED]:
        genome.nodes[1].bias = normal(1.5, .1)  # Initially normally distributed around bias of other output
    else:
        genome.nodes[1].bias = random() * bias_range + cfg.genome.bias_min_value  # Uniformly sampled bias
    
    # Setup the recurrent unit
    if pop_name in [P_GRU_NR]:
        genome.nodes[2] = GruNoResetNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden
        genome.nodes[2].bias_h = rand_arr((2,)) * bias_range + cfg.genome.bias_min_value
        genome.nodes[2].weight_xh_full = rand_arr((2, 1)) * rnn_range + cfg.genome.weight_min_value
        genome.nodes[2].weight_hh = rand_arr((2, 1)) * rnn_range + cfg.genome.weight_min_value
    else:
        genome.nodes[2] = GruNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
        genome.nodes[2].bias_h = rand_arr((3,)) * bias_range + cfg.genome.bias_min_value
        genome.nodes[2].weight_xh_full = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
        genome.nodes[2].weight_hh = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    # Create the connections
    genome.connections = dict()
    
    # input2gru - Uniformly sampled on the positive spectrum
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    if pop_name in [P_BIASED]:
        genome.connections[key].weight = 6  # Maximize connection, GRU can always lower values flowing through
    else:
        genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    # gru2output - Uniformly sampled on the positive spectrum
    key = (2, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    if pop_name in [P_BIASED]: genome.connections[key].weight = abs(genome.connections[key].weight)  # Always positive!
    genome.connections[key].enabled = True
    
    # input2output - Uniformly sampled
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    if pop_name in [P_BIASED]: genome.connections[key].weight = -abs(genome.connections[key].weight)  # Always negative!
    genome.connections[key].enabled = True
    
    # Enforce the topology constraints
    enforce_topology(pop_name=pop_name, genome=genome)
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def enforce_topology(pop_name, genome: Genome):
    """
    Enforce the fixed parameters of topology2. It is assumed that topology hasn't changed.
    
    Ideology: The GRU can always scale the input and outputs itself, so making its corresponding connections a single
        value doesn't change much as long as it doesn't take away its capabilities (i.e. maximize, scaling down is
        always possible!)
    """
    genome.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    if pop_name in [P_BIASED]:
        genome.connections[(-1, 1)].weight = -abs(genome.connections[(-1, 1)].weight)
        genome.connections[(-1, 2)].weight = 6
        genome.connections[(2, 1)].weight = abs(genome.connections[(2, 1)].weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pop_name', type=str)  # ID of the used topology
    parser.add_argument('--version', type=int, default=0)  # Version of the population
    parser.add_argument('--unused_cpu', type=int, default=2)  # Number of CPU cores not used during evaluation
    parser.add_argument('--use_backup', type=bool, default=False)  # Use the backup-data
    args = parser.parse_args()
    
    # Run the program
    main(
            pop_name=args.pop_name,
            version=args.version,
            unused_cpu=args.unused_cpu,
            use_backup=args.use_backup,
    )
