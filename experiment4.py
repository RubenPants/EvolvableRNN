"""
experiment4.py

Run the fourth experiment, which uses the mazes of the first experiment.
"""
import argparse
import multiprocessing as mp
import traceback

from neat.six_util import iteritems, itervalues

from config import Config
from environment.env_multi import get_multi_env
from main import evaluate, get_folder, get_game_ids, get_name, gru_analysis, monitor, training_overview, \
    visualize_genome
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness
from process_killer import main as process_killer
from utils.dictionary import *


def main(fitness,
         prob_gru: float,
         prob_gru_nr: float,
         prob_gru_nu: float,
         prob_simple_rnn: float,
         train_iterations=0,
         version=0,
         unused_cpu=1,
         ):
    """
    Run a population's configuration.

    :param fitness: Fitness function used to evaluate the population
    :param prob_gru: Probability of mutating towards a GRU-node
    :param prob_gru_nr: Probability of mutating towards a GRU-NR-node
    :param prob_gru_nu: Probability of mutating towards a GRU-NU-node
    :param prob_simple_rnn: Probability of mutating towards a SimpleRNN-node
    :param train_iterations: Number of training generations
    :param version: Version of the model
    :param unused_cpu: Number of CPUs not used during training
    """
    # Re-configure the config-file
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.game.duration = 60  # 60 seconds should be enough to reach the target from each starting orientation
    cfg.genome.node_add_prob = 0  # Do not change number of hidden nodes
    cfg.genome.node_disable_prob = 0  # Do not change number of hidden nodes
    cfg.population.pop_size = 512
    cfg.population.compatibility_thr = 1.  # Very small since all architectures have strictly one hidden node
    
    # Let inputs apply to configuration
    cfg.genome.rnn_prob_gru = prob_gru
    cfg.genome.rnn_prob_gru_nr = prob_gru_nr
    cfg.genome.rnn_prob_gru_nu = prob_gru_nu
    cfg.genome.rnn_prob_simple_rnn = prob_simple_rnn
    cfg.evaluation.fitness = fitness
    cfg.update()
    
    # Create the population
    folder = get_folder(experiment_id=4)
    name = get_name(cfg=cfg, version=version)
    pop = Population(
            name=name,
            config=cfg,
            folder_name=folder,
            use_backup=False,
    )
    
    # Make sure that all of the genomes in the initial population have exactly one hidden node
    if pop.generation == 0:
        for g in pop.population.values():
            g.mutate_add_node(pop.config.genome)
    
    # Give overview of population
    gru = cfg.genome.rnn_prob_gru
    gru_nr = cfg.genome.rnn_prob_gru_nr
    gru_nu = cfg.genome.rnn_prob_gru_nu
    rnn = cfg.genome.rnn_prob_simple_rnn
    msg = f"\n\n\n\n\n===> RUNNING EXPERIMENT 4 FOR THE FOLLOWING CONFIGURATION: <===" \
          f"\n\t> fitness:             {cfg.evaluation.fitness}" \
          f"\n\t> GRU enabled:         {gru > 0}  (probability={round(gru, 2)})" \
          f"\n\t> GRU-NR enabled:      {gru_nr > 0}  (probability={round(gru_nr, 2)})" \
          f"\n\t> GRU-NU enabled:      {gru_nu > 0}  (probability={round(gru_nu, 2)})" \
          f"\n\t> SRU enabled:         {rnn > 0}  (probability={round(rnn, 2)})" \
          f"\n\t> Saving under folder: {folder}" \
          f"\n\t> Training iterations: {train_iterations}\n"
    pop.log(msg)
    
    # Set games used for evaluation
    games_train, games_eval = get_game_ids(experiment_id=4)
    
    # Execute the requested segments
    try:
        train(
                game_config=cfg,
                games=games_train,
                iterations=train_iterations,
                population=pop,
                unused_cpu=unused_cpu,
        )
        
        # Evaluate the trained population
        evaluate(
                game_config=cfg,
                games=games_eval,
                population=pop,
                unused_cpu=unused_cpu,
        )
        training_overview(
                population=pop,
        )
        visualize_genome(
                genome=pop.best_genome,
                population=pop,
        )
        
        # Perform GRU-analysis if population is NEAT-GRU
        if gru > 0:
            gru_analysis(
                    population=pop,
                    unused_cpu=unused_cpu,
                    experiment_id=4,
            )
            monitor(
                    game_cfg=cfg,
                    game_id=games_eval[0],
                    population=pop,
                    genome=pop.best_genome,
            )
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
    finally:
        process_killer('run_population.py')  # Close all the terminated files


def train(population: Population,
          game_config: Config,
          games: list,
          iterations: int,
          unused_cpu: int = 0,
          save_interval: int = 10,
          ):
    """Train the population on the requested number of iterations. Manual adaptation of main's train()."""
    
    population.log("\n===> TRAINING <===\n")
    
    multi_env = get_multi_env(pop=population,
                              game_config=game_config)
    msg = f"Repetitive evaluating on games: {games} for {iterations} iterations"
    population.log(msg, print_result=False)
    
    # Iterate and evaluate over the games
    saved = True
    for iteration in range(iterations):
        # Set and randomize the games
        multi_env.set_games(games, noise=True)
        
        # Prepare the generation's reporters for the generation
        population.reporters.start_generation(gen=population.generation, logger=population.log)
        
        # Fetch the dictionary of genomes
        genomes = list(iteritems(population.population))
        
        # Initialize the evaluation-pool
        pool = mp.Pool(mp.cpu_count() - unused_cpu)
        manager = mp.Manager()
        return_dict = manager.dict()
        
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict))
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        
        # Calculate the fitness from the given return_dict
        fitness = calc_pop_fitness(
                fitness_cfg=population.config.evaluation,
                game_cfg=game_config.game,
                game_obs=return_dict,
                gen=population.generation,
        )
        for i, genome in genomes:
            genome.fitness = fitness[i]
        
        # Gather and report statistics
        best = None
        for g in itervalues(population.population):
            if best is None or g.fitness > best.fitness: best = g
        population.reporters.post_evaluate(population=population.population,
                                           species=population.species,
                                           best_genome=best,
                                           logger=population.log)
        
        # Update the population's best_genome
        genomes = sorted(population.population.items(), key=lambda x: x[1].fitness, reverse=True)
        population.best_fitness[population.generation] = genomes[0][1].fitness
        population.best_genome_hist[population.generation] = genomes[0]
        population.best_genome = best
        
        # Let population evolve
        population.evolve()
        
        # Update the genomes such all have one hidden node
        for g in population.population.values():
            n_hidden, _ = g.size()
            while n_hidden < 1:
                g.mutate_add_connection(population.config.genome)
                n_hidden, _ = g.size()
        
        # End generation
        population.reporters.end_generation(population=population.population,
                                            name=str(population),
                                            species_set=population.species,
                                            logger=population.log)
        
        # Save the population
        if (iteration + 1) % save_interval == 0:
            population.save()
            saved = True
        else:
            saved = False
    
    # Make sure that last iterations saves
    if not saved: population.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prob_gru', type=float, default=0)
    parser.add_argument('--prob_gru_nr', type=float, default=0)
    parser.add_argument('--prob_gru_nu', type=float, default=0)
    parser.add_argument('--prob_simple_rnn', type=float, default=0)
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--unused_cpu', type=int, default=2)
    args = parser.parse_args()
    
    main(
            fitness=D_DISTANCE,
            prob_gru=args.prob_gru,
            prob_gru_nr=args.prob_gru_nr,
            prob_gru_nu=args.prob_gru_nu,
            prob_simple_rnn=args.prob_simple_rnn,
            train_iterations=args.iterations,
            version=args.version,
            unused_cpu=args.unused_cpu,
    )
