"""
env_training.py

Train and evaluate the population on a provided set of training games.
"""
import multiprocessing as mp
import warnings

from neat.six_util import iteritems, itervalues
from tqdm import tqdm

from config import Config
from environment.env_multi import get_multi_env
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness


class TrainingEnv:
    """This class is responsible evaluating and evolving the population across a set of games."""
    
    __slots__ = {
        'game_config', 'unused_cpu',
        'games', 'batch_size',
    }
    
    def __init__(self, game_config: Config, games: list, unused_cpu: int = 0):
        """
        The evaluator is given a population which it then evaluates using the MultiEnvironment.
        
        :param game_config: Config file used to configure the variable parts of the gaming environment
        :param games: List of game-IDs on which the training will proceed
        :param unused_cpu: Number of CPU-cores NOT utilized during training
        """
        # Load in current configuration
        self.game_config = game_config
        self.unused_cpu = unused_cpu
        
        #  Create a list of all the possible games
        if len(games) > 20:
            warnings.warn(f"It is not advised to train a population on more than 20 games ({len(games)})")
        self.games = None
        self.batch_size = 0
        self.set_games(games)
    
    def set_games(self, games: list):
        """
        Set the game-IDs that will be used to evaluate the population. The full game-set as defined by the configuration
        file will be used if games=None.
        
        :param games: List of integers representing the game-IDs
        """
        self.games = games
        self.batch_size = len(games)
    
    def evaluate_and_evolve(self,
                            pop: Population,
                            n: int = 1,
                            parallel=True,
                            save_interval: int = 1,
                            ):
        """
        Evaluate the population on the same set of games.
        
        :param pop: Population object
        :param n: Number of generations
        :param parallel: Parallel the code (disable parallelization for debugging purposes)
        :param save_interval: Indicates how often a population gets saved
        """
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        msg = f"Repetitive evaluating on games: {self.games} for {n} iterations"
        pop.log(msg, print_result=False)
        
        # Iterate and evaluate over the games
        saved = True
        for iteration in range(n):
            # Set and randomize the games
            multi_env.set_games(self.games, noise=True)
            
            # Prepare the generation's reporters for the generation
            pop.reporters.start_generation(gen=pop.generation, logger=pop.log)
            
            # Fetch the dictionary of genomes
            genomes = list(iteritems(pop.population))
            
            if parallel:
                # Initialize the evaluation-pool
                pool = mp.Pool(mp.cpu_count() - self.unused_cpu)
                manager = mp.Manager()
                return_dict = manager.dict()
                
                for genome in genomes:
                    pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict))
                pool.close()  # Close the pool
                pool.join()  # Postpone continuation until everything is finished
            else:
                return_dict = dict()
                for genome in tqdm(genomes, desc="sequential training"):
                    multi_env.eval_genome(genome, return_dict)
            
            # Calculate the fitness from the given return_dict
            fitness = calc_pop_fitness(
                    fitness_cfg=pop.config.evaluation,
                    game_cfg=self.game_config.game,
                    game_obs=return_dict,
                    gen=pop.generation,
            )
            for i, genome in genomes:
                genome.fitness = fitness[i]
            
            # Gather and report statistics
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
            
            # End generation
            pop.reporters.end_generation(population=pop.population,
                                         name=str(pop),
                                         species_set=pop.species,
                                         logger=pop.log)
            
            # Save the population
            if (iteration + 1) % save_interval == 0:
                pop.save()
                saved = True
            else:
                saved = False
        
        # Make sure that last iterations saves
        if not saved: pop.save()
