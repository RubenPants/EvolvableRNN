"""
env_visualizing.py

Visualize the performance of a population.
"""
import multiprocessing as mp

from neat.six_util import iteritems
from tqdm import tqdm

from config import Config
from environment.game import get_game
from environment.env_multi import get_multi_env
from population.population import Population
from population.utils.genome import Genome
from population.utils.visualizing.population_visualizer import create_blueprints, create_traces
from utils.myutils import get_subfolder


class VisualizingEnv:
    """This class is responsible evaluating and evolving the population across a set of games."""
    
    __slots__ = {
        'game_config', 'unused_cpu',
        'games', 'batch_size',
    }
    
    def __init__(self, game_config: Config, games: list, unused_cpu: int = 0):
        """ The evaluator is given a population which it then evaluates using the MultiEnvironment. """
        # Load in current configuration
        self.game_config = game_config
        self.unused_cpu = unused_cpu
        
        #  Create a list of all the possible games
        self.games = None
        self.batch_size = 0
        self.set_games(games)
    
    def set_games(self, games: list):
        """
        Set the game-IDs that will be used to evaluate the population. The full game-set as defined by the configuration
        file will be used if games=None.

        :param games: List of integers
        """
        self.games = games
        self.batch_size = len(games)
    
    def blueprint_genomes(self, pop: Population, parallel: bool = True):
        """
        Create blueprints for all the requested mazes.

        :param pop: Population object
        :param parallel: Evaluate the population in parallel
        """
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        if len(self.games) > 100:
            raise Exception("It is not advised to evaluate on more than 100 at once")
        
        multi_env.set_games(self.games, noise=False)
        
        # Fetch the dictionary of genomes
        genomes = list(iteritems(pop.population))
        
        if parallel:
            # Initialize the evaluation-pool
            pool = mp.Pool(mp.cpu_count() - self.unused_cpu)
            manager = mp.Manager()
            return_dict = manager.dict()
            
            # Evaluate the genomes
            for genome in genomes:
                pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict))
            pool.close()  # Close the pool
            pool.join()  # Postpone continuation until everything is finished
        else:  # Evaluate sequentially
            return_dict = dict()
            for genome in genomes:
                multi_env.eval_genome(genome, return_dict)
        
        # Create blueprint of final result
        game_objects = [get_game(g, cfg=self.game_config) for g in self.games]
        path = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/",
                             'images')
        path = get_subfolder(path, 'games')
        create_blueprints(
                final_observations=return_dict,
                games=game_objects,
                gen=pop.generation,
                save_path=path,
        )
    
    def trace_genomes(self, pop: Population, given_genome: Genome = None, parallel: bool = True):
        """
        Create blueprints that contain the walking-traces for all the requested mazes.

        :param pop: Population object
        :param given_genome: Single genomes for which the trace must be made
        :param parallel: Create the traces in parallel
        """
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        if len(self.games) > 20 and given_genome is None:
            raise Exception("It is not advised to evaluate on more than 20 at once")
        elif len(self.games) > 100:
            raise Exception("It is not advised to evaluate on more than 100 at once")
        
        # Set the games for which traces will be made
        multi_env.set_games(self.games, noise=False)
        
        # Fetch the dictionary of genomes
        genomes = [(given_genome.key, given_genome)] if given_genome else list(iteritems(pop.population))
        
        if parallel:
            # Initialize the evaluation-pool
            pool = mp.Pool(mp.cpu_count() - self.unused_cpu)
            manager = mp.Manager()
            return_dict = manager.dict()
            
            # Evaluate the genomes
            for genome in genomes:
                pool.apply_async(func=multi_env.trace_genome, args=(genome, return_dict))
            pool.close()  # Close the pool
            pool.join()  # Postpone continuation until everything is finished
        else:  # Train sequentially
            return_dict = dict()
            for genome in tqdm(genomes, desc="sequential evaluating"):
                multi_env.trace_genome(genome, return_dict)
        
        # Create blueprint of final result
        game_objects = [get_game(g, cfg=self.game_config) for g in self.games]
        path = get_subfolder(f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/",
                             'images')
        path = get_subfolder(path, 'games')
        create_traces(
                traces=return_dict,
                games=game_objects,
                gen=pop.generation,
                save_path=path,
                save_name=f'trace_{given_genome.key}' if given_genome else 'trace',
        )
