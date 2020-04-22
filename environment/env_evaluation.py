"""
evaluation_env.py

Evaluate a certain set of genomes on the evaluation mazes.
"""
import multiprocessing as mp

from tqdm import tqdm

from config import Config
from environment.env_training import get_multi_env
from population.population import Population
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.dictionary import *


class EvaluationEnv:
    """ This class is responsible evaluating the population across a set of games. """
    
    __slots__ = {
        'game_config', 'unused_cpu',
        'games', 'batch_size',
    }
    
    def __init__(self, game_config: Config, games: list, unused_cpu: int = 0):
        """
        The evaluator used to evaluate the population.
        
        :param game_config: Configuration file for the games
        :param games: List of integers representing the game-IDs
        :param unused_cpu: Number of CPU-cores NOT utilized during training
        """
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
    
    def evaluate_genome_list(self,
                             genome_list,
                             pop: Population,
                             parallel: bool = True,
                             ):
        """
        Evaluate the population for a single evaluation-process.

        :param genome_list: List of genomes that will be evaluated
        :param pop: The population to which the genomes belong (used to setup the network and query the config)
        :param parallel: Evaluate the given genomes in parallel
        """
        # Create the environment which is responsible for evaluating the genomes
        multi_env = get_multi_env(pop=pop, game_config=self.game_config)
        
        # Evaluate on all the games
        multi_env.set_games(self.games, noise=False)
        
        # Fetch requested genomes
        genomes = [(g.key, g) for g in genome_list]
        
        if parallel:
            pool = mp.Pool(mp.cpu_count() - self.unused_cpu)
            manager = mp.Manager()
            return_dict = manager.dict()
            
            for genome in genomes:
                pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict))
            pool.close()  # Close the pool
            pool.join()  # Postpone continuation until everything is finished
        else:
            return_dict = dict()
            for genome in tqdm(genomes, desc="sequential evaluating"):
                multi_env.eval_genome(genome, return_dict)
        
        # Create the evaluation for each of the genomes
        eval_result = dict()
        for k in return_dict.keys():
            # Create answer based on game.close()
            eval_result[str(k)] = create_answer(return_dict[k])
            
            # Append fitness to answer
            eval_result[str(k)][D_FITNESS] = calc_pop_fitness(fitness_cfg=pop.config.evaluation,
                                                              game_cfg=pop.config.game,
                                                              game_obs={k: return_dict[k]},
                                                              gen=pop.generation)[k]
        pop.add_evaluation_result(eval_result)


def create_answer(games: list):
    answer = dict()
    
    answer[D_FINISHED] = round(100 * len([g for g in games if g[D_DONE]]) / len(games), 2)
    
    answer[D_SCORE_MIN] = round(min([g[D_SCORE] for g in games]), 2)
    answer[D_SCORE_AVG] = round(sum([g[D_SCORE] for g in games]) / len(games), 2)
    answer[D_SCORE_MAX] = round(max([g[D_SCORE] for g in games]), 2)
    
    answer[D_DISTANCE_MIN] = round(min([g[D_DIST_TO_TARGET] for g in games]), 2)
    answer[D_DISTANCE_AVG] = round(sum([g[D_DIST_TO_TARGET] for g in games]) / len(games), 2)
    answer[D_DISTANCE_MAX] = round(max([g[D_DIST_TO_TARGET] for g in games]), 2)
    
    answer[D_TIME_MIN] = round(min([g[D_TIME_TAKEN] for g in games]), 2)
    answer[D_TIME_AVG] = round(sum([g[D_TIME_TAKEN] for g in games]) / len(games), 2)
    answer[D_TIME_MAX] = round(max([g[D_TIME_TAKEN] for g in games]), 2)
    return answer
