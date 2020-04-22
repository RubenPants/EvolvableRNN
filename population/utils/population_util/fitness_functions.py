"""
fitness_functions.py

This file contains multiple possible fitness functions. Each of the fitness functions takes in a dictionary with as key
 the ID of the corresponding candidate, and as value a list of all its final observations (i.e. list of game.close()
 dictionaries). Based on this input, a suitable fitness value for each of the candidates is determined.
"""
import sys

from numpy import clip, mean
from scipy.stats import gmean

from config import GameConfig
from configs.evaluation_config import EvaluationConfig
from utils.dictionary import *

if 'linux' in sys.platform:
    from utils.cy.line2d_cy import Line2dCy as Line
else:
    from utils.line2d import Line2d as Line


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #


def calc_pop_fitness(fitness_cfg: EvaluationConfig, game_cfg: GameConfig, game_obs: dict, gen: int = 0):
    """
    Determine the fitness out of the given game_observation dictionary. This happens in two stages:
     1) Evaluate the candidate's fitness for each of the games individually, thus resolving in a list of floats
         (fitness) for each candidate for each of the games
     2) Combine all the fitness-values of all the games for every individual candidate to get the candidate's overall
         fitness score
    
    :param fitness_cfg: EvaluationConfig object
    :param game_cfg: GameConfig object representing the evaluated set of games' configuration
    :param game_obs: Dictionary containing for each genome the list of all its game.close() results
    :param gen: Current generation of the population
    :return: { genome_key: combined_fitness_float }
    """
    # 1) Evaluate fitness for each of the games
    intermediate_observations = fitness_per_game(
            fitness_cfg=fitness_cfg,
            game_cfg=game_cfg,
            game_obs=game_obs,
            gen=gen,
    )
    
    # 2) Combine the fitness-functions
    return fitness_averaged(fitness_cfg=fitness_cfg, fitness=intermediate_observations)


def fitness_averaged(fitness_cfg: EvaluationConfig, fitness: dict):
    """
    
    :param fitness_cfg: Configuration dictionary that contains a tag specifying in which way the fitness scores get
     combined (min, avg, max)
    :param fitness: { genome_key : [fitness_floats] }
    :return: Adjusted fitness dictionary: { genome_key: combined_fitness_float }
    """
    t = fitness_cfg.fitness_comb
    assert (t in [D_MIN, D_AVG, D_MAX, D_GMEAN])
    f = min if t == D_MIN else max if t == D_MAX else mean if D_AVG else gmean
    for k in fitness.keys(): fitness[k] = f(fitness[k])
    return fitness


def fitness_per_game(fitness_cfg: EvaluationConfig, game_cfg: GameConfig, game_obs: dict, gen: int):
    """
    General fitness-function called by the evaluator environment containing all the possible attributes. Based on the
     given 'tag', determined by the population's config, a suitable fitness function is called.
    
    :param fitness_cfg: EvaluationConfig object
    :param game_cfg: GameConfig object representing the evluated set of games' configuration
    :param game_obs: Dictionary containing for each genome the list of all its game.close() results
    :param gen: Current generation of the population
    :return: Dictionary: { genome_key: [fitness_floats] }
    """
    tag = fitness_cfg.fitness
    if tag == D_DISTANCE:
        return distance(
                game_cfg=game_cfg,
                game_obs=game_obs,
        )
    elif tag == D_DISTANCE_SCORE:
        return distance_score(
                game_cfg=game_cfg,
                game_obs=game_obs,
        )
    elif tag == D_NOVELTY:
        return novelty_search(
                game_obs=game_obs,
                k=fitness_cfg.nn_k,
                safe_zone=fitness_cfg.safe_zone,
        )
    elif tag == D_DIVERSITY:
        return diversity(
                game_cfg=game_cfg,
                game_obs=game_obs,
                gen=gen,
                k=fitness_cfg.nn_k,
        )
    else:
        raise Exception(f"{tag} is not supported")


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #


def distance(game_cfg: GameConfig, game_obs: dict):
    """
    The distance-fitness is two-fold:
     * The distance to target when the game has finished
     * The time spent to reach the target (this sub-score equals zero when no target is found)
    Since 'reaching the target' implies a lower 'time spent to reach', the fitness measure is weighted as follows:
     * Reaching the target = 80%
     * Time spent = 20%
    The reason why this weighting is chosen is since 'reducing time' is only fine-tuning, where the real goal is still
     to find the target in the first place.
    
    :param game_cfg: GameConfig object representing the evaluated set of games' configuration
    :param game_obs: Dictionary containing for each genome the list of all its game.close() results
    :return: { genome_id, [fitness_floats] }
    """
    
    def get_distance_score(d, reached, init_dist):
        """Get a score for the given distance."""
        return 1 if reached else clip_f((1 - (d - game_cfg.target_reached) / init_dist)) ** 2
    
    def get_time_score(time_taken):
        """Get a score based on the time taken by the robot to finish the simulation."""
        return 1 - time_taken / game_cfg.duration
    
    def get_score(d, reached, init_d, time):
        return 0.8 * get_distance_score(d, reached, init_d) + 0.2 * get_time_score(time)
    
    fitness = dict()
    for k, v in game_obs.items():  # Iterate over the candidates
        fitness[k] = [get_score(o[D_DIST_TO_TARGET], o[D_DONE], o[D_INIT_DIST], o[D_TIME_TAKEN]) for o in v]
    return fitness


def distance_score(game_cfg: GameConfig, game_obs: dict):
    """
    Give a score based on the number of targets found plus how much distance there was left until the next target was
    reached. Note that this implies that the final fitness score is greater than 1 once a target is reached.
    
    :param game_cfg: GameConfig object representing the evaluated set of games' configuration
    :param game_obs: Dictionary containing for each genome the list of all its game.close() results
    :return: { genome_id, [fitness_floats] }
    """
    
    def get_distance_score(d, init_dist):
        """Get a score for the given distance."""
        return clip_f((1 - (d - game_cfg.target_reached) / init_dist)) ** 2
    
    def get_score(d, init_d, score):
        return score + get_distance_score(d, init_d)
    
    fitness = dict()
    for k, v in game_obs.items():  # Iterate over the candidates
        fitness[k] = [get_score(o[D_DIST_TO_TARGET], o[D_INIT_DIST], o[D_SCORE]) for o in v]
    return fitness


def diversity(game_cfg: GameConfig, game_obs: dict, gen: int, k: int = 3, safe_zone: float = 1):
    """
    Every end of 10 generations, filter out the most fit candidates based on their distance towards the target,
     otherwise for enforce novelty.
    
    :param game_cfg: GameConfig object representing the evaluated set of games' configuration
    :param game_obs: Dictionary containing for each genome the list of all its game.close() results
    :param gen: Population's current generation
    :param k: The number of neighbours taken into account
    :param safe_zone: The range surrounding a genome in which other neighbours are taken into account
    :return: { genome_id, [fitness_floats] }
    """
    if (gen + 1) % 10 == 0:
        return distance(game_cfg=game_cfg, game_obs=game_obs)
    else:
        return novelty_search(game_obs=game_obs, k=k, safe_zone=safe_zone)


def novelty_search(game_obs: dict, k: int = 3, safe_zone: int = 1):
    """
    Rate a genome based on its novelty. A 'more novel' genomes is further placed away from its peers than another
     genome. This novelty is based on the final position of the genome. A genome gets a perfect score if no other
     genomes are within a 1 meter range of the genome's center or the genome reached the target.
    
    :param game_obs: Dictionary containing for each genome (key) the list of all its game.close() results
    :param k: Number of closest neighbours taken into account
    :param safe_zone: The range surrounding a genome in which other neighbours are taken into account
    :return: { genome_id, [fitness_floats] }
    """
    # For each game, create a dictionary of the genome-id mapped to its position
    position_dict = dict()
    sample = list(game_obs.values())[0]
    for game_id in range(len(sample)): position_dict[game_id] = dict()
    for genome_id, observations in game_obs.items():
        for game_id, observation in enumerate(observations):
            position_dict[game_id][genome_id] = observation[D_POS]
    
    # Define the fitness for each genome at each game
    distance_dict = dict()
    for game_id, positions in position_dict.items():
        distance_dict[game_id] = dict()
        
        # Go over each genome to measure its lengths towards the other genomes
        cache = DistanceCache(safe_zone=safe_zone)
        for genome_id, genome_pos in positions.items():
            # Go over all the other genomes
            dist = set()
            for other_genome_id, other_genome_pos in positions.items():
                if genome_id == other_genome_id: continue
                d = cache.distance(pos1=genome_pos,
                                   pos2=other_genome_pos)
                if d < safe_zone: dist.add(d)
            
            # Add the k neighbours that are closest by
            distance_dict[game_id][genome_id] = sorted(dist)[:k]
    
    # Stitch results together such that each genome is mapped to a fitness-list
    fitness_dict = dict()
    for genome_id in game_obs.keys():
        fitness_dict[genome_id] = []
        for game_id in range(len(sample)):
            score = (sum(distance_dict[game_id][genome_id]) / (safe_zone * k)) ** 2
            assert 0 <= score <= 1.0
            fitness_dict[genome_id].append(score)
    return fitness_dict


def clip_f(v):
    """Clip the value between 0 and 1."""
    return clip(v, a_min=0, a_max=1)


class DistanceCache:
    """Cache for the distance-checks."""
    
    def __init__(self, safe_zone):
        self.distances = dict()
        self.range = safe_zone
    
    def distance(self, pos1, pos2):
        """Determine the distance between two positions. If the"""
        # Check cache
        if pos2 < pos1: pos2, pos1 = pos1, pos2
        l = Line(pos1, pos2)
        if l in self.distances: return self.distances[l]
        
        # Add the distance if the other genome is in the 1m-zone
        self.distances[l] = l.get_length() if l.get_length() < self.range else float('inf')
        return self.distances[l]
