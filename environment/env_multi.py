"""
env_multi.py

Environment where a single genome gets evaluated over multiple games. This environment will be called in a process.
"""
import sys

import numpy as np

from config import Config
from environment.game import get_game
from population.population import Population
from population.utils.network_util.feed_forward_net import make_net
from utils.dictionary import D_DONE, D_SENSOR_LIST


class MultiEnvironment:
    """This class provides an environment to evaluate a single genome on multiple games."""
    
    __slots__ = {
        'batch_size', 'game_config', 'games', 'max_steps', 'pop_config',
    }
    
    def __init__(self,
                 game_config: Config,
                 pop_config: Config,
                 ):
        """
        Create an environment in which the genomes get evaluated across different games.
        
        :param game_config: Config file for game-creation
        :param pop_config: Config file specifying how genome's network will be made
        """
        self.batch_size = 0
        self.games = np.asarray([])
        self.max_steps = game_config.game.duration * game_config.game.fps
        self.game_config = game_config
        self.pop_config = pop_config
    
    def eval_genome(self,
                    genome,
                    return_dict=None,
                    ):
        """
        Evaluate a single genome in a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param return_dict: Dictionary used to return observations corresponding the genome
        """
        # Split up genome by id and genome itself
        genome_id, genome = genome
        
        # Ask for each of the games the starting-state
        states = np.asarray([g.reset()[D_SENSOR_LIST] for g in self.games])
        
        # Finished-state for each of the games is set to false
        finished = np.repeat(False, self.batch_size)
        
        # Create the network used to query on, initialize it with the first-game's readings (good approximation)
        net = make_net(genome=genome,
                       genome_config=self.pop_config.genome,
                       batch_size=self.batch_size,
                       initial_read=states[0],
                       )
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
            actions = net(states)
            
            # Check if each game received an action
            assert len(actions) == len(self.games)
            
            for i, (g, a, f) in enumerate(zip(self.games, actions, finished)):
                # Ignore if game has finished
                if not f:
                    # Proceed the game with one step, based on the predicted action
                    obs = g.step(l=a[0], r=a[1])
                    finished[i] = obs[D_DONE]
                    
                    # Update the candidate's current state
                    states[i] = obs[D_SENSOR_LIST]
            
            # Stop if agent reached target in all the games
            if all(finished): break
            step_num += 1
        
        # Return the final observations
        if return_dict is not None: return_dict[genome_id] = [g.close() for g in self.games]
    
    def trace_genome(self,
                     genome,
                     return_dict=None,
                     ):
        """
        Get the trace of a single genome for a pre-defined game-environment. Due to performance reasons, only one
        trace-point is saved each second of simulation-time.
        
        :param genome: Tuple (genome_id, genome_class)
        :param return_dict: Dictionary used to return the traces corresponding the genome-game combination
        """
        # Split up genome by id and genome itself
        genome_id, genome = genome
        
        # Ask for each of the games the starting-state
        states = np.asarray([g.reset()[D_SENSOR_LIST] for g in self.games])
        
        # Initialize the traces
        traces = [[g.player.pos.get_tuple()] for g in self.games]
        
        # Finished-state for each of the games is set to false
        finished = np.repeat(False, self.batch_size)
        
        # Create the network used to query on, initialize it with the first-game's readings (good approximation)
        net = make_net(genome=genome,
                       genome_config=self.pop_config.genome,
                       batch_size=self.batch_size,
                       initial_read=states[0],
                       )
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
            actions = net(states)
            
            # Check if each game received an action
            assert len(actions) == len(self.games)
            
            for i, (g, a, f) in enumerate(zip(self.games, actions, finished)):
                # Do not advance the player if target is reached
                if f:
                    if step_num % self.game_config.game.fps == 0: traces[i].append(g.player.pos.get_tuple())
                    continue
                
                # Proceed the game with one step, based on the predicted action
                obs = g.step(l=a[0], r=a[1])
                finished[i] = obs[D_DONE]
                
                # Update the candidate's current state
                states[i] = obs[D_SENSOR_LIST]
                
                # Update the trace
                if step_num % self.game_config.game.fps == 0: traces[i].append(g.player.pos.get_tuple())
            
            # Next step
            step_num += 1
        
        # Return the final observations
        if return_dict is not None: return_dict[genome_id] = traces
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def set_games(self, games: list, noise: bool = False):
        """
        Set the games-set with new games.
        
        :param games: List of Game-IDs
        :param noise: Add noise to the games
        """
        self.games = np.asarray([get_game(g, cfg=self.game_config, noise=noise) for g in games])
        self.batch_size = len(games)
        if noise: [g.randomize() for g in self.games]


def get_multi_env(pop: Population, game_config: Config):
    """Create a multi-environment used to evaluate a population on."""
    if 'linux' in sys.platform:
        from environment.cy.env_multi_cy import MultiEnvironmentCy
        return MultiEnvironmentCy(
                game_config=game_config,
                pop_config=pop.config,
        )
    elif 'win32' in sys.platform:
        return MultiEnvironment(
                game_config=game_config,
                pop_config=pop.config,
        )
    else:
        raise Exception(f"Platform {sys.platform} not yet supported!")
