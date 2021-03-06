"""
game.py

Game class which contains the player, target, and all the walls.
"""
from math import cos, sin

import matplotlib.collections as mc
import pylab as plt
from numpy import asarray, pi

from config import Config
from environment.robot import Robot
from utils.dictionary import *
from utils.myutils import load_pickle, store_pickle
from utils.vec2d import Vec2d


class Game:
    """
    A game environment is built up from the following segments:
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    
    __slots__ = {
        'bot_config', 'done', 'game_config', 'id', 'init_distance', 'noise', 'player', 'player_angle_noise',
        'save_path', 'score', 'silent', 'spawn_function', 'steps_taken', 'stop_if_reached', 'target', 'wall_bound',
        'x_axis', 'y_axis'
    }
    
    def __init__(self,
                 game_id: int,
                 config: Config,
                 player_noise: float = 0,
                 noise: bool = True,
                 overwrite: bool = False,
                 save_path: str = '',
                 silent: bool = True,
                 spawn_func=None,
                 stop_if_reached: bool = True,
                 wall_bound: bool = True,
                 ):
        """
        Define a new game.

        :param game_id: Game id
        :param config: Configuration file (only needed to pass during creation)
        :param player_noise: The maximum noise added to the player's initial location
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param save_path: Save and load the game from different directories
        :param silent: Do not print anything
        :param spawn_func: Function that determines which target-position should spawn
        :param stop_if_reached: Stop the simulation when agent reaches target
        :param wall_bound: Bound the position of the agent to be within the walls of the game
        """
        assert type(game_id) == int
        
        # Set the game's configuration
        self.bot_config = config.bot
        self.game_config = config.game
        
        # Environment specific parameters
        self.noise: bool = noise  # Add noise to the game-environment
        self.silent: bool = silent  # True: Do not print out statistics
        self.save_path: str = save_path if save_path else 'environment/games_db/'
        self.wall_bound: bool = wall_bound  # Permit robot to go outside of the boundaries
        self.stop_if_reached: bool = stop_if_reached  # Terminate the simulation ones the target is found
        self.player_angle_noise: float = player_noise  # The noise added to the player's initial orientation
        
        # Placeholders for parameters
        self.done: bool = False  # Game has finished
        self.id: int = game_id  # Game's ID-number
        self.init_distance: float = 0  # Denotes the initial distance from target
        self.player: Robot = None  # Candidate-robot
        self.score: int = 0  # Denotes the number of targets found
        self.spawn_function = None  # Function determining which targets to spawn
        self.steps_taken: int = 0  # Number of steps taken by the agent
        self.target: Vec2d = None  # Target-robot
        self.x_axis: int = 0  # Width of the game
        self.y_axis: int = 0  # Height of the game
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load():
            assert spawn_func is not None
            self.create_empty_game(spawn_func)
    
    def __str__(self):
        return f"game_{self.id:05d}"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def close(self):
        """Final state of the agent's statistics."""
        return {
            D_DIST_TO_TARGET: self.get_distance_to_target(),
            D_DONE:           self.done,
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_SCORE:          self.score,
            D_TIME_TAKEN:     self.steps_taken / self.game_config.fps,
            D_INIT_DIST:      self.init_distance,
        }
    
    def get_observation(self):
        """Get the current observation of the game in the form of a dictionary."""
        return {
            D_DONE:        self.done,
            D_SENSOR_LIST: self.player.get_sensor_readings(),
        }
    
    def randomize(self):
        """Randomize the maze."""
        self.player.randomize(max_noise=self.player_angle_noise)
        self.spawn_function.randomize()
        self.sample_target()
    
    def reset(self):
        """Reset the game and return initial observations."""
        self.done = False
        self.score = 0
        self.steps_taken = 0
        self.spawn_function.reset()
        self.sample_target()
        self.player.reset(noise=self.noise)
        obs = self.get_observation()
        self.init_distance = self.get_distance_to_target()  # The sensor-values must be read in first!
        return obs
    
    def step(self, l: float, r: float):
        """
        Progress one step in the game.

        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        dt = 1.0 / self.game_config.fps
        return self.step_dt(dt=dt, l=l, r=r)
    
    def step_dt(self, dt: float, l: float, r: float):
        """
        Progress one step in the game based on a predefined delta-time. This method should only be used for debugging or
        visualization purposes.

        :param dt: Delta time
        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        self.steps_taken += 1
        self.player.drive(dt, lw=l, rw=r)
        
        # Check if player is not outside of playing-field if the game is wall-bound
        if self.wall_bound and \
                (not (self.player.radius <= self.player.pos[0] <= self.x_axis - self.player.radius) or
                 not (self.player.radius <= self.player.pos[1] <= self.y_axis - self.player.radius)):
            self.player.set_back()
        
        # Check if target reached
        if self.get_distance_to_target() <= self.game_config.target_reached:
            self.score += 1
            if self.stop_if_reached:
                self.done = True
            else:
                self.sample_target()
        
        # Return the current observations
        return self.get_observation()
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_empty_game(self, spawn_func):
        """Create an empty game."""
        self.x_axis = self.game_config.x_axis
        self.y_axis = self.game_config.y_axis
        self.spawn_function = spawn_func
        self.player = Robot(game=self)
        self.set_player_init_angle(a=pi / 2)
        self.set_player_init_pos(p=Vec2d(self.game_config.x_axis / 2, self.game_config.y_axis / 2))
        
        # Save the new game
        self.save()
        if not self.silent: print(f"New game created under id: {self.id}")
    
    def get_distance_to_target(self):
        """Get the distance between robot and target."""
        return (self.target - self.player.pos).get_length()
    
    def sample_target(self):
        """Sample a target from the target_list."""
        self.target = Vec2d().load_tuple(self.spawn_function())
    
    def set_player_init_angle(self, a: float):
        """Set a new initial angle for the player."""
        self.player.set_init_angle(a=a)
    
    def set_player_init_pos(self, p: Vec2d):
        """Set a new initial position for the player."""
        self.player.set_init_pos(p=p)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def save(self):
        """Save the current state's state."""
        persist_dict = dict()
        persist_dict.update({D_X_AXIS: self.x_axis})
        persist_dict.update({D_Y_AXIS: self.y_axis})
        persist_dict.update({D_WALL_BOUND: self.wall_bound})
        persist_dict.update({D_TARGET_REACHED: self.stop_if_reached})
        persist_dict.update({D_ANGLE: self.player.init_angle})  # Initial angle of player
        persist_dict.update({D_ANGLE_NOISE: self.player_angle_noise})  # Noise added to the initial angle of the player
        persist_dict.update({D_POS: tuple(self.player.init_pos)})  # Initial position of player
        persist_dict.update({D_SPAWN_F: self.spawn_function})  # Function deciding on which target to use
        store_pickle(persist_dict, f'{self.save_path}{self}')
    
    def load(self):
        """Load in a game, specified by its current id and return True if successful."""
        try:
            game = load_pickle(f'{self.save_path}{self}')
            self.x_axis = game.get(D_X_AXIS)
            self.y_axis = game.get(D_Y_AXIS)
            self.wall_bound = game.get(D_WALL_BOUND)
            self.stop_if_reached = game.get(D_TARGET_REACHED)
            self.player = Robot(game=self)  # Create a dummy-player to set values on
            self.set_player_init_angle(game.get(D_ANGLE))
            self.player_angle_noise = game.get(D_ANGLE_NOISE)
            self.set_player_init_pos(Vec2d().load_tuple(game.get(D_POS)))
            self.spawn_function = game.get(D_SPAWN_F)
            self.spawn_function.reset()
            self.sample_target()
            if not self.silent: print(f"Existing game loaded with id: {self.id}")
            return True
        except FileNotFoundError:
            return False
    
    def get_blueprint(self, ax=None, show_player: bool = False, annotate: bool = True):
        """The blueprint map of the board (matplotlib Figure)."""
        if not ax: fig, ax = plt.subplots()
        
        # Draw the (implicit) boundary walls
        if self.wall_bound:
            walls = []
            corners = asarray([(0, 0), (0, self.y_axis), (self.x_axis, self.y_axis), (self.x_axis, 0)])
            for c in range(4):
                walls.append([corners[c], corners[(c + 1) % 4]])
            lc = mc.LineCollection(walls, linewidths=5, colors='k')
            ax.add_collection(lc)
        
        # Add all possible targets to map
        if "locations" in self.spawn_function.__slots__:
            for i, t in enumerate(self.spawn_function.locations):
                plt.plot(t[0], t[1], 'go')
                if annotate and type(self.spawn_function.locations) == list:
                    plt.annotate(str(i + 1), xy=(t[0] + 0.1, t[1] + 0.1))
        
        # Add player to map if requested
        if show_player:
            x = self.player.init_pos[0]
            y = self.player.init_pos[1]
            dx = cos(self.player.noisy_init_angle)
            dy = sin(self.player.noisy_init_angle)
            plt.arrow(x, y, dx, dy, head_width=0.1, length_includes_head=True)
        
        # Adjust the boundaries
        plt.xlim(0, self.x_axis)
        plt.ylim(0, self.y_axis)
        
        # Return the figure in its current state
        return ax


def get_game(i: int, cfg: Config = None, noise: bool = True):
    """
    Create a game-object.
    
    :param i: Game-ID
    :param cfg: Config object
    :param noise: Add noise to the game
    :return: Game or GameCy object
    """
    config = cfg if cfg else Config()
    return Game(
            game_id=i,
            config=config,
            noise=noise,
            silent=True,
    )
