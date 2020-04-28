"""
range.py

Spawning function that will always spawn its targets random within a range of the initial position.
"""
from math import sqrt
from random import random
from warnings import warn

from configs.game_config import GameConfig


class SpawnRange:
    """Spawn random target positions based on the robot's current position."""
    
    __slots__ = {
        'train', 'location_idx', 'locations', 'game_config', 'r_max', 'r_min',
    }
    
    def __init__(self, game_config: GameConfig, r_max: float = 10, r_min: float = 2, train: bool = True):
        # Set spawning parameters
        self.game_config = game_config
        self.location_idx = 0
        self.locations = []
        self.r_max = r_max  # Maximum range a target may spawn from the initial position
        self.r_min = r_min  # Minimal range a target may spawn from the initial position
        self.train = train
        
        # Create initial locations
        self.create_location()
    
    def __call__(self):
        loc = self.locations[self.location_idx]
        self.location_idx = (self.location_idx + 1) % len(self.locations)
        return loc
    
    def __str__(self):
        return f"SpawnRandom(n_targets={len(self.locations)})"
    
    def __repr__(self):
        return str(self)
    
    def create_location(self):
        """Create 'size' random target-locations. Only one location is created each call."""
        # Start from a clean slate
        self.locations = []
        
        # Initialize with a valid position based on the robot's initial position
        init_pos = (self.game_config.x_axis / 2, self.game_config.y_axis / 2)  # Agent always starts in the middle
        self.locations.append(get_valid_spawn(self.game_config, pos=init_pos, r_max=self.r_max, r_min=self.r_min))
    
    def randomize(self):
        if self.train:
            self.create_location()
        else:
            warn("Tried to randomize an evaluation game")
    
    def reset(self):
        self.location_idx = 0


def get_valid_spawn(game_config: GameConfig, pos, r_max: float, r_min: float):
    """
    Spawn a new target that is within the maze (+some additional boundary) and at least 1m away from the agent.

    :param game_config: Used to know maze-dimensions
    :param pos: Position around which the target is spawn.
    :param r_max: Maximum range a target may spawn from the initial position
    :param r_min: Minimal range a target may spawn from the initial position
    """
    target_pos = None
    while target_pos is None:
        # Get random position within the maze
        random_x = game_config.x_axis * random()
        random_y = game_config.y_axis * random()
        
        # Check if target is between r_min and r_max from given position
        d = sqrt((pos[0] - random_x) ** 2 + (pos[1] - random_y) ** 2)
        if r_min <= d <= r_max: target_pos = (random_x, random_y)
    return target_pos
