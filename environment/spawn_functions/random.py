"""
random.py

Spawning function that will always spawn its targets completely random.
"""
from math import sqrt
from random import random
from warnings import warn

from configs.game_config import GameConfig


class SpawnRandom:
    """Spawn random target positions based on the robot's current position."""
    
    __slots__ = {
        'train', 'location_idx', 'locations', 'game_config',
    }
    
    def __init__(self, game_config: GameConfig, train: bool = True):
        # Set spawning parameters
        self.game_config = game_config
        self.location_idx = 0
        self.locations = []
        self.train = train
        
        # Create initial locations
        self.create_locations()
    
    def __call__(self):
        loc = self.locations[self.location_idx]
        self.location_idx = (self.location_idx + 1) % len(self.locations)
        return loc
    
    def __str__(self):
        return f"SpawnRandom(n_targets={len(self.locations)})"
    
    def __repr__(self):
        return str(self)
    
    def create_locations(self, size: int = 20):  # 20 random targets should definitely suffice
        """Create 'size' random target-locations."""
        # Start from a clean slate
        self.locations = []
        
        # Initialize with a valid position based on the robot's initial position
        init_pos = (self.game_config.x_axis / 2, self.game_config.y_axis / 2)  # Agent always starts in the middle
        self.locations.append(get_valid_spawn(game_config=self.game_config, pos=init_pos))
        
        # Fill the rest of locations with suiting locations (based on previous target position)
        for _ in range(size - 1):
            self.locations.append(get_valid_spawn(game_config=self.game_config, pos=self.locations[-1]))
    
    def randomize(self):
        if self.train:
            self.create_locations()
        else:
            warn("Tried to randomize an evaluation game")
    
    def reset(self):
        self.location_idx = 0


def get_valid_spawn(game_config: GameConfig, pos, r: float = 1):
    """
    Spawn a new target that is within the maze (+some additional boundary) and at least 1m away from the agent.
    
    :param game_config: Used to know maze-dimensions
    :param pos: Position around which the target is spawn.
    :param r: Minimum range from which the target must be spawned away from given position
    """
    target_pos = None
    while target_pos is None:
        # Get random position within the maze (+boundary of 2)
        random_x = 2 + (game_config.x_axis - 4) * random()
        random_y = 2 + (game_config.y_axis - 4) * random()
        
        # Check if at least 1m away from given position
        if sqrt((pos[0] - random_x) ** 2 + (pos[1] - random_y) ** 2) > r:
            target_pos = (random_x, random_y)
    
    return target_pos
