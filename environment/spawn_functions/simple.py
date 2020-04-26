"""
simple.py

Simple spawning function, sampling one target at random each time it is called.
"""
from random import shuffle
from warnings import warn

from configs.game_config import GameConfig


class SpawnSimple:
    """Keeps a set of target-locations (tuples) and picks one randomly each call."""
    
    __slots__ = {
        'train', 'location_idx', 'locations', 'game_config',
    }
    
    def __init__(self, game_config: GameConfig, train: bool = True):
        # Set spawning parameters
        self.game_config = game_config
        self.location_idx = 0
        self.locations = []
        self.train = train
    
    def __call__(self):
        loc = self.locations[self.location_idx]
        self.location_idx = (self.location_idx + 1) % len(self.locations)
        return loc
    
    def __str__(self):
        return f"SpawnSimple(n_targets={len(self.locations)})"
    
    def __repr__(self):
        return str(self)
    
    def add_location(self, loc: tuple):
        """Add a new location to the set."""
        assert type(loc) == tuple
        self.locations.append(loc)
    
    def randomize(self):
        if self.train:
            shuffle(self.locations)  # Keep the same targets but change order
        else:
            warn("Tried to randomize an evaluation game")
    
    def reset(self):
        self.location_idx = 0
