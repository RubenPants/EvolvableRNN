"""
simple.py

Simple spawning function, sampling one target at random each time it is called.
"""
from random import choice

from configs.game_config import GameConfig


class SpawnSimple:
    """Keeps a set of target-locations (tuples) and picks one randomly each call."""
    
    __slots__ = {
        'locations',
    }
    
    def __init__(self):
        self.locations = set()  # Keep a set of possible valid points
    
    def __call__(self, game_config: GameConfig, player_pos: tuple):
        return choice(tuple(self.locations))
    
    def __str__(self):
        return f"SpawnSimple(n_locations={len(self.locations)})"
    
    def __repr__(self):
        return str(self)
    
    def add_location(self, loc: tuple):
        """Add a new location to the set."""
        assert type(loc) == tuple
        self.locations.add(loc)
    
    def reset(self):
        pass
