"""
random_eval.py

Copy of the SpawnRandom function found in random.py, with as only difference that it fixes its targets once created.
"""
from configs.game_config import GameConfig
from environment.spawn_functions.random import get_valid_spawn


class SpawnRandomEval:
    """Create a fixed set of random target positions based on the position of the previous target."""
    
    __slots__ = {
        'location_idx', 'locations',
    }
    
    def __init__(self, game_config: GameConfig, size: int = 10):  # A list-length of 10 should suffice
        self.location_idx = 0
        self.locations = []
        self.create_locations(game_config=game_config, size=size)
    
    def __call__(self, game_config: GameConfig, player_pos: tuple):
        loc = self.locations[self.location_idx]
        self.location_idx = (self.location_idx + 1) % len(self.locations)
        return loc
    
    def __str__(self):
        return f"SpawnRandom(targets={self.locations})"
    
    def __repr__(self):
        return str(self)
    
    def create_locations(self, game_config, size: int):
        """Create 'size' random target-locations."""
        # Initialize with a valid position based on the robot's initial position
        init_pos = (game_config.x_axis / 2, game_config.y_axis / 2)  # Agent always starts in the middle
        self.locations.append(get_valid_spawn(game_config=game_config, pos=init_pos))
        
        # Fill the rest of locations with suiting locations (based on previous target position)
        for _ in range(size - 1):
            self.locations.append(get_valid_spawn(game_config=game_config, pos=self.locations[-1]))
    
    def reset(self):
        self.location_idx = 0
