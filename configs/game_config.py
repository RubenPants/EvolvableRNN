"""
game_config.py

Configuration file relating to the game.
"""
from configs.base_config import BaseConfig


class GameConfig(BaseConfig):
    """Game-specific configuration parameters."""
    
    __slots__ = {
        'duration', 'fps', 'p2m', 'target_reached', 'x_axis', 'y_axis',
    }
    
    def __init__(self):
        # Number of seconds it takes for one game to complete  [def=60]  TODO
        self.duration: int = 60
        # Number of frames each second  [def=10]
        self.fps: int = 10
        # Pixel-to-meters: number of pixels that represent one meter  [def=50]
        self.p2m: int = 35
        # Target is reached when within this range, expressed in meters  [def=.5]
        self.target_reached: float = .5
        # Number of meters the x-axis represents  [def=14]
        self.x_axis: int = 14
        # Number of meters the y-axis represents  [def=14]
        self.y_axis: int = 14
