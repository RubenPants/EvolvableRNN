"""
bot_config.py

Robot configuration file.
"""
from configs.base_config import BaseConfig


class BotConfig(BaseConfig):
    """Robot-specific configuration parameters."""
    
    __slots__ = {
        'driving_speed', 'radius', 'turning_speed',
        'angular_dir', 'delta_dist_enabled', 'dist_enabled',
    }
    
    def __init__(self):
        # Maximal driving speed (driving straight) of the robot expressed in m/s  [def=0.5]
        self.driving_speed: float = 0.5
        # Radius of the bot expressed in meters  [def=0.085]
        self.radius: float = 0.1
        # Maximal turning speed of the robot expressed in radians per second  [def=2.5=0.5/0.2]
        self.turning_speed: float = 2.5
        
        # Sensor-configurations  TODO
        # The clockwise directions for the angular sensors  [def=[]]
        self.angular_dir = []
        # The delta-distance sensor  [def=False]
        self.delta_dist_enabled = False
        # The delta-distance sensor  [def=True]
        self.dist_enabled = True
