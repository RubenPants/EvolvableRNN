"""
random.py

Spawning function that will always spawn its targets completely random.
"""
from math import cos, pi, sin
from random import random

from configs.game_config import GameConfig


class SpawnRandom:
    """Spawn random target positions based on the robot's current position."""
    
    __slots__ = {}
    
    def __init__(self):
        pass  # No state is preserved
    
    def __call__(self, game_config: GameConfig, player_pos: tuple):
        return get_valid_spawn(game_config=game_config, pos=player_pos)
    
    def __str__(self):
        return f"SpawnRandom()"
    
    def __repr__(self):
        return str(self)
    
    def reset(self):
        pass


def get_valid_spawn(game_config: GameConfig, pos):
    """
    Spawn a target on a valid position based on the given parameters. A valid spawn will position itself on the maze,
    with a distance of 4 to 8 meters of the given position.
    
    :param game_config: Used to know maze-dimensions
    :param pos: Position around which the target is spawn.
    """
    valid = False
    target_pos = None
    invalid_count = 0
    while not valid:
        random_radius = 4 + 4 * random()
        random_angle = 2 * pi * random()
        target_pos = (pos[0] + random_radius * cos(random_angle), pos[1] + random_radius * sin(random_angle))
        valid = in_maze(game_config=game_config, pos=target_pos)
        invalid_count += 1
        if invalid_count >= 100:
            break
    
    # Robot is most likely out of maze-area, spawn randomly new target
    if not valid:
        random_radius = 4 + 4 * random()
        random_angle = 2 * pi * random()
        target_pos = (game_config.x_axis / 2 + random_radius * cos(random_angle),
                      game_config.y_axis / 2 + random_radius * sin(random_angle))
    
    return target_pos


def in_maze(game_config: GameConfig, pos, boundary: float = 2):
    """Checks if the given position is within the maze (+some additional boundary)."""
    x_min = 0 + boundary
    x_max = game_config.x_axis - boundary
    y_min = 0 + boundary
    y_max = game_config.x_axis - boundary
    return (x_min <= pos[0] <= x_max) and (y_min <= pos[1] <= y_max)
