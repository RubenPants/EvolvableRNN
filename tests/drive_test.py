"""
drive_test.py

Tests concerning the game mechanics.
"""
import os
import unittest
from random import random

import numpy as np

from config import Config
from environment.game import Game
from environment.spawn_functions.simple import SpawnSimple
from utils.vec2d import Vec2d

EPSILON = 0.05  # 5 centimeter offset allowed


class GameWallCollision(unittest.TestCase):
    """Test the collision mechanism of the game."""
    
    def test_wall_force(self):
        """> Check if drone cannot force itself through a wall."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        init_pos = game.player.init_pos.__copy__()
        
        # Drive forward for 100 seconds
        for _ in range(50 * game.game_config.fps): game.step(l=1, r=1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, init_pos[0], delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, game.y_axis - game.bot_config.radius, delta=EPSILON)
    
    def test_wall_force_reverse(self):
        """> Check if a drone cannot force itself through a wall while driving backwards."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        init_pos = game.player.init_pos.__copy__()
        
        # Drive forward for 100 seconds
        for _ in range(10 * game.game_config.fps):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, init_pos[0], delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, float(game.bot_config.radius), delta=EPSILON)


class GameDrive(unittest.TestCase):
    """Test the robot's drive mechanics."""
    
    def test_360(self):
        """> Let bot spin 360s and check if position has changed."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        init_pos = game.player.init_pos.__copy__()
        
        # Keep spinning around
        for _ in range(10 * game.game_config.fps):
            game.step(l=-1, r=1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, init_pos[0], delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, init_pos[1], delta=EPSILON)
    
    def test_remain_in_box(self):
        """> Set drone in small box in the middle of the game to check if it stays in this box."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Do 10 different initialized tests
        for _ in range(10):
            # Set player init positions
            game.set_player_init_pos(Vec2d(0.5, 0.5))
            game.player.angle = random() * np.pi
            
            # Run for one twenty seconds
            for _ in range(20 * game.game_config.fps):
                l = random() * 1.5 - 0.5  # [-0.5..1]
                r = random() * 1.5 - 0.5  # [-0.5..1]
                game.step(l=l, r=r)
            
            # Check if bot still in box
            self.assertTrue(0 <= game.player.pos.x <= 1)
            self.assertTrue(0 <= game.player.pos.y <= 1)


def get_game():
    cfg = Config()
    cfg.game.fps = 60
    cfg.game.x_axis = 1
    cfg.game.y_axis = 1
    cfg.update()
    spawn = SpawnSimple(game_config=cfg)
    spawn.add_location((10, 10))  # Dummy location
    game = Game(
            config=cfg,
            game_id=0,
            noise=False,
            overwrite=True,
            save_path="tests/games_db/",
            silent=True,
            spawn_func=spawn,
            wall_bound=True,
    )
    game.set_player_init_pos(Vec2d(0.5, 0.5))
    game.sample_target()
    game.reset()
    return game


def main():
    # Test wall collision
    gwc = GameWallCollision()
    gwc.test_wall_force()
    gwc.test_wall_force_reverse()
    
    # Test driving mechanics
    gd = GameDrive()
    gd.test_360()
    gd.test_remain_in_box()


if __name__ == '__main__':
    unittest.main()
