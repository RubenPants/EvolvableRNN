"""
sensor_test_cy.py

Test of all the sensors.
"""
import os
import unittest

from numpy import pi, sqrt

from config import Config
from environment.cy.game_cy import GameCy
from environment.spawn_functions.simple import SpawnSimple
from utils.cy.vec2d_cy import Vec2dCy
from utils.dictionary import D_SENSOR_LIST

# Parameters
EPSILON_ANGLE = 0.0001  # 0.0001 radian offset allowed (~0.02 degrees)
EPSILON_DISTANCE = 0.001  # 1 millimeter offset allowed
EPSILON_DISTANCE_L = 0.1  # 10 centimeter offset allowed


class AngularSensorTestCy(unittest.TestCase):
    """Test the angular sensor."""
    
    def test_front(self):
        """> Test angular sensors when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)
        
        # Update the player's sensor-set
        game.player.sensors = dict()
        game.player.add_angular_sensors(clockwise=True)
        game.player.add_angular_sensors(clockwise=False)
        
        # The third and second last sensors
        sensor_values = game.player.get_sensor_readings()
        self.assertEqual(len(sensor_values), 2)
        for s in sensor_values:
            self.assertAlmostEqual(s, 0.0, delta=EPSILON_ANGLE)
    
    def test_left_angle(self):
        """> Test the angular sensors when target on the left."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)  # Looking to the right
        
        # Update the player's sensor-set
        game.player.sensors = dict()
        game.player.add_angular_sensors(clockwise=True)
        game.player.add_angular_sensors(clockwise=False)
        
        sensor_values = game.player.get_sensor_readings()
        self.assertEqual(len(sensor_values), 2)
        self.assertAlmostEqual(sensor_values[0], (-pi / 4) / pi, delta=EPSILON_ANGLE)  # Clockwise
        self.assertAlmostEqual(sensor_values[1], (pi / 4) / pi, delta=EPSILON_ANGLE)  # Anti-clockwise


class DistanceSensorTestCy(unittest.TestCase):
    """Test the distance sensor."""
    
    def test_front(self):
        """> Test the distance sensor when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.get_distance_to_target(), 1.0, delta=EPSILON_DISTANCE)
    
    def test_left_angle(self):
        """> Test distance sensor when target under an angle (towards the left)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.get_distance_to_target(), sqrt(2), delta=EPSILON_DISTANCE)
    
    def test_negative(self):
        """> Test the distance when the position of the agent is negative."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_init_pos(Vec2dCy(-1, -1))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.get_distance_to_target(), sqrt(8), delta=EPSILON_DISTANCE)


class DeltaDistanceSensorTestCy(unittest.TestCase):
    """Test the delta distance sensor."""
    
    def test_front(self):
        """> Test the distance sensor when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)
        
        # Update the sensors used to only include the delta-distance sensor
        game.player.sensors = dict()
        game.player.add_delta_distance_sensor()
        
        # Initially the sensor should read zero
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.0, delta=EPSILON_DISTANCE)
        
        # Advance the player's position by 0.1 meters and test sensor-reading
        game.player.pos = Vec2dCy(0.1, 0)
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.1, delta=EPSILON_DISTANCE)
        
        # Advance the player's position by 0.001 meters backwards and test sensor-reading
        game.player.pos = Vec2dCy(0.0999999, 0)
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], -0.0000001, delta=EPSILON_DISTANCE)
    
    def test_equal_side(self):
        """> Test distance sensor when target on the sides with equal distance."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_init_pos(Vec2dCy(0.999, 0))
        game.set_player_init_angle(0)
        
        # Update the sensors used to only include the delta-distance sensor
        game.player.sensors = dict()
        game.player.add_delta_distance_sensor()
        
        # Initially the sensor should read zero
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.0, delta=EPSILON_DISTANCE)
        
        # Advance the player's position to a symmetric position with equal distance
        game.player.pos = Vec2dCy(1.001, 0)
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.0, delta=EPSILON_DISTANCE)
    
    def test_none_zero_drive(self):
        """Test if the delta-distance sensor is non-zero when driving with high frame-rate."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        game.game_config.fps = 100  # Put greater FPS to test the extremes
        
        # Update player and target position
        game.target = Vec2dCy(10, 1)
        game.set_player_init_pos(Vec2dCy(1, 1))
        game.set_player_init_angle(0)
        
        # Update the sensors used to only include the delta-distance sensor
        game.player.sensors = dict()
        game.player.add_delta_distance_sensor()
        game.player.add_distance_sensor()  # Last sensor must always be the distance sensor
        
        # Drive forward for 10 simulated seconds
        start = True
        for _ in range(10 * game.game_config.fps):
            obs = game.step(l=1, r=1)
            if start:  # Cold start, reading of 0
                self.assertAlmostEqual(obs[D_SENSOR_LIST][0], 0.0, delta=EPSILON_DISTANCE)
                start = False
            else:
                self.assertGreater(obs[D_SENSOR_LIST][0], 0.0)  # Must be strictly greater than 0


def get_game():
    cfg = Config()
    spawn = SpawnSimple(game_config=cfg)
    spawn.add_location((1, 1))  # Dummy location
    return GameCy(
            config=cfg,
            game_id=0,
            noise=False,
            overwrite=True,
            save_path="tests/games_db/",
            silent=True,
            spawn_func=spawn,
    )


def main():
    # Test angular sensors
    ast = AngularSensorTestCy()
    ast.test_front()
    ast.test_left_angle()
    
    # Test distance sensor
    dst = DistanceSensorTestCy()
    dst.test_front()
    dst.test_left_angle()
    dst.test_negative()
    
    # Test delta distance sensor
    delta_dst = DeltaDistanceSensorTestCy()
    delta_dst.test_front()
    delta_dst.test_equal_side()
    delta_dst.test_none_zero_drive()


if __name__ == '__main__':
    unittest.main()
