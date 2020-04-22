"""
robots.py

Robots used to manoeuvre around in the Game-environment.
"""
from random import random

from numpy import pi, sqrt, asarray

from configs.bot_config import BotConfig
from environment.sensors import AngularSensor, DeltaDistanceSensor, DistanceSensor
from utils.vec2d import angle_to_vec, Vec2d


class Robot:
    """Differential drive robot."""
    
    __slots__ = {
        'game',
        'pos', 'prev_pos', 'init_pos', 'init_angle', 'noisy_init_angle', 'angle', 'prev_angle', 'radius',
        'n_angular', 'n_delta_distance', 'n_distance', 'sensors',
    }
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 r: float = 0
                 ):
        """
        Create a new Robot object.
        
        :param game: Reference to the game in which the robot is created [Game]
        :param r: Radius of the circular robot
        """
        # Game specific parameter
        self.game = game  # Game in which robot runs
        
        # Robot specific parameters (Placeholders)
        self.pos = Vec2d(0, 0)  # Current position
        self.init_pos = Vec2d(0, 0)  # Initial position
        self.prev_pos = Vec2d(0, 0)  # Previous current position
        self.angle: float = 0  # Current angle
        self.init_angle: float = 0  # Initial angle
        self.noisy_init_angle: float = 0  # Initial angle with noise
        self.prev_angle: float = 0  # Previous angle
        self.radius: float = r if r else game.bot_config.radius  # Radius of the bot
        
        # Container of all the sensors
        self.sensors: dict = dict()
        
        # Counters for number of sensors used
        self.n_angular: int = 0
        self.n_delta_distance: int = 0
        self.n_distance: int = 0
        
        # Create the sensors (fixed order!)
        self.create_angular_sensors(cfg=game.bot_config)
        self.create_delta_distance_sensor(cfg=game.bot_config)
        self.create_distance_sensor(cfg=game.bot_config)
    
    def __str__(self):
        return f"Robot(pos={self.pos}, angle={round(self.angle, 2)})"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def drive(self, dt: float, lw: float, rw: float):
        """
        Update the robot's position and orientation based on the action of the wheels.
        
        :param dt: Delta time (must always be positioned first)
        :param lw: Speed of the left wheel, float [-1,1]
        :param rw: Speed of the right wheel, float [-1,1]
        """
        # Constraint the inputs
        lw = max(min(lw, 1), -1)
        rw = max(min(rw, 1), -1)
        
        # Update previous state
        self.prev_pos.x, self.prev_pos.y = self.pos.x, self.pos.y
        self.prev_angle = self.angle
        
        # Update angle is determined by the speed of both wheels
        self.angle += (rw - lw) * self.game.bot_config.turning_speed * dt
        self.angle %= 2 * pi
        
        # Update position is the average of the two wheels times the maximum driving speed
        self.pos += angle_to_vec(self.angle) * float((((lw + rw) / 2) * self.game.bot_config.driving_speed * dt))
    
    def set_back(self):
        """Set the robot back to its previous position and orientation."""
        self.pos.x = self.prev_pos.x
        self.pos.y = self.prev_pos.y
        self.angle = self.prev_angle
    
    def get_sensor_readings(self):
        """Numpy array of the current sensory-readings."""
        return asarray([self.sensors[i].measure() for i in sorted(self.sensors)])
    
    def reset(self, noise: bool = True):
        """Put the robot back to its initial parameters."""
        self.pos.x = self.init_pos.x
        self.pos.y = self.init_pos.y
        self.prev_pos.x = self.init_pos.x
        self.prev_pos.y = self.init_pos.y
        if noise:
            self.angle = self.noisy_init_angle
            self.prev_angle = self.noisy_init_angle
        else:
            self.angle = self.init_angle
            self.prev_angle = self.init_angle
    
    def randomize(self, max_noise: float = pi / 4):
        """Randomize the initial angle of the robot."""
        self.noisy_init_angle = self.init_angle + max_noise * (2 * random() - 1)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def set_init_angle(self, a: float):
        """Set the initial angle of the robot."""
        self.angle = a
        self.prev_angle = a
        self.init_angle = a
        self.noisy_init_angle = a
    
    def set_init_pos(self, p: Vec2d):
        """Set the initial position of the robot."""
        self.init_pos = p.__copy__()
        self.pos = p.__copy__()
        self.prev_pos = p.__copy__()
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    def add_angular_sensors(self, clockwise: bool = True):
        """
        Add an angular sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is the
        first sensor that is added.
        """
        self.sensors[len(self.sensors)] = AngularSensor(
                sensor_id=len(self.sensors),
                game=self.game,
                clockwise=clockwise)
        self.n_angular += 1
    
    def add_delta_distance_sensor(self):
        """Single distance sensor which determines distance between agent's center and target's center."""
        self.sensors[len(self.sensors)] = DeltaDistanceSensor(
                sensor_id=len(self.sensors),
                game=self.game)
        self.n_delta_distance += 1
    
    def add_distance_sensor(self):
        """Single distance sensor which determines distance between agent's center and target's center."""
        self.sensors[len(self.sensors)] = DistanceSensor(
                sensor_id=len(self.sensors),
                normalizer=sqrt(self.game.game_config.x_axis ** 2 + self.game.game_config.y_axis ** 2),
                game=self.game)
        self.n_distance += 1
    
    def create_angular_sensors(self, cfg: BotConfig):
        """
        Two angular sensors that define the angle between the orientation the agent is heading and the agent towards the
        target 'in crows flight'. One measures this angle in clockwise, the other counterclockwise.
        """
        for clockwise in cfg.angular_dir: self.add_angular_sensors(clockwise=clockwise)
    
    def create_delta_distance_sensor(self, cfg: BotConfig):
        """Add a delta-distance sensor which measures the difference in distance to the target each time-point."""
        if cfg.delta_dist_enabled: self.add_delta_distance_sensor()
    
    def create_distance_sensor(self, cfg: BotConfig):
        """Add a distance sensor which measures the distance to target."""
        if cfg.dist_enabled: self.add_distance_sensor()


def get_number_of_sensors(cfg: BotConfig):
    """Get the number of sensors mounted on the robot. The robot always has a distance sensor."""
    return len(cfg.angular_dir) + int(cfg.delta_dist_enabled) + int(cfg.dist_enabled)


def get_snapshot(cfg: BotConfig):
    """
    Get the snapshot of the current robot-configuration. This method mimics the 'Create the sensors' section in the
    MarXBot creation process.
    """
    sorted_names = []
    
    # Angular sensors
    for cw in cfg.angular_dir: sorted_names.append(str(AngularSensor(game=None, clockwise=cw)))
    
    # Delta distance sensor
    if cfg.delta_dist_enabled: sorted_names.append(str(DeltaDistanceSensor(game=None)))
    
    # Distance sensor
    if cfg.dist_enabled: sorted_names.append(str(DistanceSensor(game=None, normalizer=1)))
    
    # Negate all the keys to create the snapshot
    snapshot = dict()
    for i, name in enumerate(sorted_names): snapshot[-len(sorted_names) + i] = name
    
    return snapshot
