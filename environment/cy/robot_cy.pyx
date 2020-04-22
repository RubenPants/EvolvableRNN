"""
robots_cy.pyx

Robots used to manoeuvre around in the Game-environment.
"""
from random import random

cimport numpy as np
from numpy import pi, sqrt, asarray

from sensors_cy cimport AngularSensorCy, DeltaDistanceSensorCy, DistanceSensorCy
from utils.cy.vec2d_cy cimport angle_to_vec, Vec2dCy


cdef class RobotCy:
    """Differential drive robot."""
    
    __slots__ = {
        'game',
        'pos', 'prev_pos', 'init_pos', 'init_angle', 'noisy_init_angle', 'angle', 'prev_angle', 'radius',
        'n_angular', 'n_delta_distance', 'n_distance', 'sensors',
    }
    
    def __init__(self,
                 GameCy game,
                 float r=0
                 ):
        """
        Create a new Robot object.
        
        :param game: Reference to the game in which the robot is created [Game]
        :param r: Radius of the circular robot
        """
        # Game specific parameter
        self.game = game  # Game in which robot runs
        
        # Robot specific parameters (Placeholders)
        self.pos = Vec2dCy(0, 0)  # Current position
        self.init_pos = Vec2dCy(0, 0)  # Initial position
        self.prev_pos = Vec2dCy(0, 0)  # Previous current position
        self.angle = 0  # Current angle
        self.init_angle = 0  # Initial angle
        self.noisy_init_angle = 0  # Initial angle with noise
        self.prev_angle = 0  # Previous angle
        self.radius = r if r else game.bot_config.radius  # Radius of the bot
        
        # Container of all the sensors
        self.sensors = dict()
        
        # Counters for number of sensors used
        self.n_angular = 0
        self.n_delta_distance = 0
        self.n_distance = 0
        
        # Create the sensors (fixed order!)
        self.create_angular_sensors(cfg=game.bot_config)
        self.create_delta_distance_sensor(cfg=game.bot_config)
        self.create_distance_sensor(cfg=game.bot_config)
    
    def __str__(self):
        return f"Robot(pos={self.pos}, angle={round(self.angle, 2)})"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef void drive(self, float dt, float lw, float rw):
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
    
    cpdef void set_back(self):
        """Set the robot back to its previous position and orientation."""
        self.pos.x = self.prev_pos.x
        self.pos.y = self.prev_pos.y
        self.angle = self.prev_angle
    
    cpdef np.ndarray get_sensor_readings(self):
        """Numpy array of the current sensory-readings."""
        return asarray([self.sensors[i].measure() for i in sorted(self.sensors)])
    
    cpdef void reset(self, bint noise=True):
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
    
    cpdef void randomize(self, float max_noise=pi / 4):
        """Randomize the initial angle of the robot."""
        self.noisy_init_angle = self.init_angle + max_noise * (2 * random() - 1)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void set_init_angle(self, float a):
        """Set the initial angle of the robot."""
        self.angle = a
        self.prev_angle = a
        self.init_angle = a
        self.noisy_init_angle = a
    
    cpdef void set_init_pos(self, Vec2dCy p):
        """Set the initial position of the robot."""
        self.init_pos = p.__copy__()
        self.pos = p.__copy__()
        self.prev_pos = p.__copy__()
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    cpdef void add_angular_sensors(self, bint clockwise=True):
        """
        Add an angular sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is the
        first sensor that is added.
        """
        self.sensors[len(self.sensors)] = AngularSensorCy(
                sensor_id=len(self.sensors),
                game=self.game,
                clockwise=clockwise)
        self.n_angular += 1
    
    cpdef void add_delta_distance_sensor(self):
        """Single distance sensor which determines distance between agent's center and target's center."""
        self.sensors[len(self.sensors)] = DeltaDistanceSensorCy(
                sensor_id=len(self.sensors),
                game=self.game)
        self.n_delta_distance += 1
    
    cpdef void add_distance_sensor(self):
        """Single distance sensor which determines distance between agent's center and target's center."""
        self.sensors[len(self.sensors)] = DistanceSensorCy(
                sensor_id=len(self.sensors),
                normalizer=sqrt(self.game.game_config.x_axis ** 2 + self.game.game_config.y_axis ** 2),
                game=self.game)
        self.n_distance += 1
    
    cpdef void create_angular_sensors(self, cfg):
        """
        Two angular sensors that define the angle between the orientation the agent is heading and the agent towards the
        target 'in crows flight'. One measures this angle in clockwise, the other counterclockwise.
        """
        cdef bint clockwise
        for clockwise in cfg.angular_dir: self.add_angular_sensors(clockwise=clockwise)
    
    cpdef void create_delta_distance_sensor(self, cfg):
        """Add a delta-distance sensor which measures the difference in distance to the target each time-point."""
        if cfg.delta_dist_enabled: self.add_delta_distance_sensor()
    
    cpdef void create_distance_sensor(self, cfg):
        """Add a distance sensor which measures the distance to target."""
        if cfg.dist_enabled: self.add_distance_sensor()
