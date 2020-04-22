"""
sensors.py

Sensor classes used by the bots. The different types of sensors are:
 * Angular: Measures the burden angle between the robot and the target
 * Distance: Measures the distance in crows flight between the robot and the target
"""
from numpy import pi


class Sensor:
    """The baseclass used by all sensors."""
    
    __slots__ = {
        "game",
        "id", "value",
    }
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0,
                 ):
        """
        Basic characteristics of a sensor.
        
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        # Game-object
        self.game = game
        
        # Default sensor attributes
        self.id = sensor_id
        self.value = 0.0  # Zero value for initialized sensors
    
    def __str__(self):
        """Name of the sensor """
        raise NotImplemented
    
    def measure(self):
        """Store the sensor's current value in self.value."""
        raise NotImplemented


class AngularSensor(Sensor):
    """Angle deviation between bot and wanted direction in 'crows flight'."""
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0,
                 clockwise: bool = True):
        """
        :param clockwise: Calculate the angular difference in clockwise direction
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        # noinspection PyCompatibility
        super().__init__(game=game, sensor_id=sensor_id)
        self.clockwise = clockwise
    
    def __str__(self):
        return f"angular {'right' if self.clockwise else 'left'}"
    
    def measure(self):
        """Update self.value, result is a float between 0 and 2*PI."""
        # Get relative angle
        start_a = self.game.player.angle
        req_a = (self.game.target - self.game.player.pos).get_angle()
        
        # Constraint the value-range
        self.value = 2 * pi + start_a - req_a
        self.value %= 2 * pi
        
        # Check direction
        if not self.clockwise:
            self.value = abs(2 * pi - self.value)
        self.value = ((self.value + pi) % (2 * pi)) - pi  # constraint value between -pi and pi

        # Normalize
        self.value /= pi
        return self.value


class DeltaDistanceSensor(Sensor):
    """Difference in distance from bot to the target in 'crows flight' between current and the previous time-point."""
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        super().__init__(game=game, sensor_id=sensor_id)
        self.distance: float = 0.0
        self.prev_distance: float = 0.0
    
    def __str__(self):
        return "delta_distance"
    
    def measure(self):
        """Update self.value to difference between previous distance and current distance."""
        self.prev_distance = self.distance  # Save previous distance
        start_p = self.game.player.pos
        end_p = self.game.target
        self.distance = (start_p - end_p).get_length()  # Get current measure
        if self.prev_distance == 0.0: self.prev_distance = self.distance  # Disable cold start
        self.value = self.prev_distance - self.distance  # Positive value == closer to target
        return self.value


class DistanceSensor(Sensor):
    """Distance from bot to the target in 'crows flight'."""
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 normalizer: float,
                 sensor_id: int = 0):
        """
        :param game: Reference to the game in which the sensor is used
        :param normalizer: The constant by which the distance-value is normalized
        :param sensor_id: Identification number for the sensor
        """
        super().__init__(game=game, sensor_id=sensor_id)
        self.normalizer = normalizer
    
    def __str__(self):
        return "distance"
    
    def measure(self):
        """Update self.value to current distance between target and robot's center coordinate."""
        start_p = self.game.player.pos
        end_p = self.game.target
        self.value = (start_p - end_p).get_length() / self.normalizer
        return self.value
