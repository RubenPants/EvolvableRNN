"""
robots_cy.pxd

Used to declare all the methods and classes inside of robots_cy that must be callable from outside of other objects.
"""
cimport numpy as np

from environment.cy.game_cy cimport GameCy
from utils.cy.vec2d_cy cimport Vec2dCy

cdef class RobotCy:
    """Differential drive robot."""

    cdef GameCy game
    cdef public Vec2dCy pos, prev_pos, init_pos
    cdef public float angle, init_angle, noisy_init_angle, radius
    cdef float prev_angle
    cdef public dict sensors
    cdef int n_angular, n_delta_distance, n_distance
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef void drive(self, float dt, float lw, float rw)
    
    cpdef np.ndarray get_sensor_readings(self)
    
    cpdef void set_back(self)
    
    cpdef void reset(self, bint noise=?)
    
    cpdef void randomize(self, float max_noise=?)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void set_init_angle(self, float a)
    
    cpdef void set_init_pos(self, Vec2dCy p)
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    cpdef void add_angular_sensors(self, bint clockwise=?)
    
    cpdef void add_delta_distance_sensor(self)
    
    cpdef void add_distance_sensor(self)
    
    cpdef void create_angular_sensors(self, cfg)
    
    cpdef void create_delta_distance_sensor(self, cfg)
    
    cpdef void create_distance_sensor(self, cfg)
