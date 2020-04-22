"""
sensors_cy.pxd

Used to declare all the methods and classes inside of sensors_cy that must be callable from outside of other objects.
"""
from environment.cy.game_cy cimport GameCy


cdef class SensorCy:
    """The baseclass used by all sensors."""
    
    cdef GameCy game
    cdef public int id
    cdef public float value
    
    cpdef float measure(self)


cdef class AngularSensorCy(SensorCy):
    """Angle deviation between bot and wanted direction in 'crows flight'."""
    
    cdef public bint clockwise
    
    cpdef float measure(self)


cdef class DeltaDistanceSensorCy(SensorCy):
    """Difference in distance from bot to the target in 'crows flight' between current and the previous time-point."""
    
    cdef public float distance, prev_distance
    
    cpdef float measure(self)


cdef class DistanceSensorCy(SensorCy):
    """Distance from bot to the target in 'crows flight'."""
    
    cdef public float normalizer
    
    cpdef float measure(self)
