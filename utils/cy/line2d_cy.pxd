"""
line2d_cy.pxd

Used to declare all the methods of Line2dCy that must be callable from outside of other objects.
"""
from utils.cy.vec2d_cy cimport Vec2dCy


cdef class Line2dCy:
    cdef public Vec2dCy x, y
    
    cpdef float get_length(self)
    
    cpdef float get_orientation(self)
