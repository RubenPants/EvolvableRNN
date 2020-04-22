"""
game_cy.pxd

Used to declare all the game_cy class and method that must be callable from outside of other objects.
"""
from environment.cy.robot_cy cimport RobotCy
from utils.cy.vec2d_cy cimport Vec2dCy

cdef class GameCy:
    """
    A game environment is built up from the following segments:
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    cdef public bot_config, game_config
    cdef bint wall_bound, noise, silent, stop_if_reached, done
    cdef str save_path
    cdef public int x_axis, y_axis
    cdef int id, score, steps_taken
    cdef float init_distance, player_angle_noise
    cdef public RobotCy player
    cdef public Vec2dCy target
    cdef public spawn_function
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self)
    
    cpdef dict get_observation(self)
    
    cpdef void randomize(self)
    
    cpdef dict reset(self)
    
    cpdef step(self, float l, float r)
    
    cpdef step_dt(self, float dt, float l, float r)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void create_empty_game(self, spawn_func)
    
    cpdef float get_distance_to_target(self)
    
    cpdef void sample_target(self)
    
    cpdef void set_player_init_angle(self, float a)
    
    cpdef void set_player_init_pos(self, Vec2dCy p)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    cpdef void save(self)
    
    cpdef bint load(self)
    
    cpdef get_blueprint(self, ax=?, bint show_player=?)

cpdef GameCy get_game_cy(int i, cfg=?, bint noise=?)
