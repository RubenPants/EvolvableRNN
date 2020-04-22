"""
env_multi_cy.py

Environment where a single genome gets evaluated over multiple games. This environment will be called in a process.
"""
cimport numpy as np

cdef class MultiEnvironmentCy:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    cdef int batch_size, max_steps
    cdef np.ndarray games
    cdef game_config
    cdef pop_config
    
    cpdef void eval_genome(self, genome, return_dict=?)
    
    cpdef void trace_genome(self, genome, return_dict=?)
    
    cpdef void set_games(self, list games, bint noise=?)
