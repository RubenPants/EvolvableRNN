"""
berkeley_gru_no_reset_cy.py

The Berkeley implementation of the GRU cell, with the lack of a reset-gate.
"""
cimport numpy as np


cdef class GRUCellNoResetCy:
    """Copy of the PyTorch implementation of the single GRU-cell."""
    cdef public int hidden_size, input_size
    cdef public np.ndarray bias, weight_hh, weight_xh, hx
