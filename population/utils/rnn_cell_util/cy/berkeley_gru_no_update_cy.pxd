"""
berkeley_gru_no_update_cy.py

The Berkeley implementation of the GRU cell, with the lack of an update-gate.
"""
cimport numpy as np


cdef class GRUCellNoUpdateCy:
    """Copy of the PyTorch implementation of the single GRU-cell."""
    cdef public int hidden_size, input_size
    cdef public np.ndarray bias, weight_hh, weight_xh, hx
