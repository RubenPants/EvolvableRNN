"""
berkeley_gru_cy.py

The Berkeley implementation of the GRU cell.
"""
cimport numpy as np


cdef class GRUCellCy:
    """Copy of the PyTorch implementation of the single GRU-cell."""
    cdef public int hidden_size, input_size
    cdef public np.ndarray bias, weight_hh, weight_xh, hx
