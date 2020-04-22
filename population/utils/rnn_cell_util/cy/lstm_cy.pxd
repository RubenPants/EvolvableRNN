"""
lstm_cy.py

The LSTM-cell implementation.
"""
cimport numpy as np


cdef class LSTMCellCy:
    """Custom implementation of the single LSTM-cell."""
    cdef public int hidden_size, input_size
    cdef public np.ndarray bias, weight_hh, weight_xh, hx, c
