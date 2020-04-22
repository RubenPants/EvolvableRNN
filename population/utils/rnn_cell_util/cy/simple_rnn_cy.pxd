"""
elman_rnn_cy.py

The simple RNN-cell implementation.
"""
cimport numpy as np


cdef class RNNCellCy:
    """Small variation of the PyTorch implementation of the simple RNN-cell."""
    cdef public int hidden_size, input_size
    cdef public np.ndarray bias, weight_hh, weight_xh, hx
