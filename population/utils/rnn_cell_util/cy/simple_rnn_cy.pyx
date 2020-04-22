"""
simple_rnn_cy.py

The simple RNN-cell implementation.
"""
import numpy as np
cimport numpy as np


cdef class RNNCellCy:
    """Small variation of the PyTorch implementation of the simple RNN-cell."""
    
    def __init__(self, int input_size, np.ndarray bias, np.ndarray weight_hh, np.ndarray weight_xh):
        """
        Create the RNN-cell with the provided parameters.

        :param input_size: Number of inputs going into the cell
        :param bias: Bias for the internal node
        :param weight_hh: Weight of the hidden-state connection
        :param weight_xh: Weight-vector from input to hidden state
        """
        assert bias.shape[0] == 1  # Problem with tuple unpacking in cython otherwise
        assert weight_hh.shape[0] == 1
        assert weight_hh.shape[1] == 1
        
        self.hidden_size = 1  # Fixed
        self.input_size = input_size
        self.bias = bias
        self.weight_hh = weight_hh
        self.weight_xh = weight_xh
        self.hx = np.asarray([])  # Cell takes responsibility in remembering hidden state (no pass at call)
    
    def __call__(self, np.ndarray x):
        """
        Forward the network by one iteration and return the updated hidden state.

        :param x: Input
        :return: Updated hidden state
        """
        if len(self.hx) == 0:  # (batch_size, hidden_size)
            self.hx = np.zeros((x.shape[0], self.hidden_size), dtype=np.float64)
        self.hx = np.tanh(np.matmul(self.hx, self.weight_hh.transpose()) +
                          np.matmul(x, self.weight_xh.transpose()) +
                          self.bias)
        return self.hx
