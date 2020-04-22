"""
berkeley_gru_no_reset_cy.py

The Berkeley implementation of the GRU cell, with the lack of a reset-gate.
"""
import numpy as np
cimport numpy as np

from population.utils.network_util.cy.activations_cy import sigmoid_cy


cdef class GRUCellNoResetCy:
    """Berkeley implementation of the single GRU-cell, with the lack of a reset-gate."""
    
    def __init__(self, int input_size, np.ndarray bias, np.ndarray weight_hh, np.ndarray weight_xh):
        """
        Create the GRU-cell with the provided parameters.
        
        :param input_size: Number of inputs going into the cell
        :param bias: Bias-vector from each of the internal nodes
        :param weight_hh: Weight-vector from hidden to hidden states
        :param weight_xh: Weight-vector from input to hidden states
        """
        assert bias.shape[0] == 2  # Problem with tuple unpacking in cython otherwise
        assert weight_hh.shape[0] == 2
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
        cdef np.ndarray W_xh, W_hh, R_t, Z_t
        if len(self.hx) == 0:  # (batch_size, hidden_size)
            self.hx = np.zeros((x.shape[0], self.hidden_size), dtype=np.float64)
        xh = np.matmul(x, self.weight_xh.transpose())
        hh = np.matmul(self.hx, self.weight_hh.transpose())
        Z_t = sigmoid_cy(xh[:, 0:1] + hh[:, 0:1] + self.bias[0:1])
        self.hx = (1 - Z_t) * np.tanh(xh[:, 1:2] + self.bias[1:2]) + Z_t * self.hx
        return self.hx