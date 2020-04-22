"""
berkeley_gru.py

The Berkeley implementation of the GRU cell.
"""
import numpy as np

from population.utils.network_util.activations import sigmoid


class GRUCell:
    """Berkeley implementation of the single GRU-cell."""
    
    def __init__(self, input_size: int, bias: np.ndarray, weight_hh: np.ndarray, weight_xh: np.ndarray):
        """
        Create the GRU-cell with the provided parameters.
        
        :param input_size: Number of inputs going into the cell
        :param bias: Bias-vector from each of the internal nodes
        :param weight_hh: Weight-vector from hidden to hidden states
        :param weight_xh: Weight-vector from input to hidden states
        """
        assert bias.shape[0] == 3  # Problem with tuple unpacking in cython otherwise
        assert weight_hh.shape[0] == 3
        assert weight_hh.shape[1] == 1
        
        self.hidden_size: int = 1  # Fixed
        self.input_size: int = input_size
        self.bias: np.ndarray = bias
        self.weight_hh: np.ndarray = weight_hh
        self.weight_xh: np.ndarray = weight_xh
        self.hx: np.ndarray = np.asarray([])  # Cell takes responsibility in remembering hidden state (no pass at call)
    
    def __call__(self, x: np.ndarray):
        """
        Forward the network by one iteration and return the updated hidden state.
        
        :note: H_tilde is NOT COMPUTED CORRECTLY if hx is larger than one, but is correct when hx.size==(1,1), which is
               always the case for our usage.
               Correct implementation: http://courses.d2l.ai/berkeley-stat-157/slides/4_9/19-RNN.pdf
        
        :param x: Input
        :return: Updated hidden state
        """
        if len(self.hx) == 0:  # (batch_size, hidden_size)
            self.hx = np.zeros((x.shape[0], self.hidden_size), dtype=np.float64)
        xh = np.matmul(x, self.weight_xh.transpose())
        hh = np.matmul(self.hx, self.weight_hh.transpose())
        R_t = sigmoid(xh[:, 0:1] + hh[:, 0:1] + self.bias[0:1])
        Z_t = sigmoid(xh[:, 1:2] + hh[:, 1:2] + self.bias[1:2])
        # H_tilde = tanh(W_xh[:, 2:3] + R_t * W_hh[:, 2:3] + self.bias[2:3])  # Reducing 1 extra variable
        self.hx = (1 - Z_t) * np.tanh(xh[:, 2:3] + R_t * hh[:, 2:3] + self.bias[2:3]) + Z_t * self.hx
        return self.hx
