"""
lstm.py

The LSTM-cell implementation.
"""
import numpy as np

from population.utils.network_util.activations import sigmoid


class LSTMCell:
    """Custom implementation of the single LSTM-cell."""
    
    def __init__(self, input_size: int, bias: np.ndarray, weight_hh: np.ndarray, weight_xh: np.ndarray):
        """
        Create the LSTM-cell with the provided parameters.
        
        :param input_size: Number of inputs going into the cell
        :param bias: Bias-vector from each of the internal nodes
        :param weight_hh: Weight-vector from hidden to hidden states
        :param weight_xh: Weight-vector from input to hidden states
        """
        assert bias.shape[0] == 4  # Problem with tuple unpacking in cython otherwise
        assert weight_hh.shape[0] == 4
        assert weight_hh.shape[1] == 1
        
        self.hidden_size: int = 1  # Fixed
        self.input_size: int = input_size
        self.bias: np.ndarray = bias
        self.weight_hh: np.ndarray = weight_hh
        self.weight_xh: np.ndarray = weight_xh
        self.hx: np.ndarray = np.asarray([])  # Cell takes responsibility in remembering hidden state (no pass at call)
        self.c: np.ndarray = np.asarray([])  # Cell takes responsibility in remembering candidate states
    
    def __call__(self, x: np.ndarray):
        """
        Forward the network by one iteration and return the updated hidden state.
        
        :param x: Input
        :return: Updated hidden state
        """
        if len(self.hx) == 0:  # (batch_size, hidden_size)
            self.hx = np.zeros((x.shape[0], self.hidden_size), dtype=np.float64)
        if len(self.c) == 0:  # (batch_size, hidden_size)
            self.c = np.zeros((x.shape[0], self.hidden_size), dtype=np.float64)
        xh = np.matmul(x, self.weight_xh.transpose())
        hh = np.matmul(self.hx, self.weight_hh.transpose())
        F_t = sigmoid(xh[:, 0:1] + hh[:, 0:1] + self.bias[0:1])
        I_t = sigmoid(xh[:, 1:2] + hh[:, 1:2] + self.bias[1:2])
        O_t = sigmoid(xh[:, 2:3] + hh[:, 2:3] + self.bias[2:3])
        # C_tilde = np.tanh(W_xh[:, 3:4] + W_hh[:, 3:4] + self.bias[3:4])  # Reducing 1 extra variable
        self.c = F_t * self.c + I_t * np.tanh(xh[:, 3:4] + hh[:, 3:4] + self.bias[3:4])
        self.hx = O_t * np.tanh(self.c)
        return self.hx
