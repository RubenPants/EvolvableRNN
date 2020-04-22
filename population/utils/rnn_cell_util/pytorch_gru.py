"""
pytorch_self.py

A literal copy of the PyTorch implementation of the GRU-cell.
"""
from torch import float64, sigmoid, tanh, tensor, zeros
from torch.nn import Module


class GRUCell(Module):
    """Copy of the PyTorch implementation of the single GRU-cell."""
    
    def __init__(self, input_size, bias_hh, bias_ih, weight_hh, weight_ih):
        """
        Create the GRU-cell with the provided parameters.
        
        :param input_size: Number of inputs going into the cell
        :param bias_hh: Bias-vector from the hidden state to the hidden state
        :param bias_ih: Bias-vector from the input state to the hidden state
        :param weight_hh: Weight-vector from the hidden state to the hidden state
        :param weight_ih: Weight-vector from the input state to the hidden state
        """
        super(GRUCell, self).__init__()
        self.hidden_size = 1  # Fixed
        self.input_size = input_size
        self.bias_hh = tensor(bias_hh, dtype=float64)
        self.bias_ih = tensor(bias_ih, dtype=float64)
        self.weight_hh = tensor(weight_hh, dtype=float64)
        self.weight_ih = tensor(weight_ih, dtype=float64)
        self.hx = None
    
    def forward(self, inp: tensor):
        if self.hx is None:  # (batch_size, hidden_size)
            self.hx = zeros(inp.size(0), self.hidden_size, dtype=inp.dtype)
        ih = inp.mm(self.weight_ih.t())
        hh = self.hx.mm(self.weight_hh.t())
        r = sigmoid(ih[:, 0:1] + self.bias_ih[0:1] + hh[:, 0:1] + self.bias_hh[0:1])
        z = sigmoid(ih[:, 1:2] + self.bias_ih[1:2] + hh[:, 1:2] + self.bias_hh[1:2])
        n = tanh(ih[:, 2:3] + self.bias_ih[2:3] + r * (hh[:, 2:3] + self.bias_hh[2:3]))
        self.hx = (1 - z) * n + z * self.hx
        return self.hx
