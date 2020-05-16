"""
fixed_rnn.py

No real recurrent unit, just a hand-made delay between input and output.
"""
from random import random

import numpy as np
from numpy.linalg import norm

from configs.genome_config import GenomeConfig
from population.utils.attributes import rnn
from population.utils.gene_util.rnn import RnnNodeGene


class FixedRnnNodeGene(RnnNodeGene):
    """Custom Fixed RNN cell implementation."""
    
    __slots__ = {
        'key', 'delay', 'scale', 'input_keys', 'bias',
    }
    
    def __init__(self, key, cfg: GenomeConfig, input_keys=None):
        super().__init__(
                key=key,
                hid_dim=1,
                cfg=cfg,
                input_keys=input_keys,
                input_keys_full=input_keys,
        )
        self.key = key
        self.bias_h = rnn.init(cfg, hid_dim=len(input_keys), bias=True)
        self.delay = 50 + int(random() * 20)  # Delay until distance info is used
        self.scale = rnn.init(cfg, hid_dim=len(input_keys))  # Scales the inputs before returning them
        self.input_keys = input_keys
        self.bias = 0
    
    def __str__(self):
        return f"FixedRnnNodeGene(\n" \
               f"\tkey={self.key}\n" \
               f"\tdelay={self.bias_h[0]:.5f},\n" \
               f"\tdelay={self.delay},\n" \
               f"\tscale={self.scale[0]:.5f},\n" \
               f"\tinput_keys={self.input_keys!r})"
    
    def __repr__(self):
        return f"FixedRnnNodeGene(bias={self.bias_h[0]:.3f},delay={self.delay},scale={self.scale[0]:.3f})"
    
    def mutate(self, cfg: GenomeConfig):
        self.bias_h = rnn.mutate_1d(self.bias_h, cfg=cfg, bias=True)
        self.scale = rnn.mutate_1d(self.scale, cfg=cfg)
        if random() < cfg.bias_mutate_rate:
            self.delay += int(np.random.normal() * 5)
        if self.delay < 2: self.delay = 2
    
    def distance(self, other, cfg: GenomeConfig):
        d = 0
        d += norm(self.bias_h - other.bias_h)
        d += norm(self.scale - other.scale)
        d += abs(self.delay - other.delay) / 10
        return d
    
    def update_weight_xh(self):
        pass
    
    def get_rnn(self, mapping=None):
        """Return a RNNCell based on current configuration. The mapping denotes which columns to use."""
        cell = FixedCell(
                input_size=len(mapping[mapping]) if mapping is not None else len(self.input_keys),
                bias=self.bias_h,
                delay=self.delay,
                scale=self.scale,
        )
        return cell


class FixedCell:
    """Small variation of the PyTorch implementation of the simple RNN-cell."""
    
    def __init__(self, input_size: int, bias, delay: int, scale):
        """
        Create the RNN-cell with the provided parameters.

        :param input_size: Number of inputs going into the cell
        :param delay: Bias for the internal node
        """
        self.input_size: int = input_size
        self.bias = bias
        self.delay: int = delay
        self.scale = scale
        self.hx: np.ndarray = None
    
    def __call__(self, x: np.ndarray):
        """
        Forward the network by one iteration and return the updated hidden state.

        :param x: Input
        :return: Updated hidden state
        """
        if self.hx is None:  # (batch_size, hidden_size)
            self.hx = np.zeros((x.shape[0], 1), dtype=np.float64)
        self.hx = np.concatenate((self.hx, x), axis=1)
        if self.hx.shape[1] <= self.delay + 1:  # As long as delay does not kick in, use first reading
            return self.hx[:, 1:2] * self.scale[0] + self.bias[0]
        else:
            return self.hx[:, -self.delay - 1:-self.delay] * self.scale[0] + self.bias[0]
