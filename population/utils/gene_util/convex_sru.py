"""
convex_sru.py

Modification of the SRU that has a convex recurrent weight (weight_xh and _hh are ignored).
"""

import numpy as np
from numpy.linalg import norm

from configs.genome_config import GenomeConfig
from population.utils.attributes import rnn
from population.utils.gene_util.rnn import RnnNodeGene


class ConvexSruNodeGene(RnnNodeGene):
    """SRU modification to ensure convex connection."""
    
    __slots__ = {
        'weight'
    }
    
    def __init__(self, key, cfg: GenomeConfig, input_keys=None):
        super().__init__(
                key=key,
                hid_dim=1,
                cfg=cfg,
                input_keys=input_keys,
                input_keys_full=input_keys,
        )
        self.weight = np.clip(np.random.normal(0.975, 0.01), a_min=0, a_max=1)
    
    def __str__(self):
        return f"ConvexSruNodeGene(\n" \
               f"\tkey={self.key}\n" \
               f"\tdelay={self.bias_h[0]:.5f},\n" \
               f"\tweight={self.weight:.5f},\n" \
               f"\tinput_keys={self.input_keys!r})"
    
    def __repr__(self):
        return f"ConvexSruNodeGene(bias={self.bias_h[0]:.3f},weight={self.weight:.3f})"
    
    def mutate(self, cfg: GenomeConfig):
        self.bias_h = rnn.mutate_1d(self.bias_h, cfg=cfg, bias=True)
        self.weight = np.clip(self.weight + np.random.normal(0, 0.001), a_min=0, a_max=1)  # Very sensitive!
    
    def distance(self, other, cfg: GenomeConfig):
        d = 0
        d += norm(self.bias_h - other.bias_h)
        d += 10 * abs(self.weight - other.weight)
        d /= 2  # Average distance per component
        return d
    
    def update_weight_xh(self):
        pass
    
    def get_rnn(self, mapping=None):
        """Return a RNNCell based on current configuration. The mapping denotes which columns to use."""
        cell = FixedCell(
                input_size=len(mapping[mapping]) if mapping is not None else len(self.input_keys),
                bias=self.bias_h,
                weight=self.weight,
        )
        return cell


class FixedCell:
    """Small variation of the PyTorch implementation of the simple RNN-cell."""
    
    def __init__(self, input_size: int, bias, weight: float):
        """
        Create the RNN-cell with the provided parameters.

        :param input_size: Number of inputs going into the cell
        :param weight: Convex weight of the unit's connections
        """
        self.input_size: int = input_size
        self.bias = bias
        self.weight = weight
        self.hx: np.ndarray = None
    
    def __call__(self, x: np.ndarray):
        """
        Forward the network by one iteration and return the updated hidden state.

        :param x: Input
        :return: Updated hidden state
        """
        if self.hx is None:  # (batch_size, hidden_size)
            self.hx = np.zeros((x.shape[0], 1), dtype=np.float64)
        self.hx = np.tanh(self.weight * self.hx + (1 - self.weight) * x)
        return self.hx
