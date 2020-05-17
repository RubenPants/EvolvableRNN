"""
rnn.py

RNN attributes, used for both biases as weights, represented by a numpy tensor which is used during neuro-evolution.
Vector are represented by numpy-arrays.
"""
from random import gauss, random

from numpy import clip, float64, zeros

from configs.genome_config import GenomeConfig
from population.utils.attributes.bias import init as bias_init


def init(cfg: GenomeConfig, hid_dim: int, input_size: int = None, bias: bool = False):
    """Initialize a RNN-vector"""
    t = zeros((hid_dim, input_size), dtype=float64) if input_size is not None else zeros((hid_dim,), dtype=float64)
    
    # Query the FloatAttribute for each initialization of the tensor's parameters
    for t_index in range(len(t)):
        t[t_index] = single_init(cfg) if not bias else bias_init(cfg)
    return t


def single_init(cfg: GenomeConfig):
    """Random initialized floating RNN value, calculated via a normal distribution."""
    return clip(gauss(cfg.rnn_init_mean, cfg.rnn_init_stdev), a_min=cfg.rnn_min_value, a_max=cfg.rnn_max_value)


def mutate_1d(v, cfg: GenomeConfig, bias: bool = False):
    """Mutate a 1-dimensional RNN-vector."""
    for i, elem in enumerate(v):
        v[i] = mutate(elem, cfg=cfg, bias=bias)
    return v


def mutate_2d(v, cfg: GenomeConfig, mapping=None):
    """Mutate a 2-dimensional RNN-vector. If mapping is given, it should denote which columns to mutate."""
    for col_i in range(v.shape[1]):
        if mapping and not mapping[col_i]: continue
        v[:, col_i] = mutate_1d(v[:, col_i], cfg=cfg)
    return v


def mutate(v, cfg: GenomeConfig, bias: bool = False):
    """Mutate the given RNN-value based on the provided GenomeConfig file."""
    # Check if value must mutate
    r = random()
    if r < cfg.rnn_mutate_rate:
        return clip(v + gauss(0.0, cfg.rnn_mutate_power),
                    a_min=cfg.bias_min_value if bias else cfg.rnn_min_value,
                    a_max=cfg.bias_max_value if bias else cfg.rnn_max_value)
    
    # Check if value must be replaced
    elif r < cfg.rnn_replace_rate + cfg.rnn_mutate_rate:
        return single_init(cfg)
    
    # No changes, return original value
    return v
