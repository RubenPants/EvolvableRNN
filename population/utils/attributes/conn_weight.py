"""
conn_weight.py

Connection weight attribute represented by a float which is used during neuroevolution.
"""
from random import gauss, random

from numpy import clip

from configs.genome_config import GenomeConfig


def cross(v1, v2, ratio: float = 0.5):
    """
    Inherit one of the two weight attributes from the given parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    return v1 if random() < ratio else v2


def init(cfg: GenomeConfig):
    """Random initialized weight value, calculated via a normal distribution."""
    return clip(gauss(cfg.weight_init_mean, cfg.weight_init_stdev),
                a_min=cfg.weight_min_value,
                a_max=cfg.weight_max_value)


def mutate(v, cfg: GenomeConfig):
    """Mutate the given weight-value based on the provided GenomeConfig file."""
    # Check if value is mutated
    r = random()
    if r < cfg.weight_mutate_rate:
        return clip(v + gauss(0.0, cfg.weight_mutate_power), a_min=cfg.weight_min_value, a_max=cfg.weight_max_value)
    
    # Check if value is replaced
    elif r < cfg.weight_replace_rate + cfg.weight_mutate_rate:
        return init(cfg)
    
    # No changes, return original value
    return v
