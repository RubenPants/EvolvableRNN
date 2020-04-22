"""
conn_enabled.py

Connection enabled attribute represented by a boolean which is used during neuroevolution.
"""
from random import random

from configs.genome_config import GenomeConfig


def cross(v1, v2, ratio: float = 0.5):
    """
    Inherit one of the two enabled attributes from the given parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    return v1 if random() < ratio else v2


def init(cfg: GenomeConfig):
    """Put enabled-state as requested by the config, 50-50 chance if not specified."""
    default = str(cfg.enabled_default).lower()
    if default in ('1', 'true'):
        return True
    elif default in ('0', 'false'):
        return False
    elif default in ('random', 'none'):
        return bool(random() < 0.5)
    raise RuntimeError(f"Unknown default value {default!r} for connection enabled attribute")


def mutate(v, cfg: GenomeConfig):
    """
    Mutate the given enabled-value based on the provided GenomeConfig file. Note that this mutations reflects the
    mutations seen in the other attributes since it *may* change the value.
    """
    r = random()
    if r < cfg.enabled_mutate_rate:
        return random() < 0.5
    return v
