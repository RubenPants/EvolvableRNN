"""
simple_node.py

Gene representation of a simple hidden node.
"""
from numpy import round

from configs.genome_config import GenomeConfig
from population.utils.attributes import bias
from population.utils.gene_util.base import BaseGene


class SimpleNodeGene(BaseGene):
    """Simple node configuration, as specified by the Python-NEAT documentation."""
    
    __slots__ = {
        'bias',
    }
    
    def __init__(self, key, cfg: GenomeConfig):
        assert isinstance(key, int), f"SimpleNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.bias: float = bias.init(cfg)
    
    def __str__(self):
        return f"SimpleNodeGene(\n" \
               f"\tkey={self.key}\n" \
               f"\tbias={round(self.bias, 2)})"
    
    def __repr__(self):
        return f"SimpleNodeGene(bias={round(self.bias, 2)})"
    
    def distance(self, other, cfg: GenomeConfig):
        d = abs(self.bias - other.bias)
        return d * cfg.compatibility_weight
    
    def mutate(self, cfg: GenomeConfig):
        self.bias = bias.mutate(self.bias, cfg=cfg)
