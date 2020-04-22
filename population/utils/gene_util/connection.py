"""
connection.py

Gene-representation of a connection.
"""
from numpy import round

from configs.genome_config import GenomeConfig
from population.utils.attributes import conn_enabled, conn_weight
from population.utils.gene_util.base import BaseGene


class ConnectionGene(BaseGene):
    """Connection configuration, as specified by the Python-NEAT documentation."""
    
    __slots__ = {
        'enabled', 'weight',
    }
    
    def __init__(self, key, cfg: GenomeConfig):
        assert isinstance(key, tuple), f"ConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.enabled = conn_enabled.init(cfg)
        self.weight = conn_weight.init(cfg)
    
    def __str__(self):
        return f"ConnectionGene(\n" \
               f"\tkey={self.key}\n" \
               f"\tenabled={self.enabled}\n" \
               f"\tweight={round(self.weight, 2)})"
    
    def __repr__(self):
        return f"ConnectionGene(weight={round(self.weight, 2)}, enabled={self.enabled})"
    
    def distance(self, other, cfg):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled: d += 1.0
        return d * cfg.compatibility_weight
    
    def mutate(self, cfg: GenomeConfig):
        """
        Mutate the connection's attributes. Mutate the enabled state as last since it returns a value.

        :param cfg: GenomeConfig object
        :return: None: 'enabled' hasn't changed
                 True: 'enabled' is set to True
                 False: 'enabled' is set to False
        """
        self.weight = conn_weight.mutate(self.weight, cfg=cfg)
        pre = self.enabled
        
        # Enabling connection (if allowed) is done by the genome itself
        enabled = conn_enabled.mutate(self.enabled, cfg=cfg)
        if pre != enabled:
            return enabled
        return None
