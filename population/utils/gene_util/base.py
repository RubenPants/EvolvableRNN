"""
base.py

The base-gene, containing the shared gene-parameters.
"""
from abc import abstractmethod
from copy import deepcopy

from configs.genome_config import GenomeConfig


class BaseGene(object):
    """BaseGene specifies which methods a gene-object must implement."""
    
    __slots__ = {
        'key',
    }
    
    def __init__(self, key):
        """Key to identify the gene. Each gene has a distinct key."""
        self.key = key
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError(f"__str__ not set for gene with key {self.key}")
    
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError(f"__repr__ not set for gene with key {self.key}")
    
    def __lt__(self, other):
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key
    
    def copy(self, cfg):
        new_gene = self.__class__(self.key, cfg=cfg)
        for param in self.__slots__:
            attr = getattr(self, param)
            
            # Data-types that carry no references
            if type(attr) in [int, float, bool, str, complex]:
                setattr(new_gene, param, attr)
            
            # Deepcopy the attributes that carry references
            else:
                setattr(new_gene, param, deepcopy(attr))
        return new_gene
    
    @abstractmethod
    def distance(self, other, cfg: GenomeConfig):
        raise NotImplementedError(f"Distance is not implemented for gene {self.key}")
    
    @abstractmethod
    def mutate(self, cfg: GenomeConfig):
        raise NotImplementedError(f"Mutation is not implemented for gene {self.key}")
