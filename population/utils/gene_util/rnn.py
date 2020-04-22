"""
rnn.py

Global gene-representation of a hidden RNN node (LSTM, GRU, SimpleRNN).
"""
from numpy import concatenate as cat, zeros
from numpy.linalg import norm

from configs.genome_config import GenomeConfig
from population.utils.attributes import rnn
from population.utils.gene_util.base import BaseGene


class RnnNodeGene(BaseGene):
    """Custom GRU cell implementation."""
    
    __slots__ = {
        'bias', 'bias_h', 'hid_dim', 'input_keys', 'input_keys_full', 'weight_hh', 'weight_xh', 'weight_xh_full',
    }
    
    def __init__(self, key: int, hid_dim: int, cfg: GenomeConfig, input_keys=None, input_keys_full=None):
        assert isinstance(key, int), f"GruNodeGene key must be an int, not {key!r}"
        if input_keys and input_keys_full:
            for k in input_keys: assert k in input_keys_full
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.bias = 0  # Needed for feed-forward-network
        self.bias_h = rnn.init(cfg, hid_dim=hid_dim)
        self.hid_dim = hid_dim  # The hidden dimension of the network (number of hidden gates)
        self.input_keys: list = sorted(input_keys) if input_keys else []
        self.input_keys_full: list = sorted(input_keys_full) if input_keys_full else []
        self.weight_hh = rnn.init(cfg, hid_dim=hid_dim, input_size=1)
        self.weight_xh = None  # Updated via update_weight_xh
        self.weight_xh_full = rnn.init(cfg, hid_dim=hid_dim, input_size=len(self.input_keys_full))
        
        # Make sure that the GRU-cell is initialized correct
        self.update_weight_xh()
    
    def __str__(self):
        raise NotImplementedError(f"Implement string representation in child class {self.__class__}.")
    
    def __repr__(self):
        raise NotImplementedError(f"Implement node's representation in child class {self.__class__}.")
    
    def distance(self, other, cfg: GenomeConfig):
        """Calculate the average distance between two GRU nodes, which is determined by its coefficients."""
        d = 0
        d += norm(self.bias_h - other.bias_h)
        d += norm(self.weight_hh - other.weight_hh)
        
        # Compare only same input keys
        key_set = sorted(set(self.input_keys + other.input_keys))
        s = zeros((self.hid_dim, len(key_set)), dtype=float)
        o = zeros((self.hid_dim, len(key_set)), dtype=float)
        for i, k in enumerate(key_set):
            if k in self.input_keys: s[:, i] = self.weight_xh_full[:, self.input_keys_full.index(k)]
            if k in other.input_keys: o[:, i] = other.weight_xh_full[:, other.input_keys_full.index(k)]
        d += norm(s - o)
        d /= 3  # Divide by three since the average node weight-difference should be calculated
        return d * cfg.compatibility_weight
    
    def mutate(self, cfg: GenomeConfig):
        self.bias_h = rnn.mutate_1d(self.bias_h, cfg=cfg, bias=True)
        self.weight_hh = rnn.mutate_2d(self.weight_hh, cfg=cfg)
        self.weight_xh_full = rnn.mutate_2d(self.weight_xh_full, cfg=cfg,
                                            mapping=[k in self.input_keys for k in self.input_keys_full])
    
    def update_weight_xh(self):
        """Update weight_xh to be conform with the current input_keys-set."""
        self.weight_xh = zeros((self.hid_dim, len(self.input_keys)), dtype=float)
        for i, k in enumerate(self.input_keys):
            self.weight_xh[:, i] = self.weight_xh_full[:, self.input_keys_full.index(k)]
    
    def get_rnn(self, mapping=None):
        raise NotImplementedError(f"Implement the numpy-representation of the gene in child class {self.__class__}.")
    
    def add_input_key(self, cfg: GenomeConfig, k: int):
        """Extend the input-key list with the given key, and expand the corresponding weights."""
        # Update self.weight_xh_full if key never seen before
        if k not in self.input_keys_full:
            # Find the index to insert the key
            lst = [i + 1 for i in range(len(self.input_keys_full)) if self.input_keys_full[i] < k]  # List of indices
            i = lst[-1] if lst else 0  # Index to insert key in
            
            # Save key to list
            self.input_keys_full.insert(i, k)
            
            # Update weight_xh_full correspondingly by inserting random initialized tensor in correct position
            new_tensor = rnn.init(cfg, hid_dim=self.hid_dim, input_size=1)
            assert new_tensor.shape == (self.hid_dim, 1)
            self.weight_xh_full = cat((self.weight_xh_full[:, :i], new_tensor, self.weight_xh_full[:, i:]), axis=1)
        
        # Update input_keys (current key-set) analogously
        if k not in self.input_keys:
            lst = [i + 1 for i in range(len(self.input_keys)) if self.input_keys[i] < k]  # List of indices
            i = lst[-1] if lst else 0
            self.input_keys.insert(i, k)
    
    def remove_input_key(self, k):
        """Delete one of the input_keys, input_keys_full and weight_xh_full are left unchanged."""
        if k in self.input_keys: self.input_keys.remove(k)
