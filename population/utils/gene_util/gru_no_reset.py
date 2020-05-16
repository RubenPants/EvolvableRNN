"""
gru_no_reset.py

Gene-representation of a hidden GRU node that lacks a reset gate.
"""
import sys

from numpy import round

from configs.genome_config import GenomeConfig
from population.utils.gene_util.rnn import RnnNodeGene

if 'linux' in sys.platform:
    from population.utils.rnn_cell_util.cy.berkeley_gru_no_reset_cy import GRUCellNoResetCy as GRU
else:
    from population.utils.rnn_cell_util.berkeley_gru_no_reset import GRUCellNoReset as GRU


class GruNoResetNodeGene(RnnNodeGene):
    """Custom GRU cell implementation."""
    
    __slots__ = {
        'bias', 'bias_h', 'input_keys', 'input_keys_full', 'weight_hh', 'weight_xh', 'weight_xh_full',
    }
    
    def __init__(self, key, cfg: GenomeConfig, input_keys=None, input_keys_full=None):
        super().__init__(
                key=key,
                hid_dim=2,
                cfg=cfg,
                input_keys=input_keys,
                input_keys_full=input_keys_full,
        )
    
    def __str__(self):
        self.update_weight_xh()
        bias = str(round(self.bias_h, 5)).replace('\n', ',')
        weight_hh = str(round(self.weight_hh, 5)).replace('\n', ',')
        weight_xh = str(round(self.weight_xh, 5)).replace('\n', ',')
        return f"GruNoResetNodeGene(\n" \
               f"\tkey={self.key}\n" \
               f"\tbias={bias},\n" \
               f"\tinput_keys={self.input_keys},\n" \
               f"\tweight_hh={weight_hh},\n" \
               f"\tweight_xh={weight_xh})"
    
    def __repr__(self):
        return f"GruNoResetNodeGene(inputs={self.input_keys!r})"
    
    def get_rnn(self, mapping=None):
        """Return a GRUCell based on current configuration. The mapping denotes which columns to use."""
        self.update_weight_xh()
        cell = GRU(
                input_size=len(mapping[mapping]) if mapping is not None else len(self.input_keys),
                bias=self.bias_h,
                weight_hh=self.weight_hh,
                weight_xh=self.weight_xh[:, mapping] if mapping is not None else self.weight_xh,
        )
        return cell
