"""
lstm.py

Gene-representation of a hidden LSTM node.
"""
import sys

from numpy import round

from configs.genome_config import GenomeConfig
from population.utils.gene_util.rnn import RnnNodeGene

if 'linux' in sys.platform:
    from population.utils.rnn_cell_util.cy.lstm_cy import LSTMCellCy as LSTM
else:
    from population.utils.rnn_cell_util.lstm import LSTMCell as LSTM


class LstmNodeGene(RnnNodeGene):
    """Custom LSTM cell implementation."""
    
    __slots__ = {
        'bias', 'bias_h', 'input_keys', 'input_keys_full', 'weight_hh', 'weight_xh', 'weight_xh_full',
    }
    
    def __init__(self, key, cfg: GenomeConfig, input_keys=None, input_keys_full=None):
        super().__init__(
                key=key,
                hid_dim=4,
                cfg=cfg,
                input_keys=input_keys,
                input_keys_full=input_keys_full,
        )
    
    def __str__(self):
        bias = str(round(self.bias_h, 2)).replace('\n', ',')
        weight_hh = str(round(self.weight_hh, 2)).replace('\n', ',')
        weight_xh = str(round(self.weight_xh, 2)).replace('\n', ',')
        return f"LstmNodeGene(\n" \
               f"\tkey={self.key}\n" \
               f"\tbias={bias},\n" \
               f"\tinput_keys={self.input_keys},\n" \
               f"\tweight_hh={weight_hh},\n" \
               f"\tweight_xh={weight_xh})"
    
    def __repr__(self):
        return f"LstmNodeGene(inputs={self.input_keys!r})"
    
    def get_rnn(self, mapping=None):
        """Return a LSTMCell based on current configuration. The mapping denotes which columns to use."""
        self.update_weight_xh()
        cell = LSTM(
                input_size=len(mapping[mapping]) if mapping is not None else len(self.input_keys),
                bias=self.bias_h,
                weight_hh=self.weight_hh,
                weight_xh=self.weight_xh[:, mapping] if mapping is not None else self.weight_xh,
        )
        return cell
