"""
genome_config.py

Configuration file corresponding the creation and mutation of the genomes.
"""
from itertools import count

from neat.six_util import iterkeys

from configs.base_config import BaseConfig


class GenomeConfig(BaseConfig):
    """Genome-specific configuration parameters."""
    
    __slots__ = {
        'bias_init_mean', 'bias_init_stdev', 'bias_max_value', 'bias_min_value', 'bias_mutate_power',
        'bias_mutate_rate', 'bias_replace_rate', 'compatibility_disjoint_conn', 'compatibility_disjoint_node',
        'compatibility_weight', 'conn_add_prob', 'conn_disable_prob', 'enabled_default', 'enabled_mutate_rate',
        'keys_input', 'keys_output', 'node_add_prob', 'node_disable_prob', 'node_indexer', 'num_inputs', 'num_outputs',
        'rnn_init_mean', 'rnn_init_stdev', 'rnn_max_value', 'rnn_min_value', 'rnn_mutate_power', 'rnn_mutate_rate',
        'rnn_prob_gru', 'rnn_prob_gru_nr', 'rnn_prob_gru_nu', 'rnn_prob_lstm', 'rnn_prob_simple_rnn',
        'rnn_replace_rate', 'weight_init_mean', 'weight_init_stdev', 'weight_max_value', 'weight_min_value',
        'weight_mutate_power', 'weight_mutate_rate', 'weight_replace_rate',
    }
    
    def __init__(self):
        # The mean of the gaussian distribution, used to select the bias attribute values for new nodes  [def=0]
        self.bias_init_mean: float = 0.
        # Standard deviation of gaussian distribution, used to select the bias attribute values of new nodes  [def=1]
        self.bias_init_stdev: float = 1.5
        # The maximum allowed bias value, biases above this threshold will be clamped to this value  [def=2]
        self.bias_max_value: float = 3.
        # The minimum allowed bias value, biases below this threshold will be clamped to this value  [def=-2]
        self.bias_min_value: float = -3.
        # The standard deviation of the zero-centered gaussian from which a bias value mutation is drawn  [def=0.5] TODO
        self.bias_mutate_power: float = .5
        # The probability that mutation will change the bias of a node by adding a random value  [def=0.1]  TODO
        self.bias_mutate_rate: float = .1
        # The probability that mutation will replace the bias of a node with a completely random value  [def=0.05]
        self.bias_replace_rate: float = .05
        # Weight of disjoint and excess connections on the genomic distance  [def=1.25]
        self.compatibility_disjoint_conn: float = 1.25
        # Weight of disjoint and excess nodes on the genomic distance  [def=2.5]
        self.compatibility_disjoint_node: float = 2.5
        # Coefficient for each weight or bias difference contribution to the genomic distance  [def=0.2]
        self.compatibility_weight: float = .2
        # Probability of adding a connection between existing nodes during mutation (each generation)  [def=0.02]  TODO
        self.conn_add_prob: float = .02
        # Probability of deleting an existing connection during mutation (each generation)  [def=0.02]  TODO
        self.conn_disable_prob: float = .02
        # Initial enabled-state of a connection  [def=True]
        self.enabled_default: bool = True
        # The probability that mutation will replace the 'enabled status' of a connection  [def=0.]
        self.enabled_mutate_rate: float = 0.
        # Input-keys, which are by convention negative starting from -1 and descending, set in update()  [def=/]
        self.keys_input = None
        # Output-keys, which start by convention from 0 and increment with each output, set in update()  [def=/]
        self.keys_output = None
        # Probability of adding a node during mutation (each generation)  [def=.005]  TODO
        self.node_add_prob: float = .005
        # Probability of removing a node during mutation (each generation)  [def=.005]  TODO
        self.node_disable_prob: float = .005
        # Node-indexer helps with the generation of node-keys  [def=/]
        self.node_indexer = None
        # Number of inputs, which are the robot's sensors  [def=/]
        self.num_inputs: int = 0
        # Number of output nodes, which are the wheels: [left_wheel, right_wheel]  [def=2]
        self.num_outputs: int = 2
        # Mean of the gaussian distribution used to select the RNN attribute values  [def=0]
        self.rnn_init_mean: float = 0.
        # Standard deviation of the gaussian used to select the RNN attributes values  [def=1]
        self.rnn_init_stdev: float = 1.5
        # The maximum allowed RNN-weight value, values above this will be clipped  [def=3]
        self.rnn_max_value: float = 6.
        # The minimum allowed RNN-weight value, values below this will be clipped  [def=-3]
        self.rnn_min_value: float = -6.
        # The standard deviation of the zero-centered gaussian from which a RNN value mutation is drawn  [def=0.2]
        self.rnn_mutate_power: float = .2
        # Probability of a RNN value to mutate  [def=0.2]  TODO
        self.rnn_mutate_rate: float = .2
        # The probability that a new node mutates towards a GRU-node  [def=0.]
        self.rnn_prob_gru = 0.
        # The probability that a new node mutates towards a GRU-NR-node (No Reset)  [def=0.]
        self.rnn_prob_gru_nr = 0.
        # The probability that a new node mutates towards a GRU-NU-node (No Update)  [def=0.]
        self.rnn_prob_gru_nu = 0.
        # The probability that a new node mutates towards a LSTM-node  [def=0.]
        self.rnn_prob_lstm = 0.
        # The probability that a new node mutates towards a simple RNN-node  [def=0.]
        self.rnn_prob_simple_rnn = 0.
        # Probability of assigning (single) random value in RNN, based on rnn_init_mean and rnn_init_stdev  [def=0.05]
        self.rnn_replace_rate: float = .05
        # Mean of the gaussian distribution used to select the weight attribute values for new connections  [def=0]
        self.weight_init_mean: float = 0.
        # Standard deviation of the gaussian used to select the weight attributes values for new connections  [def=3]
        self.weight_init_stdev: float = 3.
        # The maximum allowed weight value, weights above this value will be clipped to this value  [def=2]
        self.weight_max_value: float = 6.
        # The minimum allowed weight value, weights below this value will be clipped to this value  [def=-2]
        self.weight_min_value: float = -6.
        # The standard deviation of the zero-centered gaussian from which a weight value mutation is drawn [def=.5]TODO
        self.weight_mutate_power: float = .5
        # Probability of a weight (connection) to mutate  [def=0.1]  TODO
        self.weight_mutate_rate: float = .1
        # Probability of assigning completely new value, based on weight_init_mean and weight_init_stdev  [def=0.05]
        self.weight_replace_rate: float = .05
    
    def update(self, main_config):
        """Reload the current number of input sensors."""
        from environment.robot import get_number_of_sensors
        self.num_inputs: int = get_number_of_sensors(cfg=main_config.bot)
        self.keys_input = [-i - 1 for i in range(self.num_inputs)]
        self.keys_output = [i for i in range(self.num_outputs)]
        
        # The total probability of choosing one of the hidden nodes must be at most one
        assert sum([self.rnn_prob_lstm,
                    self.rnn_prob_gru,
                    self.rnn_prob_gru_nr,
                    self.rnn_prob_gru_nu,
                    self.rnn_prob_simple_rnn]) <= 1
    
    def get_new_node_key(self, node_dict):
        if self.node_indexer is None: self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)
        new_id = next(self.node_indexer)
        assert new_id not in node_dict
        return new_id
