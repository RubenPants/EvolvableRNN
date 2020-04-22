"""
feed_forward_net_cy.py

Create a simple feedforward network. The network will propagate one iteration at a time, doing the following:
 1) Update the hidden nodes their state taking the input and other hidden nodes into account
 2) Execute the RNNs such that their current state is updated (input=current_state)
 2) Update the output nodes their state taking the input and hidden nodes into account
"""
import numpy as np
cimport numpy as np

from population.utils.gene_util.rnn import RnnNodeGene
from population.utils.network_util.cy.activations_cy import sigmoid_cy
from population.utils.network_util.graphs import required_for_output

cdef class FeedForwardNetCy:
    """Custom representation of a feedforward network used by the genomes to make predictions."""
    
    __slots__ = {
        'act_f', 'bs', 'dtype', 'rnn_array', 'rnn_idx', 'rnn_map', 'rnn_state', 'hid2hid', 'hid2out', 'hidden_act',
        'hidden_biases', 'in2hid', 'in2out', 'n_rnn', 'n_hidden', 'n_inputs', 'n_outputs', 'output_act',
        'output_biases',
    }
    
    def __init__(self,
                 np.ndarray input_idx, np.ndarray hidden_idx, np.ndarray rnn_idx, np.ndarray output_idx,
                 tuple in2hid, tuple in2out,
                 tuple hid2hid, tuple hid2out,
                 np.ndarray hidden_biases, np.ndarray output_biases,
                 np.ndarray rnn_array, np.ndarray rnn_map,
                 activation,
                 int batch_size=1,
                 np.ndarray initial_read=None,
                 dtype=np.float64,
                 ):
        """
        Create a simple feedforward network used as the control-mechanism for the drones.
        
        :param input_idx: Input indices (sensors)
        :param hidden_idx: Hidden simple-node indices (DefaultGeneNode) in the network
        :param rnn_idx: Hidden RNN-node indices (DefaultGeneNode) in the network
        :param output_idx: Output indices (the two differential wheels)
        :param in2hid: Connections connecting the input nodes to the hidden nodes
        :param in2out: Connections directly connecting from the inputs to the outputs
        :param hid2hid: Connections between the hidden nodes
        :param hid2out: Connections from hidden nodes towards the outputs
        :param rnn_array: Array of RNN cell objects (length equals len(rnn_idx))
        :param rnn_map: Boolean matrix mapping raw inputs to inputs used by RNN cell for a single batch
        :param activation: The used activation function (squishing)
        :param batch_size: Needed to setup network-dimensions
        :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
        :param dtype: Value-type used in the tensors
        """
        # Storing the input arguments (needed later on)
        self.act_f = activation
        self.dtype = dtype
        self.rnn_idx = rnn_idx
        self.n_inputs = len(input_idx)
        self.n_hidden = len(hidden_idx)
        self.n_rnn = len(rnn_idx)
        self.n_outputs = len(output_idx)
        self.bs = batch_size
        
        # Setup the rnn_map
        cdef list rnn_map_temp = []  # Numpy-append has unwanted behaviour
        for i, m in enumerate(rnn_map):
            rnn_map_temp.append(np.tile(rnn_map[i], (batch_size, 1)))
        self.rnn_map = np.asarray(rnn_map_temp, dtype=bool)
        
        # Placeholders, initialized during reset
        self.rnn_state = None  # State of the RNNs
        self.hidden_act = None  # Activations of the hidden nodes
        self.output_act = None  # Activations of the output nodes
        
        # Do not create the hidden-related matrices if hidden-nodes do not exist
        #  If they do not exist, a single matrix directly mapping inputs to outputs is only used
        if self.n_hidden > 0:
            self.in2hid = dense_from_coo((self.n_hidden, self.n_inputs), in2hid, dtype=dtype)
            self.hid2hid = dense_from_coo((self.n_hidden, self.n_hidden), hid2hid, dtype=dtype)
            self.hid2out = dense_from_coo((self.n_outputs, self.n_hidden), hid2out, dtype=dtype)
            self.rnn_array = rnn_array
        self.in2out = dense_from_coo((self.n_outputs, self.n_inputs), in2out, dtype=dtype)
        
        # Fill in the biases
        if self.n_hidden > 0:
            self.hidden_biases = np.asarray(hidden_biases, dtype=dtype)
        self.output_biases = np.asarray(output_biases, dtype=dtype)
        
        # Put network to initial (default) state
        self.reset(initial_read=initial_read)
    
    def __call__(self, np.ndarray inputs):
        """
        Activate the network. This is used during the call of "query-net". It will feed the inputs into the network and
        return the resulting outputs.

        :param inputs: (batch_size, n_inputs)
        :return: The output-values (i.e. floats for the differential wheels) of shape (batch_size, n_outputs)
        """
        cdef int i, rnn_idx
        cdef np.ndarray output_inputs
        
        # Denote the impact the inputs have directly on the outputs
        output_inputs = np.matmul(self.in2out, inputs.transpose()).transpose()
        
        # Denote the impact hidden nodes have on the outputs, if there are hidden nodes
        if self.n_hidden > 0:
            # Nice to know:
            #  - np.transpose() will transpose the tensor
            #  - np.matmul(tensor1, tensor2) will perform a matrix multiplication between tensor and tensor2
            
            # The activation is defined by:
            #  - the inputs mapping to the hidden nodes
            #  - the hidden nodes mapping to themselves
            #  - the hidden nodes' biases
            
            # 1) Propagate the hidden nodes
            self.hidden_act = self.act_f(np.matmul(self.in2hid, inputs.transpose()).transpose() +
                                         np.matmul(self.hid2hid, self.hidden_act.transpose()).transpose() +
                                         self.hidden_biases)
            
            # 2) Execute the RNN nodes if they exists (updating current hidden state)
            for i, rnn_idx in enumerate(self.rnn_idx):
                self.rnn_state[:, i] = self.rnn_array[i](
                        np.concatenate((self.in2hid[rnn_idx] * inputs,
                                        self.hid2hid[rnn_idx] * self.hidden_act),
                                       axis=1)[self.rnn_map[i]].reshape(self.bs, self.rnn_array[i].input_size)
                )
                self.hidden_act[:, rnn_idx] = self.rnn_state[:, i, 0]
            
            # 3) Propagate hidden-values to the outputs
            output_inputs += np.matmul(self.hid2out, self.hidden_act.transpose()).transpose()
        
        # Define the values of the outputs, which is the sum of their received inputs and their corresponding bias
        self.output_act = self.act_f(output_inputs + self.output_biases)
        return self.output_act
    
    cpdef void reset(self, np.ndarray initial_read=None):
        """
        Set the network back to initial state.
        
        :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
        """
        cdef int i, rnn_idx
        cdef np.ndarray inputs, outputs
        
        # Reset the network back to zero inputs
        self.rnn_state = np.zeros((self.bs, self.n_rnn, 1), dtype=self.dtype)  # RNN outputs are single float
        self.hidden_act = np.zeros((self.bs, self.n_hidden), dtype=self.dtype) if self.n_hidden > 0 else None
        self.output_act = np.zeros((self.bs, self.n_outputs), dtype=self.dtype)
        
        # Initialize the network on maximum sensory inputs
        if (initial_read is not None) and self.n_hidden > 0:
            for _ in range(self.n_hidden + 20):  # RNNs need a little bit of warmth to start up
                # Code below is straight up stolen from 'activate(self, inputs)'
                inputs = np.asarray([initial_read] * self.bs, dtype=self.dtype)  # TODO: Improve
                output_inputs = np.matmul(self.in2out, inputs.transpose()).transpose()
                self.hidden_act = self.act_f(np.matmul(self.in2hid, inputs.transpose()).transpose() +
                                             np.matmul(self.hid2hid, self.hidden_act.transpose()).transpose() +
                                             self.hidden_biases)
                for i, rnn_idx in enumerate(self.rnn_idx):
                    self.rnn_state[:, i] = self.rnn_array[i](
                            np.concatenate((self.in2hid[rnn_idx] * inputs,
                                            self.hid2hid[rnn_idx] * self.hidden_act),
                                           axis=1)[self.rnn_map[i]].reshape(self.bs, self.rnn_array[i].input_size))
                    self.hidden_act[:, rnn_idx] = self.rnn_state[:, i, 0]
                output_inputs += np.matmul(self.hid2out, self.hidden_act.transpose()).transpose()
                self.output_act = self.act_f(output_inputs + self.output_biases)


cpdef int k2i(int k, dict input_k2i, np.ndarray input_keys, dict output_k2i, np.ndarray output_keys, dict hidden_k2i):
        """Convert key to their corresponding index."""
        return input_k2i.get(k) if k in input_keys \
            else output_k2i.get(k) if k in output_keys \
            else hidden_k2i.get(k)


cpdef FeedForwardNetCy make_net_cy(genome,
                                   genome_config,
                                   int batch_size=1,
                                   np.ndarray initial_read=None,
                                   logger=None):
    """
    This class will unravel the genome and create a feed-forward network based on it. In other words, it will create
    the phenotype (network) suiting the given genome.
    
    :param genome: The genome for which a network must be created
    :param genome_config: GenomeConfig object
    :param batch_size: Batch-size needed to setup network dimension
    :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
    :param logger: A population's logger
    """
    cdef set used_inp, used_hid, used_out, used_nodes
    cdef dict used_conn
    cdef np.ndarray input_keys, hidden_keys, rnn_keys, output_keys
    cdef np.ndarray hidden_biases, output_biases
    cdef dict input_k2i, hidden_k2i, output_k2i
    cdef int k, i
    cdef np.ndarray input_idx, hidden_idx, rnn_idx, output_idx
    cdef tuple in2hid, hid2hid, in2out, hid2out
    cdef int i_key, o_key
    cdef list idxs, vals
    cdef np.ndarray rnn_array, rnn_map
    cdef list rnn_map_temp
    cdef int rnn_key
    cdef np.ndarray mapping, weight_map
    
    # Collect the nodes whose state is required to compute the final network output(s), this excludes the inputs
    used_inp, used_hid, used_out, used_conn = required_for_output(
            inputs=set(genome_config.keys_input),
            outputs=set(genome_config.keys_output),
            connections=genome.connections
    )
    used_nodes = used_inp | used_hid | used_out
    if initial_read is not None: assert len(genome_config.keys_input) == len(initial_read)
    
    # Get a list of all the (used) input, (used) hidden, and output keys
    input_keys = np.asarray(sorted(genome_config.keys_input))
    hidden_keys = np.asarray([k for k in genome.nodes.keys() if
                              (k not in genome_config.keys_output and k in used_nodes)])
    rnn_keys = np.asarray([k for k in hidden_keys if issubclass(genome.nodes[k].__class__, RnnNodeGene)])
    output_keys = np.asarray(genome_config.keys_output)
    
    # Define the biases, note that inputs do not have a bias (since they aren't actually nodes!)
    hidden_biases = np.asarray([genome.nodes[k].bias for k in hidden_keys])
    output_biases = np.asarray([genome.nodes[k].bias for k in output_keys])
    
    # Create a mapping of a node's key to their index in their corresponding list
    input_k2i = {k: i for i, k in enumerate(input_keys)}
    hidden_k2i = {k: i for i, k in enumerate(hidden_keys)}
    output_k2i = {k: i for i, k in enumerate(output_keys)}
    
    # Position-encode (index) the keys
    input_idx = np.asarray([k2i(k, input_k2i, input_keys, output_k2i, output_keys, hidden_k2i)
                            for k in input_keys])
    hidden_idx = np.asarray([k2i(k, input_k2i, input_keys, output_k2i, output_keys, hidden_k2i)
                             for k in hidden_keys])
    rnn_idx = np.asarray([k2i(k, input_k2i, input_keys, output_k2i, output_keys, hidden_k2i)
                          for k in rnn_keys])
    output_idx = np.asarray([k2i(k, input_k2i, input_keys, output_k2i, output_keys, hidden_k2i)
                             for k in output_keys])
    
    # Only feed-forward connections considered, these lists contain the connections and their weights respectively
    #  Note that the connections are index-based and not key-based!
    in2hid = ([], [])
    hid2hid = ([], [])
    in2out = ([], [])
    hid2out = ([], [])
    
    # Convert the key-based connections to index-based connections one by one, also save their weights
    #  At this point, it is already known that all connections are used connections
    for conn in used_conn.values():
        # Convert to index-based
        i_key, o_key = conn.key
        i_idx = k2i(i_key, input_k2i, input_keys, output_k2i, output_keys, hidden_k2i)
        o_idx = k2i(o_key, input_k2i, input_keys, output_k2i, output_keys, hidden_k2i)
        
        # Store
        if i_key in input_keys and o_key in hidden_keys:
            idxs, vals = in2hid
        elif i_key in hidden_keys and o_key in hidden_keys:
            idxs, vals = hid2hid
        elif i_key in input_keys and o_key in output_keys:
            idxs, vals = in2out
        elif i_key in hidden_keys and o_key in output_keys:
            idxs, vals = hid2out
        else:
            msg = f"{genome}" \
                  f"\ni_key: {i_key}, o_key: {o_key}" \
                  f"\ni_key in input_keys: {i_key in input_keys}" \
                  f"\ni_key in hidden_keys: {i_key in hidden_keys}" \
                  f"\ni_key in output_keys: {i_key in output_keys}" \
                  f"\no_key in input_keys: {o_key in input_keys}" \
                  f"\no_key in hidden_keys: {o_key in hidden_keys}" \
                  f"\no_key in output_keys: {o_key in output_keys}"
            if logger: logger(msg)
            raise ValueError(f'Invalid connection from key {i_key} to key {o_key}')
        
        # Append to the lists of the right tuple
        idxs.append((o_idx, i_idx))  # Connection: to, from
        vals.append(conn.weight)  # Connection: weight
    
    # Create the RNN cells and put them in a list
    rnn_array = np.asarray([])
    rnn_map_temp = []  # Keep, otherwise errors occur
    for rnn_key in rnn_keys:
        # Query the node that contains the RNN cell's weights
        node = genome.nodes[rnn_key]
        
        # Create a map of all inputs/hidden nodes to the ones used by the RNN cell (as inputs)
        mapping = np.asarray([], dtype=bool)
        for k in input_keys:
            mapping = np.append(mapping, True if k in node.input_keys else False)
        for k in hidden_keys:
            mapping = np.append(mapping, True if k in node.input_keys else False)
        weight_map = np.asarray([k in np.append(input_keys, hidden_keys) for k in node.input_keys])
        
        # Add the RNN cell and its corresponding mapping to the list of used RNN cells
        rnn_array = np.append(rnn_array, node.get_rnn(mapping=weight_map))
        assert len(mapping[mapping]) == rnn_array[-1].input_size
        rnn_map_temp.append(mapping)
    rnn_map = np.asarray(rnn_map_temp, dtype=bool)
    
    return FeedForwardNetCy(
            input_idx=input_idx, hidden_idx=hidden_idx, rnn_idx=rnn_idx, output_idx=output_idx,
            in2hid=in2hid, in2out=in2out,
            hid2hid=hid2hid, hid2out=hid2out,
            hidden_biases=hidden_biases, output_biases=output_biases,
            rnn_array=rnn_array, rnn_map=rnn_map,
            batch_size=batch_size,
            initial_read=initial_read,
            activation=sigmoid_cy,
    )


# noinspection PyArgumentList
cpdef np.ndarray dense_from_coo(shape, conns, dtype=np.float64):
    """
    Create a dense matrix based on the coordinates it will represent.

    :param shape: Tuple (output_size, input_size) ~ (nr_rows, nr_cols)
    :param conns: The connections that are being represented by the matrix, these connections are index-based
    :param dtype: Tensor type
    :return: PyTorch tensor
    """
    # Initialize an empty matrix of correct shape
    mat = np.zeros(shape, dtype=dtype)
    # Split the connections-tuple in its corresponding indices- and weight-lists
    idxs, weights = conns
    # If no indices (i.e. no connections), return the empty matrix
    if len(idxs) == 0: return mat
    # Split the idxs (e.g. [(A, B)]) to rows ([A]) and cols ([B])
    rows, cols = np.array(idxs).transpose()
    # Put the weights on the correct spots in the empty tensor
    mat[rows, cols] = np.asarray(weights, dtype=dtype)
    return mat

