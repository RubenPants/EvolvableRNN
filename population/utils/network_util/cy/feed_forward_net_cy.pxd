"""
feed_forward_net_cy.py

Create a simple feedforward network. The network will propagate one iteration at a time, doing the following:
 1) Update the hidden nodes their state taking the input and other hidden nodes into account
 2) Execute the RNNs such that their current state is updated (input=current_state)
 2) Update the output nodes their state taking the input and hidden nodes into account
"""
cimport numpy as np

cdef class FeedForwardNetCy:
    """Custom cython representation of a feedforward network used by the genomes to make predictions."""
    cdef act_f, dtype
    cdef np.ndarray rnn_idx
    cdef int n_inputs, n_hidden, n_rnn, n_outputs, bs
    cdef np.ndarray rnn_map
    cdef np.ndarray rnn_state, hidden_act, output_act
    cdef np.ndarray in2out, in2hid, hid2hid, hid2out
    cdef np.ndarray rnn_array
    cdef np.ndarray hidden_biases, output_biases
    
    cpdef void reset(self, np.ndarray initial_read=?)

cpdef int k2i(int k, dict input_k2i, np.ndarray input_keys, dict output_k2i, np.ndarray output_keys, dict hidden_k2i)

cpdef FeedForwardNetCy make_net_cy(genome, genome_config, int batch_size=?, np.ndarray initial_read=?, logger=?)

cpdef np.ndarray dense_from_coo(shape, conns, dtype=?)
