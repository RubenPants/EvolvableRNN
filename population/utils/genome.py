"""
genome.py

Handles genomes (individuals in the population). A single genome has two types of genes:
 * node gene: specifies the configuration of a single node (e.g. activation function)
 * connection gene: specifies a single connection between two neurons (e.g. weight)
"""
from __future__ import division, print_function

from random import choice, random

from neat.six_util import iteritems, iterkeys

from configs.bot_config import BotConfig
from configs.genome_config import GenomeConfig
from environment.robot import get_snapshot
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.gru_no_reset import GruNoResetNodeGene
from population.utils.gene_util.gru_no_update import GruNoUpdateNodeGene
from population.utils.gene_util.lstm import LstmNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.rnn import RnnNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene
from population.utils.gene_util.simple_rnn import SimpleRnnNodeGene
from population.utils.network_util.graphs import creates_cycle, required_for_output
from utils.myutils import load_pickle, store_pickle


class Genome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world; pins are either input or
            output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's output and a pin/node
            input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique neuron by an implicit connection with
            weight one. This connection is permanently enabled.
        2. The output pin's key is always the same as the key for its associated neuron.
        3. Output neurons can be modified but not disabled.
        4. The input values are applied to the input pins unmodified.
    """
    
    def __init__(self, key, num_outputs, bot_config: BotConfig):
        # Unique identifier for a genome instance.
        self.key = key
        
        # Placeholders for the elementary parts of a genomes
        self.connections: dict = dict()  # Container for all the connections present in the genome
        self.nodes: dict = dict()  # Container for all the nodes (hidden and output)
        self.num_outputs: int = num_outputs  # Number of outputs in the genome
        self.fitness = None  # Container for the genome's fitness
        
        # Get snapshot of current robot configuration
        self.robot_snapshot = get_snapshot(cfg=bot_config)
    
    def configure_new(self, config: GenomeConfig):
        """Configure a new genome based on the given configuration."""
        # Create node genes for the output pins
        for node_key in config.keys_output: self.nodes[node_key] = self.create_output_node(config, node_key)
        
        # Add connections based on initial connectivity type
        self.connect_full_direct(config)
    
    def configure_crossover(self, config: GenomeConfig, genome1, genome2):
        """Configure a new genome by crossover from two parent genomes."""
        # Rank the parents based on fitness
        assert isinstance(genome1.fitness, (int, float))  # (key, fitness)
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness >= genome2.fitness:
            p1, p2 = genome1, genome2
        else:
            p1, p2 = genome2, genome1
        
        # Get the fitness ratio of the two parents (determines from which parent a child is most likely to inherit from)
        #  If very similar fitness values, ratio will be fixed to 0.5 (prevent division by ~0+0)
        ratio = 0.5 if abs(p1.fitness - p2.fitness) < 0.001 else p1.fitness / (p1.fitness + p2.fitness)
        
        # Inherit connection genes of the most fit genome, crossover the connection if present at both parents
        for key, cg1 in iteritems(p1.connections):
            cg2 = p2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy(cfg=config)
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cfg=config,
                                                      other=cg2,
                                                      ratio=ratio)
        
        # Inherit node genes
        for key, ng1 in iteritems(p1.nodes):
            ng2 = p2.nodes.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy(cfg=config)
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(cfg=config,
                                                other=ng2,
                                                ratio=ratio)
        
        # Make sure that all RNN-nodes are correctly configured (input_keys)
        self.update_rnn_nodes(config)
    
    def mutate(self, config: GenomeConfig):
        """Mutates this genome."""
        if random() < config.node_add_prob: self.mutate_add_node(config)
        if random() < config.node_disable_prob: self.mutate_disable_node(config)
        if random() < config.conn_add_prob: self.mutate_add_connection(config)
        if random() < config.conn_disable_prob: self.mutate_disable_connection()
        
        # Mutate connection genes
        for cid, cg in self.connections.items():
            mut_enabled = cg.mutate(config)
            if mut_enabled is not None:
                self.enable_connection(config=config, key=cid) if mut_enabled else self.disable_connection(key=cid)
        
        # Mutate node genes (bias etc.)
        for ng in self.nodes.values():
            ng.mutate(config)
    
    def mutate_add_node(self, config: GenomeConfig):
        """Add (or enable) a node as part of a mutation."""
        used_connections = self.get_used_connections()
        if not used_connections: return
        
        # Choose a random connection to split
        conn_to_split = choice(list(used_connections.values()))
        node_id = config.get_new_node_key(self.nodes)
        
        # Choose type of node to mutate to and add the node, must be done before adding the connection!
        r = random()
        if r <= config.rnn_prob_lstm:
            ng = self.create_lstm_node(config, node_id)
        elif r <= config.rnn_prob_lstm + config.rnn_prob_gru:
            ng = self.create_gru_node(config, node_id)
        elif r <= config.rnn_prob_lstm + config.rnn_prob_gru + config.rnn_prob_gru_nr:
            ng = self.create_gru_nr_node(config, node_id)
        elif r <= config.rnn_prob_lstm + config.rnn_prob_gru + config.rnn_prob_gru_nr + config.rnn_prob_gru_nu:
            ng = self.create_gru_nu_node(config, node_id)
        elif r <= config.rnn_prob_lstm + config.rnn_prob_gru + config.rnn_prob_gru_nr + config.rnn_prob_gru_nu + \
                config.rnn_prob_simple_rnn:
            ng = self.create_simple_rnn_node(config, node_id)
        else:
            ng = self.create_node(config, node_id)
        self.nodes[node_id] = ng
        
        # Disable this connection and create two new connections joining its nodes via the given node. The first
        # connection will simply forward its inputs (i.e. weight=1.0), whereas the second connection tries to mimic the
        # original (split) connection.
        self.disable_connection(key=conn_to_split.key)
        i, o = conn_to_split.key
        self.create_connection(config=config, input_key=i, output_key=node_id, weight=1.0)
        self.create_connection(config=config, input_key=node_id, output_key=o, weight=conn_to_split.weight)
    
    def mutate_disable_node(self, config: GenomeConfig):
        """Disable a node as part of a mutation, this is done by disabling all the node's adjacent connections."""
        # Get a list of all possible nodes to deactivate (i.e. all the hidden, non-output, nodes)
        available_nodes = [k for k in iterkeys(self.nodes) if k not in config.keys_output]
        used_connections = self.get_used_connections()
        if not available_nodes:
            return
        
        # Find all the adjacent connections and disable those
        disable_key = choice(available_nodes)
        connections_to_disable = set()
        for _, v in iteritems(used_connections):
            if disable_key in v.key:
                connections_to_disable.add(v.key)
        
        # Check if any connections left after disabling node
        for k in connections_to_disable: used_connections.pop(k)
        _, _, _, used_conn = required_for_output(
                inputs={a for (a, _) in used_connections if a < 0},
                outputs={i for i in range(self.num_outputs)},
                connections=used_connections,
        )
        
        # There are still connections left after disabling the nodes, disable connections for real
        if len(used_conn) > 0:
            for key in connections_to_disable:
                self.disable_connection(key=key, safe_disable=False)
    
    def create_connection(self, config: GenomeConfig, input_key: int, output_key: int, weight: float = None):
        """Add a connection to the genome."""
        # Create the connection
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0  # output_key is not one of the inputs (sensor readings)
        assert input_key not in config.keys_output
        key = (input_key, output_key)
        connection = ConnectionGene(key, cfg=config)
        
        if weight:
            assert isinstance(weight, float)
            connection.weight = weight
        
        self.connections[key] = connection
        self.enable_connection(config=config, key=key)
    
    def mutate_add_connection(self, config: GenomeConfig):
        """
        Attempt to add a new connection. A connection starts in the input_node and ends in the output_node.
        The restrictions laid on the mutation are:
         - An output of the network may never be an input_node (sending-end)
         - An input of the network may never be an output_node (receiving-end)
        """
        # List all the keys that are possible output nodes (i.e. all output and hidden nodes)
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)
        
        # List all the keys that are possible input-nodes (i.e. all input and hidden nodes)
        possible_inputs = [i for i in possible_outputs + config.keys_input if i not in config.keys_output]
        in_node = choice(possible_inputs)
        
        # Check if the new connection would create a cycle, discard if so
        key = (in_node, out_node)
        if creates_cycle(list(iterkeys(self.connections)), key):
            return
        
        # Don't duplicate connections
        if key in self.connections:
            self.enable_connection(config=config, key=key)
            return
        
        # Create the new connection
        self.create_connection(config, in_node, out_node)
    
    def mutate_disable_connection(self):
        """Disable the connection as part of a mutation."""
        used_connections = self.get_used_connections()
        if used_connections:
            key = choice(list(used_connections.keys()))
            self.disable_connection(key=key)
    
    def enable_connection(self, config: GenomeConfig, key: tuple):
        """Enable the connection, and ripple this through to its potential RNN cell."""
        assert key in self.connections
        self.connections[key].enabled = True
        if issubclass(self.nodes[key[1]].__class__, RnnNodeGene):
            self.nodes[key[1]].add_input_key(config, k=key[0])
    
    def disable_connection(self, key: tuple, safe_disable: bool = True):
        """Disable the connection, and ripple this through to its potential RNN cell."""
        assert key in self.connections
        # Test if other used connections remain when this connection is disabled
        if safe_disable:
            connections = self.get_used_connections()
            if key in connections: connections.pop(key)
            if len(connections) == 0: return
        
        # Perennial connections exist, disable chosen connection
        self.connections[key].enabled = False
        if issubclass(self.nodes[key[1]].__class__, RnnNodeGene):
            self.nodes[key[1]].remove_input_key(k=key[0])
    
    def size(self):
        """Returns genome 'complexity', taken to be (number of hidden nodes, number of enabled connections)"""
        inputs = {a for (a, _) in self.connections.keys() if a < 0}
        _, used_hid, _, used_conn = required_for_output(
                inputs=inputs,
                outputs={i for i in range(self.num_outputs)},
                connections=self.connections,
        )
        return len(used_hid), len(used_conn)
    
    def get_used_nodes(self):
        """Get all of the nodes currently used by the genome."""
        used_inp, used_hid, used_out, _ = required_for_output(
                inputs={a for (a, _) in self.connections if a < 0},
                outputs={i for i in range(self.num_outputs)},
                connections=self.connections,
        )
        # used_nodes only is a set of node-IDs, transform this to a node-dictionary
        return {nid: n for (nid, n) in self.nodes.items() if nid in (used_inp | used_hid | used_out)}
    
    def get_used_connections(self):
        """Get all of the connections currently used by the genome."""
        connections = self.connections.copy()
        _, _, _, used_conn = required_for_output(
                inputs={a for (a, _) in connections if a < 0},
                outputs={i for i in range(self.num_outputs)},
                connections=connections,
        )
        return used_conn
    
    def __str__(self):
        s = f"Key: {self.key}\n" \
            f"Fitness: {self.fitness}\n" \
            f"Nodes:\n"
        for k, ng in sorted(self.get_used_nodes().items(), key=lambda x: x[0]):
            s += f"\t{k} - {repr(ng)!s}\n"
        s += "Connections:\n"
        for k, cg in sorted(self.get_used_connections().items(), key=lambda x: x[0]):
            s += f"\t{k} - {repr(cg)!s}\n"
        return s
    
    def __repr__(self):
        return f"DefaultGenome(key={self.key}, " \
               f"n_nodes={len(self.get_used_nodes())}, " \
               f"n_connections={len(self.get_used_connections())})"
    
    @staticmethod
    def create_node(config: GenomeConfig, node_id: int):
        node = SimpleNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_output_node(config: GenomeConfig, node_id: int):
        node = OutputNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_lstm_node(config: GenomeConfig, node_id: int):
        node = LstmNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_gru_node(config: GenomeConfig, node_id: int):
        node = GruNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_gru_nr_node(config: GenomeConfig, node_id: int):
        node = GruNoResetNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_gru_nu_node(config: GenomeConfig, node_id: int):
        node = GruNoUpdateNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_simple_rnn_node(config: GenomeConfig, node_id: int):
        node = SimpleRnnNodeGene(node_id, cfg=config)
        return node
    
    def update_rnn_nodes(self, config: GenomeConfig):
        """Update all the hidden RNN-nodes such that their input_keys are correct."""
        for (key, node) in self.nodes.items():
            if issubclass(node.__class__, RnnNodeGene):
                # Get all the input-keys
                input_keys = set(a for (a, b) in self.get_used_connections().keys() if b == key)
                
                # Remove older inputs that aren't inputs anymore
                for k in reversed(node.input_keys):
                    if k not in input_keys: node.remove_input_key(k)
                
                # Add new inputs that were not yet inputs
                for k in input_keys:
                    if k not in node.input_keys: node.add_input_key(config, k)
                
                # Change in input_keys results in a change in weight_ih
                node.update_weight_xh()
                assert len(node.input_keys) == len(input_keys)
    
    def compute_full_connections(self, config: GenomeConfig, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each input connected to all hidden nodes (and
        output nodes if ``direct`` is set or there are no hidden nodes), each hidden node connected to all output nodes.
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.keys_output]
        output = [i for i in iterkeys(self.nodes) if i in config.keys_output]
        connections = []
        if hidden:
            for input_id in config.keys_input:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.keys_input:
                for output_id in output:
                    connections.append((input_id, output_id))
        
        return connections
    
    def connect_full_direct(self, config: GenomeConfig):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            self.create_connection(config, input_id, output_id)
    
    def save(self, path=None):
        """Method to solely save the genome."""
        if path is None: path = f'genome{self.key}'
        store_pickle(self, path)
    
    def load(self, path):
        """Load in a genome."""
        genome = load_pickle(path)
        self.key = genome.key
        self.connections = genome.connections
        self.nodes = genome.nodes
        self.num_outputs = genome.num_outputs
        self.fitness = genome.fitness
        self.robot_snapshot = genome.robot_snapshot
