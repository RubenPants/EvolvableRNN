"""
required_for_output_test.py

This test-case solely focuses on the 'required_for_output' method, found in graphs.py .
"""
import os
import unittest

from config import Config
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene
from population.utils.genome import Genome
from population.utils.network_util.graphs import required_for_output


def get_config():
    """Get a shared config-file."""
    cfg = Config()
    
    # Bot with three inputs
    cfg.bot.angular_dir = [True, False]
    cfg.bot.delta_dist_enabled = False
    
    # Genomes have two outputs
    cfg.genome.compatibility_disjoint_conn = 1  # Easier to calculate with
    cfg.genome.compatibility_disjoint_node = 1  # Easier to calculate with
    cfg.genome.compatibility_weight = 1  # Easier to calculate with
    cfg.genome.num_outputs = 2
    
    cfg.update()
    return cfg


def get_valid0(cfg: Config):
    """
    Simple network with all inputs and all outputs used.
    Configuration:
        0   1
       /    |
      2     \
     /  \    |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=0,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (-2, 2), (2, 0), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_valid1(cfg: Config):
    """
    Simple network with only one input and one output used.
    Configuration:
         0   1
             |
             |
             |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=1,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_valid2(cfg: Config):
    """
    Network with a recurrent connection (at node 2).
    Configuration:
         0   1
        /
       2>
      / \
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=2,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (-2, 2), (2, 0), (2, 2)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_invalid0(cfg: Config):
    """
    Genome with no connections.
    Configuration:
       0   1
       
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=0,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in []:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_invalid1(cfg: Config):
    """
    Genome with only one recurrent connection (at node 2).
    Configuration:
       0   1
       
         2>
       
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=1,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(2, 2)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_invalid2(cfg: Config):
    """
    Genome without connections to the output nodes.
    Configuration:
       0   1
       
         2>
         |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=2,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-2, 2), (2, 2)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_invalid3(cfg: Config):
    """
    Genome without connections to the input nodes.
    Configuration:
       0   1
           |
           2>
          
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=3,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(2, 2), (2, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_invalid4(cfg: Config):
    """
    Genome with connection from start to recurrent node, and from another recurrent node to the output.
    Configuration:
       0   1
           |
     2>    3>
     |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=4,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (2, 2), (3, 3), (3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_invalid5(cfg: Config):
    """
    Genome with connections between the hidden nodes and to one output node.
    Configuration:
       0   1
           |
        2--3
     
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=4,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(2, 3), (3, 2), (3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_pruned0(cfg: Config):
    """
    Genome with partially valid connections and nodes (dangling node on output).
    Configuration:
       0   1
      /    |
     2     3>
     |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=0,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (2, 0), (3, 1), (3, 3)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_pruned1(cfg: Config):
    """
    Genome with partially valid connections and nodes (dangling node on input).
    Configuration:
       0   1
      /
     2       3>
     |       |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=1,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (2, 0), (-3, 3), (3, 3)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_pruned2(cfg: Config):
    """
    Genome with partially valid connections and nodes (dangling node on other hidden node).
    Configuration:
       0   1
      /
     2---3>
     |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=2,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (2, 0), (2, 3), (3, 3)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_circular0(cfg: Config):
    """
    Genome with hidden nodes connected to the outputs.
    Configuration:
       0   1
       |   |
       2---3
     
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=2,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(2, 0), (2, 3), (3, 2), (3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_circular1(cfg: Config):
    """
    Genome with circular connections, not connected to any input or output node.
    Configuration:
       0   1
         
         4
        / \
       2---3
     
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=2,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    genome.nodes[4] = SimpleNodeGene(key=4, cfg=cfg.genome)  # Hidden node
    genome.nodes[4].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(2, 3), (3, 4), (4, 2)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_circular2(cfg: Config):
    """
    Genome with circular connections, not connected to the output genome.
    Configuration:
       0   1
       
     2---3
     |   |
    -1  -2  -3
    """
    # Create a dummy genome
    genome = Genome(
            key=2,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    # Reset the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 0
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0
    genome.nodes[2] = SimpleNodeGene(key=2, cfg=cfg.genome)  # Hidden node
    genome.nodes[2].bias = 0
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 2), (2, 3), (3, 2), (-2, 3)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


class ValidGenomes(unittest.TestCase):
    """Test when the given genomes are valid."""
    
    def test_valid0(self):
        """> Test a simple fully connected network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_valid0(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are three inputs, two outputs and one hidden node
        self.assertEqual(len(used_inp), 3)
        self.assertEqual(len(used_hid), 1)
        self.assertEqual(len(used_out), 2)
        # Number of connections are 4
        self.assertEqual(len(used_conn), 4)
    
    def test_valid1(self):
        """> Test a simple partial connected network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_valid1(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are one input (only one used), two outputs (always present, even though not used!)
        self.assertEqual(len(used_inp), 1)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # Only one connection present in the network
        self.assertEqual(len(used_conn), 1)
    
    def test_valid2(self):
        """> Test a partially connected network with a recurrent connection."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_valid2(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are two inputs, two outputs and one hidden node
        self.assertEqual(len(used_inp), 2)
        self.assertEqual(len(used_hid), 1)
        self.assertEqual(len(used_out), 2)
        # Three simple connections, and one recurrent
        self.assertEqual(len(used_conn), 3 + 1)


class InvalidGenomes(unittest.TestCase):
    """Test when the given genomes are invalid."""
    
    def test_invalid0(self):
        """> Test a completely unconnected genome."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_invalid0(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are only the two outputs
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections
        self.assertEqual(len(used_conn), 0)
    
    def test_invalid1(self):
        """> Test a unconnected network that only has one hidden node with a recurrent connection attached."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_invalid1(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are only the two outputs
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)
    
    def test_invalid2(self):
        """> Test a unconnected network that only has one hidden node with a recurrent connection attached."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_invalid2(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are only the two outputs
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)
    
    def test_invalid3(self):
        """> Test a unconnected complex network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_invalid3(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are only the two outputs
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)
    
    def test_invalid4(self):
        """> Test a unconnected complex network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_invalid4(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are only the two outputs
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)
    
    def test_invalid5(self):
        """> Test a unconnected complex network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_invalid5(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Number of nodes are only the two outputs
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)


class CircularGenomes(unittest.TestCase):
    """Test when the given genomes are invalid."""
    
    def test_circular0(self):
        """> Test a genome with circular hidden nodes connected to only the outputs."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_circular0(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Invalid genome
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)
    
    def test_circular1(self):
        """> Test a genome with circular hidden nodes, not connected to any input or output."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_circular1(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Invalid genome
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)
    
    def test_circular2(self):
        """> Test a genome with circular hidden nodes, only connected to the inputs."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_circular2(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Invalid genome
        self.assertEqual(len(used_inp), 0)
        self.assertEqual(len(used_hid), 0)
        self.assertEqual(len(used_out), 2)
        # No connections that were used to compute the outputs
        self.assertEqual(len(used_conn), 0)


class PrunedGenomes(unittest.TestCase):
    """Test when the given genomes are invalid."""
    
    def test_pruned0(self):
        """> Test a genome with a hidden recurrent node pruned from the output."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_pruned0(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Two outputs, one used input, one used hidden
        self.assertEqual(len(used_inp), 1)
        self.assertEqual(len(used_hid), 1)
        self.assertEqual(len(used_out), 2)
        # Two used connections
        self.assertEqual(len(used_conn), 2)
    
    def test_pruned1(self):
        """> Test a genome with a hidden recurrent node pruned from the input."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_pruned1(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Two outputs, one used input, one used hidden
        self.assertEqual(len(used_inp), 1)
        self.assertEqual(len(used_hid), 1)
        self.assertEqual(len(used_out), 2)
        # Two used connections
        self.assertEqual(len(used_conn), 2)
    
    def test_pruned2(self):
        """> Test a genome with a hidden recurrent node pruned from another hidden recurrent node."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genome = get_pruned2(cfg)
        
        # Test the required nodes
        used_inp, used_hid, used_out, used_conn = required_for_output(
                inputs={a for (a, _) in genome.connections if a < 0},
                outputs={i for i in range(cfg.genome.num_outputs)},
                connections=genome.connections,
        )
        
        # Two outputs, one used input, one used hidden
        self.assertEqual(len(used_inp), 1)
        self.assertEqual(len(used_hid), 1)
        self.assertEqual(len(used_out), 2)
        # Two used connections
        self.assertEqual(len(used_conn), 2)


def main():
    vg = ValidGenomes()
    vg.test_valid0()
    vg.test_valid1()
    vg.test_valid2()
    
    ig = InvalidGenomes()
    ig.test_invalid0()
    ig.test_invalid1()
    ig.test_invalid2()
    ig.test_invalid3()
    ig.test_invalid4()
    ig.test_invalid5()
    
    cg = CircularGenomes()
    cg.test_circular0()
    cg.test_circular1()
    cg.test_circular2()
    
    pg = PrunedGenomes()
    pg.test_pruned0()
    pg.test_pruned1()
    pg.test_pruned2()


if __name__ == '__main__':
    unittest.main()
