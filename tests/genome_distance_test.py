"""
genome_distance_test.py

Test on the genome-distance calculations. These tests mainly evaluate the two used Cache-classes:
 * GenomeDistanceCache
 * NodeComparingCache
"""
import os
import unittest

from config import Config
from population.utils.cache.genome_distance import GenomeDistanceCache
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene
from population.utils.genome import Genome


def get_genome0(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
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


def get_genome1(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0   1
       /    |
      3     \
     /  \    |
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
    genome.nodes[3] = SimpleNodeGene(key=3, cfg=cfg.genome)  # Hidden node
    genome.nodes[3].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 3), (-2, 3), (3, 0), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_genome2(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0   1
       /    |
      2     \
     /       |
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
    for key in [(-1, 2), (2, 0), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_genome3(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0   1
       /    |
      4     \
     /       |
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
    genome.nodes[4] = SimpleNodeGene(key=4, cfg=cfg.genome)  # Hidden node
    genome.nodes[4].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 4), (4, 0), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_deep_genome0(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0       1
        |       |
        7       |
        |       |
        6       |
        |       |
        5       |
        |       |
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
    for k in [5, 6, 7]:
        genome.nodes[k] = SimpleNodeGene(key=k, cfg=cfg.genome)  # Hidden node
        genome.nodes[k].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 5), (5, 6), (6, 7), (7, 0), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_deep_genome1(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0       1
        |       |
        10      |
        |       |
        9       |
        |       |
        8       |
        | \     |
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
    for k in [8, 9, 10]:
        genome.nodes[k] = SimpleNodeGene(key=k, cfg=cfg.genome)  # Hidden node
        genome.nodes[k].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 8), (8, 9), (9, 10), (10, 0), (-2, 8), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_deep_genome2(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0       1
        |       |
        13      |
        |       |
        12      |
        |  \    |
        11  |   |
        |   |   |
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
    for k in [11, 12, 13]:
        genome.nodes[k] = SimpleNodeGene(key=k, cfg=cfg.genome)  # Hidden node
        genome.nodes[k].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 11), (11, 12), (12, 13), (13, 0), (-2, 12), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_deep_genome3(cfg: Config):
    """
    Genome with all biases set to 0, only simple hidden nodes used, all connections enabled with weight 1.
    Configuration:
        0       1
        |       |
        16      |
        |  \    |
        15  |   |
        |   |   |
        14  |   |
        |   |   |
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
    for k in [14, 15, 16]:
        genome.nodes[k] = SimpleNodeGene(key=k, cfg=cfg.genome)  # Hidden node
        genome.nodes[k].bias = 0
    
    # Reset the connections
    genome.connections = dict()
    for key in [(-1, 14), (14, 15), (15, 16), (16, 0), (-2, 16), (-3, 1)]:
        genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
        genome.connections[key].weight = 1
        genome.connections[key].enabled = True
    
    return genome


def get_config():
    """Get a shared config-file."""
    cfg = Config()
    
    # Bot with three inputs
    cfg.bot.angular_dir = [True, False]
    cfg.bot.delta_dist_enabled = False
    
    # Genomes have two outputs
    # cfg.genome.compatibility_disjoint_conn = 1  # Easier to calculate with
    # cfg.genome.compatibility_disjoint_node = 1  # Easier to calculate with
    # cfg.genome.compatibility_weight = 1  # Easier to calculate with
    cfg.genome.num_outputs = 2
    
    cfg.update()
    return cfg


class SameGenome(unittest.TestCase):
    """Test the distance between two exact same genomes."""
    
    def test_zero_distance(self):
        """> Test if False is returned when comparing nodes of different type."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes (each type of genome twice)
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_genome0(cfg)
        genomes[1] = get_genome0(cfg)
        genomes[1].key = 1  # Update
        genomes[2] = get_genome1(cfg)
        genomes[3] = get_genome1(cfg)
        genomes[3].key = 3  # Update
        genomes[4] = get_genome2(cfg)
        genomes[5] = get_genome2(cfg)
        genomes[5].key = 5  # Update
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Get the genome-distance
        self.assertEqual(cache(genome0=genomes[0], genome1=genomes[1]), 0)
        self.assertEqual(cache(genome0=genomes[2], genome1=genomes[3]), 0)
        self.assertEqual(cache(genome0=genomes[4], genome1=genomes[5]), 0)
    
    def test_distance_similar_genome(self):
        """> Test if two different genomes with exact same architecture have zero distance."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_genome0(cfg)
        genomes[1] = get_genome1(cfg)
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Get the genome-distance
        self.assertEqual(cache(genome0=genomes[0], genome1=genomes[1]), 0)
    
    def test_same_internal_representation(self):
        """> Test if the merged hidden nodes have the same internal representation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_genome0(cfg)
        genomes[1] = get_genome1(cfg)
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Hidden nodes with keys 2 and 3 should have the same internal representation since they represent the same node
        self.assertEqual(cache.node_cache.index_map[2], cache.node_cache.index_map[3])


class DifferentGenome(unittest.TestCase):
    """Test the distance between two exact same genomes."""
    
    def test_distance(self):
        """> Test if the distance holds for two completely different genomes."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_genome0(cfg)
        genomes[3] = get_genome3(cfg)
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Get the genome-distance
        result = cache(genome0=genomes[0], genome1=genomes[3])
        self.assertGreater(result, 0)
        
        # Calculation:
        #  2 disjoint nodes
        #  2 identical nodes (outputs)
        #  5 disjoint connections (1 completely removed, 2x2 others connected to different genes)
        #  1 identical connection
        node_distance = (2 * cfg.genome.compatibility_disjoint_node + 2 * 0) / 3
        conn_distance = (5 * cfg.genome.compatibility_disjoint_conn + 1 * 0) / 4
        self.assertEqual(result, node_distance + conn_distance)


class DifferentDeepGenome(unittest.TestCase):
    """Test the distance between two exact same genomes."""
    
    def test_genomes01(self):
        """> Test the distance and internal representations of the two given nodes."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_deep_genome0(cfg)
        genomes[1] = get_deep_genome1(cfg)
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Get the genome-distance
        result = cache(genome0=genomes[0], genome1=genomes[1])
        self.assertGreater(result, 0)
        
        # Calculation:
        #  6 disjoint nodes
        #  2 identical nodes (outputs)
        #  9 disjoint connections
        #  1 identical connection
        node_distance = (6 * cfg.genome.compatibility_disjoint_node + 2 * 0) / 5
        conn_distance = (9 * cfg.genome.compatibility_disjoint_conn + 1 * 0) / 6
        self.assertEqual(result, node_distance + conn_distance)
        
        # Check if all different representations
        self.assertNotEqual(cache.node_cache.index_map[5], cache.node_cache.index_map[8])
        self.assertNotEqual(cache.node_cache.index_map[6], cache.node_cache.index_map[9])
        self.assertNotEqual(cache.node_cache.index_map[7], cache.node_cache.index_map[10])
    
    def test_genomes02(self):
        """> Test the distance and internal representations of the two given nodes."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_deep_genome0(cfg)
        genomes[2] = get_deep_genome2(cfg)
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Get the genome-distance
        result = cache(genome0=genomes[0], genome1=genomes[2])
        self.assertGreater(result, 0)
        
        # Calculation:
        #  4 disjoint nodes
        #  3 identical nodes (outputs)
        #  7 disjoint connections (1 completely removed, 2x2 others connected to different genes)
        #  2 identical connections
        node_distance = (4 * cfg.genome.compatibility_disjoint_node + 3 * 0) / 5
        conn_distance = (7 * cfg.genome.compatibility_disjoint_conn + 2 * 0) / 6
        self.assertEqual(result, node_distance + conn_distance)
        
        # Check if all different representations
        self.assertEqual(cache.node_cache.index_map[5], cache.node_cache.index_map[11])
        self.assertNotEqual(cache.node_cache.index_map[6], cache.node_cache.index_map[12])
        self.assertNotEqual(cache.node_cache.index_map[7], cache.node_cache.index_map[13])
    
    def test_genomes03(self):
        """> Test the distance and internal representations of the two given nodes."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Fetch the used genomes
        cfg = get_config()
        genomes = dict()
        genomes[0] = get_deep_genome0(cfg)
        genomes[3] = get_deep_genome3(cfg)
        
        # Setup an initial genome-cache
        cache = GenomeDistanceCache(cfg.genome)
        cache.warm_up(genomes)
        
        # Get the genome-distance
        result = cache(genome0=genomes[0], genome1=genomes[3])
        self.assertGreater(result, 0)
        
        # Calculation:
        #  2 disjoint nodes
        #  4 identical nodes (outputs)
        #  5 disjoint connections (1 completely removed, 2x2 others connected to different genes)
        #  4 identical connections
        node_distance = (2 * cfg.genome.compatibility_disjoint_node + 4 * 0) / 5
        conn_distance = (5 * cfg.genome.compatibility_disjoint_conn + 4 * 0) / 6
        self.assertEqual(result, node_distance + conn_distance)
        
        # Check if all different representations
        self.assertEqual(cache.node_cache.index_map[5], cache.node_cache.index_map[14])
        self.assertEqual(cache.node_cache.index_map[6], cache.node_cache.index_map[15])
        self.assertNotEqual(cache.node_cache.index_map[7], cache.node_cache.index_map[16])


def main():
    sg = SameGenome()
    sg.test_zero_distance()
    sg.test_distance_similar_genome()
    sg.test_same_internal_representation()
    
    dg = DifferentGenome()
    dg.test_distance()
    
    ddg = DifferentDeepGenome()
    ddg.test_genomes01()
    ddg.test_genomes02()
    ddg.test_genomes03()


if __name__ == '__main__':
    unittest.main()
