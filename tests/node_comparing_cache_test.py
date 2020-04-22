"""
node_distance_cache_test.py

Test the NodeComparingCache component in the genome.py file. This class is used to measure distances between nodes.
"""
import os
import unittest

from config import Config
from configs.genome_config import GenomeConfig
from population.utils.cache.node_comparing import NodeComparingCache


# TODO: Test that two nodes with different keys are always comparable
# TODO: More elaborate testing (?) --> merge-keys (first not comparable, then they are)
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene


def get_connections(receiving_key: int, sending_keys: set, cfg: GenomeConfig):
    """Create a dictionary of connection from the sending_keys to the receiving_key."""
    conn = dict()
    for k in sending_keys:
        conn[(k, receiving_key)] = ConnectionGene(key=(k, receiving_key), cfg=cfg)
    return conn


class Distance(unittest.TestCase):
    """Test if the distance-measure is computed correctly."""
    
    def test_different_type(self):
        """> Test if False is returned when comparing nodes of different type."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Setup the nodes to compare
        cfg = Config()
        output_node = OutputNodeGene(key=0, cfg=cfg.genome)
        simple_node = SimpleNodeGene(key=1, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Fetch the result of the distance-measure
        result = cache(
                node0=output_node,
                node1=simple_node,
                conn0={},
                conn1={},
                cfg=cfg.genome,
        )
        self.assertEqual(result, (False, None))
    
    def test_different_connections(self):
        """> Test if False is returned when comparing nodes of different type."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Setup the nodes to compare
        cfg = Config()
        node0 = SimpleNodeGene(key=0, cfg=cfg.genome)
        node1 = SimpleNodeGene(key=1, cfg=cfg.genome)
        conn_set1 = get_connections(receiving_key=0, sending_keys={-1, -2}, cfg=cfg.genome)
        conn_set2 = get_connections(receiving_key=1, sending_keys={-2, -3}, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Fetch the result of the distance-measure
        result = cache(
                node0=node0,
                node1=node1,
                conn0=conn_set1,
                conn1=conn_set2,
                cfg=cfg.genome,
        )
        self.assertEqual(result, (False, None))
    
    def test_successful_distance(self):
        """> Test when a distance-measure should be performed."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Setup the nodes to compare
        cfg = Config()
        cfg.genome.compatibility_weight = 1
        node0 = SimpleNodeGene(key=0, cfg=cfg.genome)
        node0.bias = 0
        node1 = SimpleNodeGene(key=1, cfg=cfg.genome)
        node1.bias = 1
        conn_set1 = get_connections(receiving_key=0, sending_keys={-1, -2}, cfg=cfg.genome)
        conn_set2 = get_connections(receiving_key=1, sending_keys={-1, -2}, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Fetch the result of the distance-measure
        result = cache(
                node0=node0,
                node1=node1,
                conn0=conn_set1,
                conn1=conn_set2,
                cfg=cfg.genome,
        )
        self.assertEqual(result, (True, 1.0))  # Only distance-difference is in the bias
    
    def test_successful_distance_multiple(self):
        """> Test when multiple distance-measures are performed."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Setup the nodes to compare
        cfg = Config()
        cfg.genome.compatibility_weight = 1
        key = 0
        node0 = SimpleNodeGene(key=key, cfg=cfg.genome)
        node0.bias = 0
        conn_set1 = get_connections(receiving_key=key, sending_keys={-1, -2}, cfg=cfg.genome)
        
        key = 1
        node1 = SimpleNodeGene(key=key, cfg=cfg.genome)
        node1.bias = 1
        conn_set2 = get_connections(receiving_key=key, sending_keys={-1, -2}, cfg=cfg.genome)
        
        key = 2
        node3 = SimpleNodeGene(key=key, cfg=cfg.genome)
        node3.bias = 0
        conn_set3 = get_connections(receiving_key=key, sending_keys={-1, -2, 0}, cfg=cfg.genome)
        
        key = 3
        node4 = SimpleNodeGene(key=key, cfg=cfg.genome)
        node4.bias = 1
        conn_set4 = get_connections(receiving_key=key, sending_keys={-1, -2, 1}, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Fetch the result of the distance-measure for (node0, node1)
        result = cache(
                node0=node0,
                node1=node1,
                conn0=conn_set1,
                conn1=conn_set2,
                cfg=cfg.genome,
        )
        self.assertEqual(result, (True, 1.0))  # Only distance-difference is in the bias
        
        # Fetch the result of the distance-measure for (node1, node3)
        result = cache(
                node0=node1,
                node1=node3,
                conn0=conn_set2,
                conn1=conn_set3,
                cfg=cfg.genome,
        )
        self.assertEqual(result, (False, None))  # Only distance-difference is in the bias
        
        # Fetch the result of the distance-measure for (node3, node4)
        result = cache(
                node0=node3,
                node1=node4,
                conn0=conn_set3,
                conn1=conn_set4,
                cfg=cfg.genome,
        )
        self.assertEqual(result, (True, 1.0))  # Only distance-difference is in the bias


class Parameters(unittest.TestCase):
    """Test if the parameters are set correctly."""
    
    def test_distance_failure(self):
        """> Test if the distance-parameter is updated correctly after unsuccessful read."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Setup the nodes to compare
        cfg = Config()
        output_node = OutputNodeGene(key=0, cfg=cfg.genome)
        simple_node = SimpleNodeGene(key=1, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Fetch the result of the distance-measure
        result = cache(
                node0=output_node,
                node1=simple_node,
                conn0={},
                conn1={},
                cfg=cfg.genome,
        )
        
        # Check the distance-parameter
        self.assertEqual(result, (False, None))
        self.assertEqual(cache.comparable[(0, 1)], False)
        self.assertFalse((1, 0) in cache.comparable)
    
    def test_distance_success(self):
        """> Test if the distance-parameter is updated correctly after successful read."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Setup the nodes to compare
        cfg = Config()
        cfg.genome.compatibility_weight = 1
        node0 = SimpleNodeGene(key=0, cfg=cfg.genome)
        node0.bias = 0
        node1 = SimpleNodeGene(key=1, cfg=cfg.genome)
        node1.bias = 1
        conn_set1 = get_connections(receiving_key=0, sending_keys={-1, -2}, cfg=cfg.genome)
        conn_set2 = get_connections(receiving_key=1, sending_keys={-1, -2}, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Cache the result of the distance-measure
        result = cache(
                node0=node0,
                node1=node1,
                conn0=conn_set1,
                conn1=conn_set2,
                cfg=cfg.genome,
        )
        
        # Check the distance-parameter
        self.assertEqual(result, (True, 1.0))
        self.assertEqual(cache.comparable[(0, 1)], True)
        self.assertFalse((1, 0) in cache.comparable)


class Indexer(unittest.TestCase):
    """Test if the indexer functions correctly."""
    
    def test_increasing_nodes(self):
        """> Test if the index-parameter keeps increasing with each newly fed node."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Get the nodes
        cfg = Config()
        cfg.genome.compatibility_weight = 1
        node0 = SimpleNodeGene(key=0, cfg=cfg.genome)
        node1 = SimpleNodeGene(key=1, cfg=cfg.genome)
        node3 = SimpleNodeGene(key=2, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Feed the nodes to the NodeComparingCache (successful matches)
        r1 = cache(node0, node1, {}, {}, cfg.genome)
        r2 = cache(node0, node3, {}, {}, cfg.genome)
        r3 = cache(node1, node3, {}, {}, cfg.genome)
        self.assertTrue(r1[0])
        self.assertTrue(r2[0])
        self.assertTrue(r3[0])
        
        # Get the current indexer-value
        self.assertEqual(next(cache.indexer), 3)
        
        # Check the index_map, keys should be merged
        self.assertEqual(cache.index_map[node0.key], cache.index_map[node1.key])
        self.assertEqual(cache.index_map[node1.key], cache.index_map[node3.key])


class Reset(unittest.TestCase):
    """Test if the reset-function works correctly."""
    
    def test_clean_start(self):
        """> Test if the restart indeed removes all the parameters."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Get the nodes
        cfg = Config()
        cfg.genome.compatibility_weight = 1
        node0 = SimpleNodeGene(key=0, cfg=cfg.genome)
        node1 = SimpleNodeGene(key=1, cfg=cfg.genome)
        node3 = SimpleNodeGene(key=2, cfg=cfg.genome)
        
        # Create an empty NodeComparingCache
        cache = NodeComparingCache()
        
        # Feed the nodes to the NodeComparingCache (successful matches)
        cache(node0, node1, {}, {}, cfg.genome)
        cache(node0, node3, {}, {}, cfg.genome)
        cache(node1, node3, {}, {}, cfg.genome)
        
        # Reset the cache
        cache.reset()
        self.assertEqual(cache.comparable, dict())
        self.assertEqual(next(cache.indexer), 0)
        self.assertEqual(cache.index_map, dict())


def main():
    d = Distance()
    d.test_different_type()
    d.test_different_connections()
    d.test_successful_distance()
    d.test_successful_distance_multiple()
    
    p = Parameters()
    p.test_distance_failure()
    p.test_distance_success()
    
    i = Indexer()
    i.test_increasing_nodes()
    
    r = Reset()
    r.test_clean_start()


if __name__ == '__main__':
    unittest.main()
