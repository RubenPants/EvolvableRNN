"""
genome_crossover_test.py

TODO: Finish test-cases!

Test gene-specific operations.
"""
import os
import unittest

from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene


def get_connection_gene(key, config):
    return ConnectionGene(key, config)


def get_gru_node_gene(key, config):
    return GruNodeGene(key, config)


def get_output_node_gene(key, config):
    return OutputNodeGene(key, config)


def get_simple_node_gene(key, config):
    return SimpleNodeGene(key, config)


class SimpleNode(unittest.TestCase):
    """Test connection-mutation mechanism in the genomes."""
    
    def test_mutation(self):
        """> TODO"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pass  # Dummy


def get_config():
    """Get a modified config-file."""


def main():
    # Test wall collision
    sn = SimpleNode()


if __name__ == '__main__':
    unittest.main()
