"""
genome_mutation_test.py

Test gene-specific operations.
"""
import os
import unittest

import numpy as np

from config import Config
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene


def get_connection_gene(key, config):
    return ConnectionGene(key, config)


def get_gru_node_gene(key, config):
    return GruNodeGene(key, config, input_keys=[-1], input_keys_full=[-1, -2])


def get_output_node_gene(key, config):
    return OutputNodeGene(key, config)


def get_simple_node_gene(key, config):
    return SimpleNodeGene(key, config)


class SimpleNode(unittest.TestCase):
    """Test the SimpleNodeGene's mutation operations."""
    
    def test_bias(self):
        """> Test if the bias remains inside its boundaries during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias remains inside its boundary
        cfg = Config().genome
        cfg.bias_mutate_rate = 0.5
        cfg.bias_replace_rate = 0.5
        cfg.bias_min_value = -0.1
        cfg.bias_max_value = 0.1
        gene = get_simple_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(-0.1 <= gene.bias <= 0.1)
        
        # Set mutation rate to 0, no change should happen
        cfg.bias_mutate_rate = 0
        cfg.bias_replace_rate = 0
        gene = get_simple_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)
        
        # Set mutation power to 0, no change should happen
        cfg.bias_mutate_rate = 1
        cfg.bias_replace_rate = 0
        cfg.bias_mutate_power = 0
        gene = get_simple_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)


class OutputNode(unittest.TestCase):
    """Test the OutputNodeGene's mutation operations."""
    
    def test_bias(self):
        """> Test if the bias remains inside its boundaries during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias remains inside its boundary
        cfg = Config().genome
        cfg.bias_mutate_rate = 0.5
        cfg.bias_replace_rate = 0.5
        cfg.bias_min_value = -0.1
        cfg.bias_max_value = 0.1
        gene = get_output_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(-0.1 <= gene.bias <= 0.1)
        
        # Set mutation rate to 0, no change should happen
        cfg.bias_mutate_rate = 0
        cfg.bias_replace_rate = 0
        gene = get_output_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)
        
        # Set mutation power to 0, no change should happen
        cfg.bias_mutate_rate = 1
        cfg.bias_replace_rate = 0
        cfg.bias_mutate_power = 0
        gene = get_output_node_gene(0, cfg)
        init_bias = gene.bias
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.bias == init_bias)


class GruNode(unittest.TestCase):
    """Test the GruNodeGene's mutation operations."""
    
    def test_bias(self):
        """> Test if bias is left unchanged during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias remains equal to zero
        cfg = Config().genome
        cfg.bias_mutate_rate = 0.5
        cfg.bias_replace_rate = 0.5
        gene = get_gru_node_gene(0, cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(gene.bias, 0)
    
    def test_bias_h(self):
        """> Test if bias_h behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, bias_h's values remain inside the set boundary
        cfg = Config().genome
        cfg.bias_min_value = -0.1
        cfg.bias_max_value = -0.1
        cfg.rnn_mutate_rate = 0.5
        cfg.rnn_replace_rate = 0.5
        cfg.rnn_min_value = -0.1
        cfg.rnn_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_bias_h = gene.bias_h.copy()
        for _ in range(100):
            gene.mutate(cfg)
            for value in gene.bias_h:
                self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.bias_h - init_bias_h) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.bias_mutate_rate = 0
        cfg.bias_replace_rate = 0
        cfg.rnn_mutate_rate = 0
        cfg.rnn_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_bias_h = gene.bias_h.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.bias_h - init_bias_h), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.rnn_mutate_rate = 1
        cfg.rnn_replace_rate = 0
        cfg.rnn_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_bias_h = gene.bias_h.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.bias_h - init_bias_h), 0)
    
    def test_weight_hh(self):
        """> Test if weight_hh behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, weight_hh's values remain inside the set boundary
        cfg = Config().genome
        cfg.rnn_mutate_rate = 0.5
        cfg.rnn_replace_rate = 0.5
        cfg.rnn_min_value = -0.1
        cfg.rnn_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_weight_hh = gene.weight_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            for value in gene.weight_hh:
                self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.weight_hh - init_weight_hh) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.rnn_mutate_rate = 0
        cfg.rnn_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_hh = gene.weight_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_hh - init_weight_hh), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.rnn_mutate_rate = 1
        cfg.rnn_replace_rate = 0
        cfg.rnn_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_hh = gene.weight_hh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_hh - init_weight_hh), 0)
    
    def test_weight_ih(self):
        """> Test if weight_ih behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Test on shape of weight_ih
        cfg = Config().genome
        gene = get_gru_node_gene(0, cfg)
        self.assertEqual(gene.weight_xh.shape, (3, 1))
        
        # After mutations, weight_ih's values remain inside the set boundary
        cfg.rnn_mutate_rate = 0.5
        cfg.rnn_replace_rate = 0.5
        cfg.rnn_min_value = -0.1
        cfg.rnn_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_weight_ih = gene.weight_xh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            gene.update_weight_xh()
            for value in gene.weight_xh:
                for v in value:
                    self.assertTrue(-0.1 <= value <= 0.1)
            if np.linalg.norm(gene.weight_xh - init_weight_ih) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.rnn_mutate_rate = 0
        cfg.rnn_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih = gene.weight_xh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            gene.update_weight_xh()
            self.assertEqual(np.linalg.norm(gene.weight_xh - init_weight_ih), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.rnn_mutate_rate = 1
        cfg.rnn_replace_rate = 0
        cfg.rnn_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih = gene.weight_xh.copy()
        for _ in range(100):
            gene.mutate(cfg)
            gene.update_weight_xh()
            self.assertEqual(np.linalg.norm(gene.weight_xh - init_weight_ih), 0)
    
    def test_weight_ih_full(self):
        """> Test if weight_ih_full behaves as expected"""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Test on shape of weight_ih_full
        cfg = Config().genome
        gene = get_gru_node_gene(0, cfg)
        self.assertEqual(gene.weight_xh_full.shape, (3, 2))
        
        # After mutations, weight_ih_full's values remain inside the set boundary
        cfg.rnn_mutate_rate = 0.5
        cfg.rnn_replace_rate = 0.5
        cfg.rnn_min_value = -0.1
        cfg.rnn_max_value = 0.1
        gene = get_gru_node_gene(0, cfg)
        changed = False
        init_weight_ih_full = gene.weight_xh_full.copy()
        for _ in range(100):
            gene.mutate(cfg)  # No update_weight_ih must be called
            for value in gene.weight_xh_full:
                for v in value:
                    self.assertTrue(-0.1 <= v <= 0.1)
            if np.linalg.norm(gene.weight_xh_full - init_weight_ih_full) > 0: changed = True
        self.assertTrue(changed)
        
        # Set mutation rate to 0, no change should happen
        cfg.rnn_mutate_rate = 0
        cfg.rnn_replace_rate = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih_full = gene.weight_xh_full.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_xh_full - init_weight_ih_full), 0)
        
        # Set mutation power to 0, no change should happen
        cfg.rnn_mutate_rate = 1
        cfg.rnn_replace_rate = 0
        cfg.rnn_mutate_power = 0
        gene = get_gru_node_gene(0, cfg)
        init_weight_ih_full = gene.weight_xh_full.copy()
        for _ in range(100):
            gene.mutate(cfg)
            self.assertEqual(np.linalg.norm(gene.weight_xh_full - init_weight_ih_full), 0)


class Connection(unittest.TestCase):
    """Test the ConnectionGene's mutation operations."""
    
    def test_enabled(self):
        """> Test if enabled changes during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Test the enabled defaults
        cfg = Config().genome
        cfg.enabled_default = False
        gene = get_connection_gene((-1, 0), cfg)
        self.assertFalse(gene.enabled)
        cfg.enabled_default = True
        gene = get_connection_gene((-1, 0), cfg)
        self.assertTrue(gene.enabled)
        
        # Enabled state should change during mutation
        cfg.enabled_mutate_rate = 1
        changed = False
        gene = get_connection_gene((-1, 0), cfg)
        init_enabled = gene.enabled
        for _ in range(100):
            gene.enabled = gene.mutate(cfg)
            if gene.enabled != init_enabled:
                changed = True
                break
        self.assertTrue(changed)
    
    def test_weight(self):
        """> Test if the weight remains inside its boundaries during mutation."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # After mutations, weight remains inside its boundary
        cfg = Config().genome
        cfg.weight_mutate_rate = 0.5
        cfg.weight_replace_rate = 0.5
        cfg.weight_min_value = -0.1
        cfg.weight_max_value = 0.1
        gene = get_connection_gene((-1, 0), cfg)
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(-0.1 <= gene.weight <= 0.1)
        
        # Set mutation rate to 0, no change should happen
        cfg.weight_mutate_rate = 0
        cfg.weight_replace_rate = 0
        gene = get_connection_gene((-1, 0), cfg)
        init_weight = gene.weight
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.weight == init_weight)
        
        # Set mutation power to 0, no change should happen
        cfg.weight_mutate_rate = 1
        cfg.weight_replace_rate = 0
        cfg.weight_mutate_power = 0
        gene = get_connection_gene((-1, 0), cfg)
        init_weight = gene.weight
        for _ in range(100):
            gene.mutate(cfg)
            self.assertTrue(gene.weight == init_weight)


def main():
    # Test the SimpleNodeGene
    sn = SimpleNode()
    sn.test_bias()
    
    # Test the OutputNodeGene
    on = OutputNode()
    on.test_bias()
    
    # Test the GruNodeGene
    gn = GruNode()
    gn.test_bias()
    gn.test_bias_h()
    gn.test_weight_hh()
    gn.test_weight_ih()
    gn.test_weight_ih_full()
    
    # Test the ConnectionGene
    c = Connection()
    c.test_enabled()
    c.test_weight()


if __name__ == '__main__':
    unittest.main()
