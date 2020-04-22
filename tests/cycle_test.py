"""
cycle_test.py

Test the creates_cycle method.
"""
import os
import unittest

from population.utils.network_util.graphs import creates_cycle


def get_simple_net():
    """
         0
       /  \
      1    |
      |    |
    -1    -2
    """
    connections = dict()
    connections[(-1, 1)] = 1
    connections[(1, 0)] = 1
    connections[(-2, 0)] = 1
    return connections


def get_medium_net():
    """
         0
      /  |  \
     1  /   2
     | /    |
    -1     -2
    """
    connections = dict()
    connections[(-1, 0)] = 1
    connections[(-1, 1)] = 1
    connections[(1, 0)] = 0
    connections[(-2, 2)] = 1
    connections[(2, 0)] = 1
    return connections


def get_complex_net():
    """
        0
      /  \
     3   |
     |   |
     2<--4
     |   |
     1   |
     |   |
    -1  -2
    """
    connections = dict()
    connections[(-1, 1)] = 1
    connections[(1, 2)] = 1
    connections[(2, 3)] = 1
    connections[(3, 0)] = 1
    connections[(-2, 4)] = 1
    connections[(4, 0)] = 1
    connections[(4, 2)] = 1
    return connections


class NoCycle(unittest.TestCase):
    """Test the cycle_test method when there are no cycles."""
    
    def test_simple(self):
        """> Cycle-free check in simple network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the connections (only keys matter!)
        connections = get_simple_net()
        
        # Test
        self.assertFalse(creates_cycle(connections=connections, test=(-2, 1)))
        self.assertFalse(creates_cycle(connections=connections, test=(-2, -1)))
        self.assertFalse(creates_cycle(connections=connections, test=(1, -2)))
        self.assertFalse(creates_cycle(connections=connections, test=(1, 2)))
        self.assertFalse(creates_cycle(connections=connections, test=(2, 1)))
    
    def test_medium(self):
        """> Cycle-free check in medium network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the connections (only keys matter!)
        connections = get_medium_net()
        
        # Test
        self.assertFalse(creates_cycle(connections=connections, test=(2, 1)))
        self.assertFalse(creates_cycle(connections=connections, test=(-2, 1)))
        self.assertFalse(creates_cycle(connections=connections, test=(-1, 2)))
    
    def test_complex(self):
        """> Cycle-free check in complex network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the connections (only keys matter!)
        connections = get_complex_net()
        
        # Test
        self.assertFalse(creates_cycle(connections=connections, test=(-1, 2)))
        self.assertFalse(creates_cycle(connections=connections, test=(-1, 3)))
        self.assertFalse(creates_cycle(connections=connections, test=(-1, 0)))
        self.assertFalse(creates_cycle(connections=connections, test=(-2, 0)))
        self.assertFalse(creates_cycle(connections=connections, test=(-2, 1)))
        self.assertFalse(creates_cycle(connections=connections, test=(-2, 2)))
        self.assertFalse(creates_cycle(connections=connections, test=(-2, 3)))
        self.assertFalse(creates_cycle(connections=connections, test=(1, 0)))
        self.assertFalse(creates_cycle(connections=connections, test=(2, 0)))


class Cycle(unittest.TestCase):
    """Test the cycle_test method when there are cycles."""
    
    def test_simple(self):
        """> Cycle check in simple network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the connections (only keys matter!)
        connections = get_simple_net()
        
        # Test
        self.assertTrue(creates_cycle(connections=connections, test=(1, 1)))
        self.assertTrue(creates_cycle(connections=connections, test=(1, -1)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 1)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, -1)))
    
    def test_medium(self):
        """> Cycle check in medium network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the connections (only keys matter!)
        connections = get_medium_net()
        
        # Test
        self.assertTrue(creates_cycle(connections=connections, test=(0, -1)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, -2)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 1)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 2)))
        self.assertTrue(creates_cycle(connections=connections, test=(1, -1)))
        self.assertTrue(creates_cycle(connections=connections, test=(1, 1)))
        self.assertTrue(creates_cycle(connections=connections, test=(2, 2)))
        
        # Extend network
        connections.update({(1, 2): 1})
        self.assertTrue(creates_cycle(connections=connections, test=(2, -1)))
        self.assertTrue(creates_cycle(connections=connections, test=(2, 1)))
        self.assertFalse(creates_cycle(connections=connections, test=(1, -2)))  # Allowed!
    
    def test_complex(self):
        """> Cycle check in complex network."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the connections (only keys matter!)
        connections = get_complex_net()
        
        # Test
        self.assertTrue(creates_cycle(connections=connections, test=(0, -1)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, -2)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 1)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 2)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 3)))
        self.assertTrue(creates_cycle(connections=connections, test=(0, 4)))
        self.assertTrue(creates_cycle(connections=connections, test=(3, 3)))
        self.assertTrue(creates_cycle(connections=connections, test=(3, 2)))
        self.assertTrue(creates_cycle(connections=connections, test=(3, 1)))
        self.assertTrue(creates_cycle(connections=connections, test=(3, -1)))
        self.assertTrue(creates_cycle(connections=connections, test=(3, -2)))
        self.assertTrue(creates_cycle(connections=connections, test=(3, 4)))
        self.assertTrue(creates_cycle(connections=connections, test=(-1, -1)))
        self.assertFalse(creates_cycle(connections=connections, test=(1, 4)))  # Allowed!


def main():
    nc = NoCycle()
    nc.test_simple()
    nc.test_medium()
    nc.test_complex()
    
    c = Cycle()
    c.test_simple()
    c.test_medium()
    c.test_complex()


if __name__ == '__main__':
    unittest.main()
