"""
line_2d_test.py

Test the Line2d component.
"""
import os
import unittest
from math import pi, sqrt

from utils.line2d import Line2d
from utils.vec2d import Vec2d

EPSILON = 1e-5


class Length(unittest.TestCase):
    """Test the Line2d component's length measure."""
    
    def test_simple_length(self):
        """> Test for simple line-segments."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create simple lines
        a = Vec2d(1, 1)
        b = Vec2d(1, 2)
        c = Vec2d(2, 2)
        line1 = Line2d(a, b)
        line2 = Line2d(b, c)
        line3 = Line2d(a, c)
        
        # Test the length
        self.assertTrue(1 - EPSILON <= line1.get_length() <= 1 + EPSILON)
        self.assertTrue(1 - EPSILON <= line2.get_length() <= 1 + EPSILON)
        self.assertTrue(sqrt(2) - EPSILON <= line3.get_length() <= sqrt(2) + EPSILON)
    
    def test_negative_component(self):
        """> Test for line-segments with negative components."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create simple lines
        a = Vec2d(1, 1)
        b = Vec2d(1, -1)
        c = Vec2d(-1, -1)
        d = Vec2d(-1, 1)
        line1 = Line2d(a, b)
        line2 = Line2d(b, c)
        line3 = Line2d(c, d)
        line4 = Line2d(d, a)
        diag1 = Line2d(a, c)
        diag2 = Line2d(b, d)
        
        # Test the length
        self.assertTrue(2 - EPSILON <= line1.get_length() <= 2 + EPSILON)
        self.assertTrue(2 - EPSILON <= line2.get_length() <= 2 + EPSILON)
        self.assertTrue(2 - EPSILON <= line3.get_length() <= 2 + EPSILON)
        self.assertTrue(2 - EPSILON <= line4.get_length() <= 2 + EPSILON)
        self.assertTrue(sqrt(8) - EPSILON <= diag1.get_length() <= sqrt(8.) + EPSILON)
        self.assertTrue(sqrt(8) - EPSILON <= diag2.get_length() <= sqrt(8.) + EPSILON)


class Orientation(unittest.TestCase):
    """Test the Line2d component's orientation measure."""
    
    def test_quadrant_angles(self):
        """> Check if drone cannot force itself through a wall."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the lines
        zero = Vec2d(0, 0)
        a = Vec2d(1, 0)
        b = Vec2d(0, 1)
        c = Vec2d(-1, 0)
        d = Vec2d(0, -1)
        line1 = Line2d(zero, a)
        line2 = Line2d(zero, b)
        line3 = Line2d(zero, c)
        line4 = Line2d(zero, d)
        
        # Tests
        self.assertTrue(0 - EPSILON <= line1.get_orientation() % (2 * pi) <= 0 + EPSILON)
        self.assertTrue(pi / 2 - EPSILON <= line2.get_orientation() % (2 * pi) <= pi / 2 + EPSILON)
        self.assertTrue(pi - EPSILON <= line3.get_orientation() % (2 * pi) <= pi + EPSILON)
        self.assertTrue(3 * pi / 2 - EPSILON <= line4.get_orientation() % (2 * pi) <= 3 * pi / 2 + EPSILON)
    
    def test_other_angles(self):
        """> Check if drone cannot force itself through a wall."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the lines
        zero = Vec2d(0, 0)
        a = Vec2d(1, 1)
        line1 = Line2d(zero, a)
        
        # Tests
        self.assertTrue(pi / 4 - EPSILON <= line1.get_orientation() % (2 * pi) <= pi / 4 + EPSILON)


def main():
    # Test for Line2d's length calculations
    l = Length()
    l.test_simple_length()
    l.test_negative_component()
    
    # Test for Line2d's orientation calculations
    o = Orientation()
    o.test_quadrant_angles()
    o.test_other_angles()


if __name__ == '__main__':
    unittest.main()
