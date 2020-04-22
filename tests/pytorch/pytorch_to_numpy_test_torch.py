"""
pytorch_to_numpy_test.py

Test if the conversion from pytorch tensors to numpy arrays hold. Larger than three dimensional tensors isn't tested
since these aren't used throughout the project.
"""
import os
import unittest

import numpy as np
import torch
from numpy.testing import assert_array_equal


def ones_numpy(shape):
    return np.ones(shape)


def ones_torch(shape):
    return torch.ones(shape)


class Shape(unittest.TestCase):
    """Test the shapes of both tensor representations. This is a fairly trivial test-suite."""
    
    def test_vector(self):
        """> Test the shape if both one dimensional vectors."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3,))
        t = ones_torch((3,))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (3,))
    
    def test_matrix(self):
        """> Test the shape if both two dimensional matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3, 4))
        t = ones_torch((3, 4))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (3, 4))
    
    def test_tensor(self):
        """> Test the shape if both three dimensional tensors."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3, 4, 2))
        t = ones_torch((3, 4, 2))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (3, 4, 2))


class Transpose(unittest.TestCase):
    """Test the transpose operations."""
    
    def test_vector(self):
        """> Test the transpose option if both one dimensional vectors."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3,)).transpose()
        t = ones_torch((3,)).t()
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (3,))
    
    def test_matrix(self):
        """> Test the transpose option if both two dimensional matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3, 4)).transpose()
        t = ones_torch((3, 4)).t()
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (4, 3))


class Concatenate(unittest.TestCase):
    """Test the concatenation of two arrays."""
    
    def test_vector(self):
        """> Test the concatenation of two vectors."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((3,))
        n2 = ones_numpy((3,)) * 2
        t1 = ones_torch((3,))
        t2 = ones_torch((3,)) * 2
        
        # Concatenate
        n = np.concatenate((n1, n2))
        t = torch.cat((t1, t2))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([1, 1, 1, 2, 2, 2]),
        )
        self.assertEqual(n.shape, (6,))
    
    def test_matrix(self):
        """> Test the concatenation of two matrices over their first dimension."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((3, 2))
        n2 = ones_numpy((3, 2)) * 2
        t1 = ones_torch((3, 2))
        t2 = ones_torch((3, 2)) * 2
        
        # Concatenate
        n = np.concatenate((n1, n2))
        t = torch.cat((t1, t2))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[1, 1],
                            [1, 1],
                            [1, 1],
                            [2, 2],
                            [2, 2],
                            [2, 2]]),
        )
        self.assertEqual(n.shape, (6, 2))
    
    def test_matrix_other_dimension(self):
        """> Test the concatenation of two matrices over their second dimension."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((3, 2))
        n2 = ones_numpy((3, 2)) * 2
        t1 = ones_torch((3, 2))
        t2 = ones_torch((3, 2)) * 2
        
        # Concatenate
        n = np.concatenate((n1, n2), axis=1)
        t = torch.cat((t1, t2), dim=1)
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[1, 1, 2, 2],
                            [1, 1, 2, 2],
                            [1, 1, 2, 2]]),
        )
        self.assertEqual(n.shape, (3, 4))
    
    def test_tensor(self):
        """> Test the concatenation of two tensors over their first dimension."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((1, 3, 4))
        n2 = ones_numpy((1, 3, 4)) * 2
        t1 = ones_torch((1, 3, 4))
        t2 = ones_torch((1, 3, 4)) * 2
        
        # Concatenate
        n = np.concatenate((n1, n2))
        t = torch.cat((t1, t2))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]],
                            [[2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]]]),
        )
        self.assertEqual(n.shape, (2, 3, 4))
    
    def test_tensor_second_dimension(self):
        """> Test the concatenation of two tensors over their second dimension."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((1, 3, 4))
        n2 = ones_numpy((1, 3, 4)) * 2
        t1 = ones_torch((1, 3, 4))
        t2 = ones_torch((1, 3, 4)) * 2
        
        # Concatenate
        n = np.concatenate((n1, n2), axis=1)
        t = torch.cat((t1, t2), dim=1)
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2],
                             [2, 2, 2, 2]]]),
        )
        self.assertEqual(n.shape, (1, 6, 4))
    
    def test_tensor_third_dimension(self):
        """> Test the concatenation of two tensors over their third dimension."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((1, 3, 4))
        n2 = ones_numpy((1, 3, 4)) * 2
        t1 = ones_torch((1, 3, 4))
        t2 = ones_torch((1, 3, 4)) * 2
        
        # Concatenate
        n = np.concatenate((n1, n2), axis=2)
        t = torch.cat((t1, t2), dim=2)
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[[1, 1, 1, 1, 2, 2, 2, 2],
                             [1, 1, 1, 1, 2, 2, 2, 2],
                             [1, 1, 1, 1, 2, 2, 2, 2]]]),
        )
        self.assertEqual(n.shape, (1, 3, 8))


class MatrixModifications(unittest.TestCase):
    """Test the elementwise replacement modifications."""
    
    def test_element_replacement(self):
        """> Test the element-wise modification if both two dimensional matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3, 3))
        t = ones_torch((3, 3))
        
        # Do the modification
        n[0][1] = 2
        t[0][1] = 2
        
        # Test
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[1, 2, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
        )
        
        # Do the modification
        n[2, 1] = 3
        t[2, 1] = 3
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[1, 2, 1],
                            [1, 1, 1],
                            [1, 3, 1]])
        )
        
        # Do the modification
        n[1, 2] = 5
        t[1][2] = 5
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[1, 2, 1],
                            [1, 1, 5],
                            [1, 3, 1]])
        )
    
    def test_vector_replacement(self):
        """> Test the element-wise modification if both two dimensional matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n = ones_numpy((3, 3))
        t = ones_torch((3, 3))
        
        # Do the modification
        n[0] = np.zeros((3,))
        t[0] = torch.zeros((3,))
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[0, 0, 0],
                            [1, 1, 1],
                            [1, 1, 1]])
        )
        
        # Do the modification
        n[:, 1] = np.ones((3,)) * 5
        t[:, 1] = torch.ones((3,)) * 5
        
        # Test
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
        assert_array_equal(
                np.asarray(n),
                np.asarray([[0, 5, 0],
                            [1, 5, 1],
                            [1, 5, 1]])
        )


class MatrixMultiply(unittest.TestCase):
    """Test the transpose operations."""
    
    def test_vector(self):
        """> Test matrix-multiplication between two vectors."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((1, 3))
        n2 = ones_numpy((1, 3))
        t1 = ones_torch((1, 3))
        t2 = ones_torch((1, 3))
        
        # Modify the matrices
        n1[0, 0], n2[0, 1] = 2, 3
        t1[0, 0], t2[0, 1] = 2, 3
        
        # Matrix multiply
        n = np.matmul(n1, n2.transpose())
        t = t1.mm(t2.t())
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (1, 1))
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
    
    def test_matrix(self):
        """> Test matrix-multiplication between two matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays  (note different shapes in 1 vs 2!)
        n1 = ones_numpy((2, 3))
        n2 = ones_numpy((1, 3))
        t1 = ones_torch((2, 3))
        t2 = ones_torch((1, 3))
        
        # Modify the matrices
        n1[0, 0], n2[0, 1] = 2, 3
        t1[0, 0], t2[0, 1] = 2, 3
        
        # Matrix multiply
        n = np.matmul(n1, n2.transpose())
        t = t1.mm(t2.t())
        
        # Test
        self.assertEqual(n.shape, t.shape)
        self.assertEqual(n.shape, (2, 1))
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )


class ElementMultiply(unittest.TestCase):
    """Test elementwise multiplication."""
    
    def test_vector(self):
        """> Test element-wise-multiplication between two vectors."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((1, 3)) * 2
        n2 = ones_numpy((1, 3)) * 3
        t1 = ones_torch((1, 3)) * 2
        t2 = ones_torch((1, 3)) * 3
        
        # Modify
        n1[0, 0], t1[0, 0] = 0, 0
        
        # Multiply the matrices
        n = n1 * n2
        t = t1 * t2
        
        self.assertEqual(n.shape, n1.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray([[0, 6, 6]]),
        )
        self.assertEqual(t.shape, t1.shape)
        assert_array_equal(
                np.asarray(t),
                np.asarray([[0, 6, 6]]),
        )
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
    
    def test_matrix(self):
        """> Test element-wise-multiplication between two matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((2, 3)) * 2
        n2 = ones_numpy((2, 3)) * 3
        t1 = ones_torch((2, 3)) * 2
        t2 = ones_torch((2, 3)) * 3
        
        # Modify
        n1[0, 0], t1[0, 0] = 0, 0
        n1[1, 2], t1[1, 2] = 1, 1
        
        # Multiply the matrices
        n = n1 * n2
        t = t1 * t2
        
        self.assertEqual(n.shape, n1.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray([[0, 6, 6],
                            [6, 6, 3]]),
        )
        self.assertEqual(t.shape, t1.shape)
        assert_array_equal(
                np.asarray(t),
                np.asarray([[0, 6, 6],
                            [6, 6, 3]]),
        )
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )
    
    def test_tensor(self):
        """> Test element-wise-multiplication between two matrices."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Load in the arrays
        n1 = ones_numpy((1, 2, 3)) * 2
        n2 = ones_numpy((1, 2, 3)) * 3
        t1 = ones_torch((1, 2, 3)) * 2
        t2 = ones_torch((1, 2, 3)) * 3
        
        # Modify
        n1[0, 0, 0], t1[0, 0, 0] = 0, 0
        n1[0, 1, 2], t1[0, 1, 2] = 1, 1
        
        # Multiply the matrices
        n = n1 * n2
        t = t1 * t2
        
        self.assertEqual(n.shape, n1.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray([[[0, 6, 6],
                             [6, 6, 3]]]),
        )
        self.assertEqual(t.shape, t1.shape)
        assert_array_equal(
                np.asarray(t),
                np.asarray([[[0, 6, 6],
                             [6, 6, 3]]]),
        )
        self.assertEqual(n.shape, t.shape)
        assert_array_equal(
                np.asarray(n),
                np.asarray(t),
        )


def main():
    s = Shape()
    s.test_vector()
    s.test_matrix()
    s.test_tensor()
    
    t = Transpose()
    t.test_vector()
    t.test_matrix()
    
    c = Concatenate()
    c.test_vector()
    c.test_matrix()
    c.test_matrix_other_dimension()
    c.test_tensor()
    c.test_tensor_second_dimension()
    c.test_tensor_third_dimension()
    
    mmod = MatrixModifications()
    mmod.test_element_replacement()
    mmod.test_vector_replacement()
    
    mmul = MatrixMultiply()
    mmul.test_vector()
    mmul.test_matrix()
    
    emul = ElementMultiply()
    emul.test_vector()
    emul.test_matrix()
    emul.test_tensor()


if __name__ == '__main__':
    unittest.main()
