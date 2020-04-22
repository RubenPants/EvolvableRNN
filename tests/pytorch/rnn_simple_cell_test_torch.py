"""
rnn_simple_cell_test.py

Test the correctness of the Simple-RNN-cell implementation.
"""
import os
import unittest
from random import random

from numpy import asarray, ones, zeros
from torch import float64, FloatTensor, tensor
from torch.nn import RNNCell as PytorchRNN

from population.utils.rnn_cell_util.simple_rnn import RNNCell

EPSILON = 1e-5


def get_simple_rnn(input_size):
    """Get a RNN-cell of the requested input-size, completely initialized with zeros."""
    bias_h = zeros((1,))
    weight_hh = zeros((1, 1))
    weight_xh = zeros((1, input_size))
    return RNNCell(
            input_size=input_size,
            bias=bias_h,
            weight_hh=weight_hh,
            weight_xh=weight_xh,
    )


def get_pytorch_rnn(input_size, used_rnn):
    """Load in a PyTorch RNNCell that is a copy of the currently used RNNCell."""
    rnn = PytorchRNN(input_size, 1)
    rnn.bias_hh[:] = tensor(zeros((1,)), dtype=float64)[:]
    rnn.bias_ih[:] = tensor(used_rnn.bias, dtype=float64)[:]
    rnn.weight_hh[:] = tensor(used_rnn.weight_hh, dtype=float64)[:]
    rnn.weight_ih[:] = tensor(used_rnn.weight_xh, dtype=float64)[:]
    return rnn


# noinspection PyArgumentList
class SimpleRNN(unittest.TestCase):
    """Test the simple RNN-cell."""
    
    def test_single_input_single_batch(self):
        """> Test when only one input given and batch-size is only one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' RNN
        rnn = get_simple_rnn(1)
        
        # Completely zero RNN, all inputs get ignored
        self.assertEqual(rnn(asarray([[0]])), 0)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        self.assertEqual(rnn(asarray([[1]])), 0)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        
        # Modify the RNN to have weight-arrays of one
        rnn.weight_hh = asarray(ones((1, 1)))
        rnn.weight_xh = asarray(ones((1, 1)))
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(1, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
        
        # Set bias to minus ones
        rnn.bias = ones((1,)) * -1
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(1, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
    
    def test_single_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' RNN
        rnn = get_simple_rnn(1)
        
        # Completely zero RNN, all inputs get ignored
        result = rnn(asarray([[0], [0]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        result = rnn(asarray([[1], [1]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        
        # Modify the RNN to have weight-arrays of one
        rnn.weight_hh = ones((1, 1))
        rnn.weight_xh = ones((1, 1))
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(1, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i], [i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        rnn.bias = ones((1,)) * -1
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(1, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i], [i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_single_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' RNN
        rnn = get_simple_rnn(2)
        
        # Completely zero RNN, all inputs get ignored
        self.assertEqual(rnn(asarray([[0, 0]])), 0)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        self.assertEqual(rnn(asarray([[1, 1]])), 0)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        
        # Modify the RNN to have weight-arrays of one
        rnn.weight_hh = ones((1, 1))
        rnn.weight_xh = ones((1, 2))
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(2, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i, i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        rnn.bias = ones((1,)) * -1
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(2, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i, i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' RNN
        rnn = get_simple_rnn(2)
        
        # Completely zero RNN, all inputs get ignored
        result = rnn(asarray([[0, 0], [0, 0]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        result = rnn(asarray([[1, 1], [1, 1]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        rnn.hx = asarray([])  # RNN keeps own state, reset it
        
        # Modify the RNN to have weight-arrays of one
        rnn.weight_hh = ones((1, 1))
        rnn.weight_xh = ones((1, 2))
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(2, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i, i], [i, i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        rnn.bias = ones((1,)) * -1
        
        # Load in PyTorch native RNN to compare with
        pytorch_rnn = get_pytorch_rnn(2, rnn)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = rnn(asarray([[i, i], [i, i]]))
            rnn.hx = asarray([])  # RNN keeps own state, reset it
            b = pytorch_rnn(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)


def main():
    e = SimpleRNN()
    e.test_single_input_single_batch()
    e.test_single_input_multi_batch()
    e.test_multi_input_single_batch()
    e.test_multi_input_multi_batch()


if __name__ == '__main__':
    unittest.main()
