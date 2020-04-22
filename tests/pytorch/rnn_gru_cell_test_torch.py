"""
rnn_gru_cell_test.py

Test the correctness of the GRU-cell implementation.
"""
import os
import unittest
from random import random

from numpy import asarray, ones, zeros
from torch import float64, FloatTensor, tensor
from torch.nn import GRUCell as PytorchGRU

from population.utils.rnn_cell_util.berkeley_gru import GRUCell as GRUBerkeley
from population.utils.rnn_cell_util.pytorch_gru import GRUCell as GRUPyTorch

EPSILON = 1e-5


def get_gru_pytorch_copy(input_size):
    """Get a GRU-cell of the requested input-size, completely initialized with zeros."""
    bias_hh = zeros((3,))
    bias_ih = zeros((3,))
    weight_hh = zeros((3, 1))
    weight_ih = zeros((3, input_size))
    return GRUPyTorch(
            input_size=input_size,
            bias_hh=bias_hh,
            bias_ih=bias_ih,
            weight_hh=weight_hh,
            weight_ih=weight_ih,
    )


def get_gru_berkeley(input_size):
    """Get a GRU-cell of the requested input-size, completely initialized with zeros."""
    bias_h = zeros((3,))
    weight_hh = zeros((3, 1))
    weight_xh = zeros((3, input_size))
    return GRUBerkeley(
            input_size=input_size,
            bias=bias_h,
            weight_hh=weight_hh,
            weight_xh=weight_xh,
    )


def get_pytorch_gru(input_size, used_gru):
    """Load in a PyTorch GRU that is a copy of the currently used GRU."""
    gru = PytorchGRU(input_size, 1)
    if type(used_gru) == GRUBerkeley:
        gru.bias_hh[:] = tensor(zeros((3,)), dtype=float64)[:]
        gru.bias_ih[:] = tensor(used_gru.bias, dtype=float64)[:]
        gru.weight_hh[:] = tensor(used_gru.weight_hh, dtype=float64)[:]
        gru.weight_ih[:] = tensor(used_gru.weight_xh, dtype=float64)[:]
    elif type(used_gru) == GRUPyTorch:
        gru.bias_hh[:] = tensor(used_gru.bias_hh, dtype=float64)[:]
        gru.bias_ih[:] = tensor(used_gru.bias_ih, dtype=float64)[:]
        gru.weight_hh[:] = tensor(used_gru.weight_hh, dtype=float64)[:]
        gru.weight_ih[:] = tensor(used_gru.weight_ih, dtype=float64)[:]
    else:
        raise Exception(f"Invalid input for used_gru: {used_gru}")
    return gru


# noinspection PyArgumentList
class PyTorch(unittest.TestCase):
    """Test the PyTorch-copy of the GRU-cell."""
    
    def test_single_input_single_batch(self):
        """> Test when only one input given and batch-size is only one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_pytorch_copy(1)
        
        # Completely zero GRU, all inputs get ignored
        self.assertEqual(gru(tensor([[0]], dtype=float64)), 0)
        gru.hx = None  # GRU keeps own state, reset it
        self.assertEqual(gru(tensor([[1]], dtype=float64)), 0)
        gru.hx = None  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = tensor(ones((3, 1)), dtype=float64)
        gru.weight_ih = tensor(ones((3, 1)), dtype=float64)
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
        
        # Set bias_ih to minus ones
        gru.bias_ih = tensor(ones((3,)), dtype=float64) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
    
    def test_single_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_pytorch_copy(1)
        
        # Completely zero GRU, all inputs get ignored
        result = gru(tensor([[0], [0]], dtype=float64))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = None  # GRU keeps own state, reset it
        result = gru(tensor([[1], [1]], dtype=float64))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = None  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = tensor(ones((3, 1)), dtype=float64)
        gru.weight_ih = tensor(ones((3, 1)), dtype=float64)
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i], [i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias_ih to minus ones
        gru.bias_ih = tensor(ones((3,)), dtype=float64) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i], [i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_single_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_pytorch_copy(2)
        
        # Completely zero GRU, all inputs get ignored
        self.assertEqual(gru(tensor([[0, 0]], dtype=float64)), 0)
        gru.hx = None  # GRU keeps own state, reset it
        self.assertEqual(gru(tensor([[1, 1]], dtype=float64)), 0)
        gru.hx = None  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = tensor(ones((3, 1)), dtype=float64)
        gru.weight_ih = tensor(ones((3, 2)), dtype=float64)
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i, i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias_ih to minus ones
        gru.bias_ih = tensor(ones((3,)), dtype=float64) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i, i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_pytorch_copy(2)
        
        # Completely zero GRU, all inputs get ignored
        result = gru(tensor([[0, 0], [0, 0]], dtype=float64))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = None  # GRU keeps own state, reset it
        result = gru(tensor([[1, 1], [1, 1]], dtype=float64))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = None  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = tensor(ones((3, 1)), dtype=float64)
        gru.weight_ih = tensor(ones((3, 2)), dtype=float64)
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i, i], [i, i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias_ih to minus ones
        gru.bias_ih = tensor(ones((3,)), dtype=float64) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(tensor([[i, i], [i, i]], dtype=float64))
            gru.hx = None  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)


# noinspection PyArgumentList
class Berkeley(unittest.TestCase):
    """Test the Berkeley implementation of the GRU-cell."""
    
    def test_single_input_single_batch(self):
        """> Test when only one input given and batch-size is only one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_berkeley(1)
        
        # Completely zero GRU, all inputs get ignored
        self.assertEqual(gru(asarray([[0]])), 0)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        self.assertEqual(gru(asarray([[1]])), 0)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = asarray(ones((3, 1)))
        gru.weight_xh = asarray(ones((3, 1)))
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
        
        # Set bias to minus ones
        gru.bias = ones((3,)) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
    
    def test_single_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_berkeley(1)
        
        # Completely zero GRU, all inputs get ignored
        result = gru(asarray([[0], [0]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        result = gru(asarray([[1], [1]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = ones((3, 1))
        gru.weight_xh = ones((3, 1))
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i], [i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        gru.bias = ones((3,)) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(1, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i], [i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_single_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_berkeley(2)
        
        # Completely zero GRU, all inputs get ignored
        self.assertEqual(gru(asarray([[0, 0]])), 0)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        self.assertEqual(gru(asarray([[1, 1]])), 0)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = ones((3, 1))
        gru.weight_xh = ones((3, 2))
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i, i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        gru.bias = ones((3,)) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i, i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' GRU
        gru = get_gru_berkeley(2)
        
        # Completely zero GRU, all inputs get ignored
        result = gru(asarray([[0, 0], [0, 0]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        result = gru(asarray([[1, 1], [1, 1]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        gru.hx = asarray([])  # GRU keeps own state, reset it
        
        # Modify the GRU to have weight-arrays of one
        gru.weight_hh = ones((3, 1))
        gru.weight_xh = ones((3, 2))
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i, i], [i, i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        gru.bias = ones((3,)) * -1
        
        # Load in PyTorch native GRU to compare with
        pytorch_gru = get_pytorch_gru(2, gru)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = gru(asarray([[i, i], [i, i]]))
            gru.hx = asarray([])  # GRU keeps own state, reset it
            b = pytorch_gru(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)


def main():
    pt = PyTorch()
    pt.test_single_input_single_batch()
    pt.test_single_input_multi_batch()
    pt.test_multi_input_single_batch()
    pt.test_multi_input_multi_batch()
    
    b = Berkeley()
    b.test_single_input_single_batch()
    b.test_single_input_multi_batch()
    b.test_multi_input_single_batch()
    b.test_multi_input_multi_batch()


if __name__ == '__main__':
    unittest.main()
