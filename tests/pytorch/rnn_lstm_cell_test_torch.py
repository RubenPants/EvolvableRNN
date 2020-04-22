"""
rnn_lstm_cell_test.py

Test the correctness of the LSTM-cell implementation.
"""
import os
import unittest
from random import random

from numpy import asarray, ones, zeros
from torch import float64, FloatTensor, tensor
from torch.nn import LSTMCell as PytorchLSTM

from population.utils.rnn_cell_util.lstm import LSTMCell

EPSILON = 1e-5


def get_lstm(input_size):
    """Get a LSTM-cell of the requested input-size, completely initialized with zeros."""
    bias_h = zeros((4,))
    weight_hh = zeros((4, 1))
    weight_xh = zeros((4, input_size))
    return LSTMCell(
            input_size=input_size,
            bias=bias_h,
            weight_hh=weight_hh,
            weight_xh=weight_xh,
    )


def get_pytorch_lstm(input_size, used_lstm):
    """Load in a PyTorch LSTM that is a copy of the currently used LSTM."""
    lstm = PytorchLSTM(input_size, 1)
    lstm.bias_hh[:] = tensor(zeros((4,)), dtype=float64)[:]
    lstm.bias_ih[:] = tensor(used_lstm.bias, dtype=float64)[:]
    lstm.weight_hh[:] = tensor(used_lstm.weight_hh, dtype=float64)[:]
    lstm.weight_ih[:] = tensor(used_lstm.weight_xh, dtype=float64)[:]
    return lstm


# noinspection PyArgumentList
class LSTM(unittest.TestCase):
    """Test the custom numpy implementation of the LSTM-cell."""
    
    def test_single_input_single_batch(self):
        """> Test when only one input given and batch-size is only one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' LSTM
        lstm = get_lstm(1)
        
        # Completely zero LSTM, all inputs get ignored
        self.assertEqual(lstm(asarray([[0]])), 0)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        self.assertEqual(lstm(asarray([[1]])), 0)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        
        # Modify the LSTM to have weight-arrays of one
        lstm.weight_hh = asarray(ones((4, 1)))
        lstm.weight_xh = asarray(ones((4, 1)))
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(1, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
        
        # Set bias to minus ones
        lstm.bias = ones((4,)) * -1
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(1, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i]]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(float(a) - EPSILON <= float(b) <= float(a) + EPSILON)
    
    def test_single_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' LSTM
        lstm = get_lstm(1)
        
        # Completely zero LSTM, all inputs get ignored
        result = lstm(asarray([[0], [0]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        result = lstm(asarray([[1], [1]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        
        # Modify the LSTM to have weight-arrays of one
        lstm.weight_hh = ones((4, 1))
        lstm.weight_xh = ones((4, 1))
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(1, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i], [i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        lstm.bias = ones((4,)) * -1
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(1, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i], [i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i], [i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_single_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' LSTM
        lstm = get_lstm(2)
        
        # Completely zero LSTM, all inputs get ignored
        self.assertEqual(lstm(asarray([[0, 0]])), 0)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        self.assertEqual(lstm(asarray([[1, 1]])), 0)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        
        # Modify the LSTM to have weight-arrays of one
        lstm.weight_hh = ones((4, 1))
        lstm.weight_xh = ones((4, 2))
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(2, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i, i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        lstm.bias = ones((4,)) * -1
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(2, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i, i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
    
    def test_multi_input_multi_batch(self):
        """> Test when only one input given and batch-size is more than one."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Get 'empty' LSTM
        lstm = get_lstm(2)
        
        # Completely zero LSTM, all inputs get ignored
        result = lstm(asarray([[0, 0], [0, 0]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        result = lstm(asarray([[1, 1], [1, 1]]))
        for aa, bb in zip(result, asarray([[0], [0]])):
            self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
        
        # Modify the LSTM to have weight-arrays of one
        lstm.weight_hh = ones((4, 1))
        lstm.weight_xh = ones((4, 2))
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(2, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i, i], [i, i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)
        
        # Set bias to minus ones
        lstm.bias = ones((4,)) * -1
        
        # Load in PyTorch native LSTM to compare with
        pytorch_lstm = get_pytorch_lstm(2, lstm)
        
        # Test if they continue to obtain the same results
        for _ in range(100):
            i = random()
            a = lstm(asarray([[i, i], [i, i]]))
            lstm.hx, lstm.c = asarray([]), asarray([])  # LSTM keeps own state, reset it
            (b, _) = pytorch_lstm(FloatTensor([[i, i], [i, i]]))
            self.assertEqual(a.shape, b.shape)
            for aa, bb in zip(a, b):
                self.assertTrue(float(aa) - EPSILON <= float(bb) <= float(aa) + EPSILON)


def main():
    l = LSTM()
    l.test_single_input_single_batch()
    l.test_single_input_multi_batch()
    l.test_multi_input_single_batch()
    l.test_multi_input_multi_batch()


if __name__ == '__main__':
    unittest.main()
