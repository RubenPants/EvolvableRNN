"""
activations.py

Each of the supported activation functions.
"""
import numpy as np


def sigmoid(x: np.ndarray):
    """Added *2 to ensure that sigmoid remains inside the -3..3 range."""
    return 1 / (1 + np.exp(-2 * x))


# Directly implemented
# def tanh(x: np.ndarray):
#     """Already inside the -3..3 range."""
#     return np.tanh(x)
