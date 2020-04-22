"""
activations_cy.py

Each of the supported activation functions.
"""
import numpy as np
cimport numpy as np


cpdef np.ndarray sigmoid_cy(np.ndarray x):
    """Added *2 to ensure that sigmoid remains inside the -3..3 range."""
    return 1 / (1 + np.exp(-2 * x))


# Directly implemented
# cpdef np.ndarray tanh_cy(np.ndarray x):
#     """Already inside the -3..3 range."""
#     return np.tanh(x)
