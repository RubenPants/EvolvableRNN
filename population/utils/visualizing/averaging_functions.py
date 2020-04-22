"""
averaging_functions.py

Functions used to improve visualizations.
"""
import numpy as np


def Forward(values, _):
    """Simply forwarding the values."""
    return values


def EMA(values, window: int = 5):
    """Calculates the exponential moving average over a specified time-window."""
    window = min(window, len(values))
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(values, weights)[:len(values)]
    for i in range(0, window):
        ema[i] = sum(values[:i + 1]) / (i + 1)  # Simple average
    return ema


def SMA(values, window: int = 5):
    """Calculates the simple moving average."""
    window = min(window, len(values))
    weights = np.repeat(1., window) / window
    sma = np.convolve(values, weights)[:len(values)]
    for i in range(0, window):
        sma[i] = sum(values[:i + 1]) / (i + 1)  # Simple average
    return sma
