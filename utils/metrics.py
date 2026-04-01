"""Lightweight metrics helpers."""

import numpy as np


def rolling_average(values, window):
    """Compute rolling average using convolution. Returns array of same length (padded)."""
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    # Use 'same' mode to keep length, then fix the edges
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad front so output length matches input
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def compute_win_rates(win_history, window=500):
    """
    From a list of +1/0/-1 outcomes, compute rolling win/draw/loss rates.
    Returns dict with 'win', 'draw', 'loss' keys, each a numpy array.
    """
    wins   = rolling_average([1.0 if x ==  1 else 0.0 for x in win_history], window)
    draws  = rolling_average([1.0 if x ==  0 else 0.0 for x in win_history], window)
    losses = rolling_average([1.0 if x == -1 else 0.0 for x in win_history], window)
    return {"win": wins, "draw": draws, "loss": losses}


def final_stats(win_history, last_n=2000):
    """Return final W/D/L rates over the last_n episodes."""
    recent = win_history[-last_n:]
    n = len(recent)
    return {
        "win_rate":  sum(1 for x in recent if x ==  1) / n,
        "draw_rate": sum(1 for x in recent if x ==  0) / n,
        "loss_rate": sum(1 for x in recent if x == -1) / n,
    }
