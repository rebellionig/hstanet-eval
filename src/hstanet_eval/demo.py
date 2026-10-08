"""Dummy targets for trying out the ``latency`` command without a real model."""

from __future__ import annotations

import numpy as np


def dummy_model(window: np.ndarray) -> np.ndarray:
    """Stand-in for a forward pass: mean over the temporal axis plus a matmul."""
    pooled = window.mean(axis=0)
    flat = pooled.reshape(pooled.shape[0], -1)
    return flat @ flat.T
