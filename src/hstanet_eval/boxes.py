"""Bounding-box utilities.

Boxes are expressed as ``[x1, y1, x2, y2]`` (top-left, bottom-right) in pixels.
"""

from __future__ import annotations

import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Return the area of each box. Degenerate boxes have zero area."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return w * h


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Intersection-over-Union between two sets of boxes.

    Args:
        a: array of shape ``(N, 4)``.
        b: array of shape ``(M, 4)``.

    Returns:
        Array of shape ``(N, M)`` with IoU values in ``[0, 1]``.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two single boxes."""
    return float(iou_matrix(np.array([box_a]), np.array([box_b]))[0, 0])
