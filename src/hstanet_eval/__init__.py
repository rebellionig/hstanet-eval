"""hstanet-eval: evaluation utilities for real-time video object detection."""

from .boxes import iou, iou_matrix
from .latency import benchmark
from .metrics import (
    EvalResult,
    ImageDetections,
    ImageGroundTruth,
    average_precision,
    evaluate,
)

__version__ = "0.1.0"

__all__ = [
    "EvalResult",
    "ImageDetections",
    "ImageGroundTruth",
    "average_precision",
    "benchmark",
    "evaluate",
    "iou",
    "iou_matrix",
    "__version__",
]
