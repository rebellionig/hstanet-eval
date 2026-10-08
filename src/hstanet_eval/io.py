"""JSON input/output for predictions and ground truth.

Predictions file::

    {"img_001": {"boxes": [[x1, y1, x2, y2], ...], "scores": [0.9, ...], "labels": [0, ...]}}

Ground-truth file::

    {"img_001": {"boxes": [[x1, y1, x2, y2], ...], "labels": [0, ...]}}
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import ImageDetections, ImageGroundTruth


def load_predictions(path: str | Path) -> dict[str, ImageDetections]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: ImageDetections(v["boxes"], v["scores"], v["labels"]) for k, v in data.items()}


def load_ground_truth(path: str | Path) -> dict[str, ImageGroundTruth]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: ImageGroundTruth(v["boxes"], v["labels"]) for k, v in data.items()}
