"""Detection metrics: precision, recall, F1, AP and mAP.

Matching follows the PASCAL VOC / COCO convention: predictions are processed in
descending score order and greedily matched to the unmatched ground-truth box of
the same class and image with the highest IoU, provided it reaches the threshold.
AP is the area under the monotone precision envelope (all-point interpolation).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .boxes import iou_matrix


@dataclass
class ImageDetections:
    """Predicted boxes of a single image."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=float).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)
        self.labels = np.asarray(self.labels).reshape(-1)
        if not (len(self.boxes) == len(self.scores) == len(self.labels)):
            raise ValueError("boxes, scores and labels must have the same length")


@dataclass
class ImageGroundTruth:
    """Ground-truth boxes of a single image."""

    boxes: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=float).reshape(-1, 4)
        self.labels = np.asarray(self.labels).reshape(-1)
        if len(self.boxes) != len(self.labels):
            raise ValueError("boxes and labels must have the same length")


@dataclass
class EvalResult:
    """Aggregated evaluation output."""

    iou_thresholds: list[float]
    ap: dict[float, dict[int, float]] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def ap_at(self, iou_thr: float) -> float:
        """Mean AP over classes at a single IoU threshold."""
        per_class = self.ap[iou_thr]
        return float(np.mean(list(per_class.values()))) if per_class else 0.0

    @property
    def map(self) -> float:
        """Mean of AP@IoU over all configured thresholds."""
        if not self.iou_thresholds:
            return 0.0
        return float(np.mean([self.ap_at(t) for t in self.iou_thresholds]))

    def to_dict(self) -> dict:
        return {
            "iou_thresholds": self.iou_thresholds,
            "ap": {f"AP@{t:g}": self.ap_at(t) for t in self.iou_thresholds},
            "ap_per_class": {
                f"AP@{t:g}": {str(c): v for c, v in self.ap[t].items()} for t in self.iou_thresholds
            },
            "mAP": self.map,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve (all-point interpolation)."""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match_class(
    preds: Mapping[str, ImageDetections],
    gts: Mapping[str, ImageGroundTruth],
    label: int,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(scores, is_tp, n_gt)`` for one class, sorted by descending score."""
    scores: list[float] = []
    tps: list[bool] = []
    n_gt = sum(int(np.sum(g.labels == label)) for g in gts.values())

    for image_id, det in preds.items():
        sel = det.labels == label
        if not sel.any():
            continue
        p_boxes, p_scores = det.boxes[sel], det.scores[sel]
        gt = gts.get(image_id)
        if gt is not None:
            g_boxes = gt.boxes[gt.labels == label]
        else:
            g_boxes = np.zeros((0, 4))

        order = np.argsort(-p_scores, kind="stable")
        ious = iou_matrix(p_boxes[order], g_boxes)
        used = np.zeros(len(g_boxes), dtype=bool)

        for rank, pi in enumerate(order):
            hit = False
            if ious.shape[1]:
                cand = np.where(used, -1.0, ious[rank])
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used[j] = True
                    hit = True
            scores.append(float(p_scores[pi]))
            tps.append(hit)

    order = np.argsort(-np.asarray(scores), kind="stable")
    return (
        np.asarray(scores, dtype=float)[order],
        np.asarray(tps, dtype=bool)[order],
        n_gt,
    )


def evaluate(
    predictions: Mapping[str, ImageDetections],
    ground_truth: Mapping[str, ImageGroundTruth],
    iou_thresholds: Sequence[float] = (0.5,),
    score_threshold: float = 0.0,
) -> EvalResult:
    """Evaluate detections against ground truth.

    Args:
        predictions: ``image_id -> ImageDetections``.
        ground_truth: ``image_id -> ImageGroundTruth``.
        iou_thresholds: IoU thresholds for AP (e.g. ``(0.5, 0.75)``).
        score_threshold: confidence cut-off used for precision / recall / F1,
            computed at the *first* IoU threshold. AP always uses all predictions.

    Returns:
        :class:`EvalResult` with per-class AP, mAP and micro-averaged P/R/F1.
    """
    if not iou_thresholds:
        raise ValueError("at least one IoU threshold is required")

    labels = sorted(
        {int(x) for g in ground_truth.values() for x in g.labels}
        | {int(x) for p in predictions.values() for x in p.labels}
    )
    result = EvalResult(iou_thresholds=[float(t) for t in iou_thresholds])

    for thr in result.iou_thresholds:
        result.ap[thr] = {}
        for label in labels:
            scores, tps, n_gt = _match_class(predictions, ground_truth, label, thr)
            if n_gt == 0:
                # No ground truth for this class: AP is undefined, skip it.
                continue
            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(~tps)
            recall = tp_cum / n_gt
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            result.ap[thr][label] = average_precision(recall, precision)

    first = result.iou_thresholds[0]
    tp = fp = n_gt_total = 0
    for label in labels:
        scores, tps, n_gt = _match_class(predictions, ground_truth, label, first)
        keep = scores >= score_threshold
        tp += int(np.sum(tps[keep]))
        fp += int(np.sum(~tps[keep]))
        n_gt_total += n_gt
    fn = n_gt_total - tp
    result.precision = tp / (tp + fp) if (tp + fp) else 0.0
    result.recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = result.precision + result.recall
    result.f1 = 2 * result.precision * result.recall / denom if denom else 0.0
    return result
