import numpy as np
import pytest

from hstanet_eval import ImageDetections, ImageGroundTruth, average_precision, evaluate


def det(boxes, scores, labels):
    return ImageDetections(boxes, scores, labels)


def gt(boxes, labels):
    return ImageGroundTruth(boxes, labels)


def test_perfect_detector_scores_one():
    gts = {"a": gt([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])}
    preds = {"a": det([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 1])}
    res = evaluate(preds, gts)
    assert res.map == pytest.approx(1.0)
    assert (res.precision, res.recall, res.f1) == (1.0, 1.0, 1.0)


def test_no_predictions_gives_zero():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    res = evaluate({}, gts)
    assert res.map == 0.0
    assert res.recall == 0.0 and res.f1 == 0.0


def test_wrong_class_is_not_a_match():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    preds = {"a": det([[0, 0, 10, 10]], [0.9], [1])}
    res = evaluate(preds, gts)
    assert res.ap_at(0.5) == 0.0
    assert res.precision == 0.0


def test_duplicate_detection_counts_as_false_positive():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    preds = {"a": det([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])}
    res = evaluate(preds, gts)
    assert res.recall == 1.0
    assert res.precision == pytest.approx(0.5)


def test_ap_hand_computed_example():
    # 2 GT; predictions sorted by score: TP, FP, TP
    # recall:    0.5, 0.5, 1.0   precision: 1, 0.5, 2/3
    # envelope:  1.0, 2/3, 2/3   AP = 0.5*1 + 0.5*(2/3) = 5/6
    gts = {"a": gt([[0, 0, 10, 10], [50, 50, 60, 60]], [0, 0])}
    preds = {
        "a": det(
            [[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]],
            [0.9, 0.8, 0.7],
            [0, 0, 0],
        )
    }
    assert evaluate(preds, gts).ap_at(0.5) == pytest.approx(5 / 6)


def test_higher_iou_threshold_is_stricter():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    preds = {"a": det([[0, 0, 10, 7]], [0.9], [0])}  # IoU = 0.7
    res = evaluate(preds, gts, iou_thresholds=(0.5, 0.75))
    assert res.ap_at(0.5) == pytest.approx(1.0)
    assert res.ap_at(0.75) == 0.0
    assert res.map == pytest.approx(0.5)


def test_score_threshold_affects_prf_but_not_ap():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    preds = {"a": det([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.3], [0, 0])}
    low = evaluate(preds, gts, score_threshold=0.0)
    high = evaluate(preds, gts, score_threshold=0.5)
    assert low.ap_at(0.5) == high.ap_at(0.5)
    assert low.precision == pytest.approx(0.5)
    assert high.precision == 1.0


def test_images_without_ground_truth_produce_false_positives():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    preds = {
        "a": det([[0, 0, 10, 10]], [0.5], [0]),
        "b": det([[0, 0, 10, 10]], [0.9], [0]),  # image b has no GT
    }
    res = evaluate(preds, gts)
    assert res.precision == pytest.approx(0.5)
    assert res.ap_at(0.5) == pytest.approx(0.5)


def test_class_without_ground_truth_is_excluded_from_map():
    gts = {"a": gt([[0, 0, 10, 10]], [0])}
    preds = {"a": det([[0, 0, 10, 10], [30, 30, 40, 40]], [0.9, 0.8], [0, 7])}
    res = evaluate(preds, gts)
    assert list(res.ap[0.5]) == [0]
    assert res.map == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        det([[0, 0, 1, 1]], [0.9, 0.8], [0])
    with pytest.raises(ValueError):
        gt([[0, 0, 1, 1]], [0, 1])


def test_requires_iou_threshold():
    with pytest.raises(ValueError):
        evaluate({}, {}, iou_thresholds=())


def test_average_precision_edge_cases():
    assert average_precision(np.array([]), np.array([])) == 0.0
    assert average_precision(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)
