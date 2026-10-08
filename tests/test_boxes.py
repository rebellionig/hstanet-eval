import numpy as np
import pytest

from hstanet_eval import iou, iou_matrix


def test_identical_boxes_have_iou_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_half_overlap():
    # intersection 50, union 150
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_degenerate_box_does_not_divide_by_zero():
    assert iou([5, 5, 5, 5], [5, 5, 5, 5]) == 0.0


def test_matrix_shape_and_empty_inputs():
    a = np.array([[0, 0, 10, 10], [0, 0, 5, 5]])
    b = np.array([[0, 0, 10, 10]])
    assert iou_matrix(a, b).shape == (2, 1)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 1)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)


def test_iou_is_symmetric():
    a = np.array([[0, 0, 10, 10], [3, 3, 8, 12]])
    b = np.array([[2, 2, 9, 9], [10, 10, 20, 20]])
    np.testing.assert_allclose(iou_matrix(a, b), iou_matrix(b, a).T)
