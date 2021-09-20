from fractal_cartoons.generators import CartoonGenerator
from typing import Tuple
from unittest.mock import patch
import numpy as np
import pytest


@pytest.mark.parametrize(
    "p1,p2",
    [
        ((0.4, 0.3), (0.2, 0.5)),
        ((-0.1, 0.3), (0.2, 0.4)),
        ((0.2, 0.4), (-0.3, 0.6)),
        ((1.2, 0.3), (1.4, 0.6)),
        ((0.4, 0.6), (1.1, 0.2)),
        ((0.3, -0.3), (0.5, 0.2)),
        ((0.3, 0.2), (0.5, -0.3)),
        ((0.3, 1.1), (0.5, 0.2)),
        ((0.3, 0.2), (0.5, 1.1)),
    ],
)
def test_wrong_point_values(p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
    """Test that points are enforced to be in [0, 1] range for both x and y coordinates."""
    with pytest.raises(ValueError):
        CartoonGenerator(p1, p2)


def test_dx() -> None:
    """Test dx computation."""
    p1 = (3 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    np.testing.assert_allclose(CartoonGenerator(p1, p2).dx, np.array([3 / 9, 2 / 9, 4 / 9]))


def test_dy() -> None:
    """Test dy computation."""
    p1 = (3 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    np.testing.assert_allclose(CartoonGenerator(p1, p2).dy, np.array([2 / 3, -1 / 3, 2 / 3]))


def test_h() -> None:
    """Test h computation."""
    p1 = (3 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    h = np.array([np.log(2 / 3) / np.log(3 / 9), np.log(1 / 3) / np.log(2 / 9), np.log(2 / 3) / np.log(4 / 9)])
    cartoon = CartoonGenerator(p1, p2)

    np.testing.assert_allclose(cartoon.h, h)
    np.testing.assert_allclose(cartoon.compute_h(), h)


def test_get_permutated_segments_indices() -> None:
    """Test _get_permutated_segments_indices with and without random permutation."""
    p1 = (3 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    cartoon = CartoonGenerator(p1, p2)
    idx = cartoon._get_permutated_segments_indices(randomized=False, random_state=None)
    np.testing.assert_array_equal(idx, np.array([0, 1, 2]))

    # Randomization
    idx = cartoon._get_permutated_segments_indices(randomized=True, random_state=np.random.RandomState(0))
    np.testing.assert_array_equal(idx, np.array([2, 1, 0]))


def test_generate_segment() -> None:
    """Test generate_segment from an initiator."""
    p1 = (3 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    cartoon = CartoonGenerator(p1, p2)

    with patch(
        "fractal_cartoons.generators.CartoonGenerator._get_permutated_segments_indices",
        return_value=np.array([0, 1, 2]),
    ):
        result = cartoon.generate_segment((1, 1), (2, 2))

    expected_result = np.array([1, 12 / 9, 14 / 9, 2]), np.array([1, 5 / 3, 4 / 3, 2])

    np.testing.assert_allclose(result[0], expected_result[0])
    np.testing.assert_allclose(result[1], expected_result[1])


def test_generate() -> None:
    """Test generate with 2 iterations from the 0-1 initiator"""
    p1 = (4 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    cartoon = CartoonGenerator(p1, p2)

    with patch(
        "fractal_cartoons.generators.CartoonGenerator._get_permutated_segments_indices",
        return_value=np.array([0, 1, 2]),
    ):
        result = cartoon.generate(np.array([0, 1]), np.array([0, 1]), iterations=2, resample=False)

    expected_x = np.array([0.0, 16 / 81, 20 / 81, 4 / 9, 40 / 81, 41 / 81, 5 / 9, 61 / 81, 65 / 81, 1.0])

    expected_y = np.array([0.0, 4 / 9, 2 / 9, 2 / 3, 4 / 9, 5 / 9, 1 / 3, 7 / 9, 5 / 9, 1.0])
    np.testing.assert_allclose(result[0], expected_x)
    np.testing.assert_allclose(result[1], expected_y)


def test_apply() -> None:
    """Test _apply with a 0-1 initiator"""
    p1 = (4 / 9, 2 / 3)
    p2 = (5 / 9, 1 / 3)
    cartoon = CartoonGenerator(
        p1,
        p2,
    )

    with patch(
        "fractal_cartoons.generators.CartoonGenerator._get_permutated_segments_indices",
        return_value=np.array([0, 1, 2]),
    ):
        result = cartoon._apply(np.array([0, 1]), np.array([0, 1]), randomized=False, random_state=None)

    expected_result_x = np.array([0, 4 / 9, 5 / 9, 1])
    expected_result_y = np.array([0, 2 / 3, 1 / 3, 1])

    np.testing.assert_array_equal(result[0], expected_result_x)
    np.testing.assert_array_equal(result[1], expected_result_y)
