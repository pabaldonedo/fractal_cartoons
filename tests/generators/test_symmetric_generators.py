from fractal_cartoons.symmetric_generators import SymmetricCartoonGenerator
from typing import Tuple
import numpy as np
import pytest


@pytest.mark.parametrize(
    "p1",
    [(-0.4, 0.3), (0.8, 0.1), (0.2, -0.2), (0.4, 1.1)],
)
def test_wrong_point_values(p1: Tuple[float, float]) -> None:
    """Test that point is enforce to have x in [0, .5] and y [0, 1]."""
    with pytest.raises(ValueError):
        SymmetricCartoonGenerator(p1)


def test_get_unifractal_phase() -> None:
    """Test that get_unifractal_phase returns values meeting y=x**h."""
    x, y, h = SymmetricCartoonGenerator.get_unifractal_phase()
    np.testing.assert_almost_equal(y, x ** h)
