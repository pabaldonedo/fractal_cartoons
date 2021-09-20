from .cartoon_generators import CartoonGenerator
from matplotlib.axes import Axes
from scipy import optimize
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


class SymmetricCartoonGenerator(CartoonGenerator):
    def __init__(self, p1: Tuple[float, float]) -> None:
        """Initialize SymmetricCartoonGenerator with the symmetric generator point.

        Generators consists of 3 segments: 0->p1, p1->1-p1, 1-p1-> 1.This is the same as a CartoonGenerator
        with 2 generating points that are symmetric to the point (0.5, 0.5).

        :param p1: (x, y) coordinates of the generator point.
        :raise ValueError: if generator point x coordinate outside [0, 0.5] or y coordinate outside [0, 1].
        """
        if not (0 < p1[0] < 0.5):
            raise ValueError(f"x must be in [0, 0.5], received: {p1[0]}")

        if not (0 < p1[1] < 1):
            raise ValueError(f"y must be in [0, 1], received: {p1[1]}")

        p2 = (1 - p1[0], 1 - p1[1])
        super().__init__(p1, p2)

    @staticmethod
    def get_unifractal_phase() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the coordinates of the unifractal line in the phase diagram.

        Unifractality condition is met when dy = dx^H for all (dx, dy) increments with a fixed H value. Therefore,
        given the symmetric points of this generator this function solves the following system:
            y = x^H  # For the first and third increment
            2y - 1 = (1 - 2x)^H  # For the second increment
            => 2x^H - 1 = (1 - 2x)^H  # Substitute first equation in the second

        The function samples 20 evenly-spaced points from 0.01 to 0.99 of H, and solves the equations above for each
        H value.

        :return: Tuple of x, y and H coordinates meeting the unifractality condition.
        """

        def unifractality_condition(x_value: float, h_value: float) -> float:
            """Return unifractal condition written as f(x) = 0 evaluated at x_value.

            :param x_value: x coordinate.
            :param h_value: h exponent values
            :return: unifractal condition as f(x) = 0 evaluated at x_value.
            """
            return 2 * x_value ** h_value - 1 - (1 - 2 * x_value) ** h_value

        x = []
        y = []
        h = []

        for hi in np.linspace(1e-2, 0.99, 20):
            xi = optimize.newton(unifractality_condition, 0.3, args=([hi]))
            yi = xi ** hi
            # out of domain
            if yi == 1:
                continue
            h.append(hi)
            y.append(yi)
            x.append(xi)

        return np.array(x), np.array(y), np.array(h)

    @staticmethod
    def get_phase_diagram_base(ax: Optional[Axes] = None) -> Axes:
        """Generate phase diagram with linear, unifractal, devil staircase and mesofractal lines.

        Plot a diagram indicating the different regimes of fractality depending on the x and y coordinates
        of the symmetric point of the SymmetricCartoonGenerator. It plots the following regimes:
            - Linear: generator is just a straight line.
            - Unifractal: points meeting the unifractality condition.
            - Devil staircase: generator middle segment is a flat line with no y increments.
            - Mesofractal: H2=0; H1 = H3 != 0.

        :param ax: axes to use to plot. If None, a new plot is generated.
        :return: ax updated with plotted phase diagram.
        """
        if ax is None:
            fig, ax = plt.subplots()

        blue_color = "C0"
        black_color = "k"

        # Linear
        ax.plot([0, 0.5], [0, 0.5], color=blue_color)
        # Unifractal
        x, y, h = SymmetricCartoonGenerator.get_unifractal_phase()
        ax.plot(x, y, color=blue_color)
        # Devil staircase
        ax.plot([0, 0.5], [0.5, 0.5], color=black_color)
        # Mesofractal function cartoons
        ax.plot([0.5, 0.5], [0, 0.5], color=black_color)
        # Mesofractal measure cartoons
        ax.plot([0.5, 0.5], [0.5, 1], color=black_color)
        return ax

    def get_phase_diagram(self, ax: Optional[Axes] = None, color: str = "r") -> Axes:
        """Plot the phase diagram along with the symmetric point of this generator overlayed.

        :param ax: axes to use to plot. If None, a new plot is generated.
        :param color: color to use for plotting the symmetric point of this generator.
        :return: ax updated with plotted phase diagram and symmetric point.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        ax = self.get_phase_diagram_base(ax=ax)
        ax.scatter(*self.p1, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax


def get_trading_time_generator(gen: CartoonGenerator) -> CartoonGenerator:
    """Instantiate a CartoonGenerator for trading time from a price generator.

    Solves for D the equation |dy1|^D + |dy2|^D + ...  |dyn|^D = 1 where dy_i is the ith price increment in gen.
    Then, instantiate the generator with:
        -x: clock time coming from gen.x
        -y: trading time as dT_i = |dy_i|^D

    :param gen: price generator.
    :return: trading time generator.
    """

    def equation(d: float, dys: np.ndarray) -> float:
        """Return equation |dy1|^D + |dy2|^D + ...  |dyn|^D - 1"""
        return np.sum(np.abs(dys) ** d) - 1

    d_exponent = optimize.fsolve(equation, np.ndarray([1]), args=gen.dy)
    dy = np.abs(gen.dy) ** d_exponent
    y = np.cumsum(dy)
    if abs(y[-1] - 1) > 1e-3:
        raise RuntimeError

    return CartoonGenerator((gen.p1[0], y[0]), (gen.p2[0], y[1]))
