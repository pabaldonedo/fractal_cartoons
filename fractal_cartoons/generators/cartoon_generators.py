import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple


class CartoonGenerator:
    """Generate Cartoons from a generator specified by 2 points that define 3 continuous segments, the first starting
    at 0 and the last ending at 1. This generator defines the fractal pattern to be used iteratively to generate
    a cartoon from an initiator, i.e. a segment. At each application of the generator, the 3 segments may suffer
    a random permutation to add more variability to the cartoon generator.
    """

    def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        """Initialize CartoonGenerator with the two generator points.

        Generators consists of 3 segments 0->p1, p1->p2, p2->1.

        :param p1: (x, y) coordinates of first generator point.
        :param p2: (x, y) coordinates of first generator point.
        :raise ValueError: if generator points our outside the [0, 1] range for any coordinate or
                           if p2 x coordinate > p1 x coordinate.
        """
        self.p1 = p1
        self.p2 = p2
        if (
            self.p1[0] > self.p2[0]
            or not (0 < self.p1[0] < 1)
            or not (0 < self.p1[1] < 1)
            or not (0 < self.p2[0] < 1)
            or not (0 < self.p2[1] < 1)
        ):
            raise ValueError(
                f"Generator p1 x coordinate must be smaller than p2 x coordinate and both x, y"
                f" coordinates must be in [0,1] range."
                f"\nGiven:\n\tp1: {self.p1[0]}, {self.p1[1]}\n\tp2: {self.p2[0]}, {self.p2[1]}"
            )

    @property
    def dx(self) -> np.ndarray:
        """Compute the three x coordinate increments.

        :return: np.ndarray with 3 x coordinate increments.
        """
        return np.array([self.p1[0], self.p2[0] - self.p1[0], 1 - self.p2[0]])

    @property
    def dy(self) -> np.ndarray:
        """Compute the three x coordinate increments.

        :return: np.ndarray with 3 x coordinate increments.
        """
        return np.array([self.p1[1], self.p2[1] - self.p1[1], 1 - self.p2[1]])

    @property
    def h(self) -> np.ndarray:
        """Compute Hölder exponent for each increment. Wrapper of compute_h method.

        :return: np.ndarray with 3 Hölder exponents.
        """
        return self.compute_h()

    def compute_h(self) -> np.ndarray:
        """Compute Hölder exponent for each increment.

        :return: np.ndarray with 3 Hölder exponents.
        """
        return np.log(np.abs(self.dy)) / np.log(self.dx)

    def _get_permutated_segments_indices(
        self, randomized: bool, random_state: Optional[np.random.mtrand.RandomState]
    ) -> np.ndarray:
        """Get index of dy generator segments. Optionally, they are return after a random permutation.

        :param randomized: bool whether perform a random permutation.
        :param random_state: optional random state to use in the random permutation.
        :return: array of indices of dy generator segments.
        """
        idx = np.arange(self.dy.size)

        if randomized:
            if random_state is None:
                random_state = np.random.RandomState()
            idx = random_state.permutation(idx)
        return idx

    def generate_segment(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        randomized: bool = True,
        random_state: Optional[np.random.mtrand.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate new segment from the initiator p1-p2 and the class generator.

        From a initiator defined by p1 and p2 apply one iteration of the generator self.p1, self.p2.

        Initiator:
                |------p2
                |   // |
                | //   |
                p1-----|

        Generator:
            g1 = self.p1
            g2 = self.p2

                |--------------1
                |    g1     // |
                |  //  \\  //  |
                | //    g2     |
                0--------------|

        Result:
            h = p2 - p1
            r1 = p1 + g1/h
            r2 = r1 + (g2 - g1) / h

                |------------p2
                |   r1    // |
                | //  \\ //  |
                |//    r2    |
               p1------------|

        :param p1: initial point of the initiator segment.
        :param p2: final point of the initiator segment.
        :param randomized: boolean whether to perform a random permutation of the generator segments or not.
        :param random_state: optional random state to use for the random permutation.
        :return: Tuple of x-coordinates and y-coordinates of the new segment from the initiator and generator.
        """

        range_x = p2[0] - p1[0]
        range_y = p2[1] - p1[1]

        idx = self._get_permutated_segments_indices(randomized, random_state)
        new_y = np.hstack((np.array(p1[1]), p1[1] + np.cumsum(range_y * self.dy[idx])))
        new_x = np.hstack((np.array(p1[0]), p1[0] + np.cumsum(range_x * self.dx[idx])))
        return new_x, new_y

    @staticmethod
    def _resample(x: np.ndarray, y: np.ndarray, n_samples: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply resample to the resulted curve defined by x and y coordinates.

        :param x: curve x coordinates.
        :param y: curve y coordinates
        :param n_samples: number of samples to resample. If None, n_samples = size of y.
        :return: tuple with n_sample-sized x and y coordinates arrays. of the resampled curve.
        """
        if n_samples is None:
            n_samples = y.size
        new_x = np.linspace(x[0], x[1], n_samples)
        new_y = interp1d(x, y)(new_x)
        return new_x, new_y

    def generate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iterations: int,
        randomized: bool = True,
        random_state: Optional[np.random.mtrand.RandomState] = None,
        resample: bool = True,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply n iterations of the generator given x, y input and optionally resample the result.

        Given an input of points defined by 'x' and 'y' coordinates, apply n iterations according to the
        generator. The generator samples may suffer random permutations if 'randomized' is True. Final result
        is two array of x and y coordinates respectively of the result of applying the fractal generator
        'iterations' times. If resample is True, this result is resampled in evenly spaced 'n_samples' points
        in the x coordinate. If 'n_samples' is None, then as many points as the output results are sampled.

        :param x: initial x coordinates.
        :param y: initial y coordinates.
        :param iterations: number of iterations to apply.
        :param randomized: whether to perform random permutations of the generator segments.
        :param random_state: optional random state object to perform the permutations.
        :param resample: whether to resample results in equally spaced points along the x-axis.
        :param n_samples: number of samples to resample. Only applicable if resample is True. If None,
                          the number of points of the results are kept, but it is resampled in evenly spaced
                          coordinates.
        :return: x and y coordinates of the result of applying 'iterations' times the generation algorithm.
        """
        new_x = x.copy()
        new_y = y.copy()
        for it in range(iterations):
            new_x, new_y = self._apply(new_x, new_y, randomized, random_state)

        if resample:
            return self._resample(new_x, new_y, n_samples)

        return new_x, new_y

    def _apply(
        self, x: np.ndarray, y: np.ndarray, randomized: bool, random_state: Optional[np.random.mtrand.RandomState]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply single iteration of the generation algorithm.

        :param x: initial x coordinates.
        :param y: initial y coordinates.
        :param randomized: whether to perform random permutations of the generator segments.
        :param random_state: optional random state object to perform the permutations.
        :return: x and y coordinates of the result of applying one iteration of the generation algorithm.
        """

        new_x = np.array([])
        new_y = np.array([])
        for i in range(x.size - 1):
            this_new_x, this_new_y = self.generate_segment((x[i], y[i]), (x[i + 1], y[i + 1]), randomized, random_state)
            if i == 0:
                new_x = np.hstack((new_x, this_new_x))
                new_y = np.hstack((new_y, this_new_y))
            else:
                new_x = np.hstack((new_x, this_new_x[1:]))
                new_y = np.hstack((new_y, this_new_y[1:]))

        return new_x, new_y
