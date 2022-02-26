"""
Contains primitive signal processing functionality
"""
from multiprocessing import cpu_count
from typing import Optional, Tuple

import numpy as np
import scipy.fft

import numba

try:
    import pyfftw
except ImportError:
    PYFFTW_AVAILABLE = True
    empty = pyfftw.empty_aligned
    rfft2 = pyfftw.interfaces.scipy_fft.rfft2
else:
    PYFFTW_AVAILABLE = False
    empty = np.empty
    rfft2 = scipy.fft.rfft2


def pyfftw_setup() -> None:
    if pyfftw is not None:
        pyfftw.interfaces.cache.enable()
        set_pyfftw_cores(None)


def set_pyfftw_cores(num_cores: Optional[int]) -> None:
    if pyfftw is not None:
        pyfftw.config.NUM_THREADS = num_cores or cpu_count()


# https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray
@numba.vectorize(  # type: ignore
    [
        numba.float64(numba.complex128),
        numba.float32(numba.complex64),
    ]
)
def mod_square(x: np.ndarray) -> np.ndarray:
    return x.real**2 + x.imag**2  # type: ignore


class HalfPlaneRadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""

    def __init__(self, shape: Tuple[int, int]) -> None:
        """
        A RadialAverager instance can process only arrays of a given shape,
        fixed at instanciation.
        """
        self.shape = shape

        if len(shape) != 2:
            raise ValueError("Invalid shape.")

        # Calculate a matrix of distances in frequency space
        self.dists = np.sqrt(
            np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.rfftfreq(shape[1])[None, :] ** 2
        )

        # Dump the cross
        # TODO: Why do we do this?
        self.dists[0] = 0
        self.dists[:, 0] = 0

        # Discretize distances into bins
        self.bins = np.arange(max(shape) // 2 + 1) / float(max(shape))

        # Number of pixels at each distance
        self.pixel_density = np.histogram(self.dists, self.bins)[0]

    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        """Perform and return the radial average of the spectrum"""
        assert spectrum.shape == self.dists.shape

        hw = np.histogram(self.dists, self.bins, weights=spectrum)[0]
        average: np.ndarray = hw / self.pixel_density

        return average
