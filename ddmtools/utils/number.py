from typing import Tuple

import numpy as np


def log_spaced(max_num: int, points_per_decate: int = 15) -> np.ndarray:
    """Generate an array of log spaced integers smaller than L"""
    decades = np.log10(max_num)
    series: np.ndarray = np.unique(
        np.logspace(
            start=0,
            stop=decades,
            num=int(decades * points_per_decate),
            base=10,
            endpoint=False,
        ).astype(int)
    )

    return series


def get_centre_matrix(big_matrix: np.ndarray, small_matrix_shape: Tuple[int, int]) -> np.ndarray:
    assert big_matrix.ndim == 2

    big_center = np.array(big_matrix.shape) // 2

    corner = big_center - np.array(small_matrix_shape) // 2

    aa = corner[0]
    ba = corner[1]
    ab = corner[0] + small_matrix_shape[0]
    bb = corner[1] + small_matrix_shape[1]

    small_matrix: np.ndarray = big_matrix[aa:ab, ba:bb]

    return small_matrix
