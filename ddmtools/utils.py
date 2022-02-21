from typing import Optional, Tuple

import numpy as np
import pandas as pd

from joblib import Parallel
from tqdm.auto import tqdm


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


# https://stackoverflow.com/a/61900501/5036246
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm: bool = True, total: Optional[int] = None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self) -> None:
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks

        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


# Extract nominal and stddevs from a pandas series
def pd_nom(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x.nominal_value)


def pd_sd(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x.std_dev)
