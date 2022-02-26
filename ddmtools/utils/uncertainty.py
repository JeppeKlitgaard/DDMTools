from typing import Union

import pandas as pd

from uncertainties import UFloat, ufloat

FUF = Union[float, UFloat]


def as_ufloat(value: FUF) -> UFloat:
    if isinstance(value, UFloat):
        return value

    return ufloat(value, 0.0)


# Extract nominal and stddevs from a pandas series
def pd_nom(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x.nominal_value)


def pd_sd(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x.std_dev)
