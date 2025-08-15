import pandas as pd
import numpy as np


def safe_median_datetime(x: np.ndarray, axis=None, **kwargs) -> np.datetime64:
    """
    Safely compute the median of datetime64[ns] array using pandas.

    Parameters
    ----------
    x : np.ndarray
        A 1D array of datetime64 values.

    Returns
    -------
    np.datetime64
        Median datetime or NaT if input is empty/all-NaT.
    """
    x = pd.to_datetime(x)

    if isinstance(x, pd.DatetimeIndex):
        x = pd.Series(x)

    if x.empty or x.isna().all():
        return np.datetime64("NaT")

    return x.median()


def add_datetime_secondary_xaxis(ax, position="top", rotation=45):
    import matplotlib.dates as mdates

    secax = ax.secondary_xaxis(position, functions=(lambda x: x, lambda x: x))
    secax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    secax.tick_params(rotation=rotation)
    secax.set_xlabel("Datetime")
    for label in secax.get_xticklabels():
        label.set_ha("left")
    return secax
