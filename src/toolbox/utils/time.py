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
    """
    Add a secondary datetime x-axis (on top) that mirrors the main x-axis ticks and labels.
    """
    # Create secondary axis with identity transform
    secax = ax.secondary_xaxis(position, functions=(lambda x: x, lambda x: x))

    # Copy tick locator and formatter from the main axis
    secax.xaxis.set_major_locator(ax.xaxis.get_major_locator())
    secax.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())

    # Set label and rotation
    secax.set_xlabel("Datetime")
    secax.tick_params(rotation=rotation)

    # Optional: alignment tweak
    for label in secax.get_xticklabels():
        label.set_ha("left")

    return secax
