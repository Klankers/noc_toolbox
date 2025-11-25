import numpy as np

# ----------------------------- NaN Handling ------------------------------
def find_nans(data: np.ndarray):
    """
    Handles generation of masks and location indices of nans.
    Intended for 1D arrays.

    Parameters
    ----------
    data : np.ndarray
        numpy array with nans

    Returns
    -------
    nan_mask : np.ndarray
        mask where incices with nans are True
    nan_indices : np.ndarray
        indices of locations where there are nans
    non_nan_indices : np.ndarray
        indices of locations where there are values
    """
    nan_mask = np.isnan(data)
    nan_indices = np.nonzero(nan_mask)[0]
    non_nan_indices = np.nonzero(~nan_mask)[0]
    return nan_mask, nan_indices, non_nan_indices

def interpolate_nans(coords, data):
    """
    Fills nan values in y using interpolation over x.
    x and y must have the same dimensions.

    Parameters
    ----------
    coords : np.ndarray
        1D array of size N which the data will be interpolated over
    data : np.ndarray
        1D array of size N to interpolate

    Returns
    -------
    filled_data : np.ndarray
        data with nans filled using linear interpolation
    """
    non_nan_mask = ~np.isnan(data)
    filled_data = np.interp(
        coords,
        coords[non_nan_mask],
        data[non_nan_mask]
    )
    return filled_data