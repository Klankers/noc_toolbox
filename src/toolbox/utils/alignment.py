import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import sys

from typing import Dict, List


def interpolate_DEPTH(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolate missing DEPTH values and compute 5m depth bins, all using xarray operations.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset with variables 'DEPTH', 'TIME', and 'PROFILE_NUMBER'.

    Returns
    -------
    xr.Dataset
        Same dataset with new data variables:
            - DEPTH_interpolated
            - DEPTH_bin (floored to nearest 5 m)
            - DEPTH_range (string label like "50–55m")
    """
    # Validate required variables
    required_vars = ["DEPTH", "TIME", "PROFILE_NUMBER"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Dataset must contain variable: {var}")

    # Get dimensions
    if "N_MEASUREMENTS" not in ds.dims:
        raise ValueError(
            "Dataset must have a profile-wise measurement dimension (e.g., 'N_MEASUREMENTS')."
        )

    # Interpolate missing values per PROFILE_NUMBER
    # Approach: loop over unique profile numbers using xarray vectorized selection
    if "DEPTH" not in ds:
        raise ValueError("DEPTH not found in dataset.")

    if "PROFILE_NUMBER" not in ds:
        raise ValueError(
            "PROFILE_NUMBER must be present for per-profile interpolation."
        )

    # filter PROFILE_NUMBER > 0
    # (-1 is surfacing behaviour)
    mask = ds["PROFILE_NUMBER"] > 0
    ds = ds.sel(N_MEASUREMENTS=mask)
    # convert to int
    ds["PROFILE_NUMBER"] = ds["PROFILE_NUMBER"].astype(np.int32)
    ##### DEV ONLY #####
    # For debugging, limit to first 20 profiles
    ds = ds.where(ds["PROFILE_NUMBER"].isin(range(1, 20)), drop=True)
    print("##### DEV ONLY: Limiting to first 20 profiles for debugging #####")
    ###################

    if "PROFILE_NUMBER" not in ds.coords:
        ds = ds.set_coords("PROFILE_NUMBER")

    print("[Pipeline Manager] Interpolating missing DEPTH values by PROFILE_NUMBER...")

    # Use xarray groupby to interpolate within each profile
    DEPTH_interp = (
        ds["DEPTH"]
        .groupby("PROFILE_NUMBER")
        .map(
            lambda g: g.interpolate_na(
                dim="N_MEASUREMENTS", method="linear", fill_value="extrapolate"
            ).reindex_like(g)
        )
    )

    ds = ds.assign({"DEPTH_interpolated": DEPTH_interp})

    # Compute 5 m bins
    bin_size = 5
    DEPTH_bin = (ds["DEPTH_interpolated"] // bin_size) * bin_size
    ds["DEPTH_bin"] = DEPTH_bin.astype(np.float32)

    # Add range label (as a string)
    ds["DEPTH_range"] = xr.apply_ufunc(
        lambda start, end: (
            f"{int(start)}–{int(end)}"
            if not np.isnan(start) and not np.isnan(end)
            else ""
        ),
        ds["DEPTH_bin"],
        ds["DEPTH_bin"] + bin_size,
        vectorize=True,
        dask="parallelized" if ds.chunks else False,
        output_dtypes=[str],
    )

    return ds


def aggregate_vars(ds, vars_to_aggregate):
    """
    Aggregate specified variables by PROFILE_NUMBER and DEPTH_bin using xarray only.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'PROFILE_NUMBER' and 'DEPTH_bin' as coordinates or variables.

    vars_to_aggregate : list of str
        List of variable names in the dataset to aggregate.

    Returns
    -------
    xr.Dataset
        Dataset with median-aggregated variables named as median_{var},
        indexed by PROFILE_NUMBER and DEPTH_bin.
    """
    # Ensure required keys exist
    if "PROFILE_NUMBER" not in ds:
        raise ValueError("Dataset must contain 'PROFILE_NUMBER'.")
    if "DEPTH_bin" not in ds:
        raise ValueError("Dataset must contain 'DEPTH_bin'.")

    if "PROFILE_NUMBER" not in ds.coords:
        ds = ds.set_coords("PROFILE_NUMBER")
    if "DEPTH_bin" not in ds.coords:
        ds = ds.set_coords("DEPTH_bin")

    print("[Pipeline Manager] Aggregating variables...")

    for var in vars_to_aggregate:
        if var not in ds:
            print(f"[Warning] Skipping missing variable: {var}")
            print(f"[Warning] Available variables: {list(ds.data_vars)}")
            continue

        # Group by (PROFILE_NUMBER, DEPTH_bin) and take median
        grouped = ds[var].groupby(["PROFILE_NUMBER", "DEPTH_bin"])
        median = grouped.reduce(np.nanmedian)

        # create new variable name
        new_var_name = f"median_{var}"
        ds[new_var_name] = median
        print(f"[Pipeline Manager] Aggregated {var} → {new_var_name}")
    # drop rows that have any NANs in the aggregated variables
    agg_vars = [f"median_{var}" for var in vars_to_aggregate if var in ds]
    ds = ds.dropna(dim="N_MEASUREMENTS", how="any", subset=agg_vars)
    return ds


def filter_xarray_by_profile_ids(
    ds: xr.Dataset,
    profile_id_var: str,
    valid_ids: np.ndarray | list,
) -> xr.Dataset:
    """
    Filters an xarray.Dataset to include only rows with PROFILE_NUMBER in a list of valid IDs.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to filter.

    profile_id_var : str
        The name of the variable or coordinate in ds representing the profile ID
        (e.g., 'PROFILE_NUMBER', 'Doombar_PROFILE_NUMBER', etc.)

    valid_ids : array-like
        List or array of profile ID values to retain.

    verbose : bool
        If True, prints the number of rows before and after filtering.

    Returns
    -------
    xr.Dataset
        Filtered dataset containing only rows with matching profile IDs.
    """
    if profile_id_var not in ds:
        raise KeyError(f"'{profile_id_var}' not found in dataset.")

    profile_ids = ds[profile_id_var]
    print(f"[Filter] Filtering dataset by {len(valid_ids)} valid IDs...")

    mask = profile_ids.isin(valid_ids)
    filtered = ds.where(mask, drop=True)

    original_size = ds.sizes.get("N_MEASUREMENTS", "unknown")
    new_size = filtered.sizes.get("N_MEASUREMENTS", "unknown")
    print(f"[Filter] {profile_id_var}: {original_size} → {new_size} rows retained.")

    return filtered


def merge_profile_pairs_on_depth_bin(
    paired_df: pd.DataFrame,
    target_ds: xr.Dataset,
    ancillary_ds: xr.Dataset,
    target_name: str,
    ancillary_name: str,
    bin_dim: str = "DEPTH_bin",
    max_pairs: int = None,  # for debugging
) -> xr.Dataset:
    """
    Merge binned datasets for each unique (target_profile, ancillary_profile) pair on DEPTH_bin.

    Parameters
    ----------
    paired_df : pd.DataFrame
        DataFrame with columns:
          - f"{target_name}_PROFILE_NUMBER"
          - f"{ancillary_name}_PROFILE_NUMBER"
          - 'time_diff_hr'
          - 'dist_km'

    target_ds : xr.Dataset
        Depth-binned and filtered xarray Dataset for target glider.

    ancillary_ds : xr.Dataset
        Depth-binned and filtered xarray Dataset for ancillary glider.

    target_name : str
        Target glider name (used for suffixing and column access).

    ancillary_name : str
        Ancillary glider name (used for suffixing and column access).

    bin_dim : str
        Dimension to align on (e.g., "DEPTH_bin").

    max_pairs : int, optional
        If set, limits the number of profile pairs processed (for debugging).

    Returns
    -------
    xr.Dataset
        Combined dataset with one record per (profile_pair, depth_bin), including metadata.
    """

    merged_list = []

    target_profile_col = f"{target_name}_PROFILE_NUMBER"
    ancillary_profile_col = f"{ancillary_name}_PROFILE_NUMBER"

    pairs = paired_df[
        [target_profile_col, ancillary_profile_col, "time_diff_hr", "dist_km"]
    ]

    if max_pairs is not None:
        pairs = pairs.head(max_pairs)

    for idx, row in pairs.iterrows():
        pid_target = row[target_profile_col]
        pid_anc = row[ancillary_profile_col]
        time_diff = row["time_diff_hr"]
        dist = row["dist_km"]

        # Subset each dataset to just the current profile
        tgt_sel = target_ds.sel(PROFILE_NUMBER=pid_target, drop=True)
        anc_sel = ancillary_ds.sel(PROFILE_NUMBER=pid_anc, drop=True)

        # Add suffix to avoid var name collisions
        tgt_sel = tgt_sel.rename(
            {v: f"{v}_TARGET_{target_name}" for v in tgt_sel.data_vars}
        )
        anc_sel = anc_sel.rename(
            {v: f"{v}_{ancillary_name}" for v in anc_sel.data_vars}
        )

        # Merge on DEPTH_bin
        pair_merged = xr.merge(
            [tgt_sel, anc_sel],
            join="inner",
            compat="override",
            combine_attrs="override",
        )

        # Annotate with profile metadata
        pair_merged = pair_merged.expand_dims("PAIR_INDEX")
        pair_merged["TARGET_PROFILE_NUMBER"] = pid_target
        pair_merged["ANCILLARY_PROFILE_NUMBER"] = pid_anc
        pair_merged["time_diff_hr"] = time_diff
        pair_merged["dist_km"] = dist

        merged_list.append(pair_merged)

    if not merged_list:
        raise ValueError("No valid profile-pair merges produced.")

    final_ds = xr.concat(merged_list, dim="PAIR_INDEX")
    return final_ds


def major_axis_r2(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) for Major Axis (Type II) regression.
    R² is simply the square of the Pearson correlation coefficient.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan  # Not enough points

    r, _ = pearsonr(x_clean, y_clean)
    return r**2


def compute_r2_for_merged_profiles(
    merged_df: pd.DataFrame,
    variables: List[str],
    target_name: str,
    other_names: List[str],
) -> pd.DataFrame:
    """
    Compute R² values for each variable between target and all other gliders.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of merge_depth_binned_profiles()

    variables : list of str
        List of variable names to compare (e.g., ["salinity", "temperature"])

    target_name : str
        Name of the target glider (used in suffixes)

    other_names : list of str
        Other glider names to compare against

    group_columns : list
        Columns to group by (default: PROFILE_NUMBER + DEPTH_bin)

    Returns
    -------
    pd.DataFrame
        One row per target_profile vs comparison_profile pair with R² values for each variable.
    """
    results = []

    grouped = merged_df.groupby(
        [f"{target_name}_PROFILE_NUMBER"]
        + [f"{other}_PROFILE_NUMBER" for other in other_names]
    )

    for keys, group in grouped:
        row = {f"{target_name}_PROFILE_NUMBER": keys[0]}
        for i, other in enumerate(other_names):
            row[f"{other}_PROFILE_NUMBER"] = keys[i + 1]

        for var in variables:
            col_target = f"median_{var}_TARGET_{target_name}"

            for other in other_names:
                col_other = f"median_{var}_{other}"
                if col_target in group.columns and col_other in group.columns:
                    x = group[col_target].to_numpy()
                    y = group[col_other].to_numpy()
                    row[f"r2_{var}_{other}"] = major_axis_r2(x, y)
                else:
                    row[f"r2_{var}_{other}"] = np.nan

        results.append(row)

    return pd.DataFrame(results)
