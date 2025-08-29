import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from typing import Dict, List


def interpolate_DEPTH(df):
    """Interpolate missing DEPTH values in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'DEPTH' column.

    Returns:
        pd.DataFrame: DataFrame with interpolated 'DEPTH' values.
    """
    if "DEPTH" not in df.columns:
        raise ValueError("DataFrame must contain a 'DEPTH' column.")

    # sort df  by [profile_id, TIME]
    df = df.sort_values(by=["profile_id", "TIME"])
    # Interpolate missing DEPTH values within each profile_id group
    df["DEPTH_interpolated"] = df.groupby("profile_id")["DEPTH"].apply(
        lambda group: group.interpolate(method="linear", limit_direction="both")
    )
    # add DEPTH_bin to sort into 5m bins
    df["DEPTH_bin"] = (df["DEPTH_interpolated"] // 5) * 5
    # add string range for DEPTH_bin
    df["DEPTH_range"] = df["DEPTH_bin"].map(lambda x: f"{int(x)}-{int(x)+5}m")
    return df


def aggregate_vars(df, vars_to_aggregate):
    """Aggregate specified variables by profile_id and DEPTH_bin.

    Args:
        df (pd.DataFrame): Input DataFrame with 'profile_id', 'DEPTH_bin', and variables to aggregate.
        vars_to_aggregate (list): List of variable names to aggregate.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    if "profile_id" not in df.columns or "DEPTH_bin" not in df.columns:
        raise ValueError("DataFrame must contain 'profile_id' and 'DEPTH_bin' columns.")

    # Agg over median, with alias of _media{var}
    agg_dict = {var: "median" for var in vars_to_aggregate}
    aggregated_df = df.groupby(["profile_id", "DEPTH_bin"]).agg(agg_dict).reset_index()
    # Rename columns
    aggregated_df.rename(
        columns={var: f"median_{var}" for var in vars_to_aggregate}, inplace=True
    )
    # sort by profile_id and DEPTH_bin
    aggregated_df = aggregated_df.sort_values(by=["profile_id", "DEPTH_bin"])
    # drop nulls
    aggregated_df = aggregated_df.dropna(
        subset=[f"median_{var}" for var in vars_to_aggregate]
    )
    return aggregated_df


def merge_depth_binned_profiles(
    target_name: str, binned_dfs: Dict[str, pd.DataFrame], bin_column: str = "DEPTH_bin"
) -> pd.DataFrame:
    """
    Merge depth-binned profile data for a target glider and multiple others.

    Parameters
    ----------
    target_name : str
        The name of the target glider (used as the left/base of the join).

    binned_dfs : dict
        Dictionary of {glider_name: binned_df} where each DataFrame includes
        'profile_id' and 'DEPTH_bin', and is the result of interpolate_DEPTH + aggregate_vars.

    bin_column : str
        The name of the depth bin column to join on.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with depth-binned values from target and all other gliders,
        using consistent suffixes like _TARGET_{target_name} and _{glider}.
    """
    if target_name not in binned_dfs:
        raise ValueError(f"Target '{target_name}' not found in binned_dfs.")

    # Add glider_profile_id to each glider's dataframe
    for name, df in binned_dfs.items():
        if f"{name}_profile_id" not in df.columns:
            binned_dfs[name] = df.copy()
            binned_dfs[name][f"{name}_profile_id"] = (
                df["profile_id"].astype(str) + f"_{name}"
            )

    # Use target as base
    target_df = binned_dfs[target_name].copy()
    target_profile_col = f"{target_name}_profile_id"

    # Rename target columns with _TARGET suffix (except join keys)
    target_df_renamed = target_df.rename(
        columns={
            col: f"{col}_TARGET_{target_name}"
            for col in target_df.columns
            if col not in [bin_column, "profile_id", target_profile_col]
        }
    )

    # Start with target dataframe as base
    merged_df = target_df_renamed.copy()

    for glider_name, glider_df in binned_dfs.items():
        if glider_name == target_name:
            continue

        profile_col = f"{glider_name}_profile_id"

        # Rename glider columns (excluding join keys)
        glider_df_renamed = glider_df.rename(
            columns={
                col: f"{col}_{glider_name}"
                for col in glider_df.columns
                if col not in [bin_column, "profile_id", profile_col]
            }
        )

        merged_df = merged_df.merge(
            glider_df_renamed,
            how="left",
            left_on=[target_profile_col, bin_column],
            right_on=[profile_col, bin_column],
            suffixes=("", f"_{glider_name}"),
        )

    return merged_df


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
        Columns to group by (default: profile_id + DEPTH_bin)

    Returns
    -------
    pd.DataFrame
        One row per target_profile vs comparison_profile pair with R² values for each variable.
    """
    results = []

    grouped = merged_df.groupby(
        [f"{target_name}_profile_id"] + [f"{other}_profile_id" for other in other_names]
    )

    for keys, group in grouped:
        row = {f"{target_name}_profile_id": keys[0]}
        for i, other in enumerate(other_names):
            row[f"{other}_profile_id"] = keys[i + 1]

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
