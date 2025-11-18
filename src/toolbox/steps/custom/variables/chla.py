#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


@register_step
class chla_deep_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Deep Correction"

    def run(self):
        """
        Example
        -------

        - name: "Chla Deep Correction"
          parameters:
            dark_value: null
            depth_threshold: 200
        diagnostics: true

        Returns
        -------

        """
        self.filter_qc()

        self.compute_dark_value()
        self.apply_dark_correction()

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"CHLA_FLUORESCENCE_QC": ["CHLA_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def compute_dark_value(self):
        """
        Compute dark value for chlorophyll-a correction.

        The dark value represents the sensor's baseline reading in the absence of
        chlorophyll fluorescence. Computed as the median of minimum CHLA values from
        deep profiles (>= depth_threshold).

        Parameters
        ----------
        ds : xarray.Dataset
            Glider dataset with variables: CHLA, DEPTH (or PRES), PROFILE_NUMBER
        depth_threshold : float, optional
            Minimum depth [m] for dark value calculation (default: 200)
        n_profiles : int, optional
            Number of deep profiles to use (default: 5)
        config_path : str or Path, optional
            Path to config file to check for existing dark value

        Returns
        -------
        dark_value : float
            Computed dark value
        profile_data : dict
            Dictionary containing profile information used in calculation
            Keys are profile numbers, values are dicts with 'depth', 'chla', 'min_value', 'min_depth'
        """

        # Check config file for existing dark value
        if self.dark_value:
            self.log(f"Using dark value from config: {self.dark_value}")
            return self.dark_value

        print(f"Computing dark value from profiles reaching >= {self.depth_threshold}m")

        # Get DEPTH and CHLA data
        missing_vars = {"TIME", "PROFILE_NUMBER", "DEPTH", "CHLA"} - set(self.data.data_vars)
        if missing_vars:
            raise KeyError(f"[Chla Deep Correction] {missing_vars} could not be found in the data.")

        # Convert to pandas dataframe and interpolate the DEPTH data
        interp_data = self.data[["TIME", "PROFILE_NUMBER", "DEPTH", "CHLA"]].to_pandas()
        interp_data["DEPTH"] = interp_data.set_index("TIME")["DEPTH"].interpolate().reset_index(drop=True)
        interp_data = interp_data.dropna(subset=["CHLA", "PROFILE_NUMBER"])

        # Subset the data to only deep measurements
        interp_data = interp_data[
            interp_data["DEPTH"] < self.depth_threshold
        ]

        # Remove profiles that do not have CHLA data below the threshold depth
        deep_profiles = interp_data.groupby("PROFILE_NUMBER").agg({"CHLA": "count"}).reset_index()
        deep_profiles = deep_profiles[deep_profiles["CHLA"] > 0]["PROFILE_NUMBER"].to_numpy()
        if len(deep_profiles) == 0:
            raise ValueError(
                "[Chla Deep Correction] No deep profiles could be identified. "
                "Try adjusting the 'depth_threshold' parameter."
            )
        interp_data = interp_data[interp_data["PROFILE_NUMBER"].isin(deep_profiles)]

        # Extract the profile number, depth and chla data for all chla minima per profile
        self.chla_deep_minima = interp_data.loc[
            interp_data.groupby("PROFILE_NUMBER")["CHLA"].idxmin(),
            ["TIME", "PROFILE_NUMBER", "DEPTH", "CHLA"]
        ]

        # Compute median of minimum values
        self.dark_value = np.nanmedian(self.chla_deep_minima["CHLA"])
        self.log(
            f"\nComputed dark value: {self.dark_value:.6f} "
            f"(median of {len(self.chla_deep_minima)} profile minimums)\n"
            f"Min values range: {np.min(self.chla_deep_minima["CHLA"]):.6f} "
            f"to {np.max(self.chla_deep_minima["CHLA"]):.6f}"
        )

    def apply_dark_correction(self):
        """
        Apply dark value correction to CHLA data.
        """

        # Create adjusted chlorophyll variable
        self.data["CHLA_FLUORESCENCE"] = xr.DataArray(
            self.data["CHLA"] - self.dark_value,
            dims=self.data["CHLA"].dims,
            coords=self.data["CHLA"].coords,
        )

        # Copy and update attributes
        if hasattr(self.data["CHLA"], 'attrs'):
            self.data["CHLA_FLUORESCENCE"].attrs = self.data["CHLA"].attrs.copy()
        self.data["CHLA_FLUORESCENCE"].attrs["comment"] = (
            f"CHLA fluorescence with dark value correction (dark_value={self.dark_value:.6f})"
        )
        self.data["CHLA_FLUORESCENCE"].attrs["dark_value"] = self.dark_value

        self.log(f"Applied dark value correction: CHLA_FLUORESCENCE = CHLA_FLUORESCENCE - {self.dark_value:.6f}")

    def generate_diagnostics(self):

        mpl.use("tkagg")

        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        ax.plot(
            self.chla_deep_minima["CHLA"],
            self.chla_deep_minima["DEPTH"],
            ls="",
            marker="o",
            c="b"
        )

        ax.axhline(self.depth_threshold, ls="--", c="k", label="Depth Threshold")
        ax.axvline(self.dark_value, ls="--", c="r", label="Dark Value")
        ax.legend(loc="upper right")

        ax.set(
            xlabel="CHLA",
            ylabel="DEPTH",
            title="Deep Minima Values",
        )

        fig.tight_layout()
        plt.show(block=True)

@register_step
class chla_quenching_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Quenching Correction"

    def run(self):
        """
        Example
        -------

        - name: "Chla Quenching Correction"
          parameters:
            method: "Argo"
            apply_to: "CHLA"
            mld_settings: {
              "threshold_on": "TEMP",
              "reference_depth": 10,
              "threshold": 0.2
              }
          diagnostics: true
        """

        self.filter_qc()

        # Get the function call for the specified method
        methods = {
            "argo": self.apply_xing2012_quenching_correction
        }
        if self.method.lower() not in methods.keys():
            raise KeyError(f"Method {self.method} is not supported")
        method_function = methods[self.method.lower()]

        # Subset the data
        # TODO: Remove time, lat and long from the subsetting by getting the first values of each for each profile
        # TODO: and then passing this to the solar_angle function. This will significantly improve processing speed.
        data_subset = self.data[
            list({self.mld_settings["threshold_on"],
                  "PROFILE_NUMBER",
                  "TIME",
                  "LATITUDE",
                  "LONGITUDE",
                  "DEPTH",
                  "PRES",
                  self.apply_to})
        ]

        # Apply the checks across individual profiles
        profile_numbers = np.unique(data_subset["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS"))
        for profile_number in tqdm(profile_numbers, colour="green", desc='\033[97mProgress\033[0m', unit="profile"):

            # Subset the data
            profile = data_subset.where(data_subset["PROFILE_NUMBER"] == profile_number, drop=True)

            corrected_chla = method_function(profile)

            # Stitch back into the full data
            profile_indices = np.where(self.data["PROFILE_NUMBER"] == profile_number)
            self.data[self.apply_to][profile_indices] = corrected_chla

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def calculate_mld(self, profile):

        for k, v in self.mld_settings.items():
            setattr(self, k, v)

        # Only look at values that are below the reference depth
        profile_subset = profile.where(
            profile["DEPTH"] <= self.reference_depth,
            drop=True
        ).dropna(dim="N_MEASUREMENTS", subset=["DEPTH", self.threshold_on])

        # Check there is still data to work with
        if len(profile_subset["DEPTH"]) == 0:
            return np.nan

        # Find the reference point and return nan if it cant be found near the reference depth
        reference_point = profile_subset[
            ["DEPTH", self.threshold_on]
        ].isel({"N_MEASUREMENTS": 0})
        if reference_point["DEPTH"] < 2 * self.reference_depth:
            return np.nan

        # Find the difference from the reference value along the profile
        reference_value = reference_point[self.threshold_on]
        profile_subset["delta"] = profile_subset[self.threshold_on] - reference_value

        # Filter out below-threshold points, then select the first (to pass the threshold)
        profile_subset = profile_subset.where(
            np.abs(profile_subset["delta"]) >= np.abs(self.threshold),
            drop=True
        )

        # Return the value if found. Otherwise nan.
        mld_value = np.nan
        if len(profile_subset["DEPTH"]) != 0:
            mld_value = float(profile_subset.isel({"N_MEASUREMENTS": 0})["DEPTH"])
        return mld_value

    def apply_xing2012_quenching_correction(self, profile):
        """
        Apply non-photochemical quenching (NPQ) correction following
        Xing et al. (2012, *JGR–Oceans*, 117:C01019).

        The maximum fluorescence within the mixed-layer depth (MLD)
        is taken as the non-quenched reference. All shallower
        (PRES < z_qd) values are adjusted upward to that maximum.
        Correction is only applied when solar elevation > 0°.

        Parameters
        ----------
        chlf : array-like of shape (N,)
            Uncorrected chlorophyll fluorescence profile F_Chl(PRES).
        pres : array-like of shape (N,)
            Pressure (dbar), increasing with depth.
        mld : float
            Mixed-layer depth (m or dbar).
        sun_angle : float
            Solar elevation angle (degrees). NPQ correction is applied
            only if `sun_angle > 0`.

        Returns
        -------
        chl_corr : ndarray of shape (N,)
            NPQ-corrected fluorescence profile.
        npq : ndarray of shape (N,)
            NPQ index = (chl_corr − chlf) / chlf.
        z_qd : float
            Quenching depth (dbar): pressure of maximum fluorescence
            within the MLD. NaN if not computable or if night-time.

        Notes
        -----
        • No correction is applied if solar elevation ≤ 0° (nighttime).
        • Shallower than z_qd → fluorescence set to Fmax (non-quenched reference).
        • Below MLD → unchanged.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        pres = np.asarray(profile["PRES"].values, dtype=float)
        N = len(chlf)

        # --- Calculate the MLD for this profile
        mld = self.calculate_mld(profile)

        # --- Night-time or invalid inputs: skip correction
        time_utc = pd.to_datetime(profile["TIME"][0].values).tz_localize("UTC")
        solar_position = pvlib.solarposition.get_solarposition(
            time_utc,
            profile["LATITUDE"][0].values,
            profile["LONGITUDE"][0].values
        )
        sun_angle = solar_position["elevation"].values
        if (
                sun_angle <= 0
                or N == 0
                or len(pres) != N
                or not np.isfinite(mld)
                or mld >= 0
                or np.all(np.isnan(chlf))
        ):
            return chlf

        # --- Ensure increasing pressure
        if pres[0] > pres[-1]:
            pres = pres[::-1]
            chlf = chlf[::-1]

        # --- Identify max F_Chl within MLD
        within_mld = pres <= mld
        if not np.any(within_mld):
            return chlf

        chlf_mld = np.where(within_mld, chlf, np.nan)
        if np.all(np.isnan(chlf_mld)):
            return chlf

        idx_max = np.nanargmax(chlf_mld)
        z_qd = float(pres[idx_max])
        F_max = chlf[idx_max]

        # --- Apply correction: flatten shallower than z_qd
        chl_corr = np.copy(chlf)
        chl_corr[pres <= z_qd] = F_max

        return chl_corr