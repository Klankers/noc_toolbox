"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glidertools as gt
from scipy.interpolate import interp1d


@register_step
class BlankStep(BaseStep, QCHandlingMixin):

    step_name = "Blank Step"

    def run(self):
        """
        Example
        -------
        - name: "Find Profile Direction"
          parameters:
            apply_to: "BBP700"
          diagnostics: false

        Returns
        -------

        """
        self.filter_qc()

        # Get the required variables
        data_subset = self.data[
            ["TIME",
             "PROFILE_NUMBER",
             "DEPTH",
             "TEMP",
             "PRAC_SALINITY",
             self.apply_to]
        ]

        # Interp DEPTH, TEMP and PRAC_SALINITY
        data_subset = data_subset.interp(
            coords={
                "DEPTH": data_subset["TIME"],
                "TEMP": data_subset["TIME"],
                "PRAC_SALINITY": data_subset["TIME"]
            }
        )

        # Apply the correction
        bbp_corrected = gt.flo_functions.flo_bback_total(
            data_subset[self.apply_to],
            data_subset['TEMP'],
            data_subset['PRAC_SALINITY'],
            self.theta,
            700,
            self.xfactor)

        # Stitch back into the data
        self.data[self.apply_to][:] = bbp_corrected

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass
