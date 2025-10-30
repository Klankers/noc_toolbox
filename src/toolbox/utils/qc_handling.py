import numpy as np
import xarray as xr


class QCHandlingMixin:

    def __init__(self):
        # fetch user inputs
        qc_settings = self.parameters.get('qc_handling_settings') or {}
        self.filter_settings = qc_settings.get("flag_filter_settings") or {}
        self.behaviour = qc_settings.get("reconstruction_behaviour") or "reinsert"

        self.flag_mapping = {
            flag: flag for flag in list(range(10))
        }
        if user_mappings := qc_settings.get("flag_mapping"):
            self.flag_mapping.update(user_mappings)

        # Make a copy of the data for reference
        self.data = self.context["data"]
        self.data_copy = self.data.copy()

        # Check that the variables are present for filter execusion
        missing_variables = []
        for var in self.filter_settings:
            if var not in self.data or f"{var}_QC" not in self.data:
                log(f"One or both of {var}/{var}_QC are missing from the dataset. They will be skipped.")
                missing_variables.append(var)
        for missing in missing_variables:
            self.filter_settings.pop(missing)

        # Continue method resolution order
        super().__init__()

    def print_qc_settings(self):
        self.log(
            "\n--------------------\n"
            f"Filter settings: {self.filter_settings}\n"
            f"Reconstruction behaviour: {self.behaviour}\n"
            f"Flag mappings: {self.flag_mapping}\n"
            "--------------------"
        )


    def filter_qc(self):
        for var, flags_to_nan in self.filter_settings.items():

            # find all positions where bad flags are present
            mask = ~self.data[f"{var}_QC"].isin(flags_to_nan)

            # nan-out the bad flagged data
            self.data[var] = self.data[var].where(mask, np.nan)


    def reconstruct_data(self):
        if self.behaviour == "replace":
            pass

        elif self.behaviour == "reinsert":
            for var in self.filter_settings.keys():

                # Find all of the postitions where there was bad data
                mask = self.data[f"{var}_QC"].isin(flags_to_nan)

                # Where there was a bad flag, reinsert the original values back into the data
                self.data[var] = xr.where(mask, self.data_copy[var], self.data[var])

        else:
            raise KeyError(f"Behaviour '{self.behaviour}' is not recgnised.")


    def update_qc(self):
        for var in self.filter_settings.keys():
            # Find all values that have changed during processing
            mask = (self.data[var] == self.data_copy[var])

            # Make a refference table for all possible flag updates
            updated_flags = self.data[f"{var}_QC"].map(self.flag_mapping.get)

            # Where data has changed, replace the old flag with the updated flag
            self.data[f"{var}_QC"] = xr.where(mask, self.data_copy[f"{var}_QC"], updated_flags)


    def generate_qc(self, generation_protocols: dict):
        # TODO: Make a combinatrix for updating QC


        pass