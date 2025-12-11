#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import matplotlib.pyplot as plt
import matplotlib as mpl

def check_config(self, expected_params):
    for param in expected_params:
        if not hasattr(self, param):
            raise KeyError(f"[{self.step_name}] '{param}' is missing from config")
        if "_name" in param:
            if self.param not in self.data.data_vars:
                raise KeyError(f"[{self.step_name}] {getattr(self, param)} could not be found in the data")

@register_step
class DeriveUncalibratedPhase(BaseStep, QCHandlingMixin):

    step_name = "Derive Uncalibrated Phase"

    def run(self):
        self.filter_qc()

        # Check blue_phase_name is present
        check_config(self, ("blue_phase_name",))

        # Check if the output already exists
        if "UNCAL_PHASE_DOXY" in self.data.data_vars:
            self.log_warn("UNCAL_PHASE_DOXY already exists in the data. Overwriting...")

        # Calculate Uncalibrated phase and specify what QC will be derived from
        qc_parents = [f"{self.blue_phase_name}_QC"]
        if hasattr(self, "red_phase_name"):
            check_config(self, ("red_phase_name",))
            self.data["UNCAL_PHASE_DOXY"] = self.data[self.blue_phase_name] - self.data[self.red_phase_name]
            qc_parents.append(f"{self.red_phase_name}_QC")
        else:
            self.data["UNCAL_PHASE_DOXY"] = self.data[self.blue_phase_name]

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"UNCAL_PHASE_DOXY_QC": qc_parents}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass

@register_step
class DeriveOptodeTemperature(BaseStep, QCHandlingMixin):

    step_name = "Derive Optode Temperature"

    def run(self):
        self.filter_qc()

        # Check the optode temperature voltage and calibration coefficients are present
        check_config(self, ("temp_voltage_name", "calib_coefficients"))

        # Check there are at least two coefficients for the polynomial. Fill in missing values.
        if len(self.calib_coefficients) < 2:
            raise ValueError(f"[{self.step_name}] At least two calibration coefficients are required.")
        coeffs = [0] * 6
        for i in range(len(self.calib_coefficients)):
            coeffs[i] = self.calib_coefficients[i]

        # Check if the output already exists
        if "TEMP_DOXY" in self.data.data_vars:
            self.log_warn("TEMP_DOXY already exists in the data. Overwriting...")

        # Calculate temp_doxy
        temp_doxy = 0
        for i, coeff in enumerate(coeffs):
            temp_doxy += (coeff[i] * self.data[self.temp_voltage_name]**i)
        self.data["TEMP_DOXY"] = temp_doxy

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"TEMP_DOXY_QC": [f"{self.temp_voltage_name}_QC"]}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass

@register_step
class PhasePressureCorrection(BaseStep, QCHandlingMixin):

    step_name = "Phase Pressure Correction"

    def run(self):
        self.filter_qc()

        # Check the optode pressure and correction coefficient are present and that UNCAL_PHASE_DOXY is in the data
        check_config(self, ("optode_pressure_name", "correction_coefficient"))
        if "UNCAL_PHASE_DOXY" not in self.data.data_vars:
            raise KeyError(f"[{self.step_name}] UNCAL_PHASE_DOXY required but is missing from the data")

        # Apply the correction
        self.data["UNCAL_PHASE_DOXY_PCORR"] = (
            self.data["UNCAL_PHASE_DOXY"] +
            0.001 * self.correction_coefficient * self.data[self.optode_pressure_name]
        )

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"UNCAL_PHASE_DOXY_PCORR_QC": [f"{self.optode_pressure_name}_QC", "UNCAL_PHASE_DOXY_QC"]}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass

@register_step
class DeriveCalibratedPhase(BaseStep, QCHandlingMixin):

    step_name = "Derive Calibrated Phase"

    def run(self):
        self.filter_qc()

        # Check the config satisfies requirements
        check_config(self, ("uncalibrated_phase_name", "calib_coefficients"))

        # Check there are at least two coefficients for the polynomial. Fill in missing values.
        if len(self.calib_coefficients) < 2:
            raise ValueError(f"[{self.step_name}] At least two calibration coefficients are required.")
        coeffs = [0] * 4
        for i in range(len(self.calib_coefficients)):
            coeffs[i] = self.calib_coefficients[i]

        # Check if the output already exists
        if "CAL_PHASE_DOXY" in self.data.data_vars:
            self.log_warn("CAL_PHASE_DOXY already exists in the data. Overwriting...")

        # Calculate cal_phase_doxy
        cal_phase_doxy = 0
        for i, coeff in enumerate(coeffs):
            cal_phase_doxy += (coeff[i] * self.data[self.uncalibrated_phase_name] ** i)
        self.data["CAL_PHASE_DOXY"] = cal_phase_doxy

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"CAL_PHASE_DOXY_QC": [f"{self.uncalibrated_phase_name}_QC"]}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass

@register_step
class DeriveOxygenConcentration(BaseStep, QCHandlingMixin):

    step_name = "Derive Oxygen Concentration"

    def func_poly(self):
        # Check the calibration matrix has the right shape
        if np.shape(self.calib_coefficient_matrix) != (5, 4):
            raise ValueError(
                f"[{self.step_name}] Calib coefficient matrix must be of shape (5, 4) for method 'poly'."
            )

        # Build the internal coefficient matrix
        coeffs_matrix = np.full((5, 4), 0)
        for i, row in enumerate(self.calib_coefficient_matrix):
            coeffs_matrix[i, :] = row

        # Apply the conversion
        poly_temp = np.array([self.data["temperature_name"].values**i for i in range(4)])[np.newaxis, :, :]
        molar_doxy = (
            (poly_temp * coeffs_matrix[:, :, np.newaxis]).sum(axis=1) *
            np.array([self.data["CAL_PHASE_DOXY"].values ** i for i in range(5)])
        ).sum(axis=0)

        return molar_doxy

    def func_SVU(self):
        pass

    def run(self):
        self.filter_qc()

        methods = {
            "poly": (self.func_poly, ("temperature_name", "calib_coefficient_matrix")),
            "SVU": (self.func_SVU, ("temperature_name", "calib_coefficient_matrix")),
        }

        # Check the specified method
        check_config(self, ("method",))
        if self.method not in methods.keys():
            raise ValueError(f"[{self.step_name}] Unknown method '{self.method}'")

        # Unpack the method args and functions
        func, args = methods[self.method]

        # Check the config satisfies requirements
        check_config(self, args)

        # Check if the output already exists
        if "MOLAR_DOXY" in self.data.data_vars:
            self.log_warn("MOLAR_DOXY already exists in the data. Overwriting...")

        self.data["MOLAR_DOXY"] = func()

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"MOLAR_DOXY_QC": ["CAL_PHASE_DOXY_QC", f"{self.temperature_name}_QC"]}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass