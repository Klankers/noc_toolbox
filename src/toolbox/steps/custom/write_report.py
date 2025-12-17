from toolbox.steps.base_step import BaseStep, register_step
import polars as pl

def print_junk(df, step_name, report_type):
        print(f"This step {step_name} is running.")

        print(df)

        print(report_type)

@register_step  #   Makes the step discoverable
class WriteReportStep(BaseStep):
    step_name = "Write report"

    def run(self):

        self.report_type = self.parameters["report_type"]
        self.depth_col = self.parameters["depth_column"]
        self.data = self.context["data"].copy()

        self._df = pl.from_pandas(
            self.data[["TIME", self.depth_col]].to_dataframe(), nan_to_null=False
        )

        print_junk(self._df, self.step_name, self.report_type)

        return self.context
    
    def generate_diagnostics(self):
        pass
