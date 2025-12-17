import pandas as pd
import polars as pl
import pytest

from src.toolbox.steps.custom.write_report import WriteReportStep

def test_write_report_step_runs():
    pdf = pd.DataFrame({
        "TIME": pd.date_range("2025-12-12", periods=3, freq="H"),
        "DEPTH": [5, 10, 15],
    })

    params = {
        "report_type": "VOTO_COMPREHENSIVE",
        "depth_column": "DEPTH",
    }

    context = {
        "data": pdf
    }

    # --- Instantiate and run the step ---
    step = WriteReportStep(parameters=params, context=context)
    result_context = step.run()

    # --- Basic sanity checks ---
    assert "data" in result_context
    assert isinstance(step._df, pl.DataFrame)
    assert step._df.shape == (3, 2)
    assert step._df.columns == ["TIME", "DEPTH"]

    # Optional: check the step assigned attributes correctly
    assert step.report_type == "VOTO_COMPREHENSIVE"
    assert step.depth_col == "DEPTH"
