# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A FakeExperiment for testing."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from qiskit import QuantumCircuit
from qiskit_experiments.framework import (
    BaseExperiment,
    BaseAnalysis,
    Options,
    AnalysisResultData,
    ArtifactData,
)
from qiskit_experiments.curve_analysis import ScatterTable, CurveFitResult


class FakeAnalysis(BaseAnalysis):
    """
    Dummy analysis class for test purposes only.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def _run_analysis(self, experiment_data):
        seed = self.options.get("seed", None)
        rng = np.random.default_rng(seed=seed)
        analysis_results = [
            AnalysisResultData(f"result_{i}", value) for i, value in enumerate(rng.random(3))
        ]
        scatter_table = ScatterTable.from_dataframe(pd.DataFrame(columns=ScatterTable.COLUMNS))
        fit_data = CurveFitResult(
            method="some_method",
            model_repr={"s1": "par0 * x + par1"},
            success=True,
            params={"par0": rng.random(), "par1": rng.random()},
            var_names=["par0", "par1"],
            covar=rng.random((2, 2)),
            reduced_chisq=rng.random(),
        )
        analysis_results.append(ArtifactData(name="curve_data", data=scatter_table))
        analysis_results.append(ArtifactData(name="fit_summary", data=fit_data))
        figures = None
        add_figures = self.options.get("add_figures", False)
        if add_figures:
            figures = [MatplotlibFigure()]
        return analysis_results, figures


class FakeExperiment(BaseExperiment):
    """Fake experiment class for testing."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.dummyoption = None
        return options

    def __init__(self, physical_qubits=None, backend=None, analysis=None, experiment_type=None):
        """Initialise the fake experiment."""
        if physical_qubits is None:
            physical_qubits = [0]
        if analysis is None:
            analysis = FakeAnalysis()
        super().__init__(
            physical_qubits,
            analysis=analysis,
            backend=backend,
            experiment_type=experiment_type,
        )

    def circuits(self):
        """Fake circuits."""
        return [QuantumCircuit(len(self.physical_qubits))]
