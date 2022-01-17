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
"""
Correlated readout readout_error calibration analysis classes
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qiskit.result import CorrelatedReadoutMitigator
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options


class CorrelatedReadoutErrorAnalysis(BaseAnalysis):
    r"""
    Correlated readout error characterization analysis

    # section: overview

        This class generates the full assignment matrix :math:`A` characterizing the
        readout error for the given qubits from the experiment results.

        :math:`A` is a :math:`2^n\times 2^n` matrix :math:`A` such that :math:`A_{y,x}`
        is the probability to observe :math:`y` given the true outcome should be :math:`x`.

        In the experiment, for each :math:`x`a circuit is constructed whose expected
        outcome is :math:`x`. From the observed results on the circuit, the probability for
        each :math:`y` is determined, and :math:`A_{y,x}` is set accordingly.

        Returns
            * The `Correlated readout error mitigator <https://qiskit.org/documentation/stubs/qiskit.result.CorrelatedReadoutMitigator.html>`_ object (the assignment matrix can be accessed via its ``assignment_matrix()`` method).
            * (Optional) A figure of the assignment matrix.


    # section: reference
        .. ref_arxiv:: 1 2006.14044
    """

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            ax (AxesSubplot): Optional. A matplotlib axis object to draw.
        """
        options = super()._default_options()
        options.plot = False
        options.ax = None
        return options

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        data = experiment_data.data()
        qubits = experiment_data.metadata["physical_qubits"]
        labels = [datum["metadata"]["label"] for datum in data]
        matrix = self._generate_matrix(data, labels)
        result_mitigator = CorrelatedReadoutMitigator(matrix, qubits=qubits)
        analysis_results = [AnalysisResultData("Correlated Readout Mitigator", result_mitigator)]
        if self.options.plot:
            ax = options.get("ax", None)
            figures = [self._plot_calibration(matrix, labels, ax)]
        else:
            figures = None
        return analysis_results, figures

    def _generate_matrix(self, data, labels) -> np.array:
        list_size = len(labels)
        matrix = np.zeros([list_size, list_size], dtype=float)
        # matrix[i][j] is the probability of counting i for expected j
        for datum in data:
            expected_outcome = datum["metadata"]["label"]
            j = labels.index(expected_outcome)
            total_counts = sum(datum["counts"].values())
            for measured_outcome, count in datum["counts"].items():
                i = labels.index(measured_outcome)
                matrix[i][j] = count / total_counts
        return matrix

    def _plot_calibration(self, matrix, labels, ax=None) -> "matplotlib.figure.Figure":
        """
        Plot the calibration matrix (2D color grid plot).

        Args:
            matrix: calibration matrix to plot
            ax (matplotlib.axes): settings for the graph

        Returns:
            The generated plot of the calibration matrix

        Raises:
            QiskitError: if _cal_matrices was not set.

            ImportError: if matplotlib was not installed.

        """

        if ax is None:
            ax = get_non_gui_ax()
        figure = ax.get_figure()
        ax.matshow(matrix, cmap=plt.cm.binary, clim=[0, 1])
        ax.set_xlabel("Prepared State")
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Measured State")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        return figure
