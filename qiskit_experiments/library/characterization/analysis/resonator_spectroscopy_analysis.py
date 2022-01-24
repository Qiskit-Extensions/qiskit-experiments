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

"""Spectroscopy analysis class for resonators."""

from typing import List, Tuple
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import ResonanceAnalysis
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.data_processing.processor_library import ProjectorType


class ResonatorSpectroscopyAnalysis(ResonanceAnalysis):
    """Class to analysis resonator spectroscopy."""

    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.dimensionality_reduction = ProjectorType.ABS
        options.result_parameters = [curve.ParameterRepr("freq", "meas_freq", "Hz")]
        options.plot_iq_data = True
        return options

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:
        """Wrap the analysis to optionally plot the IQ data."""
        analysis_results, figures = super()._run_analysis(experiment_data)

        if self.options.plot_iq_data:
            axis = get_non_gui_ax()
            figure = axis.get_figure()
            figure.set_size_inches(*self.options.style.figsize)

            iqs = []

            for datum in experiment_data.data():
                if "memory" in datum:
                    mem = np.array(datum["memory"])

                    # Average single-shot data.
                    if len(mem.shape) == 3:
                        iqs.append(np.average(mem.reshape(mem.shape[0], mem.shape[2]), axis=0))

            if len(iqs) > 0:
                iqs = np.array(iqs)
                axis.scatter(iqs[:, 0], iqs[:, 1], color="b")
                axis.set_xlabel(
                    "In phase [arb. units]", fontsize=self.options.style.axis_label_size
                )
                axis.set_ylabel(
                    "Quadrature [arb. units]", fontsize=self.options.style.axis_label_size
                )
                axis.tick_params(labelsize=self.options.style.tick_label_size)
                axis.grid(True)

                figures.append(figure)

        return analysis_results, figures
