# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Readout error mitigated tomography analysis
"""

from qiskit_experiments.framework import CompositeAnalysis
from qiskit_experiments.library.characterization import LocalReadoutErrorAnalysis
from .tomography_analysis import TomographyAnalysis
from .basis.pauli_basis import PauliMeasurementBasis


class MitigatedTomographyAnalysis(CompositeAnalysis):
    """Analysis for readout error mitigated tomography experiments.

    Analysis is performed as a :class:`.CompositeAnalysis` consisting
    of :class:`.LocalReadoutErrorAnalysis` to determine the local
    assignment matrices describing single qubit Z-basis readout errors,
    and then these matrices are used to automatically construct a noisy
    :class:`~.PauliMeasurementBasis` for use during tomographic
    fitting with the tomography analysis.
    """

    def __init__(self, roerror_analysis="default", tomography_analysis="default"):
        """Initialize mitigated tomography analysis"""
        if roerror_analysis == "default":
            roerror_analysis = LocalReadoutErrorAnalysis()
        if tomography_analysis == "default":
            tomography_analysis = TomographyAnalysis()
        super().__init__([roerror_analysis, tomography_analysis], flatten_results=True)

    @classmethod
    def _default_options(cls):
        """Default analysis options

        Analysis Options:
            fitter (str or Callable): The fitter function to use for reconstruction.
                This can  be a string to select one of the built-in fitters, or a callable to
                supply a custom fitter function. See the `Fitter Functions` section for
                additional information.
            fitter_options (dict): Any addition kwarg options to be supplied to the fitter
                function. For documentation of available kwargs refer to the fitter function
                documentation.
            rescale_positive (bool): If True rescale the state returned by the fitter
                to be positive-semidefinite. See the `PSD Rescaling` section for
                additional information (Default: True).
            rescale_trace (bool): If True rescale the state returned by the fitter
                have either trace 1 for :class:`~qiskit.quantum_info.DensityMatrix`,
                or trace dim for :class:`~qiskit.quantum_info.Choi` matrices (Default: True).
            measurement_qubits (Sequence[int]): Optional, the physical qubits with tomographic
                measurements. If not specified will be set to ``[0, ..., N-1]`` for N-qubit
                tomographic measurements.
            preparation_qubits (Sequence[int]): Optional, the physical qubits with tomographic
                preparations. If not specified will be set to ``[0, ..., N-1]`` for N-qubit
                tomographic preparations.
            unmitigated_fit (bool): If True also run tomography fit without readout error
                mitigation and include both mitigated and unmitigated analysis results. If
                False only compute mitigated results (Default: False)
            target (Any): Optional, target object for fidelity comparison of the fit
                (Default: None).
        """
        # Override options to be tomography options minus bases
        options = super()._default_options()
        options.fitter = "linear_inversion"
        options.fitter_options = {}
        options.rescale_positive = True
        options.rescale_trace = True
        options.measurement_qubits = None
        options.preparation_qubits = None
        options.unmitigated_fit = False
        options.target = None
        return options

    def set_options(self, **fields):
        # filter fields
        self_fields = {key: val for key, val in fields.items() if hasattr(self.options, key)}
        super().set_options(**self_fields)
        tomo_fields = {
            key: val for key, val in fields.items() if hasattr(self._analyses[1].options, key)
        }
        self._analyses[1].set_options(**tomo_fields)

    def _run_analysis(self, experiment_data):
        # Return list of experiment data containers for each component experiment
        # containing the marginalized data from the composite experiment
        roerror_analysis, tomo_analysis = self._analyses
        roerror_data, tomo_data = experiment_data.create_child_data().child_data()

        # Run readout error analysis
        roerror_analysis.run(roerror_data, replace_results=True).block_for_results()

        # Construct noisy measurement basis
        mitigator = roerror_data.analysis_results("Local Readout Mitigator").value

        # Run mitigated tomography analysis with noisy mitigated basis
        # Tomo analysis instance is internally copied by setting option with run.
        tomo_analysis.run(
            tomo_data,
            replace_results=True,
            measurement_basis=PauliMeasurementBasis(mitigator=mitigator),
            extra={"mitigated": True},
        ).block_for_results()

        # Combine results so that tomography results are ordered first
        combined_data = [tomo_data, roerror_data]

        # Run unmitigated tomography analysis
        if self.options.unmitigated_fit:
            nomit_data = tomo_analysis.run(
                tomo_data,
                replace_results=False,
                measurement_basis=PauliMeasurementBasis(),
                extra={"mitigated": False},
            ).block_for_results()
            combined_data.append(nomit_data)

        if self._flatten_results:
            return self._combine_results(combined_data)

        return [], []
