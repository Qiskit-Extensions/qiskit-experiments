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

"""Spectroscopy experiment class."""

from typing import List, Dict, Any, Union, Optional

import numpy as np
import qiskit.pulse as pulse
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend, Options
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    get_opt_value,
    get_opt_error,
)
from qiskit_experiments.autodocs import (
    OptionsField,
    CurveFitParameter,
    standard_experiment_documentation,
    standard_options_documentation,
    curve_analysis_documentation,
)
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor


@curve_analysis_documentation
class SpectroscopyAnalysis(CurveAnalysis):
    """Spectroscopy analysis."""

    __doc_overview__ = """
This analysis uses a Gaussian function to find a peak.

Note that this analysis assumes only a single peak.
If multiple peaks exist, you will get a poor reduced chi squared value.
"""

    __doc_equations__ = [r"F(x) = a \exp(-(x-f)^2/(2\sigma^2)) + b"]

    __doc_fit_params__ = [
        CurveFitParameter(
            name="a",
            description="Base line.",
            initial_guess="The maximum signal value with removed baseline.",
            bounds="[-2, 2] scaled with maximum signal value.",
        ),
        CurveFitParameter(
            name="b",
            description="Peak height.",
            initial_guess="A median value of the signal.",
            bounds="[-1, 1] scaled with maximum signal value.",
        ),
        CurveFitParameter(
            name="f",
            description="Center frequency. This is the fit parameter of main interest.",
            initial_guess="A frequency value at the peak (maximum signal).",
            bounds="[min(x), max(x)] of frequency scan range.",
        ),
        CurveFitParameter(
            name=r"\sigma",
            description="Standard deviation of Gaussian function.",
            initial_guess=r"Calculated from FWHM of peak :math:`w` such that "
            r":math:`w / \sqrt{8} \ln{2}`.",
            bounds=r"[0, :math:`\Delta x`] where :math:`\Delta x` represents frequency scan range.",
        ),
    ]

    __series__ = [
        SeriesDef(
            fit_func=lambda x, a, sigma, freq, b: fit_function.gaussian(
                x, amp=a, sigma=sigma, x0=freq, baseline=b
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls) -> Union[Options, Dict[str, OptionsField]]:
        """Return default options."""
        default_options = super()._default_options()
        default_options["p0"].default = {"a": None, "sigma": None, "freq": None, "b": None}
        default_options["bounds"].default = {"a": None, "sigma": None, "freq": None, "b": None}
        default_options["fit_reports"].default = {"freq": "frequency"}

        default_options["normalization"] = OptionsField(
            default=True,
            annotation=bool,
            description="Set ``True`` to normalize measurement data. Usually applied to "
            "Kerneled (level1) measurement data.",
        )

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        curve_data = self._data()

        b_guess = np.median(curve_data.y)
        peak_idx = np.argmax(np.abs(curve_data.y - b_guess))
        f_guess = curve_data.x[peak_idx]
        a_guess = curve_data.y[peak_idx] - b_guess

        # calculate sigma from FWHM
        halfmax = curve_data.x[np.abs(curve_data.y - b_guess) > np.abs(a_guess / 2)]
        fullwidth = max(halfmax) - min(halfmax)
        s_guess = fullwidth / np.sqrt(8 * np.log(2))

        max_abs_y = np.max(np.abs(curve_data.y))

        fit_option = {
            "p0": {
                "a": user_p0["a"] or a_guess,
                "sigma": user_p0["sigma"] or s_guess,
                "freq": user_p0["freq"] or f_guess,
                "b": user_p0["b"] or b_guess,
            },
            "bounds": {
                "a": user_bounds["a"] or (-2 * max_abs_y, 2 * max_abs_y),
                "sigma": user_bounds["sigma"] or (0.0, max(curve_data.x) - min(curve_data.x)),
                "freq": user_bounds["freq"] or (min(curve_data.x), max(curve_data.x)),
                "b": user_bounds["b"] or (-max_abs_y, max_abs_y),
            },
        }
        fit_option.update(options)

        return fit_option

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared less than 3,
            - a peak within the scanned frequency range,
            - a standard deviation that is not larger than the scanned frequency range,
            - a standard deviation that is wider than the smallest frequency increment,
            - a signal-to-noise ratio, defined as the amplitude of the peak divided by the
              square root of the median y-value less the fit offset, greater than a
              threshold of two, and
            - a standard error on the sigma of the Gaussian that is smaller than the sigma.
        """
        curve_data = self._data()

        max_freq = np.max(curve_data.x)
        min_freq = np.min(curve_data.x)
        freq_increment = np.mean(np.diff(curve_data.x))

        fit_a = get_opt_value(analysis_result, "a")
        fit_b = get_opt_value(analysis_result, "b")
        fit_freq = get_opt_value(analysis_result, "freq")
        fit_sigma = get_opt_value(analysis_result, "sigma")
        fit_sigma_err = get_opt_error(analysis_result, "sigma")

        snr = abs(fit_a) / np.sqrt(abs(np.median(curve_data.y) - fit_b))
        fit_width_ratio = fit_sigma / (max_freq - min_freq)

        criteria = [
            min_freq <= fit_freq <= max_freq,
            1.5 * freq_increment < fit_sigma,
            fit_width_ratio < 0.25,
            analysis_result["reduced_chisq"] < 3,
            (fit_sigma_err is None or fit_sigma_err < fit_sigma),
            snr > 2,
        ]

        if all(criteria):
            analysis_result["quality"] = "computer_good"
        else:
            analysis_result["quality"] = "computer_bad"

        return analysis_result


@standard_experiment_documentation
@standard_options_documentation
class QubitSpectroscopy(BaseExperiment):
    """Class that runs spectroscopy by sweeping the qubit frequency."""

    __doc_overview__ = """
The circuits produced by spectroscopy, i.e.
    
.. parsed-literal::

               ┌────────────┐ ░ ┌─┐
          q_0: ┤ Spec(freq) ├─░─┤M├
               └────────────┘ ░ └╥┘
    measure: 1/══════════════════╩═
                                 0

have a spectroscopy pulse-schedule embedded in a spectroscopy gate. The
pulse-schedule consists of a set frequency instruction followed by a GaussianSquare
pulse. A list of circuits is generated, each with a different frequency "freq".

A spectroscopy experiment run by setting the frequency of the qubit drive.
The parameters of the GaussianSquare spectroscopy pulse can be specified at run-time.
"""

    __analysis_class__ = SpectroscopyAnalysis

    # Supported units for spectroscopy.
    __units__ = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @classmethod
    def _default_experiment_options(cls) -> Union[Options, Dict[str, OptionsField]]:
        """Default option values used for the spectroscopy pulse."""
        return {
            "amp": OptionsField(
                default=0.1,
                annotation=float,
                description="Amplitude of spectroscopy pulse. Usually weak power pulse is used to "
                "suppress broadening of observed peaks.",
            ),
            "duration": OptionsField(
                default=1024,
                annotation=int,
                description="Duration of spectroscopy pulse. This may need to satisfy the "
                "hardware waveform memory constraint. "
                "The default value is represented in units of dt.",
            ),
            "sigma": OptionsField(
                default=256,
                annotation=Union[int, float],
                description="Sigma of Gaussian rising and falling edges. This value should be "
                "sufficiently smaller than the duration, "
                "otherwise waveform is distorted. "
                "The default value is represented in units of dt.",
            ),
            "width": OptionsField(
                default=0,
                annotation=Union[int, float],
                description="Width of the flat-top part of the Gaussian square envelope of "
                "spectroscopy pulse. Set width=0 to use Gaussian pulse.",
            ),
        }

    def __init__(
        self,
        qubit: int,
        frequencies: Union[List[float], np.array],
        unit: Optional[str] = "Hz",
        absolute: bool = True,
    ):
        """Create new experiment.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment.
            unit: The unit in which the user specifies the frequencies. Can be one
                of 'Hz', 'kHz', 'MHz', 'GHz'. Internally, all frequencies will be converted
                to 'Hz'.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            QiskitError: if there are less than three frequency shifts or if the unit is not known.

        """
        if len(frequencies) < 3:
            raise QiskitError("Spectroscopy requires at least three frequencies.")

        if unit not in self.__units__:
            raise QiskitError(f"Unsupported unit: {unit}.")

        super().__init__([qubit])

        self._frequencies = [freq * self.__units__[unit] for freq in frequencies]
        self._absolute = absolute
        self.set_analysis_options(xlabel=f"Frequency [{unit}]", ylabel="Signal [arb. unit]")

    def circuits(self, backend: Optional[Backend] = None):
        """Create the circuit for the spectroscopy experiment.

        The circuits are based on a GaussianSquare pulse and a frequency_shift instruction
        encapsulated in a gate.

        Args:
            backend: A backend object.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.

        Raises:
            QiskitError:
                - If relative frequencies are used but no backend was given.
                - If the backend configuration does not define dt.
        """
        # TODO this is temporarily logic. Need update of circuit data and processor logic.
        self.set_analysis_options(
            data_processor=get_to_signal_processor(
                meas_level=self.run_options.meas_level,
                meas_return=self.run_options.meas_return,
            )
        )

        if not backend and not self._absolute:
            raise QiskitError("Cannot run spectroscopy relative to qubit without a backend.")

        # Create a template circuit
        freq_param = Parameter("frequency")
        with pulse.build(backend=backend, name="spectroscopy") as sched:
            pulse.set_frequency(freq_param, pulse.DriveChannel(self.physical_qubits[0]))
            pulse.play(
                pulse.GaussianSquare(
                    duration=self.experiment_options.duration,
                    amp=self.experiment_options.amp,
                    sigma=self.experiment_options.sigma,
                    width=self.experiment_options.width,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )

        gate = Gate(name="Spec", num_qubits=1, params=[freq_param])

        circuit = QuantumCircuit(1)
        circuit.append(gate, (0,))
        circuit.add_calibration(gate, (self.physical_qubits[0],), sched, params=[freq_param])
        circuit.measure_active()

        if not self._absolute:
            center_freq = backend.defaults().qubit_freq_est[self.physical_qubits[0]]

        # Create the circuits to run
        circs = []
        for freq in self._frequencies:
            if not self._absolute:
                freq += center_freq

            assigned_circ = circuit.assign_parameters({freq_param: freq}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": freq,
                "unit": "Hz",
                "amplitude": self.experiment_options.amp,
                "duration": self.experiment_options.duration,
                "sigma": self.experiment_options.sigma,
                "width": self.experiment_options.width,
                "schedule": str(sched),
            }

            if not self._absolute:
                assigned_circ.metadata["center frequency"] = center_freq

            try:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt")
            except AttributeError as no_dt:
                raise QiskitError("Dt parameter is missing in backend configuration") from no_dt

            circs.append(assigned_circ)

        return circs
