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

from typing import List, Optional, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.exceptions import QiskitError
import qiskit.pulse as pulse
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.analysis.curve_fitting import curve_fit
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments import AnalysisResult
from qiskit_experiments import ExperimentData
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import SVD, AverageData
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SpectroscopyAnalysis(BaseAnalysis):
    """Class to analysis a spectroscopy experiment."""

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
        data_processor: Optional[callable] = None,
        meas_level: int = MeasLevel.KERNELED,
        meas_return: int = "single",
        amp_guess: Optional[float] = None,
        sigma_guesses: Optional[List[float]] = None,
        freq_guess: Optional[float] = None,
        offset_guess: Optional[float] = None,
        amp_bounds: Optional[Tuple[float, float]] = None,
        sigma_bounds: Optional[Tuple[float, float]] = None,
        freq_bounds: Optional[Tuple[float, float]] = None,
        offset_bounds: Optional[Tuple[float, float]] = None,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **kwargs,
    ) -> Tuple[AnalysisResult, None]:
        """
        Analyse a spectroscopy experiment by fitting the data to a Gaussian function.
        The fit function is:

        .. math::

            a * exp(-(x-x0)**2/(2*sigma**2)) + b

        Here, :math:`x` is the frequency. The analysis loops over the initial guesses
        of the width parameter :math:`sigma`.

        Args:
            experiment_data: The experiment data to analyze.
            data_processor: The data processor with which to process the data.
            meas_level: The measurement level of the experiment data.
            meas_return: Whether single-shot (the default) or average data is returned by the
                experiment.
            amp_guess: The amplitude of the Gaussian function, i.e. :math:`a`. If not
                provided, this will default to the maximum absolute value of the ydata.
            sigma_guesses: The guesses for the standard deviation of the Gaussian distribution.
                If it is not given this will default to an array of ten
                points linearly spaced between zero and the full width of the data.
            freq_guess: A guess for the frequency of the peak :math:`x0`. If not provided
                this guess will default to the location of the highest absolute data point.
            offset_guess: A guess for the magnitude :math:`b` offset of the fit function.
                If not provided, the initial guess defaults to the average of the ydata.
            amp_bounds: Bounds on the amplitude of the Gaussian function as a tuple of
                two floats. The default bounds are [0, 1.1*max(ydata)]
            sigma_bounds: Bounds on the standard deviation of the Gaussian function as a tuple
                of two floats. The default values are [0, frequency range].
            freq_bounds: Bounds on the center frequency as a tuple of two floats. The default
                values are 90% of the lower end of the frequency and 110% of the upper end of
                the frequency.
            offset_bounds: Bounds on the offset of the Gaussian function as a tuple of two floats.
                The default values are the minimum and maximum of the ydata.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.
            kwargs: Trailing unused function parameters.

        Returns:
            The analysis result with the estimated peak frequency.

        Raises:
            QiskitError:
                - If the measurement level is not supported.
                - If the fit fails.
        """

        # Pick a data processor.
        if data_processor is None:
            if meas_level == MeasLevel.CLASSIFIED:
                data_processor = DataProcessor("counts", [Probability("1")])
            elif meas_level == MeasLevel.KERNELED:
                if meas_return == "single":
                    data_processor = DataProcessor("memory", [AverageData(), SVD()])
                else:
                    data_processor = DataProcessor("memory", [SVD()])

                data_processor.train(experiment_data.data())
            else:
                raise QiskitError("Unsupported measurement level.")

        y_sigmas = np.array([data_processor(datum) for datum in experiment_data.data()])
        min_y, max_y = min(y_sigmas[:, 0]), max(y_sigmas[:, 0])
        ydata = (y_sigmas[:, 0] - min_y) / (max_y - min_y)
        sigmas = np.sqrt(y_sigmas[:, 1]) / (max_y - min_y)
        xdata = np.array([datum["metadata"]["xval"] for datum in experiment_data.data()])

        # Fitting will not work if any sigmas are exactly 0.
        if any(sigmas == 0.0):
            sigmas = None

        if not offset_guess:
            offset_guess = np.median(ydata)
        if not amp_guess:
            amp_guess = -1 if offset_guess > 0.5 else 1
        if not freq_guess:
            peak_idx = np.argmin(ydata) if offset_guess > 0.5 else np.argmax(ydata)
            freq_guess = xdata[peak_idx]
        if not sigma_guesses:
            sigma_guesses = np.linspace(1e-6, abs(xdata[-1] - xdata[0]), 20)
        if amp_bounds is None:
            amp_bounds = (-1, 1)
        if sigma_bounds is None:
            sigma_bounds = (0, abs(xdata[-1] - xdata[0]))
        if freq_bounds is None:
            dx = xdata[1] - xdata[0]
            freq_bounds = (xdata[0] - dx, xdata[-1] + dx)
        if offset_bounds is None:
            offset_bounds = (-2, 2)

        # Perform fit
        best_fit = None
        bounds = {"a": amp_bounds, "sigma": sigma_bounds, "freq": freq_bounds, "b": offset_bounds}

        def fit_fun(x, a, sigma, freq, b):
            return a * np.exp(-((x - freq) ** 2) / (2 * sigma ** 2)) + b

        for sigma_guess in sigma_guesses:
            init = {"a": amp_guess, "sigma": sigma_guess, "freq": freq_guess, "b": offset_guess}
            try:
                fit_result = curve_fit(fit_fun, xdata, ydata, init, sigmas, bounds)

                if not best_fit:
                    best_fit = fit_result
                else:
                    if fit_result["reduced_chisq"] < best_fit["reduced_chisq"]:
                        best_fit = fit_result

            except RuntimeError:
                pass

        if best_fit is None:
            raise QiskitError("Could not find a fit to the spectroscopy data.")

        best_fit["value"] = best_fit["popt"][2]
        best_fit["stderr"] = (best_fit["popt_err"][2],)
        best_fit["unit"] = experiment_data.data(0)["metadata"].get("unit", "Hz")
        best_fit["label"] = "Spectroscopy"
        best_fit["xdata"] = xdata
        best_fit["ydata"] = ydata
        best_fit["ydata_err"] = sigmas
        best_fit["quality"] = self._fit_quality(
            best_fit["popt"][0],
            best_fit["popt"][1],
            best_fit["popt"][2],
            best_fit["popt"][3],
            best_fit["reduced_chisq"],
            xdata,
            ydata,
            best_fit["popt_err"][1],
        )

        if plot and HAS_MATPLOTLIB:
            ax = plot_curve_fit(fit_fun, best_fit, ax=ax)
            ax = plot_scatter(xdata, ydata, ax=ax)
            self._format_plot(ax, best_fit)
            best_fit.plt = plt

        return best_fit, None

    @staticmethod
    def _fit_quality(
        fit_amp: float,
        fit_sigma: float,
        fit_freq: float,
        fit_offset: float,
        reduced_chisq: float,
        xdata: np.array,
        ydata: np.array,
        fit_sigma_err: Optional[float] = None,
    ) -> str:
        """
        Algorithmic criteria for whether the fit is good or bad.
        A good fit has:
            - a small reduced chi-squared,
            - the peak must be within the scanned frequency range,
            - a standard deviation that is not larger than the scanned frequency range and
              that is wider than the smallest frequency increment,
            - a signal-to-noise ratio, defined as the amplitude of the peak divided by the
              square root of the median y-value, greater than a threshold of two, and
            - a standard error on the sigma of the Gaussian that is smaller than the sigma.

        Args:
            fit_amp: Amplitude of the fitted peak.
            fit_freq: Frequency of the fit.
            fit_sigma: Standard deviation of the fitted Gaussian.
            fit_offset: Offset of the fit.
            reduced_chisq: Reduced chi-squared of the fit.
            xdata: x-values, i.e. the frequencies.
            ydata: y-values, i.e. the measured signal.
            fit_sigma_err: Errors on the standard deviation of the fit.

        Returns:
            computer_bad or computer_good if the fit passes or fails, respectively.
        """
        min_freq = xdata[0]
        max_freq = xdata[-1]
        freq_increment = xdata[1] - xdata[0]

        snr = abs(fit_amp) / np.sqrt(abs(np.median(ydata) - fit_offset))
        fit_width_ratio = fit_sigma / (max_freq - min_freq)

        # pylint: disable=too-many-boolean-expressions
        if (
            min_freq <= fit_freq <= max_freq
            and 1.5 * freq_increment < fit_sigma
            and fit_width_ratio < 0.25
            and reduced_chisq < 5
            and (fit_sigma_err is None or (fit_sigma_err < fit_sigma))
            and snr > 2
        ):
            return "computer_good"
        else:
            return "computer_bad"

    @classmethod
    def _format_plot(cls, ax, analysis_result):
        """Format curve fit plot."""
        ax.tick_params(labelsize=14)
        ax.set_xlabel(f"Frequency ({analysis_result['unit']})", fontsize=16)
        ax.set_ylabel("Signal [arb. unit.]", fontsize=16)
        ax.grid(True)


class Spectroscopy(BaseExperiment):
    """Class the runs spectroscopy by sweeping the qubit frequency.

    The circuits produced by spectroscopy correspond to a spectroscopy pulse-schedule
    embedded in a spectroscopy gate.

    .. parsed-literal::

                   ┌────────────┐ ░ ┌─┐
              q_0: ┤ Spec(freq) ├─░─┤M├
                   └────────────┘ ░ └╥┘
        measure: 1/══════════════════╩═
                                     0

    The pulse-schedule consists of a set frequency instruction followed by a GaussianSquare
    pulse. A list of circuits is generated, each with a different frequency "freq".

    """

    __analysis_class__ = SpectroscopyAnalysis

    # Supported units for spectroscopy.
    __units__ = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}

    # default run options
    __run_defaults__ = {"meas_level": MeasLevel.KERNELED, "meas_return": "single"}

    def __init__(
        self,
        qubit: int,
        frequencies: Union[List[float], np.array],
        unit: Optional[str] = "Hz",
        absolute: bool = True,
    ):
        """
        A spectroscopy experiment run by shifting the frequency of the qubit.
        The parameters of the GaussianSquare spectroscopy pulse are specified at run-time.
        The spectroscopy pulse has the following parameters:
        - amp: The amplitude of the pulse must be between 0 and 1, the default is 0.1.
        - duration: The duration of the spectroscopy pulse in samples, the default is 1000 samples.
        - sigma: The standard deviation of the pulse, the default is 5 x duration.
        - width: The width of the flat-top in the pulse, the default is 0, i.e. a Gaussian.

        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment.
            unit: The unit in which the user specifies the frequencies. Can be one
                of 'Hz', 'kHz', 'MHz', 'GHz'. Internally, all frequencies will be converted
                to 'Hz'.
            absolute: Boolean to specify if the frequencies are absolute or relative to the
                qubit frequency in the backend.

        Raises:
            ValueError: if there are less than three frequency shifts or if the unit is not known.

        """
        if len(frequencies) < 3:
            raise ValueError("Spectroscopy requires at least three frequencies.")

        if unit not in self.__units__:
            raise ValueError(f"Unsupported unit: {unit}.")

        self._frequencies = [freq * self.__units__[unit] for freq in frequencies]
        self._absolute = absolute

        super().__init__([qubit], circuit_options=("amp", "duration", "sigma", "width"))

    def circuits(self, backend: Optional["Backend"] = None, **circuit_options):
        """
        Create the circuit for the spectroscopy experiment. The circuits are based on a
        GaussianSquare pulse and a frequency_shift instruction encapsulated in a gate.

        Args:
            backend: A backend object.
            circuit_options: Key word arguments to run the circuits. The circuit options are
                - amp: The amplitude of the GaussianSquare pulse, defaults to 0.1.
                - duration: The duration of the GaussianSquare pulse, defaults to 1024 samples.
                - sigma: The standard deviation of the GaussianSquare pulse, defaults to one
                    fifth of the duration.
                - width: The width of the flat top in the GaussianSquare pulse, defaults to 0.

        Returns:
            circuits: The circuits that will run the spectroscopy experiment.

        Raises:
            QiskitError: If relative frequencies are used but no backend was given.
        """
        if not backend and not self._absolute:
            raise QiskitError("Cannot run spectroscopy relative to qubit without a backend.")

        amp = circuit_options.get("amp", 0.1)
        duration = circuit_options.get("duration", 1024)
        sigma = circuit_options.get("sigma", duration / 4)
        width = circuit_options.get("width", 0)

        drive = pulse.DriveChannel(self.physical_qubits[0])

        # Create a template circuit
        freq_param = Parameter("frequency")
        with pulse.build(name="spectroscopy") as sched:
            pulse.set_frequency(freq_param, drive)
            pulse.play(pulse.GaussianSquare(duration, amp, sigma, width), drive)

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
                "amplitude": amp,
                "duration": duration,
                "sigma": sigma,
                "width": width,
                "schedule": str(sched),
            }

            if not self._absolute:
                assigned_circ.metadata["center frequency"] = center_freq

            if backend:
                assigned_circ.metadata["dt"] = getattr(backend.configuration(), "dt", "n.a.")

            circs.append(assigned_circ)

        return circs
