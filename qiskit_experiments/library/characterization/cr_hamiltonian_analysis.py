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

"""Cross resonance Hamiltonian tomography experiment analysis."""

from collections import defaultdict
from itertools import product
from typing import List, Union

import numpy as np
from qiskit.utils import apply_prefix

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.data_processing as dp
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import AnalysisResultData, FitVal


# pylint: disable=line-too-long
class CrossResonanceHamiltonianAnalysis(curve.CurveAnalysis):
    r"""A class to analyze cross resonance Hamiltonian tomography experiment.

    # section: fit_model
        The following equations are used to approximate the dynamics of
        the target qubit Bloch vector.

        .. math::

            \begin{align}
                F_{x, c}(t) &= \frac{1}{\Omega_c^2} \left(
                    - p_{z, c} p_{x, c} + p_{z, c} p_{x, c} \cos(\Omega_c t') +
                    \Omega_c p_{y, c} \sin(\Omega_c t') \right) + b \tag{1} \\
                F_{y, c}(t) &= \frac{1}{\Omega_c^2} \left(
                    p_{z, c} p_{y, c} - p_{z, c} p_{y, c} \cos(\Omega_c t') -
                    \Omega_c p_{x, c} \sin(\Omega_c t') \right) + b \tag{2} \\
                F_{z, c}(t) &= \frac{1}{\Omega_c^2} \left(
                    p_{z, c}^2 + (p_{x, c}^2 + p_{y, c}^2) \cos(\Omega_c t') \right) + b \tag{3}
            \end{align}

        where :math:`t' = t + t_{\rm offset}` with :math:`t` is pulse duration to scan
        and :math:`t_{\rm offset}` is an extra fit parameter that may represent the edge effect.
        The :math:`\Omega_c = \sqrt{p_{x, c}^2+p_{y, c}^2+p_{z, c}^2}` and
        :math:`p_{x, c}, p_{y, c}, p_{z, c}, b` are also fit parameters.
        The subscript :math:`c` represents the state of control qubit :math:`c \in \{0, 1\}`.
        The fit functions :math:`F_{x, c}, F_{y, c}, F_{z, c}` approximate the Pauli expectation
        values :math:`\langle \sigma_{x, c} (t) \rangle, \langle \sigma_{y, c} (t) \rangle,
        \langle \sigma_{z, c} (t) \rangle` of the target qubit, respectively.

        Based on the fit result, cross resonance Hamiltonian coefficients can be written as

        .. math::

            ZX &= \frac{p_{x, 0} - p_{x, 1}}{2} \\
            ZY &= \frac{p_{y, 0} - p_{y, 1}}{2} \\
            ZZ &= \frac{p_{z, 0} - p_{z, 1}}{2} \\
            IX &= \frac{p_{x, 0} + p_{x, 1}}{2} \\
            IY &= \frac{p_{y, 0} + p_{y, 1}}{2} \\
            IZ &= \frac{p_{z, 0} + p_{z, 1}}{2}

        In this analysis, the initial guess is generated by the following equations.

        .. math::

            p_x &= \omega \cos(\theta) \cos(\phi) \\
            p_y &= \omega \cos(\theta) \sin(\phi) \\
            p_z &= \omega \sin(\theta)

        where :math:`\omega` is the mean oscillation frequency of eigenvalues,
        :math:`\theta = \cos^{-1}\sqrt{\frac{\max F_z - \min F_z}{2}}`
        and :math:`\phi \in [-\pi, \pi]`.

    # section: fit_parameters

        defpar t_{\rm off}:
            desc: Offset to the pulse duration. For example, if pulse envelope is
                a flat-topped Gaussian, two Gaussian edges may become an offset duration.
            init_guess: Computed as :math:`N \sqrt{2 \pi} \sigma` where the :math:`N` is number of
                pulses and :math:`\sigma` is Gaussian sigma of riring and falling edges.
                Note that this implicitly assumes the :py:class:`~qiskit.pulse.library\
                .parametric_pulses.GaussianSquare` pulse envelope.
            bounds: [0, None]

        defpar p_{x, 0}:
            desc: Fit parameter of oscillations when control qubit state is 0.
            init_guess: See fit model section.
            bounds: None

        defpar p_{y, 0}:
            desc: Fit parameter of oscillations when control qubit state is 0.
            init_guess: See fit model section.
            bounds: None

        defpar p_{z, 0}:
            desc: Fit parameter of oscillations when control qubit state is 0.
            init_guess: See fit model section.
            bounds: None

        defpar p_{x, 1}:
            desc: Fit parameter of oscillations when control qubit state is 1.
            init_guess: See fit model section.
            bounds: None

        defpar p_{y, 1}:
            desc: Fit parameter of oscillations when control qubit state is 1.
            init_guess: See fit model section.
            bounds: None

        defpar p_{z, 1}:
            desc: Fit parameter of oscillations when control qubit state is 1.
            init_guess: See fit model section.
            bounds: None

        defpar b:
            desc: Vertical offset of oscillations. This may indicate the state preparation and
                measurement error.
            init_guess: 0
            bounds: None

    # section: see_also
        qiskit_experiments.library.characterization.cr_hamiltonian.CrossResonanceHamiltonian

    """

    __series__ = [
        curve.SeriesDef(
            name="x|c=0",
            fit_func=lambda x, t_off, px0, px1, py0, py1, pz0, pz1, b: curve.fit_function.bloch_oscillation_x(
                x + t_off, px=px0, py=py0, pz=pz0, baseline=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "x"},
            plot_color="blue",
            plot_symbol="o",
            canvas=0,
        ),
        curve.SeriesDef(
            name="y|c=0",
            fit_func=lambda x, t_off, px0, px1, py0, py1, pz0, pz1, b: curve.fit_function.bloch_oscillation_y(
                x + t_off, px=px0, py=py0, pz=pz0, baseline=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "y"},
            plot_color="blue",
            plot_symbol="o",
            canvas=1,
        ),
        curve.SeriesDef(
            name="z|c=0",
            fit_func=lambda x, t_off, px0, px1, py0, py1, pz0, pz1, b: curve.fit_function.bloch_oscillation_z(
                x + t_off, px=px0, py=py0, pz=pz0, baseline=b
            ),
            filter_kwargs={"control_state": 0, "meas_basis": "z"},
            plot_color="blue",
            plot_symbol="o",
            canvas=2,
        ),
        curve.SeriesDef(
            name="x|c=1",
            fit_func=lambda x, t_off, px0, px1, py0, py1, pz0, pz1, b: curve.fit_function.bloch_oscillation_x(
                x + t_off, px=px1, py=py1, pz=pz1, baseline=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "x"},
            plot_color="red",
            plot_symbol="^",
            canvas=0,
        ),
        curve.SeriesDef(
            name="y|c=1",
            fit_func=lambda x, t_off, px0, px1, py0, py1, pz0, pz1, b: curve.fit_function.bloch_oscillation_y(
                x + t_off, px=px1, py=py1, pz=pz1, baseline=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "y"},
            plot_color="red",
            plot_symbol="^",
            canvas=1,
        ),
        curve.SeriesDef(
            name="z|c=1",
            fit_func=lambda x, t_off, px0, px1, py0, py1, pz0, pz1, b: curve.fit_function.bloch_oscillation_z(
                x + t_off, px=px1, py=py1, pz=pz1, baseline=b
            ),
            filter_kwargs={"control_state": 1, "meas_basis": "z"},
            plot_color="red",
            plot_symbol="^",
            canvas=2,
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options."""
        default_options = super()._default_options()
        default_options.data_processor = dp.DataProcessor(
            input_key="counts",
            data_actions=[dp.Probability("1"), dp.BasisExpectationValue()],
        )
        default_options.curve_plotter = "mpl_multiv_canvas"
        default_options.xlabel = "Flat top width"
        default_options.ylabel = "<X(t)>,<Y(t)>,<Z(t)>"
        default_options.xval_unit = "dt"
        default_options.style = curve.visualization.PlotterStyle(
            figsize=(8, 10),
            legend_loc="lower right",
            fit_report_rpos=(0.28, -0.10),
        )
        default_options.ylim = (-1, 1)

        return default_options

    def _t_off_initial_guess(self) -> float:
        """Return initial guess for time offset.

        This method assumes the :py:class:`~qiskit.pulse.library.parametric_pulses.GaussianSquare`
        envelope with the Gaussian rising and falling edges with the parameter ``sigma``.

        This is intended to be overridden by a child class so that rest of the analysis class
        logic can be reused for the fitting that assumes other pulse envelopes.

        Returns:
            An initial guess for time offset parameter ``t_off`` in units of dt.
        """
        n_pulses = self._extra_metadata().get("n_cr_pulses", 1)
        sigma = self._experiment_options().get("sigma", 0)

        return np.sqrt(2 * np.pi) * sigma * n_pulses

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(t_off=(0, np.inf), b=(-1, 1))

        user_opt.p0.set_if_empty(t_off=self._t_off_initial_guess(), b=1e-9)

        guesses = defaultdict(list)
        for control in (0, 1):
            x_data = self._data(series_name=f"x|c={control}")
            y_data = self._data(series_name=f"y|c={control}")
            z_data = self._data(series_name=f"z|c={control}")

            omega_xyz = []
            for data in (x_data, y_data, z_data):
                ymin, ymax = np.percentile(data.y, [10, 90])
                if ymax - ymin < 0.2:
                    # oscillation amplitude might be almost zero,
                    # then exclude from average because of lower SNR
                    continue
                fft_freq = curve.guess.frequency(data.x, data.y)
                omega_xyz.append(fft_freq)
            if omega_xyz:
                omega = 2 * np.pi * np.average(omega_xyz)
            else:
                omega = 1e-3

            zmin, zmax = np.percentile(z_data.y, [10, 90])
            theta = np.arccos(np.sqrt((zmax - zmin) / 2))

            # The FFT might be up to 1/2 bin off
            df = 2 * np.pi / ((z_data.x[1] - z_data.x[0]) * len(z_data.x))
            for omega_shifted in [omega, omega - df / 2, omega + df / 2]:
                for phi in np.linspace(-np.pi, np.pi, 5):
                    px = omega_shifted * np.cos(theta) * np.cos(phi)
                    py = omega_shifted * np.cos(theta) * np.sin(phi)
                    pz = omega_shifted * np.sin(theta)
                    guesses[control].append(
                        {
                            f"px{control}": px,
                            f"py{control}": py,
                            f"pz{control}": pz,
                        }
                    )
            if omega < df:
                # empirical guess for low frequency case
                guesses[control].append(
                    {
                        f"px{control}": omega,
                        f"py{control}": omega,
                        f"pz{control}": 0,
                    }
                )

        fit_options = []
        # combine all guesses in Cartesian product
        for p0s, p1s in product(guesses[0], guesses[1]):
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(**p0s, **p1s)
            fit_options.append(new_opt)

        return fit_options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - If chi-squared value is less than 3.
        """
        if fit_data.reduced_chisq < 3:
            return "good"

        return "bad"

    def _extra_database_entry(self, fit_data: curve.FitData) -> List[AnalysisResultData]:
        """Calculate Hamiltonian coefficients from fit values."""
        extra_entries = []

        for control in ("z", "i"):
            for target in ("x", "y", "z"):
                p0_val = fit_data.fitval(f"p{target}0")
                p1_val = fit_data.fitval(f"p{target}1")

                if control == "z":
                    coef_val = 0.5 * (p0_val.value - p1_val.value) / (2 * np.pi)
                else:
                    coef_val = 0.5 * (p0_val.value + p1_val.value) / (2 * np.pi)

                coef_err = 0.5 * np.sqrt(p0_val.stderr ** 2 + p1_val.stderr ** 2) / (2 * np.pi)

                extra_entries.append(
                    AnalysisResultData(
                        name=f"omega_{control}{target}",
                        value=FitVal(value=coef_val, stderr=coef_err, unit="Hz"),
                        chisq=fit_data.reduced_chisq,
                        device_components=[Qubit(q) for q in self._physical_qubits],
                    )
                )

        return extra_entries
