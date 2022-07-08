# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bloch trajectory analysis of a single qubit."""

from typing import List, Union, Optional

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.data_processing as dp


class BlochTrajectoryAnalysis(curve.CurveAnalysis):
    r"""A class to analyze a trajectory of the Bloch vector of a single qubit.

    # section: fit_model

        The following equations are used to approximate the dynamics of the qubit Bloch vector.

        .. math::

            \begin{align}
                F_x(t) &= \frac{1}{\Omega^2} \left(
                    - p_z p_x + p_z p_x \cos(\Omega t') +
                    \Omega p_y \sin(\Omega t') \right) + b \tag{1} \\
                F_y(t) &= \frac{1}{\Omega^2} \left(
                    p_z p_y - p_z p_y \cos(\Omega t') -
                    \Omega p_x \sin(\Omega t') \right) + b \tag{2} \\
                F_z(t) &= \frac{1}{\Omega^2} \left(
                    p_z^2 + (p_x^2 + p_y^2) \cos(\Omega t') \right) + b \tag{3}
            \end{align}

        where :math:`t' = t + t_{\rm offset}` with :math:`t` is pulse duration to scan
        and :math:`t_{\rm offset}` is an extra fit parameter that may represent an edge effect.
        Note that this analysis assumes a microwave drive with the flat top Gaussian envelope.
        :math:`p_x, p_y, p_z, b` are fit parameters, and :math:`\Omega = \sqrt{p_x^2+p_y^2+p_z^2}`.
        The fit functions :math:`F_x, F_y, F_z` approximate the Pauli expectation
        values :math:`\langle \sigma_x (t) \rangle, \langle \sigma_y (t) \rangle,
        \langle \sigma_z (t) \rangle` of the target qubit, respectively.

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
                pulses and :math:`\sigma` is Gaussian sigma of rising and falling edges.
                Note that this implicitly assumes the :py:class:`~qiskit.pulse.library\
                .parametric_pulses.GaussianSquare` pulse envelope.
            bounds: [0, None]

        defpar p_x:
            desc: Fit parameter of oscillations of the X observable.
            init_guess: See fit model section.
            bounds: None

        defpar p_y:
            desc: Fit parameter of oscillations of the Y observable.
            init_guess: See fit model section.
            bounds: None

        defpar p_z:
            desc: Fit parameter of oscillations of the Z observable.
            init_guess: See fit model section.
            bounds: None

        defpar b:
            desc: Vertical offset of oscillations. This may indicate the state preparation and
                measurement error.
            init_guess: 0
            bounds: None

    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):

        eqr_temps = {
            "x": "(-pz * px + pz * px * cos(W * X) + W * py * sin(W * X)) / W**2 + b",
            "y": "(pz * py - pz * py * cos(W * X) - W * px * sin(W * X)) / W**2 + b",
            "z": "(pz**2 + (px**2 + py**2) * cos(W * X)) / W**2 + b",
        }

        models = []
        for axis, temp_eq in eqr_temps.items():
            eq = temp_eq
            eq = eq.replace("W", "sqrt(px**2 + py**2 + pz**2)")
            eq = eq.replace("X", "(x + t_off)")
            models.append(
                lmfit.models.ExpressionModel(
                    expr=eq,
                    name=f"{axis}",
                    data_sort_key={"meas_basis": axis},
                )
            )

        super().__init__(
            models=models,
            name=name,
        )

    @classmethod
    def _default_options(cls):
        """Return the default analysis options."""
        default_options = super()._default_options()
        default_options.data_processor = dp.DataProcessor(
            input_key="counts",
            data_actions=[dp.Probability("1"), dp.BasisExpectationValue()],
        )
        default_options.curve_drawer.set_options(
            xlabel="Flat top width",
            ylabel="Pauli expectation values",
            xval_unit="s",
            ylim=(-1, 1),
        )

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        fit_options = []

        user_opt.bounds.set_if_empty(t_off=(0, np.inf), b=(-1, 1))
        user_opt.p0.set_if_empty(b=1e-9)

        x_data = curve_data.get_subset_of("x")
        y_data = curve_data.get_subset_of("y")
        z_data = curve_data.get_subset_of("z")

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
                new_opt = user_opt.copy()
                new_opt.p0.set_if_empty(
                    px=omega_shifted * np.cos(theta) * np.cos(phi),
                    py=omega_shifted * np.cos(theta) * np.sin(phi),
                    pz=omega_shifted * np.sin(theta),
                )
                fit_options.append(new_opt)
        if omega < df:
            # empirical guess for low frequency case
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(px=omega, py=omega, pz=0)
            fit_options.append(new_opt)

        return fit_options
