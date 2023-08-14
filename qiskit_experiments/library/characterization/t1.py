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
T1 Experiment class.
"""

from typing import List, Tuple, Dict, Optional, Union, Sequence

import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate, Parameter, ParameterExpression
from qiskit.providers.backend import Backend
from qiskit.utils import optionals as _optional

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t1_analysis import (
    T1Analysis,
    StarkP1SpectAnalysis,
)

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class T1(BaseExperiment):
    r"""An experiment to measure the qubit relaxation time.

    # section: overview

        This experiment estimates the :math:`T_1` relaxation time of the qubit by
        generating a series of circuits that excite the qubit then wait for different
        intervals before measurement. The resulting data of excited population versus
        wait time is fitted to an exponential curve to obtain an estimate for
        :math:`T_1`.

    # section: example

        Here's a minimal working example:

        .. jupyter-execute::
            :hide-code:

            from qiskit_experiments.test.noisy_delay_aer_simulator import NoisyDelayAerBackend

            t1 = 25e-6
            backend = NoisyDelayAerBackend([t1], [t1 / 2])

        .. jupyter-execute::

            from qiskit_experiments.library import T1
            import numpy as np

            exp = T1(physical_qubits = (0,),
                     delays = np.arange(1e-6, 30e-5, 3e-5))

            exp_data = exp.run(backend).block_for_results()
            exp_data.figure(0)

        The experiment executes a series of circuits with this form:

        .. jupyter-execute::

            exp.circuits()[0].draw("mpl")

    # section: analysis_ref
        :class:`.T1Analysis`

    # section: manual
        :doc:`/manuals/characterization/t1`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
        """
        options = super()._default_experiment_options()
        options.delays = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T1 experiment class.

        Args:
            physical_qubits: a single-element sequence containing the qubit whose T1 is to be
                estimated.
            delays: Delay times of the experiments in seconds.
            backend: Optional, the backend to run the experiment on.

        Raises:
            ValueError: If the number of delays is smaller than 3
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=T1Analysis(), backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        timing = BackendTiming(self.backend)

        circuits = []
        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(timing.round_delay(time=delay), 0, timing.delay_unit)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "unit": "s",
            }
            circ.metadata["xval"] = timing.delay_time(time=delay)

            circuits.append(circ)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata


class StarkP1Spectroscopy(BaseExperiment):
    """P1 spectroscopy experiment with Stark tone.

    # section: overview

        This experiment measures a probability of the excitation state of the qubit
        with a certain delay after excitation.
        A Stark tone is applied during this delay to move the
        qubit frequency to conduct a spectroscopy of qubit relaxation quantity.

        .. parsed-literal::

                 ┌───┐┌──────────────────┐┌─┐
              q: ┤ X ├┤ Stark(stark_amp) ├┤M├
                 └───┘└──────────────────┘└╥┘
            c: 1/══════════════════════════╩═
                                           0

        Since the qubit relaxation rate may depend on the qubit frequency due to the
        coupling to nearby energy levels, this experiment is useful to find out
        lossy operation frequency that might be harmful to the gate fidelity [1].

    # section: analysis_ref
        :py:class:`.StarkP1SpectAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2105.15201

    # section: see_also
        :class:`qiskit_experiments.library.characterization.ramsey_xy.StarkRamseyXY`
        :class:`qiskit_experiments.library.characterization.ramsey_xy.StarkRamseyXYAmpScan`

    # section: manual
        :doc:`/manuals/characterization/stark_experiment`

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """
        Initialize the T1 experiment class.

        Args:
            physical_qubits: Sequence with the index of the physical qubit.
            backend: Optional, the backend to run the experiment on.
            experiment_options: Experiment options. See the class documentation or
                ``self._default_experiment_options`` for descriptions.
        """
        self._timing = None

        super().__init__(
            physical_qubits=physical_qubits,
            analysis=StarkP1SpectAnalysis(),
            backend=backend,
        )
        self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            t1_delay (float): The T1 delay time after excitation pulse. The delay must be
                sufficiently greater than the edge duration determined by the stark_sigma.
            stark_channel (PulseChannel): Pulse channel to apply Stark tones.
                If not provided, the same channel with the qubit drive is used.
            stark_freq_offset (float): Offset of Stark tone frequency from the qubit frequency.
                This must be greater than zero not to apply Rabi drive.
            stark_sigma (float): Gaussian sigma of the rising and falling edges
                of the Stark tone, in seconds.
            stark_risefall (float): Ratio of sigma to the duration of
                the rising and falling edges of the Stark tone.
            min_stark_amp (float): Minimum Stark tone amplitude.
            max_stark_amp (float): Maximum Stark tone amplitude.
            num_stark_amps (int): Number of Stark tone amplitudes to scan.
            spacing (str): A policy for the spacing to create an amplitude list from
                ``min_stark_amp`` to ``max_stark_amp``. Either ``linear`` or ``quadratic``
                must be specified.
            stark_amps (list[float]): The list of amplitude that will be scanned in the experiment.
                If not set, then ``num_stark_amps`` amplitudes spaced according to
                the ``spacing`` policy between ``min_stark_amp`` and ``max_stark_amp`` are used.
                If ``stark_amps`` is set, these parameters are ignored.
        """
        options = super()._default_experiment_options()
        options.update_options(
            t1_delay=20e-6,
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_risefall=2,
            min_stark_amp=-1,
            max_stark_amp=1,
            num_stark_amps=201,
            spacing="quadratic",
            stark_amps=None,
        )
        options.set_validator("spacing", ["linear", "quadratic"])
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def parameters(self) -> np.ndarray:
        """Stark tone amplitudes to use in circuits.

        Returns:
            The list of amplitudes to use for the different circuits based on the
            experiment options.
        """
        opt = self.experiment_options  # alias

        if opt.stark_amps is None:
            if opt.spacing == "linear":
                params = np.linspace(opt.min_stark_amp, opt.max_stark_amp, opt.num_stark_amps)
            elif opt.spacing == "quadratic":
                min_sqrt = np.sign(opt.min_stark_amp) * np.sqrt(np.abs(opt.min_stark_amp))
                max_sqrt = np.sign(opt.max_stark_amp) * np.sqrt(np.abs(opt.max_stark_amp))
                lin_params = np.linspace(min_sqrt, max_sqrt, opt.num_stark_amps)
                params = np.sign(lin_params) * lin_params**2
            else:
                raise ValueError(f"Spacing option {opt.spacing} is not valid.")
        else:
            params = np.asarray(opt.stark_amps, dtype=float)

        return params

    def parameterized_circuits(self) -> Tuple[QuantumCircuit, ...]:
        """Create circuits with parameters for P1 experiment with Stark shift.

        Returns:
            Quantum template circuit for P1 experiment.
        """
        opt = self.experiment_options  # alias
        param = Parameter("stark_amp")
        sym_param = param._symbol_expr

        # Pulse gates
        stark = Gate("Stark", 1, [param])

        # Note that Stark tone yields negative (positive) frequency shift
        # when the Stark tone frequency is higher (lower) than qubit f01 frequency.
        # This choice gives positive frequency shift with positive Stark amplitude.
        qubit_f01 = self._backend_data.drive_freqs[self.physical_qubits[0]]
        neg_sign_of_amp = ParameterExpression(
            symbol_map={param: sym_param},
            expr=-sym.sign(sym_param),
        )
        abs_of_amp = ParameterExpression(
            symbol_map={param: sym_param},
            expr=sym.Abs(sym_param),
        )
        stark_freq = qubit_f01 + neg_sign_of_amp * opt.stark_freq_offset
        stark_channel = opt.stark_channel or pulse.DriveChannel(self.physical_qubits[0])
        sigma_dt = opt.stark_sigma / self._backend_data.dt
        delay_dt = self._timing.round_pulse(time=opt.t1_delay)

        with pulse.build() as stark_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.GaussianSquare(
                    duration=delay_dt,
                    amp=abs_of_amp,
                    sigma=sigma_dt,
                    risefall_sigma_ratio=opt.stark_risefall,
                ),
                stark_channel,
            )

        temp_t1 = QuantumCircuit(1, 1)
        temp_t1.x(0)
        temp_t1.append(stark, [0])
        temp_t1.measure(0, 0)
        temp_t1.add_calibration(
            gate=stark,
            qubits=self.physical_qubits,
            schedule=stark_schedule,
        )

        return (temp_t1,)

    def circuits(self) -> List[QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of P1 circuits with a variable Stark tone amplitudes.
        """
        (t1_circ,) = self.parameterized_circuits()
        param = next(iter(t1_circ.parameters))

        circs = []
        for amp in self.parameters():
            t1_assigned = t1_circ.assign_parameters({param: amp}, inplace=False)
            t1_assigned.metadata = {"xval": amp}
            circs.append(t1_assigned)

        return circs

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        metadata = super()._metadata()
        metadata["stark_freq_offset"] = self.experiment_options.stark_freq_offset

        return metadata
