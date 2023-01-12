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
Ramsey XY Experiment class with pulsed Stark tone drive.
"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from qiskit import pulse, circuit
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options, BackendTiming
from qiskit_experiments.library.characterization.analysis import RamseyXYAnalysis


class StarkRamseyXY(BaseExperiment):
    """Ramsey XY experiment with pulsed Stark tone.

    # section: overview

        This experiment is a variant of :class:`.RamseyXY` with the pulsed Stark tone
        and consists of following two circuits:

        .. parsed-literal::

            (Ramsey X)  The pulse before measurement rotates by pi-half around the X axis

                     ┌────┐┌────────┐┌───┐┌───────────────┐┌────────┐┌────┐┌─┐
                  q: ┤ √X ├┤ StarkV ├┤ X ├┤ StarkU(delay) ├┤ Rz(-π) ├┤ √X ├┤M├
                     └────┘└────────┘└───┘└───────────────┘└────────┘└────┘└╥┘
                c: 1/═══════════════════════════════════════════════════════╩═
                                                                            0

            (Ramsey Y) The pulse before measurement rotates by pi-half around the Y axis

                     ┌────┐┌────────┐┌───┐┌───────────────┐┌───────────┐┌────┐┌─┐
                  q: ┤ √X ├┤ StarkV ├┤ X ├┤ StarkU(delay) ├┤ Rz(-3π/2) ├┤ √X ├┤M├
                     └────┘└────────┘└───┘└───────────────┘└───────────┘└────┘└╥┘
                c: 1/══════════════════════════════════════════════════════════╩═
                                                                               0

        In principle, the sequence is a variant of :class:`RamseyXY` circuit,
        however, the delay in between √X gates is replaced with the off-resonant drive.
        This off-resonant drive causes the Stark shift on the qubit that modulates
        qubit frequency under the drive, which accumulates phase during the
        Ramsey sequence against the shifted frequency.
        This frequency shift is a function of the offset of the Stark tone from the
        qubit frequency, and magnitude of the tone.

        Note that the Stark tone takes the form of the flat-topped Gaussian envelope,
        in which the magnitude of pulse varies in time in rising and falling edges,
        resulting in the time-dependent Stark shift.
        This is hardly characterized by the Ramsey experiment, and thus an additional pulse
        involving only edges (StarkV) is added in front of the primary Stark drive (StarkU).
        The sign of phase accumulation is inverted with the X gate in the middle to
        cancel out the unwanted phase accumulation in the pulse edges.

        This technique allows the experiment to only accumulate the net phase
        during the flat-top part of the StarkU pulse with the constant magnitude.

    # section: note

        In the lower power limit, the Stark shift has a quadratic dependency on the
        pulse amplitude. This implies the resulting Stark shift is not sensitive to
        the sign of the amplitude of StarkV and StarkU pulses on one hand.
        On the other hand, the sign of the shift depends on the sign of frequency offset.
        In other words, positive (negative) Stark shift occurs when the Stark tone frequency
        is lower (higher) than the qubit frequency.

        In this experiment, the amplitude of Stark tone, i.e. `stark_amp` parameter, is
        abstracted to give positive (positive) Stark shift with positive (negative) stark_amp
        for better handling of the experiment parameters.
        The StarkV and StarkU pulses are always positive regardless of the sign of stark_amp,
        while the sign of frequency offset is implicitly determined by its sign.

    # section: analysis_ref
        :py:class:`RamseyXYAnalysis`

    # section: see_also
        qiskit_experiments.library.characterization.ramsey_xy.RamseyXY

    """

    def __init__(
        self,
        qubit: int,
        stark_amp: float,
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """Create new experiment.

        .. note::

            .. _stark_amplitude:

            The `stark_channel` in the experiment options may be necessary supplied.
            By default, the Stark tone is applied to the same channel with the qubit drive
            with frequency modulation. This frequency modulation might update the
            channel frame, which accumulates unwanted phase against the frequency difference
            between the qubit drive and Stark tone frequency in addition to
            the qubit Stark shift. You can use a dedicated Stark drive channel if available,
            otherwise you may want to reuse a control channel associated with the qubit.
            This depends on the architecture of your quantum device.

        Args:
            qubit: Index of qubit.
            stark_amp: A single float parameter to represent the magnitude of Stark tone
                and the sign of expected Stark shift. See :ref:`stark_amplitude` for details.
            backend: Optional, the backend to run the experiment on.
            experiment_options: Extra experiment options. See self.experiment_options.
        """
        self._timing = None

        super().__init__(qubits=[qubit], analysis=RamseyXYAnalysis(), backend=backend)
        self.set_experiment_options(stark_amp=stark_amp, **experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            stark_amp (float): A single float parameter to represent the magnitude of Stark tone
                and the sign of expected Stark shift. See :ref:`stark_amplitude` for details.
            stark_channel (PulseChannel): Pulse channel to apply Stark tones.
                If not provided, the same channel with the qubit drive is used.
            stark_freq_offset (float): Offset of Stark tone frequency from the qubit frequency.
                This must be greater than zero not to apply Rabi drive.
            stark_sigma (float): Gaussian sigma of the rising and falling edges
                of the Stark tone, in seconds.
            stark_risefall (float): Ratio of sigma to the duration of
                the rising and falling edges of the Stark tone.
            min_freq (float): Minimum frequency that this experiment is guaranteed to resolve.
                Note that fitter algorithm :class:`.RamseyXYAnalysis` of this experiment
                is still capable of fit experiment data with further lower frequency.
            max_freq (float): Maximum frequency that this experiment can resolve.
            delays (list[float]): The list of delays in set that will be scanned in the
                experiment. If not set, then evenly spaced delays with interval
                computed from ``min_freq`` and ``max_freq`` are used.
                If ``delays`` is set, these parameters are ignored.
        """
        options = super()._default_experiment_options()
        options.update_options(
            stark_amp=0.0,
            stark_channel=None,
            stark_freq_offset=80e6,
            stark_sigma=15e-9,
            stark_risefall=2,
            min_freq=5e6,
            max_freq=100e6,
            delays=None,
        )
        options.set_validator("stark_freq_offset", (0, np.inf))
        options.set_validator("stark_channel", pulse.channels.PulseChannel)
        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)
        self._timing = BackendTiming(backend)

    def delays(self) -> np.ndarray:
        """Delay values to use in circuits.

        .. note::

            The delays are computed with min_freq and max_freq experiment options.
            The maximum point is computed from the min_freq to guarantee the result
            contains at least one Ramsey oscillation cycle at this frequency.
            The interval is computed from the max_freq to sample at frequency
            higher than the Nyquist frequency of this frequency.

        Returns:
            The list of delays to use for the different circuits based on the
            experiment options.
        """
        opt = self.experiment_options  # alias

        if opt.delays is None:
            # Delay is longer enough to capture 1 cycle of the minmum frequency.
            # Fitter can still accurately fit samples shorter than 1 cycle.
            max_period = 1 / opt.min_freq
            # Inverse of interval should be greater than Nyquist frequency.
            sampling_freq = 2 * opt.max_freq
            interval = 1 / sampling_freq
            return np.arange(0, max_period, interval)
        return opt.delays

    def parameterized_circuits(self) -> Tuple[circuit.QuantumCircuit, circuit.QuantumCircuit]:
        """Create circuits with parameters for Ramsey XY experiment with Stark tone.

        Returns:
            Quantum template circuits for Ramsey X and Ramsey Y experiment.
        """
        opt = self.experiment_options  # alias
        param = circuit.Parameter("delay")

        # Pulse gates
        stark_v = circuit.Gate("StarkV", 1, [])
        stark_u = circuit.Gate("StarkU", 1, [param])

        # Note that Stark tone yields negative (positive) frequency shift
        # when the Stark tone frequency is higher (lower) than qubit f01 frequency.
        # This choice gives positive frequency shift with positive Stark amplitude.
        qubit_f01 = self._backend_data.drive_freqs[self.physical_qubits[0]]
        stark_freq = qubit_f01 - np.sign(opt.stark_amp) * opt.stark_freq_offset
        stark_amp = np.abs(opt.stark_amp)
        stark_channel = opt.stark_channel or pulse.DriveChannel(self.physical_qubits[0])
        ramps_dt = self._timing.round_pulse(time=2 * opt.stark_risefall * opt.stark_sigma)
        sigma_dt = ramps_dt / 2 / opt.stark_risefall

        with pulse.build() as stark_v_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.Gaussian(
                    duration=ramps_dt,
                    amp=stark_amp,
                    sigma=sigma_dt,
                ),
                stark_channel,
            )

        with pulse.build() as stark_u_schedule:
            pulse.set_frequency(stark_freq, stark_channel)
            pulse.play(
                pulse.GaussianSquare(
                    duration=ramps_dt + param,
                    amp=stark_amp,
                    sigma=sigma_dt,
                    risefall_sigma_ratio=opt.stark_risefall,
                ),
                stark_channel,
            )

        ram_x = circuit.QuantumCircuit(1, 1)
        ram_x.sx(0)
        ram_x.append(stark_v, [0])
        ram_x.x(0)
        ram_x.append(stark_u, [0])
        ram_x.rz(-np.pi, 0)
        ram_x.sx(0)
        ram_x.measure(0, 0)
        ram_x.metadata = {"series": "X"}
        ram_x.add_calibration(
            gate=stark_v,
            qubits=self.physical_qubits,
            schedule=stark_v_schedule,
        )
        ram_x.add_calibration(
            gate=stark_u,
            qubits=self.physical_qubits,
            schedule=stark_u_schedule,
        )

        ram_y = circuit.QuantumCircuit(1, 1)
        ram_y.sx(0)
        ram_y.append(stark_v, [0])
        ram_y.x(0)
        ram_y.append(stark_u, [0])
        ram_y.rz(-np.pi * 3 / 2, 0)
        ram_y.sx(0)
        ram_y.measure(0, 0)
        ram_y.metadata = {"series": "Y"}
        ram_y.add_calibration(
            gate=stark_v,
            qubits=self.physical_qubits,
            schedule=stark_v_schedule,
        )
        ram_y.add_calibration(
            gate=stark_u,
            qubits=self.physical_qubits,
            schedule=stark_u_schedule,
        )

        return ram_x, ram_y

    def circuits(self) -> List[circuit.QuantumCircuit]:
        """Create circuits.

        Returns:
            A list of circuits with a variable delay.
        """
        ramx_circ, ramy_circ = self.parameterized_circuits()
        param = next(iter(ramx_circ.parameters))

        circs = []
        dt = self._backend_data.dt
        granularity = self._backend_data.granularity
        for delay in self.delays():
            # Not using pulse_round method of the BackendTiming class
            # because this method considers the minimum pulse duration.
            # Valid delay here corresponds to the flat-top length and thus can be zero at minimum.
            valid_delay_dt = granularity * int(round(delay / dt / granularity))
            net_delay_sec = valid_delay_dt * dt

            ramx_circ_assigned = ramx_circ.assign_parameters({param: valid_delay_dt}, inplace=False)
            ramx_circ_assigned.metadata["xval"] = net_delay_sec

            ramy_circ_assigned = ramy_circ.assign_parameters({param: valid_delay_dt}, inplace=False)
            ramy_circ_assigned.metadata["xval"] = net_delay_sec

            circs.extend([ramx_circ_assigned, ramy_circ_assigned])

        return circs

    def _metadata(self) -> Dict[str, any]:
        """Return experiment metadata for ExperimentData."""
        return {
            "stark_amp": self.experiment_options.stark_amp,
            "stark_freq_offset": self.experiment_options.stark_freq_offset,
        }
