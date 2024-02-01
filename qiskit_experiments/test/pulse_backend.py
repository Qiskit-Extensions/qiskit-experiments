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

"""A Pulse simulation backend based on Qiskit-Dynamics"""

import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library.standard_gates import RZGate, SXGate, XGate, CXGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameter import Parameter
from qiskit.providers import BackendV2, QubitProperties
from qiskit.providers.options import Options
from qiskit import pulse
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms import block_to_schedule
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.result import Result, Counts
from qiskit.transpiler import InstructionProperties, Target
from qiskit.utils.deprecation import deprecate_arg

from qiskit_experiments.warnings import HAS_DYNAMICS
from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.test.utils import FakeJob

if HAS_DYNAMICS._is_available():
    from qiskit_dynamics import Solver
    from qiskit_dynamics.models import LindbladModel
    from qiskit_dynamics.pulse import InstructionToSignals


@HAS_DYNAMICS.require_in_instance
class PulseBackend(BackendV2):
    r"""Abstract base class for pulse simulation backends in Qiskit Experiments.

    This backend is designed for the tests in Qiskit Experiments as well as for the
    tutorials in Qiskit Experiments. The backend has a Qiskit Dynamics pulse simulator
    which allows it to simulate pulse schedules that are included in the calibrations
    attached to quantum circuits. In addition, sub-classes should implement a set of default
    schedules so that circuits that do not provide calibrations can also run, much like
    the hardware backends. In addition, the backends are also capable of simulating level-
    one (IQ data) and level-two (counts) data. Subclasses of these backends can have an
    optional discriminator so that they can produce counts based on sampled IQ data. If
    a discriminator is not provided then the counts will be produced from a statevector
    or density matrix.

    .. warning::

        Some of the functionality in this backend may move to Qiskit Dynamics and/or be
        refactored. These backends are not intended as a general pulse-simulator backend
        but rather to test the experiments and write short tutorials to demonstrate an
        experiment without having to run on hardware.
    """

    @deprecate_arg(
        name="static_hamiltonian",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="hamiltonian_operators",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="static_dissipators",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    def __init__(
        self,
        static_hamiltonian: Optional[np.ndarray] = None,
        hamiltonian_operators: Optional[np.ndarray] = None,
        static_dissipators: Optional[np.ndarray] = None,
        solver: Optional[Solver] = None,
        dt: float = 0.1 * 1e-9,
        solver_method="RK23",
        seed: int = 0,
        atol: float = None,
        rtol: float = None,
        **kwargs,
    ):
        """Initialize a backend with model information.

        Args:
            static_hamiltonian: Time-independent term in the Hamiltonian.
            hamiltonian_operators: List of time-dependent operators.
            static_dissipators: Constant dissipation operators. Defaults to None.
            dt: Sample rate for simulating pulse schedules. Defaults to 0.1*1e-9.
            solver_method: Numerical solver method to use. Check qiskit_dynamics for available
                methods. Defaults to "RK23".
            seed: An optional seed given to the random number generator. If this argument is not
                set then the seed defaults to 0.
            atol: Absolute tolerance during solving.
            rtol: Relative tolerance during solving.
        """
        super().__init__(
            None,
            name="PulseBackendV2",
            description="A PulseBackend simulator",
            online_date=datetime.datetime.now(datetime.timezone.utc),
            backend_version="0.0.1",
        )

        # subclasses must implement default pulse schedules
        self._defaults = None

        self._target = Target(dt=dt, granularity=16)

        # The RNG to sample IQ data.
        self._rng = np.random.default_rng(seed)

        # The instance to convert pulse schedules to signals for Qiskit Dynamics.
        self.converter = None

        self.solver_method = solver_method

        self.solve_kwargs = {}
        if atol:
            self.solve_kwargs["atol"] = atol
        if rtol:
            self.solve_kwargs["rtol"] = rtol

        if static_hamiltonian is not None and hamiltonian_operators is not None:
            # TODO deprecate soon
            solver = Solver(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                static_dissipators=static_dissipators,
                **kwargs,
            )
        self.solver = solver

        self.model_dim = self.solver.model.dim
        self.subsystem_dims = (self.model_dim,)

        if isinstance(self.solver.model, LindbladModel):
            self.y_0 = np.eye(self.model_dim**2)
            self.ground_state = np.array([1.0] + [0.0] * (self.model_dim**2 - 1))
        else:
            self.y_0 = np.eye(self.model_dim)
            self.ground_state = np.array([1.0] + [0.0] * (self.model_dim - 1))

        self._simulated_pulse_unitaries = {}

        # An internal cache of schedules to unitaries. The key is a hashed string representation.
        self._schedule_cache = {}

        # An optional discriminator that is used to create counts from IQ data.
        self._discriminator = None

    # pylint: disable=unused-argument
    @staticmethod
    def rz_gate(qubits, theta):
        """Initialize RZ gate."""
        return None

    @property
    def target(self):
        """Contains information for circuit transpilation."""
        return self._target

    @property
    def max_circuits(self):
        return None

    def defaults(self):
        """return backend pulse defaults."""
        return self._defaults

    # pylint: disable=unused-argument
    def control_channel(self, qubits: List[int]):
        return []

    @property
    def discriminator(self) -> List[BaseDiscriminator]:
        """Return the discriminators for the IQ data."""
        return self._discriminator

    @discriminator.setter
    def discriminator(self, discriminator: List[BaseDiscriminator]):
        """Set the discriminator."""
        self._discriminator = discriminator

    @classmethod
    def _default_options(cls) -> Options:
        """Returns the default options of the backend."""
        return Options(
            shots=4000,
            meas_level=MeasLevel.CLASSIFIED,
            meas_return=MeasReturnType.AVERAGE,
            memory=False,
        )

    @property
    def default_pulse_unitaries(self) -> Dict[Tuple, np.array]:
        """Return the default unitary matrices of the backend."""
        return self._simulated_pulse_unitaries

    @default_pulse_unitaries.setter
    def default_pulse_unitaries(self, unitaries: Dict[Tuple, np.array]):
        """Set the default unitary pulses this allows the tests to simulate the pulses only once."""
        self._simulated_pulse_unitaries = unitaries

    @staticmethod
    def pulse_command(qubit: int, amp: float, beta: float):
        """Utility function to create pulse instructions"""
        with pulse.build() as pulse_gate:
            pulse.play(
                pulse.library.Drag(duration=160, amp=amp, sigma=40, beta=beta, name=f"Xp_d{qubit}"),
                pulse.DriveChannel(qubit),
            )
        return pulse_gate

    @staticmethod
    def _get_info(
        circuit: QuantumCircuit, instruction: CircuitInstruction
    ) -> Tuple[Tuple[int], Tuple[float], str]:
        """Returns information that uniquely describes a circuit instruction.

        Args:
            circuit: The quantum circuit in which the instruction is located. This is needed to
                access the register in the circuit.
            instruction: A gate or operation in a QuantumCircuit.

        Returns:
            Tuple of qubit index, gate parameters and instruction name.
        """
        p_dict = instruction.operation
        qubit = tuple(int(circuit.qregs[0].index(qbit)) for qbit in instruction.qubits)
        params = tuple(float(val) for val in p_dict.params)
        return qubit, params, p_dict.name

    def _iq_data(
        self,
        state: Union[Statevector, DensityMatrix],
        meas_qubits: List,
        shots: int,
        centers: List[Tuple[float, float]],
        width: float,
        phase: Optional[float] = None,
    ) -> Tuple[List, List]:
        """Generates IQ data for each physical level.

        Args:
            state: Quantum state operator.
            shots: Number of shots.
            centers: The central I and Q points for each level.
            width: Width of IQ data distribution.
            phase: Phase of IQ data, by default 0. Defaults to None.

        Returns:
            (I,Q) data as List[shot index][qubit index] = [I,Q].

        Raises:
            QiskitError: If number of centers and levels don't match.
        """
        full_i, full_q = [], []
        for sub_idx in meas_qubits:
            probability = state.probabilities(qargs=[sub_idx])
            counts_n = self._rng.multinomial(shots, probability / sum(probability), size=1).T

            sub_i, sub_q = [], []
            if len(counts_n) != len(centers):
                raise QiskitError(
                    f"Number of centers ({len(centers)}) not equal to number of levels ({len(counts_n)})"
                )

            for idx, count_i in enumerate(counts_n):
                sub_i.append(self._rng.normal(loc=centers[idx][0], scale=width, size=count_i))
                sub_q.append(self._rng.normal(loc=centers[idx][1], scale=width, size=count_i))

            sub_i = list(chain.from_iterable(sub_i))
            sub_q = list(chain.from_iterable(sub_q))

            if phase is not None:
                complex_iq = (sub_i + 1.0j * sub_q) * np.exp(1.0j * phase)
                sub_i, sub_q = complex_iq.real, complex_iq.imag

            full_i.append(sub_i)
            full_q.append(sub_q)
        full_iq = np.array([full_i, full_q]).T
        return full_iq.tolist()

    # pylint: disable=unused-argument
    def _iq_cluster_centers(self, circuit: Optional[QuantumCircuit] = None) -> List[Tuple[float]]:
        """A function to provide the points for the IQ centers when doing readout.

        Subclasses can override this function, for instance, to provide circuit dependent
        IQ cluster centers. If this function is not overridden then the IQ cluster centers returned
        are evenly distributed on the unit sphere in the IQ plane with |0> located at IQ point (0, 1).

        Args:
            circuit: provided so that sub-classes that implement their own IQ simulation
                by overriding this method can access circuit-level data (e.g. for
                ReadoutSpectroscopy simulation).

        Returns:
            Coordinates for IQ centers.
        """
        theta = 2 * np.pi / self.subsystem_dims[0]
        return [(np.cos(idx * theta), np.sin(idx * theta)) for idx in range(self.subsystem_dims[0])]

    def _state_to_measurement_data(
        self,
        state: np.ndarray,
        shots: int,
        meas_level: MeasLevel,
        meas_return: MeasReturnType,
        memory: bool,
        circuit: QuantumCircuit,
        meas_qubits: List,
    ) -> Tuple[Union[Union[Dict, Counts, Tuple[List, List]], Any], Optional[Any]]:
        """Convert State objects to IQ data or Counts.

        The counts are produced by sampling from the state vector if no discriminator is
        present. Otherwise, IQ shots are generated and then discriminated based on the
        discriminator.

        Args:
            state: Quantum state information.
            shots: Number of repetitions of each circuit, for sampling.
            meas_level: Measurement level 1 returns IQ data. 2 returns counts.
            meas_return: "single" returns information from every shot. "avg" returns average
                measurement output (averaged over number of shots).
            circuit: The circuit is provided so that :meth:`iq_data` can leverage any circuit-level
                information that it might need to generate the IQ shots.

        Returns:
            Measurement output as either counts or IQ data depending on the run input.

        Raises:
            QiskitError: If unsuported measurement options are provided.
        """
        memory_data = None
        if isinstance(self.solver.model, LindbladModel):
            state = state.reshape(self.model_dim, self.model_dim)
            state = DensityMatrix(state / np.trace(state), self.subsystem_dims)
        else:
            state = Statevector(state / np.linalg.norm(state), self.subsystem_dims)

        if meas_level == MeasLevel.CLASSIFIED:
            if self._discriminator is None:
                if memory:
                    memory_data = state.sample_memory(shots, qargs=meas_qubits)
                    measurement_data = dict(zip(*np.unique(memory_data, return_counts=True)))
                else:
                    measurement_data = state.sample_counts(shots, qargs=meas_qubits)
            else:
                centers = self._iq_cluster_centers(circuit=circuit)
                iq_data = np.array(self._iq_data(state, meas_qubits, shots, centers, 0.2))
                memory_data = [
                    self._discriminator[qubit_idx].predict(iq_data[:, idx])
                    for idx, qubit_idx in enumerate(meas_qubits)
                ]
                memory_data = ["".join(state_label) for state_label in zip(*memory_data[::-1])]
                measurement_data = dict(zip(*np.unique(memory_data, return_counts=True)))

        elif meas_level == MeasLevel.KERNELED:
            centers = self._iq_cluster_centers(circuit=circuit)
            measurement_data = self._iq_data(state, meas_qubits, shots, centers, 0.2)
            if meas_return == "avg":
                measurement_data = np.average(np.array(measurement_data), axis=0)
        else:
            raise QiskitError(f"Unsupported measurement level {meas_level}.")

        return measurement_data, memory_data

    def solve(self, schedule: Union[ScheduleBlock, Schedule]) -> np.ndarray:
        """Solves for qubit dynamics under the action of a pulse instruction

        Args:
            schedule: Pulse signal.

        Returns:
            Time-evolution unitary operator.
        """

        if isinstance(schedule, ScheduleBlock):
            schedule = block_to_schedule(schedule)

        signal = self.converter.get_signals(schedule)
        time_f = schedule.duration * self.dt
        unitary = self.solver.solve(
            t_span=[0.0, time_f],
            y0=self.y_0,
            t_eval=[time_f],
            signals=signal,
            method=self.solver_method,
            **self.solve_kwargs,
        ).y[0]

        return unitary

    def run(self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **run_options) -> FakeJob:
        """Run method takes circuits as input and returns FakeJob with IQ data or counts.

        Args:
            run_input: Circuits to run.
            run_options: Any option that affects the way that the circuits are run. The options
                that are currently supported are ``shots``, ``meas_level``, ``meas_return``,
                and ``memory``.

        Returns:
            FakeJob with simulation data.

        Raises:
            QiskitError: If unsuported operations are performed.
        """
        self.options.update_options(**run_options)
        shots = self.options.get("shots", self._options.shots)
        meas_level = self.options.get("meas_level", self._options.meas_level)
        meas_return = self.options.get("meas_return", self._options.meas_return)
        memory = self.options.get("memory", self._options.memory)

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": self.backend_version,
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        if isinstance(run_input, QuantumCircuit):
            run_input = [run_input]

        for circuit in run_input:
            unitaries = {}
            meas_qubits = []
            # 1. Parse the calibrations and simulate any new schedule. Add U to the unitaries.
            for name, schedule in circuit.calibrations.items():
                for (qubits, params), schedule_block in schedule.items():
                    schedule_key = hash(repr(schedule_block))

                    # Simulate the schedule if not in the cache.
                    if schedule_key not in self._schedule_cache:
                        self._schedule_cache[schedule_key] = self.solve(schedule_block)

                    unitaries[(name, qubits, params)] = self._schedule_cache[schedule_key]

            # 2. Copy over any remaining instructions to the dict of unitaries.
            for key, unitary in self.default_pulse_unitaries.items():
                if key not in unitaries:
                    unitaries[key] = unitary

            # 3. Multiply the unitaries of the circuit instructions onto the ground state.
            state_t = self.ground_state.copy()
            for instruction in circuit.data:
                qubits, params, inst_name = self._get_info(circuit, instruction)
                if inst_name == "barrier":
                    continue
                if inst_name == "measure":
                    meas_qubits += [qubits[0]]
                    continue
                if inst_name == "rz":
                    unitary = self.rz_gate(qubits, params[0])
                else:
                    unitary = unitaries[(inst_name, qubits, params)]
                state_t = unitary @ state_t

            # 4. Convert the probabilities to IQ data or counts.
            measurement_data, memory_data = self._state_to_measurement_data(
                state_t, shots, meas_level, meas_return, memory, circuit, meas_qubits
            )

            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circuit.metadata},
                "meas_level": meas_level,
                "meas_return": meas_return,
                "data": {},
            }

            if meas_level == MeasLevel.CLASSIFIED:
                run_result["data"]["counts"] = measurement_data
                if memory:
                    run_result["data"]["memory"] = memory_data
            if meas_level == MeasLevel.KERNELED:
                run_result["data"]["memory"] = measurement_data

            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


@HAS_DYNAMICS.require_in_instance
class SingleTransmonTestBackend(PulseBackend):
    r"""A one qubit backend that corresponds to a three-level anharmonic transmon qubit.

    The Hamiltonian of the system is

    .. math::
        H = \hbar \sum_{j=1,2} \left[\omega_j |j\rangle\langle j| +
                \mathcal{E}(t) \lambda_j (\sigma_j^+ + \sigma_j^-)\right]

    Here, :math:`\omega_j` is the transition frequency from level :math:`0` to level
    :math:`j`. :math:`\mathcal{E}(t)` is the drive field and :math:`\sigma_j^\pm` are
    the raising and lowering operators between levels :math:`j-1` and :math:`j`.
    """

    @deprecate_arg(
        name="qubit_frequency",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="anharmonicity",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="lambda_1",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="lambda_2",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="gamma_1",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    def __init__(
        self,
        qubit_frequency: Optional[float] = None,
        anharmonicity: Optional[float] = None,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        gamma_1: Optional[float] = None,
        noise: bool = True,
        atol: float = None,
        rtol: float = None,
        **kwargs,
    ):
        """Initialise backend with hamiltonian parameters. If any of the hamiltonian
        parameters are not provided, default values will be used.

        Args:
            qubit_frequency: Frequency of the qubit (0-1). Defaults to 5e9.
            anharmonicity: Qubit anharmonicity $\\alpha$ = f12 - f01. Defaults to -0.25e9.
            lambda_1: Strength of 0-1 transition. Defaults to 1e9.
            lambda_2: Strength of 1-2 transition. Defaults to 0.8e9.
            gamma_1: Relaxation rate (1/T1) for 1-0. Defaults to 1e4.
            noise: Defaults to True. If True then T1 dissipation is included in the pulse-simulation.
                The strength is given by ``gamma_1``.
            atol: Absolute tolerance during solving.
            rtol: Relative tolerance during solving.
        """
        if anharmonicity is None:
            self.anharmonicity = 0.25e9
        else:
            self.anharmonicity = anharmonicity

        if qubit_frequency is None:
            qubit_frequency = 5e9
        if lambda_1 is None and lambda_2 is None:
            lambda_1 = 1e9
            lambda_2 = 0.8e9
        if gamma_1 is None:
            gamma_1 = 1e4

        qubit_frequency_02 = 2 * qubit_frequency + self.anharmonicity
        ket0 = np.array([[1, 0, 0]]).T
        ket1 = np.array([[0, 1, 0]]).T
        ket2 = np.array([[0, 0, 1]]).T

        sigma_m1 = ket0 @ ket1.T.conj()
        sigma_m2 = ket1 @ ket2.T.conj()

        sigma_p1 = sigma_m1.T.conj()
        sigma_p2 = sigma_m2.T.conj()

        p1 = ket1 @ ket1.T.conj()
        p2 = ket2 @ ket2.T.conj()

        drift = 2 * np.pi * (qubit_frequency * p1 + qubit_frequency_02 * p2)
        control = [
            2 * np.pi * (lambda_1 * (sigma_p1 + sigma_m1) + lambda_2 * (sigma_p2 + sigma_m2))
        ]
        r_frame = 2 * np.pi * qubit_frequency * (p1 + 2 * p2)
        t1_dissipator = np.sqrt(gamma_1) * sigma_m1

        self.rabi_rate_01 = [8.5985]
        self.rabi_rate_12 = [6.876]
        self.beta_01 = [1.551]

        if noise is True:
            evaluation_mode = "dense_vectorized"
            static_dissipators = [t1_dissipator]
        else:
            evaluation_mode = "dense"
            static_dissipators = None

        super().__init__(
            solver=Solver(
                static_hamiltonian=drift,
                hamiltonian_operators=control,
                static_dissipators=static_dissipators,
                rotating_frame=r_frame,
                rwa_cutoff_freq=1.9 * qubit_frequency,
                rwa_carrier_freqs=[qubit_frequency],
                evaluation_mode=evaluation_mode,
            ),
            atol=atol,
            rtol=rtol,
            **kwargs,
        )

        self._discriminator = [DefaultDiscriminator()]

        self._qubit_properties = [
            QubitProperties(frequency=qubit_frequency),
        ]
        self._target = Target(
            num_qubits=1,
            qubit_properties=self._qubit_properties,
            dt=self.dt,
            granularity=16,
        )

        measure_props = {
            (0,): InstructionProperties(),
        }
        x_props = {
            (0,): InstructionProperties(
                duration=160e-10,
                error=0,
                calibration=self.pulse_command(qubit=0, amp=0.5 / self.rabi_rate_01[0], beta=4),
            ),
        }
        sx_props = {
            (0,): InstructionProperties(
                duration=160e-10,
                error=0,
                calibration=self.pulse_command(qubit=0, amp=0.25 / self.rabi_rate_01[0], beta=4),
            ),
        }
        rz_props = {
            (0,): InstructionProperties(duration=0.0, error=0),
        }
        self._phi = Parameter("phi")
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(self._phi), rz_props)

        self.converter = InstructionToSignals(self.dt, carriers={"d0": qubit_frequency})

        default_schedules = [
            self.instruction_schedule_map.get("x", (0,)),
            self.instruction_schedule_map.get("sx", (0,)),
        ]
        self._simulated_pulse_unitaries = {
            (schedule.name, (0,), ()): self.solve(schedule) for schedule in default_schedules
        }

    @staticmethod
    def rz_gate(_, theta: float) -> np.ndarray:
        """Rz gate corresponding to single qubit 3 level qubit. Note: We do not try to
        model accurate qutrit dynamics.

        Args:
            theta: The angle parameter of Rz gate.

        Returns:
            Matrix of Rz(theta).
        """
        return np.diag([np.exp(1.0j * idx * theta / 2) for idx in [-1, 1, 3]])


@HAS_DYNAMICS.require_in_instance
class ParallelTransmonTestBackend(PulseBackend):
    r"""A decoupled two qubit backend. Models three-level anharmonic transmon qubits.

    The Hamiltonian of the system is

    .. math::
        H^i = \hbar \sum_{j=1,2} \left[\omega^i_j |j\rangle\langle j| +
                \mathcal{E}(t) \lambda^i_j (\sigma_j^+ + \sigma_j^-)\right]

        H = H_0 ⊗ I + I ⊗ H_1
    Here, :math:`\omega^i_j` is the ith transition frequency from level :math`0` to level
    :math:`j`. :math:`\mathcal{E}(t)` is the drive field and :math:`\sigma_j^\pm` are
    the raising and lowering operators between levels :math:`j-1` and :math:`j`.
    """

    @deprecate_arg(
        name="qubit_frequency",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="anharmonicity",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="lambda_1",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="lambda_2",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    @deprecate_arg(
        name="gamma_1",
        since="0.6",
        package_name="qiskit-experiments",
        pending=True,
    )
    def __init__(
        self,
        qubit_frequency: Optional[float] = None,
        anharmonicity: Optional[float] = None,
        lambda_1: Optional[float] = None,
        lambda_2: Optional[float] = None,
        gamma_1: Optional[float] = None,
        noise: bool = True,
        atol: float = None,
        rtol: float = None,
        **kwargs,
    ):
        """Initialise backend with hamiltonian parameters. If any of the hamiltonian
        parameters are not provided, default values will be used.

        Args:
            qubit_frequency: Frequency of the qubit (0-1). Defaults to 5e9.
            anharmonicity: Qubit anharmonicity $\\alpha$ = f12 - f01. Defaults to -0.25e9.
            lambda_1: Strength of 0-1 transition. Defaults to 1e9.
            lambda_2: Strength of 1-2 transition. Defaults to 0.8e9.
            gamma_1: Relaxation rate (1/T1) for 1-0. Defaults to 1e4.
            noise: Defaults to True. If True then T1 dissipation is included in the pulse-simulation.
                The strength is given by ``gamma_1``.
            atol: Absolute tolerance during solving.
            rtol: Relative tolerance during solving.
        """
        if anharmonicity is None:
            self.anharmonicity = [0.25e9, 0.25e9]
        else:
            self.anharmonicity = [anharmonicity, anharmonicity]

        if qubit_frequency is None:
            qubit_frequency = 5e9
        if lambda_1 is None and lambda_2 is None:
            lambda_1 = 1e9
            lambda_2 = 0.8e9
        if gamma_1 is None:
            gamma_1 = 1e4

        qubit_frequency_02 = 2 * qubit_frequency + anharmonicity
        ket0 = np.array([[1, 0, 0]]).T
        ket1 = np.array([[0, 1, 0]]).T
        ket2 = np.array([[0, 0, 1]]).T

        sigma_m1 = ket0 @ ket1.T.conj()
        sigma_m2 = ket1 @ ket2.T.conj()

        sigma_p1 = sigma_m1.T.conj()
        sigma_p2 = sigma_m2.T.conj()

        p1 = ket1 @ ket1.T.conj()
        p2 = ket2 @ ket2.T.conj()

        ident = np.eye(3)

        drift = 2 * np.pi * (qubit_frequency * p1 + qubit_frequency_02 * p2)
        drift_2q = np.kron(drift, ident) + np.kron(ident, drift)

        control = [
            2 * np.pi * (lambda_1 * (sigma_p1 + sigma_m1) + lambda_2 * (sigma_p2 + sigma_m2))
        ]
        control_2q = [np.kron(ident, control[0]), np.kron(control[0], ident)]

        r_frame = 2 * np.pi * qubit_frequency * (p1 + 2 * p2)
        r_frame_2q = np.kron(r_frame, ident) + np.kron(ident, r_frame)

        t1_dissipator0 = np.sqrt(gamma_1) * np.kron(ident, sigma_m1)
        t1_dissipator1 = np.sqrt(gamma_1) * np.kron(sigma_m1, ident)

        self.rabi_rate_01 = [8.589, 8.589]
        self.rabi_rate_12 = [6.876, 6.876]

        if noise is True:
            evaluation_mode = "dense_vectorized"
            static_dissipators = [t1_dissipator0, t1_dissipator1]
        else:
            evaluation_mode = "dense"
            static_dissipators = None

        super().__init__(
            solver=Solver(
                static_hamiltonian=drift_2q,
                hamiltonian_operators=control_2q,
                static_dissipators=static_dissipators,
                rotating_frame=r_frame_2q,
                rwa_cutoff_freq=1.9 * qubit_frequency,
                rwa_carrier_freqs=[qubit_frequency, qubit_frequency],
                evaluation_mode=evaluation_mode,
            ),
            atol=atol,
            rtol=rtol,
            **kwargs,
        )

        self._discriminator = [DefaultDiscriminator(), DefaultDiscriminator()]
        self.subsystem_dims = (3, 3)

        self._qubit_properties = [
            QubitProperties(frequency=qubit_frequency),
            QubitProperties(frequency=qubit_frequency),
        ]
        self._target = Target(
            num_qubits=2,
            qubit_properties=self._qubit_properties,
            dt=self.dt,
            granularity=16,
        )

        measure_props = {
            (0,): InstructionProperties(),
            (1,): InstructionProperties(),
        }
        x_props = {
            (0,): InstructionProperties(
                duration=160e-10,
                error=0,
                calibration=self.pulse_command(qubit=0, amp=0.5 / self.rabi_rate_01[0], beta=4),
            ),
            (1,): InstructionProperties(
                duration=160e-10,
                error=0,
                calibration=self.pulse_command(qubit=1, amp=0.5 / self.rabi_rate_01[1], beta=4),
            ),
        }
        sx_props = {
            (0,): InstructionProperties(
                duration=160e-10,
                error=0,
                calibration=self.pulse_command(qubit=0, amp=0.25 / self.rabi_rate_01[0], beta=4),
            ),
            (1,): InstructionProperties(
                duration=160e-10,
                error=0,
                calibration=self.pulse_command(qubit=1, amp=0.25 / self.rabi_rate_01[1], beta=4),
            ),
        }
        rz_props = {
            (0,): InstructionProperties(duration=0.0, error=0),
            (1,): InstructionProperties(duration=0.0, error=0),
        }
        cx_props = {
            (0, 1): InstructionProperties(duration=0, error=0),
            (1, 0): InstructionProperties(duration=0, error=0),
        }

        self._phi = Parameter("phi")
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(self._phi), rz_props)
        self._target.add_instruction(CXGate(), cx_props)

        self.converter = InstructionToSignals(
            self.dt,
            carriers={"d0": qubit_frequency, "d1": qubit_frequency},
            channels=["d0", "d1"],
        )

        default_schedules = [
            self.instruction_schedule_map.get("x", (0,)),
            self.instruction_schedule_map.get("sx", (0,)),
            self.instruction_schedule_map.get("x", (1,)),
            self.instruction_schedule_map.get("sx", (1,)),
        ]

        self._simulated_pulse_unitaries = {
            (schedule.name, (schedule.channels[0].index,), ()): self.solve(schedule)
            for schedule in default_schedules
        }

    @staticmethod
    def rz_gate(qubits: List[int], theta: float) -> np.ndarray:
        """Rz gate corresponding to single qubit 3 level qubit. Note: We do not try to
        model accurate qutrit dynamics.

        Args:
            qubits: Qubit index of gate.
            theta: The angle parameter of Rz gate.

        Returns:
            Matrix of Rz(theta).
        """
        rz_1q = np.diag([np.exp(1.0j * idx * theta / 2) for idx in [-1, 1, 3]])
        if qubits[0] == 0:
            rz = np.kron(np.eye(3), rz_1q)
        elif qubits[0] == 1:
            rz = np.kron(rz_1q, np.eye(3))
        return rz


class DefaultDiscriminator(BaseDiscriminator):
    """Default Discriminator used for ``meas_level=2`` in ``SingleTransmonTestBackend``
    and ``ParallelTransmonTestBackend``.
    """

    x_0 = 0.25  # empirical threshold

    def predict(self, data: List):
        """The function used to predict the labels of the data."""
        return ["0" if iq[0] > self.x_0 else "1" for iq in data]

    def config(self) -> Dict[str, Any]:
        """Return the configuration of the discriminator."""
        return {"x_0": self.x_0}

    def is_trained(self) -> bool:
        """Return True if this discriminator has been trained on data."""
        return True
