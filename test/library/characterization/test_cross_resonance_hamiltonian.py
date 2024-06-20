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

# pylint: disable=invalid-name

"""Spectroscopy tests."""
from test.base import QiskitExperimentsTestCase
import functools
import io
from unittest.mock import patch

import numpy as np
from ddt import ddt, data, unpack

from qiskit import QuantumCircuit, pulse, qpy, quantum_info as qi
from qiskit.circuit import Gate

# TODO: remove old path after we stop supporting the relevant version of Qiskit
try:
    from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate
except ModuleNotFoundError:
    from qiskit.extensions.hamiltonian_gate import HamiltonianGate

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBogotaV2

from qiskit_experiments.library.characterization import cr_hamiltonian


class SimulatableCRGate(HamiltonianGate):
    """Hamiltonian Gate for simulation."""

    def __init__(self, width, hamiltonian, sigma, dt):
        super().__init__(data=hamiltonian, time=np.sqrt(2 * np.pi) * sigma * dt + width)


@ddt
class TestCrossResonanceHamiltonian(QiskitExperimentsTestCase):
    """Test for cross resonance Hamiltonian tomography."""

    def test_circuit_generation(self):
        """Test generated circuits."""
        backend = FakeBogotaV2()

        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            physical_qubits=(0, 1),
            amp=0.1,
            sigma=64,
            risefall=2,
            durations=[backend.dt * (1000 + 4 * 64)],
        )
        expr.backend = backend

        with pulse.build(default_alignment="left", name="cr") as ref_cr_sched:
            pulse.play(
                pulse.GaussianSquare(
                    duration=1256,
                    amp=0.1,
                    sigma=64,
                    width=1000,
                ),
                pulse.ControlChannel(0),
            )
            pulse.delay(1256, pulse.DriveChannel(0))
            pulse.delay(1256, pulse.DriveChannel(1))

        width_sec = 1000 * backend.dt
        cr_gate = cr_hamiltonian.CrossResonanceHamiltonian.CRPulseGate(width=width_sec)
        expr_circs = expr.circuits()

        x0_circ = QuantumCircuit(2, 1)
        x0_circ.append(cr_gate, [0, 1])
        x0_circ.rz(np.pi / 2, 1)
        x0_circ.sx(1)
        x0_circ.measure(1, 0)

        x1_circ = QuantumCircuit(2, 1)
        x1_circ.x(0)
        x1_circ.append(cr_gate, [0, 1])
        x1_circ.rz(np.pi / 2, 1)
        x1_circ.sx(1)
        x1_circ.measure(1, 0)

        y0_circ = QuantumCircuit(2, 1)
        y0_circ.append(cr_gate, [0, 1])
        y0_circ.sx(1)
        y0_circ.measure(1, 0)

        y1_circ = QuantumCircuit(2, 1)
        y1_circ.x(0)
        y1_circ.append(cr_gate, [0, 1])
        y1_circ.sx(1)
        y1_circ.measure(1, 0)

        z0_circ = QuantumCircuit(2, 1)
        z0_circ.append(cr_gate, [0, 1])
        z0_circ.measure(1, 0)

        z1_circ = QuantumCircuit(2, 1)
        z1_circ.x(0)
        z1_circ.append(cr_gate, [0, 1])
        z1_circ.measure(1, 0)

        ref_circs = [x0_circ, y0_circ, z0_circ, x1_circ, y1_circ, z1_circ]
        for c in ref_circs:
            c.add_calibration(cr_gate, (0, 1), ref_cr_sched)

        self.assertListEqual(expr_circs, ref_circs)

    def test_circuit_generation_no_backend(self):
        """User can check experiment circuit without setting backend."""

        class FakeCRGate(HamiltonianGate):
            """Hamiltonian Gate for simulation."""

            def __init__(self, width):
                super().__init__(data=np.eye(4), time=width)

        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            physical_qubits=(0, 1),
            cr_gate=FakeCRGate,
            amp=0.1,
            sigma=64,
            risefall=2,
            durations=[1256],
        )

        # Not raise an error
        expr.circuits()

    @data(
        [1e6, 2e6, 1e3, -3e6, -2e6, 1e4],
        [-1e6, -2e6, 1e3, 3e6, 2e6, 1e4],
        [1e4, 2e4, 1e3, 5e6, 1e6, 2e3],
        [1e4, -1e3, 1e3, 5e5, 1e3, -1e3],  # low frequency test case 1
        [-1.0e5, 1.2e5, 1.0e3, 1.5e5, -1.1e5, -1.0e3],  # low frequency test case 2
    )
    @unpack
    def test_integration(self, ix, iy, iz, zx, zy, zz):
        """Integration test for Hamiltonian tomography."""
        delta = 3e4

        dt = 0.222e-9
        sigma = 64

        backend = AerSimulator(seed_simulator=123, shots=2000)
        backend._configuration.dt = dt

        # Note that Qiskit is Little endian, i.e. [q1, q0]
        hamiltonian = (
            2
            * np.pi
            * (
                ix * qi.Operator.from_label("XI") / 2
                + iy * qi.Operator.from_label("YI") / 2
                + iz * qi.Operator.from_label("ZI") / 2
                + zx * qi.Operator.from_label("XZ") / 2
                + zy * qi.Operator.from_label("YZ") / 2
                + zz * qi.Operator.from_label("ZZ") / 2
            )
        )

        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            physical_qubits=(0, 1),
            sigma=sigma,
            # A hack to avoid local function in pickle, i.e. in transpile.
            cr_gate=functools.partial(
                SimulatableCRGate, hamiltonian=hamiltonian, sigma=sigma, dt=dt
            ),
        )
        expr.backend = backend

        exp_data = expr.run()
        self.assertExperimentDone(exp_data, timeout=1000)

        self.assertEqual(exp_data.analysis_results("omega_ix").quality, "good")

        # These values are computed from other analysis results in post hook.
        # Thus at least one of these values should be round-trip tested.
        res_ix = exp_data.analysis_results("omega_ix")
        self.assertAlmostEqual(res_ix.value.n, ix, delta=delta)
        self.assertRoundTripSerializable(res_ix.value)
        self.assertEqual(res_ix.extra["unit"], "Hz")

        self.assertAlmostEqual(exp_data.analysis_results("omega_iy").value.n, iy, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_iz").value.n, iz, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zx").value.n, zx, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zy").value.n, zy, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zz").value.n, zz, delta=delta)

    def test_integration_backward_compat(self):
        """Integration test for Hamiltonian tomography."""
        ix, iy, iz, zx, zy, zz = 1e6, 2e6, 1e3, -3e6, -2e6, 1e4

        delta = 3e4

        dt = 0.222e-9
        sigma = 64

        backend = AerSimulator(seed_simulator=123, shots=2000)
        backend._configuration.dt = dt

        # Note that Qiskit is Little endian, i.e. [q1, q0]
        hamiltonian = (
            2
            * np.pi
            * (
                ix * qi.Operator.from_label("XI") / 2
                + iy * qi.Operator.from_label("YI") / 2
                + iz * qi.Operator.from_label("ZI") / 2
                + zx * qi.Operator.from_label("XZ") / 2
                + zy * qi.Operator.from_label("YZ") / 2
                + zz * qi.Operator.from_label("ZZ") / 2
            )
        )

        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            (0, 1),
            np.linspace(0, 700, 50),
            sigma=sigma,
            # A hack to avoild local function in pickle, i.e. in transpile.
            cr_gate=functools.partial(
                SimulatableCRGate, hamiltonian=hamiltonian, sigma=sigma, dt=dt
            ),
        )
        expr.backend = backend

        exp_data = expr.run()
        self.assertExperimentDone(exp_data, timeout=1000)

        self.assertEqual(exp_data.analysis_results("omega_ix").quality, "good")

        self.assertAlmostEqual(exp_data.analysis_results("omega_ix").value.n, ix, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_iy").value.n, iy, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_iz").value.n, iz, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zx").value.n, zx, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zy").value.n, zy, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zz").value.n, zz, delta=delta)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = cr_hamiltonian.CrossResonanceHamiltonian(
            physical_qubits=[0, 1],
            durations=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        loaded_exp = cr_hamiltonian.CrossResonanceHamiltonian.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = cr_hamiltonian.CrossResonanceHamiltonian(
            physical_qubits=[0, 1],
            durations=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        self.assertRoundTripSerializable(exp)

    def test_circuit_serialization(self):
        """Test generated circuits."""
        backend = FakeBogotaV2()

        with patch.object(
            cr_hamiltonian.CrossResonanceHamiltonian.CRPulseGate,
            "base_class",
            Gate,
        ):
            # Monkey patching the Instruction.base_class property of the CRPulseGate.
            # QPY loader is not aware of Gate subclasses defined outside Qiskit core,
            # and a Gate subclass instance is reconstructed as a Gate class instance.
            # This results in the failure in comparison of structurally same circuits.
            # In this context, CRPulseGate looks like a Gate class.
            expr = cr_hamiltonian.CrossResonanceHamiltonian(
                physical_qubits=(0, 1),
                amp=0.1,
                sigma=64,
                risefall=2,
            )
            expr.backend = backend

            width_sec = 1000 * backend.dt
            cr_gate = cr_hamiltonian.CrossResonanceHamiltonian.CRPulseGate(width=width_sec)
            circuits = expr._transpiled_circuits()

            x0_circ = QuantumCircuit(2, 1)
            x0_circ.append(cr_gate, [0, 1])
            x0_circ.rz(np.pi / 2, 1)
            x0_circ.sx(1)
            x0_circ.measure(1, 0)

            circuits.append(x0_circ)

            with io.BytesIO() as buff:
                qpy.dump(circuits, buff)
                buff.seek(0)
                serialized_data = buff.read()

            with io.BytesIO() as buff:
                buff.write(serialized_data)
                buff.seek(0)
                decoded = qpy.load(buff)

            self.assertListEqual(circuits, decoded)
