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
import numpy as np
from ddt import ddt, data, unpack
from qiskit import QuantumCircuit, pulse, quantum_info as qi
from qiskit.test.mock import FakeBogota
from qiskit.extensions.hamiltonian_gate import HamiltonianGate
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.library.characterization.two_qubit import cr_hamiltonian


class SimulatableCRGate(HamiltonianGate):
    """Cross resonance unitary that can be simulated with Aer simulator."""

    def __init__(self, width, t_off, wix, wiy, wiz, wzx, wzy, wzz, dt=1e-9):
        # Note that Qiskit is Little endien, i.e. [q1, q0]
        hamiltonian = (
            wix * qi.Operator.from_label("XI") / 2
            + wiy * qi.Operator.from_label("YI") / 2
            + wiz * qi.Operator.from_label("ZI") / 2
            + wzx * qi.Operator.from_label("XZ") / 2
            + wzy * qi.Operator.from_label("YZ") / 2
            + wzz * qi.Operator.from_label("ZZ") / 2
        )
        super().__init__(data=hamiltonian, time=(t_off + width) * dt)


@ddt
class TestCrossResonanceHamiltonian(QiskitExperimentsTestCase):
    """Test for cross resonance Hamiltonian tomography."""

    def test_circuit_generation(self):
        """Test generated circuits."""
        backend = FakeBogota()

        # Add granularity to check duration optimization logic
        setattr(
            backend.configuration(),
            "timing_constraints",
            {"granularity": 16},
        )

        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=(0, 1),
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        expr.backend = backend

        nearlest_16 = 1248

        with pulse.build(default_alignment="left", name="cr") as ref_cr_sched:
            pulse.play(
                pulse.GaussianSquare(
                    nearlest_16,
                    amp=0.1,
                    sigma=64,
                    width=1000,
                ),
                pulse.ControlChannel(0),
            )
            pulse.delay(nearlest_16, pulse.DriveChannel(0))
            pulse.delay(nearlest_16, pulse.DriveChannel(1))

        cr_gate = cr_hamiltonian.CrossResonanceHamiltonian.CRPulseGate(width=1000)
        expr_circs = expr.circuits()

        x0_circ = QuantumCircuit(2, 1)
        x0_circ.append(cr_gate, [0, 1])
        x0_circ.h(1)
        x0_circ.measure(1, 0)

        x1_circ = QuantumCircuit(2, 1)
        x1_circ.x(0)
        x1_circ.append(cr_gate, [0, 1])
        x1_circ.h(1)
        x1_circ.measure(1, 0)

        y0_circ = QuantumCircuit(2, 1)
        y0_circ.append(cr_gate, [0, 1])
        y0_circ.sdg(1)
        y0_circ.h(1)
        y0_circ.measure(1, 0)

        y1_circ = QuantumCircuit(2, 1)
        y1_circ.x(0)
        y1_circ.append(cr_gate, [0, 1])
        y1_circ.sdg(1)
        y1_circ.h(1)
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
        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=(0, 1),
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
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
        backend = AerSimulator(seed_simulator=123, shots=2000)
        backend._configuration.dt = 1e-9
        delta = 3e4

        sigma = 20
        t_off = np.sqrt(2 * np.pi) * sigma

        # Hack: transpiler calls qiskit.parallel but local object cannot be picked
        cr_gate = functools.partial(
            SimulatableCRGate,
            t_off=t_off,
            wix=2 * np.pi * ix,
            wiy=2 * np.pi * iy,
            wiz=2 * np.pi * iz,
            wzx=2 * np.pi * zx,
            wzy=2 * np.pi * zy,
            wzz=2 * np.pi * zz,
            dt=1e-9,
        )

        durations = np.linspace(0, 700, 50)
        expr = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=(0, 1),
            flat_top_widths=durations,
            sigma=sigma,
            risefall=2,
            cr_gate=cr_gate,
        )
        expr.backend = backend

        exp_data = expr.run()
        self.assertExperimentDone(exp_data, timeout=600)

        self.assertEqual(exp_data.analysis_results(0).quality, "good")

        # These values are computed from other analysis results in post hook.
        # Thus at least one of these values should be round-trip tested.
        res_ix = exp_data.analysis_results("omega_ix")
        self.assertAlmostEqual(res_ix.value.n, ix, delta=delta)
        self.assertRoundTripSerializable(res_ix.value, check_func=self.ufloat_equiv)
        self.assertEqual(res_ix.extra["unit"], "Hz")

        self.assertAlmostEqual(exp_data.analysis_results("omega_iy").value.n, iy, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_iz").value.n, iz, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zx").value.n, zx, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zy").value.n, zy, delta=delta)
        self.assertAlmostEqual(exp_data.analysis_results("omega_zz").value.n, zz, delta=delta)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=[0, 1],
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        loaded_exp = cr_hamiltonian.CrossResonanceHamiltonian.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = cr_hamiltonian.CrossResonanceHamiltonian(
            qubits=[0, 1],
            flat_top_widths=[1000],
            amp=0.1,
            sigma=64,
            risefall=2,
        )
        self.assertRoundTripSerializable(exp, self.json_equiv)
