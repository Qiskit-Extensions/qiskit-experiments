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

"""Test the fine amplitude characterization and calibration experiments."""
from test.base import QiskitExperimentsTestCase
import numpy as np
from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit.library import XGate, SXGate
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse

from qiskit_experiments.library import (
    FineXAmplitude,
    FineSXAmplitude,
    FineZXAmplitude,
    FineXAmplitudeCal,
    FineSXAmplitudeCal,
)
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineAmpHelper as FineAmpHelper


@ddt
class TestFineAmpEndToEnd(QiskitExperimentsTestCase):
    """Test the fine amplitude experiment."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        self.qubit = 0
        self.backend = SingleTransmonTestBackend(noise=False)

    @data(-0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08)
    def test_end_to_end(self, pi_ratio):
        """Test the experiment end to end."""

        backend_inst_map = self.backend.defaults().instruction_schedule_map
        x_pulse = backend_inst_map.get("x", (self.qubit,)).instructions[0][1].pulse

        with pulse.build(name="x") as x_sched:
            pulse.play(
                pulse.Drag(
                    x_pulse.duration, x_pulse.amp * (1 + pi_ratio), x_pulse.sigma, x_pulse.beta
                ),
                pulse.DriveChannel(self.qubit),
            )

        inst_map = pulse.InstructionScheduleMap()
        inst_map.add("x", (self.qubit,), x_sched)

        amp_exp = FineXAmplitude(self.qubit, backend=self.backend)
        amp_exp.set_transpile_options(basis_gates=["x", "sx"], inst_map=inst_map)

        error = -np.pi * pi_ratio

        expdata = amp_exp.run()
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        d_theta = result.value.n

        tol = 0.003

        self.assertAlmostEqual(d_theta, error, delta=tol)


@ddt
class TestFineZXAmpEndToEnd(QiskitExperimentsTestCase):
    """Test the fine amplitude experiment.

    Does not use the SingleTransmonTestBackend as this is a two-qubit experiment.
    """

    @data(-0.08, -0.03, -0.02, 0.02, 0.06, 0.07)
    def test_end_to_end(self, pi_ratio):
        """Test the experiment end to end."""

        error = -np.pi * pi_ratio
        amp_exp = FineZXAmplitude((0, 1))
        backend = MockIQBackend(FineAmpHelper(error, np.pi / 2, "szx"))

        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        d_theta = result.value.n

        tol = 0.04

        self.assertAlmostEqual(d_theta, error, delta=tol)
        self.assertEqual(result.quality, "good")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineZXAmplitude((0, 1))
        loaded_exp = FineZXAmplitude.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))


class TestFineAmplitudeCircuits(QiskitExperimentsTestCase):
    """Test the circuits."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=0.0), DriveChannel(0))

        with pulse.build(name="x90p") as x90p:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=0.0), DriveChannel(0))

        self.x_plus = xp
        self.x_90_plus = x90p
        self.qubit = 0
        self.backend = SingleTransmonTestBackend(noise=False)

    def test_xp(self):
        """Test a circuit with the x gate."""

        amp_cal = FineXAmplitude(self.qubit, backend=self.backend)
        circs = amp_cal.circuits()

        self.assertTrue(circs[0].data[0][0].name == "measure")
        self.assertTrue(circs[1].data[0][0].name == "x")

        for idx, circ in enumerate(circs[2:]):
            self.assertTrue(circ.data[0][0].name == "sx")
            self.assertEqual(circ.count_ops().get("x", 0), idx + 1)

    def test_x90p(self):
        """Test circuits with an x90p pulse."""

        amp_cal = FineSXAmplitude(self.qubit, backend=self.backend)

        expected = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        for idx, circ in enumerate(amp_cal.circuits()):
            self.assertEqual(circ.count_ops().get("sx", 0), expected[idx])


@ddt
class TestSpecializations(QiskitExperimentsTestCase):
    """Test the options of the specialized classes."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        self.qubit = 0
        self.backend = SingleTransmonTestBackend(noise=False)

    def test_fine_x_amp(self):
        """Test the fine X amplitude."""

        exp = FineXAmplitude(self.qubit, backend=self.backend)

        self.assertTrue(exp.experiment_options.add_cal_circuits)
        self.assertDictEqual(
            exp.analysis.options.fixed_parameters,
            {"angle_per_gate": np.pi, "phase_offset": np.pi / 2},
        )
        self.assertEqual(exp.experiment_options.gate, XGate())

    def test_x_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineXAmplitude(self.qubit, backend=self.backend)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_fine_sx_amp(self):
        """Test the fine SX amplitude."""

        exp = FineSXAmplitude(self.qubit, backend=self.backend)

        self.assertFalse(exp.experiment_options.add_cal_circuits)

        expected = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        self.assertEqual(exp.experiment_options.repetitions, expected)
        self.assertDictEqual(
            exp.analysis.options.fixed_parameters,
            {"angle_per_gate": np.pi / 2, "phase_offset": np.pi},
        )
        self.assertEqual(exp.experiment_options.gate, SXGate())

    def test_sx_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineSXAmplitude(self.qubit, backend=self.backend)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    @data((2, 3), (3, 1), (0, 1))
    def test_measure_qubits(self, qubits):
        """Test that the measurement is on the logical qubits."""

        fine_amp = FineZXAmplitude(qubits)
        for circuit in fine_amp.circuits():
            self.assertEqual(circuit.num_qubits, 2)
            self.assertEqual(circuit.data[-1][0].name, "measure")
            self.assertEqual(circuit.data[-1][1][0], circuit.qregs[0][1])


class TestFineAmplitudeCal(QiskitExperimentsTestCase):
    """A class to test the fine amplitude calibration experiments."""

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        self.qubit = 0
        self.backend = SingleTransmonTestBackend(noise=False)
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])

    def test_cal_options(self):
        """Test that the options are properly propagated."""

        # Test the X gate cal
        amp_cal = FineXAmplitudeCal(self.qubit, self.cals, "x", self.backend)

        exp_opt = amp_cal.experiment_options

        self.assertEqual(exp_opt.gate.name, "x")
        self.assertTrue(exp_opt.add_cal_circuits)
        self.assertEqual(exp_opt.result_index, -1)
        self.assertEqual(exp_opt.group, "default")
        self.assertTrue(np.allclose(exp_opt.target_angle, np.pi))

        # Test the SX gate cal
        amp_cal = FineSXAmplitudeCal(self.qubit, self.cals, "sx", self.backend)

        exp_opt = amp_cal.experiment_options

        self.assertEqual(exp_opt.gate.name, "sx")
        self.assertFalse(exp_opt.add_cal_circuits)
        self.assertEqual(exp_opt.result_index, -1)
        self.assertEqual(exp_opt.group, "default")
        self.assertTrue(np.allclose(exp_opt.target_angle, np.pi / 2))

    def test_run_x_cal(self):
        """Test that we can transpile in the calibrations before and after update.

        If this test passes then we were successful in running a calibration experiment,
        updating a pulse parameter, having this parameter propagated to the schedules
        for use the next time the experiment is run.
        """

        # Initial pulse amplitude
        init_amp = 0.5

        amp_cal = FineXAmplitudeCal(0, self.cals, "x", self.backend)

        circs = transpile(amp_cal.circuits(), self.backend, inst_map=self.cals.default_inst_map)

        with pulse.build(name="x") as expected_x:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(0))

        with pulse.build(name="sx") as expected_sx:
            pulse.play(pulse.Drag(160, 0.25, 40, 0), pulse.DriveChannel(0))

        self.assertEqual(circs[5].calibrations["x"][((0,), ())], expected_x)
        self.assertEqual(circs[5].calibrations["sx"][((0,), ())], expected_sx)

        # run the calibration experiment. This should update the amp parameter of x which we test.
        exp_data = amp_cal.run()
        self.assertExperimentDone(exp_data)
        d_theta = exp_data.analysis_results(1).value.n
        new_amp = init_amp * np.pi / (np.pi + d_theta)

        circs = transpile(amp_cal.circuits(), self.backend, inst_map=self.cals.default_inst_map)

        x_cal = circs[5].calibrations["x"][((0,), ())]

        # Requires allclose due to numerical precision.
        self.assertTrue(np.allclose(x_cal.blocks[0].pulse.amp, new_amp))
        self.assertFalse(np.allclose(x_cal.blocks[0].pulse.amp, init_amp))
        self.assertEqual(circs[5].calibrations["sx"][((0,), ())], expected_sx)

    def test_run_sx_cal(self):
        """Test that we can transpile in the calibrations before and after update.

        If this test passes then we were successful in running a calibration experiment,
        updating a pulse parameter, having this parameter propagated to the schedules
        for use the next time the experiment is run.
        """

        # Initial pulse amplitude
        init_amp = 0.25

        amp_cal = FineSXAmplitudeCal(0, self.cals, "sx", self.backend)

        circs = transpile(amp_cal.circuits(), self.backend, inst_map=self.cals.default_inst_map)

        with pulse.build(name="sx") as expected_sx:
            pulse.play(pulse.Drag(160, 0.25, 40, 0), pulse.DriveChannel(0))

        self.assertEqual(circs[5].calibrations["sx"][((0,), ())], expected_sx)

        # run the calibration experiment. This should update the amp parameter of x which we test.
        exp_data = amp_cal.run()
        self.assertExperimentDone(exp_data)
        d_theta = exp_data.analysis_results(1).value.n
        new_amp = init_amp * (np.pi / 2) / (np.pi / 2 + d_theta)

        circs = transpile(amp_cal.circuits(), self.backend, inst_map=self.cals.default_inst_map)

        sx_cal = circs[5].calibrations["sx"][((0,), ())]

        # Requires allclose due to numerical precision.
        self.assertTrue(np.allclose(sx_cal.blocks[0].pulse.amp, new_amp))
        self.assertFalse(np.allclose(sx_cal.blocks[0].pulse.amp, init_amp))

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineSXAmplitudeCal(0, self.cals, "sx", self.backend)
        loaded_exp = FineSXAmplitudeCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineSXAmplitudeCal(0, self.cals, "sx", self.backend)
        self.assertRoundTripSerializable(exp, self.json_equiv)
