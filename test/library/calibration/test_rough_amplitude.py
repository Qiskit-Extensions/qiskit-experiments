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

"""Test rough amplitude calibration experiment classes."""

from test.base import QiskitExperimentsTestCase
import numpy as np

from qiskit import pulse
from qiskit.circuit import Parameter

from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import EFRoughXSXAmplitudeCal, RoughXSXAmplitudeCal
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend


class TestRoughAmpCal(QiskitExperimentsTestCase):
    """A class to test the rough amplitude calibration experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        library = FixedFrequencyTransmon()

        self.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])

    def test_circuits(self):
        """Test the quantum circuits."""
        test_amps = [-0.5, 0, 0.5]
        rabi = RoughXSXAmplitudeCal([0], self.cals, amplitudes=test_amps, backend=self.backend)

        circs = rabi._transpiled_circuits()

        for circ, amp in zip(circs, test_amps):
            self.assertEqual(circ.count_ops()["Rabi"], 1)

            d0 = pulse.DriveChannel(0)
            with pulse.build(name="x") as expected_x:
                pulse.play(pulse.Drag(160, amp, 40, 0), d0)

            self.assertEqual(circ.calibrations["Rabi"][((0,), (amp,))], expected_x)

    def test_update(self):
        """Test that the calibrations update properly."""

        tol = 0.01
        default_amp = 0.5 / self.backend.rabi_rate_01[0]

        rabi = RoughXSXAmplitudeCal(
            [0], self.cals, amplitudes=np.linspace(-0.1, 0.1, 11), backend=self.backend
        )
        expdata = rabi.run()
        self.assertExperimentDone(expdata)

        self.assertAlmostEqual(self.cals.get_parameter_value("amp", 0, "x"), default_amp, delta=tol)
        self.assertAlmostEqual(
            self.cals.get_parameter_value("amp", 0, "sx"), default_amp / 2, delta=tol
        )

        self.cals.add_parameter_value(int(4 * 160 / 5), "duration", (), schedule="x")
        rabi = RoughXSXAmplitudeCal(
            [0], self.cals, amplitudes=np.linspace(-0.1, 0.1, 11), backend=self.backend
        )
        expdata = rabi.run()
        self.assertExperimentDone(expdata)

        self.assertTrue(
            abs(self.cals.get_parameter_value("amp", 0, "x") * (4 / 5) - default_amp) < tol
        )
        self.assertTrue(
            abs(self.cals.get_parameter_value("amp", 0, "sx") * (4 / 5) - default_amp / 2) < tol
        )

    def test_circuit_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        test_amps = [-0.5, 0]
        rabi = RoughXSXAmplitudeCal([0], self.cals, amplitudes=test_amps, backend=self.backend)
        self.assertRoundTripSerializable(rabi._transpiled_circuits())

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = RoughXSXAmplitudeCal([0], self.cals)
        config = exp.config()
        loaded_exp = RoughXSXAmplitudeCal.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config())


class TestSpecializations(QiskitExperimentsTestCase):
    """Test the specialized versions of the calibration."""

    @classmethod
    def setUpClass(cls):
        """Setup the tests"""
        super().setUpClass()

        library = FixedFrequencyTransmon()

        cls.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)
        cls.cals = Calibrations.from_backend(cls.backend, libraries=[library])

        # Add some pulses on the 1-2 transition.
        d0 = pulse.DriveChannel(0)
        with pulse.build(name="x12") as x12:
            with pulse.frequency_offset(-300e6, d0):
                pulse.play(pulse.Drag(Parameter("duration"), Parameter("amp"), 40, 0.0), d0)

        with pulse.build(name="sx12") as sx12:
            with pulse.frequency_offset(-300e6, d0):
                pulse.play(pulse.Drag(Parameter("duration"), Parameter("amp"), 40, 0.0), d0)

        cls.cals.add_schedule(x12, 0)
        cls.cals.add_schedule(sx12, 0)
        cls.cals.add_parameter_value(0.4, "amp", 0, "x12")
        cls.cals.add_parameter_value(0.2, "amp", 0, "sx12")
        cls.cals.add_parameter_value(160, "duration", 0, "x12")
        cls.cals.add_parameter_value(160, "duration", 0, "sx12")

    def test_ef_circuits(self):
        """Test that we get the expected circuits with calibrations for the EF experiment."""

        test_amps = [-0.5, 0, 0.5]
        rabi_ef = EFRoughXSXAmplitudeCal([0], self.cals, amplitudes=test_amps, backend=self.backend)

        circs = rabi_ef._transpiled_circuits()

        for circ, amp in zip(circs, test_amps):

            self.assertEqual(circ.count_ops()["x"], 1)
            self.assertEqual(circ.count_ops()["Rabi"], 1)

            d0 = pulse.DriveChannel(0)
            with pulse.build(name="x") as expected_x:
                pulse.play(pulse.Drag(160, 0.5, 40, 0), d0)

            with pulse.build(name="x12") as expected_x12:
                with pulse.frequency_offset(-300e6, d0):
                    pulse.play(pulse.Drag(160, amp, 40, 0), d0)

            self.assertEqual(circ.calibrations["x"][((0,), ())], expected_x)
            self.assertEqual(circ.calibrations["Rabi"][((0,), (amp,))], expected_x12)

    def test_ef_update(self):
        """Test that we properly update the pulses on the 1<->2 transition."""

        tol = 0.05
        default_amp = 0.5 / self.backend.rabi_rate_12[0]

        rabi_ef = EFRoughXSXAmplitudeCal(
            [0], self.cals, amplitudes=np.linspace(-0.1, 0.1, 11), backend=self.backend
        )
        rabi_ef.set_run_options(shots=200)
        expdata = rabi_ef.run()
        self.assertExperimentDone(expdata)

        self.assertAlmostEqual(
            self.cals.get_parameter_value("amp", 0, "x12"), default_amp, delta=tol
        )
        self.assertAlmostEqual(
            self.cals.get_parameter_value("amp", 0, "sx12"), default_amp / 2, delta=tol
        )

        self.cals.add_parameter_value(int(4 * 160 / 5), "duration", 0, "x12")
        self.cals.add_parameter_value(int(4 * 160 / 5), "duration", 0, "sx12")
        rabi_ef = EFRoughXSXAmplitudeCal([0], self.cals, amplitudes=np.linspace(-0.1, 0.1, 11))
        rabi_ef.set_run_options(shots=200)
        expdata = rabi_ef.run(self.backend)
        self.assertExperimentDone(expdata)

        self.assertTrue(
            abs(self.cals.get_parameter_value("amp", 0, "x12") * (4 / 5) - default_amp) < tol
        )
        self.assertTrue(
            abs(self.cals.get_parameter_value("amp", 0, "sx12") * (4 / 5) - default_amp / 2) < tol
        )
