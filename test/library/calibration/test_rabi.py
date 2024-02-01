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

"""Test Rabi amplitude Experiment class."""
from test.base import QiskitExperimentsTestCase
import unittest
import numpy as np

from qiskit import QuantumCircuit, pulse, transpile
from qiskit.exceptions import QiskitError
from qiskit.circuit import Parameter
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.framework import ExperimentData, ParallelExperiment
from qiskit_experiments.library import Rabi, EFRabi

from qiskit_experiments.curve_analysis.standard_analysis.oscillation import OscillationAnalysis
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
from qiskit_experiments.framework.experiment_data import ExperimentStatus


class TestRabiEndToEnd(QiskitExperimentsTestCase):
    """Test the rabi experiment."""

    @classmethod
    def setUpClass(cls):
        """Setup the tests."""
        super().setUpClass()

        cls.qubit = 0

        with pulse.build(name="x") as sched:
            pulse.play(pulse.Drag(160, Parameter("amp"), 40, 0.4), pulse.DriveChannel(cls.qubit))

        cls.sched = sched
        cls.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)

    # pylint: disable=no-member
    def test_rabi_end_to_end(self):
        """Test the Rabi experiment end to end."""

        test_tol = 0.15

        rabi = Rabi([self.qubit], self.sched, backend=self.backend)
        rabi.set_run_options(shots=200)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.1, 0.1, 21))
        expdata = rabi.run()
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(0)

        self.assertEqual(result.quality, "good")
        # The comparison is made against the object that exists in the backend for accurate testing
        self.assertAlmostEqual(
            result.value.params["freq"], self.backend.rabi_rate_01[0], delta=test_tol
        )

    def test_wrong_processor(self):
        """Test that we can override the data processing by giving a faulty data processor."""
        rabi = Rabi([self.qubit], self.sched, backend=self.backend)
        fail_key = "fail_key"

        rabi.analysis.set_options(data_processor=DataProcessor(fail_key, []))
        # pylint: disable=no-member
        rabi.set_run_options(shots=2)
        data = rabi.run()
        result = data.analysis_results()

        self.assertEqual(data.status(), ExperimentStatus.ERROR)
        self.assertEqual(len(result), 0)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = Rabi([self.qubit], self.sched)
        loaded_exp = Rabi.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    @unittest.skip("Schedules are not yet JSON serializable")
    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = Rabi([self.qubit], self.sched)
        self.assertRoundTripSerializable(exp)


class TestEFRabi(QiskitExperimentsTestCase):
    """Test the ef_rabi experiment."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        self.qubit = 0
        self.backend = SingleTransmonTestBackend(noise=False, atol=1e-4)
        self.anharmonicity = self.backend.anharmonicity
        with pulse.build(name="x") as sched:
            with pulse.frequency_offset(self.anharmonicity, pulse.DriveChannel(self.qubit)):
                pulse.play(
                    pulse.Drag(160, Parameter("amp"), 40, 0.4), pulse.DriveChannel(self.qubit)
                )

        self.sched = sched

    # pylint: disable=no-member
    def test_ef_rabi_end_to_end(self):
        """Test the EFRabi experiment end to end."""

        test_tol = 0.05

        # Note that the backend is not sophisticated enough to simulate an e-f
        # transition so we run the test with a tiny frequency shift, still driving the e-g transition.
        rabi = EFRabi([self.qubit], self.sched, backend=self.backend)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.1, 0.1, 11))
        expdata = rabi.run()
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)

        self.assertEqual(result.quality, "good")
        self.assertTrue(abs(result.value.n - self.backend.rabi_rate_12[0]) < test_tol)

    def test_ef_rabi_circuit(self):
        """Test the EFRabi experiment end to end."""
        anharm = self.anharmonicity

        with pulse.build() as sched:
            pulse.shift_frequency(anharm, pulse.DriveChannel(2))
            pulse.play(pulse.Gaussian(160, Parameter("amp"), 40), pulse.DriveChannel(2))
            pulse.shift_frequency(-anharm, pulse.DriveChannel(2))

        rabi12 = EFRabi([2], sched)
        rabi12.set_experiment_options(amplitudes=[0.5])
        circ = rabi12.circuits()[0]

        with pulse.build() as expected:
            pulse.shift_frequency(anharm, pulse.DriveChannel(2))
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(2))
            pulse.shift_frequency(-anharm, pulse.DriveChannel(2))

        self.assertEqual(circ.calibrations["Rabi"][((2,), (0.5,))], expected)
        self.assertEqual(circ.data[0][0].name, "x")
        self.assertEqual(circ.data[1][0].name, "Rabi")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = EFRabi([0], self.sched)
        loaded_exp = EFRabi.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    @unittest.skip("Schedules are not yet JSON serializable")
    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = EFRabi([0], self.sched)
        self.assertRoundTripSerializable(exp)


class TestRabiCircuits(QiskitExperimentsTestCase):
    """Test the circuits generated by the experiment and the options."""

    def setUp(self):
        """Setup tests."""
        super().setUp()

        with pulse.build() as sched:
            pulse.play(pulse.Gaussian(160, Parameter("amp"), 40), pulse.DriveChannel(2))

        self.sched = sched

    def test_default_schedule(self):
        """Test the default schedule."""
        rabi = Rabi([2], self.sched)
        rabi.set_experiment_options(amplitudes=[0.5])
        circs = rabi.circuits()

        with pulse.build() as expected:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(2))

        self.assertEqual(circs[0].calibrations["Rabi"][((2,), (0.5,))], expected)
        self.assertEqual(len(circs), 1)

    def test_user_schedule(self):
        """Test the user given schedule."""

        amp = Parameter("my_double_amp")
        with pulse.build() as my_schedule:
            pulse.play(pulse.Drag(160, amp, 40, 10), pulse.DriveChannel(2))
            pulse.play(pulse.Drag(160, amp, 40, 10), pulse.DriveChannel(2))

        rabi = Rabi([2], self.sched)
        rabi.set_experiment_options(schedule=my_schedule, amplitudes=[0.5])
        circs = rabi.circuits()

        assigned_sched = my_schedule.assign_parameters({amp: 0.5}, inplace=False)
        self.assertEqual(circs[0].calibrations["Rabi"][((2,), (0.5,))], assigned_sched)

    def test_circuits_roundtrip_serializable(self):
        """Test circuits serialization of the experiment."""
        rabi = Rabi([2], self.sched)
        rabi.set_experiment_options(amplitudes=[0.5])
        self.assertRoundTripSerializable(rabi._transpiled_circuits())


class TestOscillationAnalysis(QiskitExperimentsTestCase):
    """Class to test the fitting."""

    def simulate_experiment_data(self, thetas, amplitudes, shots=1024):
        """Generate experiment data for Rx rotations with an arbitrary amplitude calibration."""
        circuits = []
        for theta in thetas:
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.measure_all()
            circuits.append(qc)

        sim = QasmSimulatorPy()
        circuits = transpile(circuits, sim)
        job = sim.run(circuits, shots=shots, seed_simulator=10)
        result = job.result()
        data = [
            {
                "counts": self._add_uncertainty(result.get_counts(i)),
                "metadata": {
                    "xval": amplitudes[i],
                    "meas_level": MeasLevel.CLASSIFIED,
                    "meas_return": "avg",
                },
            }
            for i, theta in enumerate(thetas)
        ]
        return data

    @staticmethod
    def _add_uncertainty(counts):
        """Ensure that we always have a non-zero sigma in the test."""
        for label in ["0", "1"]:
            if label not in counts:
                counts[label] = 1

        return counts

    def test_good_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(-np.pi, np.pi, 31)
        amplitudes = np.linspace(-0.25, 0.25, 31)
        expected_rate, test_tol = 2.0, 0.2

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=400))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        experiment_data = OscillationAnalysis().run(
            experiment_data, data_processor=data_processor, plot=False
        )
        result = experiment_data.analysis_results(0)
        self.assertEqual(result.quality, "good")
        self.assertAlmostEqual(result.value.params["freq"], expected_rate, delta=test_tol)

    def test_bad_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(0.0, np.pi / 4, 31)
        amplitudes = np.linspace(0.0, 0.95, 31)

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=200))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        experiment_data = OscillationAnalysis().run(
            experiment_data, data_processor=data_processor, plot=False
        )
        result = experiment_data.analysis_results()

        self.assertEqual(result[0].quality, "bad")


class TestCompositeExperiment(QiskitExperimentsTestCase):
    """Test composite Rabi experiment."""

    def test_calibrations(self):
        """Test that the calibrations are preserved and that the circuit transpiles."""

        experiments = []
        for qubit in range(3):
            with pulse.build() as sched:
                pulse.play(pulse.Gaussian(160, Parameter("amp"), 40), pulse.DriveChannel(qubit))

            experiments.append(Rabi([qubit], sched, amplitudes=[0.5]))

        par_exp = ParallelExperiment(experiments, flatten_results=False)
        par_circ = par_exp.circuits()[0]

        # If the calibrations are not there we will not be able to transpile
        try:
            transpile(par_circ, basis_gates=["rz", "sx", "x", "cx"])
        except QiskitError as error:
            self.fail("Failed to transpile with error: " + str(error))

        # Assert that the calibration keys are in the calibrations of the composite circuit.
        for qubit in range(3):
            rabi_circuit = experiments[qubit].circuits()[0]
            cal_key = next(iter(rabi_circuit.calibrations["Rabi"].keys()))

            self.assertEqual(cal_key[0], (qubit,))
            self.assertTrue(cal_key in par_circ.calibrations["Rabi"])
