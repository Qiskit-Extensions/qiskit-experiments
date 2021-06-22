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

from typing import Tuple
from test.mock_iq_backend import MockIQBackend
import numpy as np

from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.test import QiskitTestCase
from qiskit.qobj.utils import MeasLevel
import qiskit.pulse as pulse

from qiskit_experiments import ExperimentData
from qiskit_experiments.calibration.experiments.rabi import RabiAnalysis, Rabi
from qiskit_experiments.calibration.calibrations import Calibrations
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.calibration.update_library import Amplitude


class RabiBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the Rabi tests."""

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        amplitude_to_angle: float = np.pi,
    ):
        """Initialize the rabi backend."""
        self._amplitude_to_angle = amplitude_to_angle

        super().__init__(iq_cluster_centers, iq_cluster_width)

    @property
    def rabi_rate(self) -> float:
        """Returns the rabi rate."""
        return self._amplitude_to_angle / np.pi

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the rotation angle and amplitude_to_angle."""
        amp = next(iter(circuit.calibrations["Rabi"].keys()))[1][0]
        return np.sin(self._amplitude_to_angle * amp) ** 2


class TestRabiEndToEnd(QiskitTestCase):
    """Test the rabi experiment."""

    def test_rabi_end_to_end(self):
        """Test the Rabi experiment end to end."""

        test_tol = 0.01
        backend = RabiBackend()

        rabi = Rabi(3)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.95, 0.95, 21))
        result = rabi.run(backend).analysis_result(0)

        self.assertEqual(result["quality"], "computer_good")
        self.assertTrue(abs(result["popt"][1] - backend.rabi_rate) < test_tol)

        backend = RabiBackend(amplitude_to_angle=np.pi / 2)

        rabi = Rabi(3)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.95, 0.95, 21))
        result = rabi.run(backend).analysis_result(0)
        self.assertEqual(result["quality"], "computer_good")
        self.assertTrue(abs(result["popt"][1] - backend.rabi_rate) < test_tol)

        backend = RabiBackend(amplitude_to_angle=2.5 * np.pi)

        rabi = Rabi(3)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.95, 0.95, 101))
        result = rabi.run(backend).analysis_result(0)

        self.assertEqual(result["quality"], "computer_good")
        self.assertTrue(abs(result["popt"][1] - backend.rabi_rate) < test_tol)

    def test_calibrations_integration(self):
        """Test that we can update the value in calibrations."""

        cals = Calibrations()

        amp = Parameter("amp")
        chan = Parameter("ch0")
        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(duration=160, amp=amp, sigma=40), pulse.DriveChannel(chan))

        amp = Parameter("amp")
        with pulse.build(name="x90p") as x90p:
            pulse.play(pulse.Gaussian(duration=160, amp=amp, sigma=40), pulse.DriveChannel(chan))

        cals.add_schedule(xp)
        cals.add_schedule(x90p)

        backend = RabiBackend()

        rabi = Rabi(3)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.95, 0.95, 21))
        exp_data = rabi.run(backend)

        for qubit in [0, 3]:
            with self.assertRaises(CalibrationError):
                cals.get_schedule("xp", qubits=qubit)

        to_update = [(np.pi, "amp", "xp"), (np.pi / 2, "amp", x90p)]

        self.assertEqual(len(cals.parameters_table()), 0)

        Amplitude.update(cals, exp_data, angles_schedules=to_update)

        with self.assertRaises(CalibrationError):
            cals.get_schedule("xp", qubits=0)

        self.assertEqual(len(cals.parameters_table()), 2)

        # Now check the corresponding schedules
        result = exp_data.analysis_result(-1)
        rate = 2 * np.pi * result["popt"][1]
        amp = np.round(np.pi / rate, decimals=8)
        with pulse.build(name="xp") as expected:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(3))

        self.assertEqual(cals.get_schedule("xp", qubits=3), expected)

        amp = np.round(0.5 * np.pi / rate, decimals=8)
        with pulse.build(name="xp") as expected:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(3))

        self.assertEqual(cals.get_schedule("x90p", qubits=3), expected)


class TestRabiCircuits(QiskitTestCase):
    """Test the circuits generated by the experiment and the options."""

    def test_default_schedule(self):
        """Test the default schedule."""

        rabi = Rabi(2)
        rabi.set_experiment_options(amplitudes=[0.5])
        circs = rabi.circuits(RabiBackend())

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

        rabi = Rabi(2)
        rabi.set_experiment_options(schedule=my_schedule, amplitudes=[0.5])
        circs = rabi.circuits(RabiBackend())

        assigned_sched = my_schedule.assign_parameters({amp: 0.5}, inplace=False)
        self.assertEqual(circs[0].calibrations["Rabi"][((2,), (0.5,))], assigned_sched)


class TestRabiAnalysis(QiskitTestCase):
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
        result = execute(circuits, sim, shots=shots, seed_simulator=10).result()
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

        result = RabiAnalysis().run(experiment_data, data_processor=data_processor, plot=False)

        self.assertEqual(result[0]["quality"], "computer_good")
        self.assertTrue(abs(result[0]["popt"][1] - expected_rate) < test_tol)

    def test_bad_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(0.0, np.pi / 4, 31)
        amplitudes = np.linspace(0.0, 0.95, 31)

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=200))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        result = RabiAnalysis().run(experiment_data, data_processor=data_processor, plot=False)

        self.assertEqual(result[0]["quality"], "computer_bad")
