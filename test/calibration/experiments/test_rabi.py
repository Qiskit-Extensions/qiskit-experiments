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
import numpy as np

from qiskit import QuantumCircuit, execute
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.test import QiskitTestCase
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments import ExperimentData
from qiskit_experiments.calibration.experiments.rabi import RabiAnalysis, Rabi
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.test.mock_iq_backend import IQTestBackend


class RabiBackend(IQTestBackend):
    """A simple and primitive backend, to be run by the T1 tests."""

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        amplitude_to_angle=np.pi,
    ):
        """Initialize the rabi backend."""
        self._amplitude_to_angle = amplitude_to_angle

        super().__init__(iq_cluster_centers, iq_cluster_width)

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the rotation angle and amplitude_to_angle."""
        amp = next(iter(circuit.calibrations["rx"].keys()))[1][0]
        return np.sin(self._amplitude_to_angle * amp) ** 2


class TestRabiEndToEnd(QiskitTestCase):
    """Test the rabi experiment."""

    def test_rabi_end_to_end(self):
        """Test the Rabi experiment end to end."""

        backend = RabiBackend()

        rabi = Rabi(3, np.linspace(-0.95, 0.95, 21))
        result = rabi.run(backend).analysis_result(0)

        self.assertEqual(result["quality"], "computer_good")


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
        amplitudes = np.linspace(-np.pi / 4, np.pi / 4, 31)

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=400))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        result = RabiAnalysis().run(experiment_data, data_processor=data_processor, plot=False)

        self.assertEqual(result["quality"], "computer_good")
        self.assertTrue(3.9 < result["value"] < 4.1)

    def test_bad_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(0.0, np.pi / 4, 31)
        amplitudes = np.linspace(0.0, 0.95, 31)

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=200))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        result = RabiAnalysis().run(experiment_data, data_processor=data_processor, plot=False)

        self.assertEqual(result["quality"], "computer_bad")
