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
Test readout angle experiment
"""

from test.base import QiskitExperimentsTestCase
from typing import Dict, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit_experiments.library import ReadoutAngle
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


def compute_probability_half_angle(circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
    """Returns the probability based on the beta, number of gates, and leakage."""

    output_dict_list = []
    for circuit in circuits:
        probability_output_dict = {"1": 1 - circuit.metadata["xval"]}
        probability_output_dict["0"] = 1 - probability_output_dict["1"]
        output_dict_list.append(probability_output_dict)
    return output_dict_list


class TestReadoutAngle(QiskitExperimentsTestCase):
    """
    Test the readout angle experiment
    """

    def test_readout_angle_end2end(self):
        """
        Test readout angle experiment using a simulator.
        """

        backend = MockIQBackend(
            iq_cluster_centers=[((-3.0, 3.0), (5.0, 5.0))],
            compute_probabilities=compute_probability_half_angle,
        )
        exp = ReadoutAngle(0)
        expdata = exp.run(backend, shots=100000)
        self.assertExperimentDone(expdata)
        res = expdata.analysis_results(0)
        self.assertAlmostEqual(res.value % (2 * np.pi), np.pi / 2, places=2)

        backend = MockIQBackend(
            iq_cluster_centers=[((0, -3.0), (5.0, 5.0))],
            compute_probabilities=compute_probability_half_angle,
        )
        exp = ReadoutAngle(0)
        expdata = exp.run(backend, shots=100000)
        self.assertExperimentDone(expdata)
        res = expdata.analysis_results(0)
        self.assertAlmostEqual(res.value % (2 * np.pi), 15 * np.pi / 8, places=2)
