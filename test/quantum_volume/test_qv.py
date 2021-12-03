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
A Tester for the Quantum Volume experiment
"""
from test.base import QiskitExperimentsTestCase
import json
import os
from qiskit.quantum_info.operators.predicates import matrix_equal

from qiskit import Aer
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import QuantumVolume
from qiskit_experiments.framework import ExperimentDecoder

SEED = 42


class TestQuantumVolume(QiskitExperimentsTestCase):
    """Test Quantum Volume experiment"""

    def test_qv_circuits_length(self):
        """
        Test circuit generation - check the number of circuits generated
        and the amount of qubits in each circuit
        """

        qubits_lists = [[0, 1, 2], [0, 1, 2, 4]]
        ntrials = [2, 3, 5]

        for qubits in qubits_lists:
            for trials in ntrials:
                qv_exp = QuantumVolume(qubits)
                qv_exp.set_experiment_options(trials=trials)
                qv_circs = qv_exp.circuits()

                self.assertEqual(
                    len(qv_circs),
                    trials,
                    "Number of circuits generated does not match the number of trials",
                )

                self.assertEqual(
                    len(qv_circs[0].qubits),
                    qv_exp.num_qubits,
                    "Number of qubits in the Quantum Volume circuit does not match the"
                    " number of qubits in the experiment",
                )

    def test_qv_ideal_probabilities(self):
        """
        Test the probabilities of ideal circuit
        Compare between simulation and statevector calculation
        and compare to pre-calculated probabilities with the same seed
        """
        num_of_qubits = 3
        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        # set number of trials to a low number to make the test faster
        qv_exp.set_experiment_options(trials=20)
        qv_circs = qv_exp.circuits()
        simulation_probabilities = [qv_circ.metadata["ideal_probabilities"] for qv_circ in qv_circs]
        # create the circuits again, but this time disable simulation so the
        # ideal probabilities will be calculated using statevector
        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        qv_exp.set_experiment_options(trials=20)
        qv_exp._simulation_backend = None
        qv_circs = qv_exp.circuits()
        statevector_probabilities = [
            qv_circ.metadata["ideal_probabilities"] for qv_circ in qv_circs
        ]

        self.assertTrue(
            matrix_equal(simulation_probabilities, statevector_probabilities),
            "probabilities calculated using simulation and " "statevector are not the same",
        )
        # compare to pre-calculated probabilities
        dir_name = os.path.dirname(os.path.abspath(__file__))
        probabilities_json_file = "qv_ideal_probabilities.json"
        with open(os.path.join(dir_name, probabilities_json_file), "r") as json_file:
            probabilities = json.load(json_file, cls=ExperimentDecoder)
        self.assertTrue(
            matrix_equal(simulation_probabilities, probabilities),
            "probabilities calculated using simulation and "
            "pre-calculated probabilities are not the same",
        )

    def test_qv_sigma_decreasing(self):
        """
        Test that the sigma is decreasing after adding more trials
        """
        num_of_qubits = 3
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        # set number of trials to a low number to make the test faster
        qv_exp.set_experiment_options(trials=2)
        expdata1 = qv_exp.run(backend)
        result_data1 = expdata1.analysis_results(0)
        expdata2 = qv_exp.run(backend, analysis=False)
        expdata2.add_data(expdata1.data())
        qv_exp.analysis.run(expdata2)
        result_data2 = expdata2.analysis_results(0)

        self.assertTrue(result_data1.extra["trials"] == 2, "number of trials is incorrect")
        self.assertTrue(
            result_data2.extra["trials"] == 4,
            "number of trials is incorrect" " after adding more trials",
        )
        self.assertTrue(
            result_data2.value.stderr <= result_data1.value.stderr,
            "sigma did not decreased after adding more trials",
        )

    def test_qv_failure_insufficient_trials(self):
        """
        Test that the quantum volume is unsuccessful when:
            there is less than 100 trials
        """
        dir_name = os.path.dirname(os.path.abspath(__file__))
        insufficient_trials_json_file = "qv_data_70_trials.json"
        with open(os.path.join(dir_name, insufficient_trials_json_file), "r") as json_file:
            insufficient_trials_data = json.load(json_file, cls=ExperimentDecoder)

        num_of_qubits = 3
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        exp_data = ExperimentData(experiment=qv_exp, backend=backend)
        exp_data.add_data(insufficient_trials_data)

        qv_exp.analysis.run(exp_data)
        qv_result = exp_data.analysis_results(1)
        self.assertTrue(
            qv_result.extra["success"] is False and qv_result.value == 1,
            "quantum volume is successful with less than 100 trials",
        )

    def test_qv_failure_insufficient_hop(self):
        """
        Test that the quantum volume is unsuccessful when:
            there are more than 100 trials, but the heavy output probability mean is less than 2/3
        """
        dir_name = os.path.dirname(os.path.abspath(__file__))
        insufficient_hop_json_file = "qv_data_high_noise.json"
        with open(os.path.join(dir_name, insufficient_hop_json_file), "r") as json_file:
            insufficient_hop_data = json.load(json_file, cls=ExperimentDecoder)

        num_of_qubits = 4
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        exp_data = ExperimentData(experiment=qv_exp, backend=backend)
        exp_data.add_data(insufficient_hop_data)

        qv_exp.analysis.run(exp_data)
        qv_result = exp_data.analysis_results(1)
        self.assertTrue(
            qv_result.extra["success"] is False and qv_result.value == 1,
            "quantum volume is successful with heavy output probability less than 2/3",
        )

    def test_qv_failure_insufficient_confidence(self):
        """
        Test that the quantum volume is unsuccessful when:
            there are more than 100 trials, the heavy output probability mean is more than 2/3
            but the confidence is not high enough
        """
        dir_name = os.path.dirname(os.path.abspath(__file__))
        insufficient_confidence_json = "qv_data_moderate_noise_100_trials.json"
        with open(os.path.join(dir_name, insufficient_confidence_json), "r") as json_file:
            insufficient_confidence_data = json.load(json_file, cls=ExperimentDecoder)

        num_of_qubits = 4
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        exp_data = ExperimentData(experiment=qv_exp, backend=backend)
        exp_data.add_data(insufficient_confidence_data)

        qv_exp.analysis.run(exp_data)
        qv_result = exp_data.analysis_results(1)
        self.assertTrue(
            qv_result.extra["success"] is False and qv_result.value == 1,
            "quantum volume is successful with insufficient confidence",
        )

    def test_qv_success(self):
        """
        Test a successful run of quantum volume.
        Compare the results to a pre-run experiment
        """
        dir_name = os.path.dirname(os.path.abspath(__file__))
        successful_json_file = "qv_data_moderate_noise_300_trials.json"
        with open(os.path.join(dir_name, successful_json_file), "r") as json_file:
            successful_data = json.load(json_file, cls=ExperimentDecoder)

        num_of_qubits = 4
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
        exp_data = ExperimentData(experiment=qv_exp, backend=backend)
        exp_data.add_data(successful_data)

        qv_exp.analysis.run(exp_data)
        results_json_file = "qv_result_moderate_noise_300_trials.json"
        with open(os.path.join(dir_name, results_json_file), "r") as json_file:
            successful_results = json.load(json_file, cls=ExperimentDecoder)

        results = exp_data.analysis_results()
        for result, reference in zip(results, successful_results):
            self.assertEqual(
                result.value,
                reference["value"],
                "result value is not the same as precalculated analysis",
            )
            self.assertEqual(
                result.name,
                reference["name"],
                "result name is not the same as precalculated analysis",
            )
            for key, value in reference["extra"].items():
                if isinstance(value, float):
                    self.assertAlmostEqual(
                        result.extra[key],
                        value,
                        msg="result " + str(key) + " is not the same as the "
                        "pre-calculated analysis",
                    )
                else:
                    self.assertTrue(
                        result.extra[key] == value,
                        "result " + str(key) + " is not the same as the " "pre-calculated analysis",
                    )

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = QuantumVolume([0, 1, 2], seed=42)
        loaded_exp = QuantumVolume.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = QuantumVolume([0, 1, 2], seed=42)
        self.assertRoundTripSerializable(exp, self.experiments_equiv)
