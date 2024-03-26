# -*- coding: utf-8 -*-
#
# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A Tester for the GST experiment
"""
import unittest

import numpy as np
import ast
from qiskit_experiments.library.gateset_tomography.gst_experiment import GateSetTomography
from qiskit_experiments.library.gateset_tomography.gatesetbasis import (
    default_gateset_basis,
    gatesetbasis_constrction,
)
from qiskit.extensions import HGate, SXGate, XGate, U2Gate, YGate, IGate
from qiskit.test import QiskitTestCase
from qiskit_experiments.exceptions import QiskitError
from qiskit import Aer
from qiskit.quantum_info import PTM, Choi
import json
from qiskit.test.mock import FakeParis
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel


class TestGatesetTomography(unittest.TestCase):
    """Test GateSetTomography"""

    def gst_experiment(
        self,
        gateset="default",
        backend=Aer.get_backend("aer_simulator"),
        noise_model=None,
        qubits=[0],
        fitter="default",
        rescale_TP=False,
        rescale_CP=False,
        only_basis_gates=False,
    ):
        """Runs gst experiment and returns the GST results data"""

        gstexp = GateSetTomography(
            qubits=qubits, gateset=gateset, only_basis_gates=only_basis_gates
        )
        gstexp.set_analysis_options(fitter=fitter, rescale_TP=rescale_TP, rescale_CP=rescale_CP)
        gstdata = gstexp.run(backend=backend, noise_model=noise_model).block_for_results()
        return gstdata

    @staticmethod
    def hs_distance(A, B):
        return sum([np.abs(x) ** 2 for x in np.nditer(A - B)])

    def compare_gates(self, expected_gates, result_gates, labels, delta=0.1):
        for label in labels:
            expected_gate = expected_gates[label]
            result_gate = result_gates[label]

            distance = self.hs_distance(expected_gate, result_gate)
            d = sum(len(row) for row in result_gate)

            msg = (
                "distance ={}, distance/d ={}, \n Failure on gate {}: Expected gate = \n{}\n"
                "vs Actual gate = \n{}".format(
                    distance, distance / d, label, expected_gate, result_gate
                )
            )

            self.assertAlmostEqual(distance / d, 0, delta=delta, msg=msg)

    def test_spam_gates_construction(self):
        """Tests SPAM gates construction from only basis gates"""

        # single qubit basis gates
        basis_1qubits_2 = {
            "I": lambda circ, qubit: None,
            "H": lambda circ, qubit: circ.append(HGate(), [qubit]),
            "Y": lambda circ, qubit: circ.append(YGate(), [qubit]),
        }
        basis_1qubits_1 = {
            "I": lambda circ, qubit: None,
            "H": lambda circ, qubit: circ.append(HGate(), [qubit]),
            "Y": lambda circ, qubit: circ.append(YGate(), [qubit]),
            "SX": lambda circ, qubit: circ.append(SXGate(), [qubit]),
        }
        # 2-qubits basis gates
        basis_2qubits_1 = {
            "I I": lambda circ, qubit1, qubit2: None,
            "X I": lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit2]),
            "Y I": lambda circ, qubit1, qubit2: circ.append(YGate(), [qubit2]),
            "I X": lambda circ, qubit1, qubit2: circ.append(XGate(), [qubit1]),
            "I Y": lambda circ, qubit1, qubit2: circ.append(YGate(), [qubit1]),
            "H I": lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit2]),
            "I H": lambda circ, qubit1, qubit2: circ.append(HGate(), [qubit1]),
            "I SX": lambda circ, qubit1, qubit2: circ.append(SXGate(), [qubit1]),
            "SX I": lambda circ, qubit1, qubit2: circ.append(SXGate(), [qubit2]),
        }
        # expected_spam_gates
        spam_1qubits_1 = [["I"], ["H"], ["Y"], ["Y", "SX"]]
        spam_2qubits_1 = [
            ["I I"],
            ["X I"],
            ["I X"],
            ["H I"],
            ["I H"],
            ["X I", "I X"],
            ["X I", "I H"],
            ["Y I", "SX I"],
            ["I X", "H I"],
            ["I Y", "I SX"],
            ["H I", "I H"],
            ["X I", "I Y", "I SX"],
            ["Y I", "I X", "SX I"],
            ["Y I", "I H", "SX I"],
            ["I Y", "H I", "I SX"],
            ["Y I", "I Y", "I SX", "SX I"],
        ]

        msg = "Spam construction failed"

        # checking cases construction should be ok
        spam1 = [
            list(i) for i in gatesetbasis_constrction(basis_1qubits_1, 1)[0].spam_spec.values()
        ]
        spam2 = [
            list(i) for i in gatesetbasis_constrction(basis_2qubits_1, 2)[0].spam_spec.values()
        ]
        for i in range(len(spam_1qubits_1)):
            for j in range(len(spam_1qubits_1[i])):
                self.assertEqual(spam1[i][j], spam_1qubits_1[i][j], msg=msg)
        for i in range(len(spam_2qubits_1)):
            for j in range(len(spam_2qubits_1[i])):
                self.assertEqual(spam2[i][j], spam_2qubits_1[i][j], msg=msg)

        # checking cases construction should fail
        self.assertRaises(QiskitError, gatesetbasis_constrction, basis_1qubits_2, 1)

    def test_linear_inversion_noiseless_single_qubit(self):
        """Tests linear inversion for a noiseless single qubit"""

        gateset = default_gateset_basis(1)
        gstdata = self.gst_experiment(fitter="linear_inversion_gst")
        result_gates = {}
        for label in gateset.gate_labels:
            result_gates[label] = np.real(
                PTM(gstdata.analysis_results("gst estimation of " + label).value).data
            )
        with open("1qubit_default_linv.json") as json_file:
            data = json.load(json_file)
            self.compare_gates(data, result_gates, data.keys())

    def test_linear_inversion_noiseless_two_qubits(self):
        """Tests linear inversion for noiseless two qubits"""

        gateset = default_gateset_basis(2)
        gstdata = self.gst_experiment(qubits=[0, 1], fitter="linear_inversion_gst")
        # result_gates = {}
        result_gates_without_E_rho = {}
        for label in gateset.gate_labels:
            i = gstdata.analysis_results("gst estimation of " + label).value
            if label not in ["E", "rho"]:
                result_gates_without_E_rho[label] = np.real(PTM(i).data)

        with open("2qubits_default_linv.json") as json_file:
            data = json.load(json_file)
            self.compare_gates(data, result_gates_without_E_rho, data.keys())

    def test_linear_inversion_noiseless_single_qubit_rescale(self):
        """Tests rescaling linear inversion results for a noiseless single qubit"""

        gateset = default_gateset_basis(1)
        gstdata = self.gst_experiment(
            fitter="linear_inversion_gst", rescale_TP=True, rescale_CP=True
        )
        for label in gateset.gate_labels:
            result_gate_ptm = np.real(
                PTM(gstdata.analysis_results("gst estimation of " + label).value).data
            )
            result_gate_choi = np.real(gstdata.analysis_results("gst estimation of " + label).value)
            rescale_tp = result_gate_ptm[0][0]
            rescale_cp = np.trace(result_gate_choi) / 2
            self.assertAlmostEqual(
                rescale_tp, 1, delta=0.01, msg="GST result for gate " + label + " is not TP"
            )
            self.assertAlmostEqual(
                rescale_cp, 1, delta=0.01, msg="GST result for gate " + label + " is not CP"
            )

    def test_mle_results_noiseless(self):
        """Tests MLE fitter results for a noiseless single qubit"""

        gateset = default_gateset_basis(1)
        gstdata = self.gst_experiment()
        fid_threshold = 0.95
        result_gates = {}
        for label in gateset.gate_labels:
            result_gates[label] = np.real(
                PTM(gstdata.analysis_results("gst estimation of " + label).value).data
            )
            fid = gstdata.analysis_results("gst estimation of " + label).extra[
                "Average gate fidelity"
            ]
            self.assertGreater(fid, fid_threshold, msg="fit fidelity is low")
        with open("1qubits_default_mle_noiseless.json") as json_file:
            data = json.load(json_file)
            self.compare_gates(data, result_gates, data.keys())

    def test_mle_results_fake_backend(self):
        """Tests MLE fitter results for a single qubit subjected to a fake noise"""

        backend_fakeparis = AerSimulator.from_backend(FakeParis())
        gateset = default_gateset_basis(1)
        gstdata = self.gst_experiment(backend=backend_fakeparis)
        fid_threshold = 0.95
        result_gates = {}
        for label in gateset.gate_labels:
            result_gates[label] = np.real(
                PTM(gstdata.analysis_results("gst estimation of " + label).value).data
            )
            fid = gstdata.analysis_results("gst estimation of " + label).extra[
                "Average gate fidelity"
            ]
            self.assertGreater(fid, fid_threshold, msg="fidelity of gate " + label + " is low")
        with open("1qubits_default_mle_fakeparis.json") as json_file:
            data = json.load(json_file)
            self.compare_gates(data, result_gates, data.keys())

    def test_mle_results_AD_noise(self):
        """Tests MLE fitter results for a single qubit subjected to an amplitude damping noise"""

        gateset = {
            "I": lambda circ, qubit: None,
            "H": lambda circ, qubit: circ.append(HGate(), [qubit]),
            "Y": lambda circ, qubit: circ.append(YGate(), [qubit]),
            "SX": lambda circ, qubit: circ.append(SXGate(), [qubit]),
        }
        # Noise model: Amplitude damping applied on each qubit after each gate. Th PTM representation of
        # amplitude damping channel:
        gamma = 0.05
        fid_threshold = 0.95
        noise_ptm_AD = PTM(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.sqrt(1 - gamma), 0, 0],
                    [0, 0, np.sqrt(1 - gamma), 0],
                    [gamma, 0, 0, 1 - gamma],
                ]
            )
        )
        target_set = {
            "I": Choi(IGate()),
            "H": Choi(HGate()),
            "Y": Choi(YGate()),
            "SX": Choi(SXGate()),
        }
        # As the gates after the noise in PTM representation is simply the multiplication of the noise
        # channel by each gate channel, the noisy target set can be obtained in the PTM representation via:
        target_set_noisy = {}
        for key in target_set:
            if key != "I":
                target_set_noisy[key] = np.real(
                    PTM(np.dot(noise_ptm_AD, PTM(target_set[key]).data)).data
                )

        # Noise model
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise_ptm_AD, ["sx", "x", "rz"])

        # GST experiment:
        gstdata = self.gst_experiment(
            gateset=gateset, noise_model=noise_model, only_basis_gates=True
        )

        result_gates = {}
        for label in target_set_noisy.keys():
            result_gates[label] = np.real(
                PTM(gstdata.analysis_results("gst estimation of " + label).value).data
            )
            fid = gstdata.analysis_results("gst estimation of " + label).extra[
                "Average gate fidelity"
            ]
            self.assertGreater(fid, fid_threshold, msg="fidelity of gate " + label + " is low")
        self.compare_gates(target_set_noisy, result_gates, target_set_noisy.keys())

    def test_mle_results_DC_noise(self):
        """Tests MLE fitter results for a single qubit subjected to depolarization noise"""

        gateset = {
            "I": lambda circ, qubit: None,
            "H": lambda circ, qubit: circ.append(HGate(), [qubit]),
            "Y": lambda circ, qubit: circ.append(YGate(), [qubit]),
            "SX": lambda circ, qubit: circ.append(SXGate(), [qubit]),
        }
        # Noise model: DC applied on each qubit after each gate. Th PTM representation of
        # DC:
        gamma = 0.05
        fid_threshold = 0.95
        noise_ptm_DC = PTM(
            np.array(
                [[1, 0, 0, 0], [0, 1 - gamma, 0, 0], [0, 0, 1 - gamma, 0], [0, 0, 0, 1 - gamma]]
            )
        )

        target_set = {
            "I": Choi(IGate()),
            "H": Choi(HGate()),
            "Y": Choi(YGate()),
            "SX": Choi(SXGate()),
        }

        # As the gates after the noise in PTM representation is simply the multiplication of the noise
        # channel by each gate channel, the noisy target set can be obtained in the PTM representation via:
        target_set_noisy = {}
        for key in target_set:
            if key != "I":
                target_set_noisy[key] = np.real(
                    PTM(np.dot(noise_ptm_DC, PTM(target_set[key]).data)).data
                )

        # Noise model
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise_ptm_DC, ["sx", "x", "rz"])

        # GST experiment:
        gstdata = self.gst_experiment(
            gateset=gateset, noise_model=noise_model, only_basis_gates=True
        )
        result_gates = {}
        for label in target_set_noisy.keys():
            result_gates[label] = np.real(
                PTM(gstdata.analysis_results("gst estimation of " + label).value).data
            )
            fid = gstdata.analysis_results("gst estimation of " + label).extra[
                "Average gate fidelity"
            ]
            self.assertGreater(fid, fid_threshold, msg="fidelity of gate " + label + " is low")
        self.compare_gates(target_set_noisy, result_gates, target_set_noisy.keys())


if __name__ == "__main__":
    unittest.main()
