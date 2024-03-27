# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for layer fidelity experiments."""
from test.base import QiskitExperimentsTestCase
from test.library.randomized_benchmarking.mixin import RBTestMixin
import copy
import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit.library import SXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity, LayerFidelityAnalysis


@ddt
class TestLayerFidelity(QiskitExperimentsTestCase, RBTestMixin):
    """Test for LayerFidelity without running the experiments."""

    # ### Tests for configuration ###
    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[10, 20, 30],
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["rz", "sx", "x"],
        )
        loaded_exp = LayerFidelity.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_invalid_two_qubit_layers(self):
        """Test raise error when creating experiment with invalid configs."""
        valid_kwargs = {
            "lengths": [10, 20, 30],
            "two_qubit_gate": "cx",
            "one_qubit_basis_gates": ["rz", "sx", "x"],
        }
        # not disjoit
        with self.assertRaises(QiskitError):
            LayerFidelity(
                physical_qubits=(0, 1, 2, 3), two_qubit_layers=[[(0, 1), (1, 2)]], **valid_kwargs
            )
        # no 2q-gate on the qubits (FakeManilaV2 has no cx gate on (0, 3))
        with self.assertRaises(QiskitError):
            LayerFidelity(
                physical_qubits=(0, 1, 2, 3),
                two_qubit_layers=[[(0, 3)]],
                backend=FakeManilaV2(),
                **valid_kwargs,
            )

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[10, 20, 30],
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["rz", "sx", "x"],
        )
        self.assertRoundTripSerializable(exp, strict_type=False)

    def test_circuit_roundtrip_serializable(self):
        """Test circuits round trip JSON serialization"""
        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[10, 20, 30],
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["rz", "sx", "x"],
        )
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = LayerFidelityAnalysis(layers=[[(1, 0), (2, 3)], [(1, 2), (0,), (3,)]])
        loaded = LayerFidelityAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    # ### Tests for circuit generation ###
    @data(
        [(1, 2), [[(1, 2)]]],
        [(1, 3, 4), [[(3, 4)]]],
        [(4, 3, 2, 1, 0), [[(0, 1), (3, 2)], [(1, 2), (3, 4)]]],
    )
    @unpack
    def test_generate_circuits(self, qubits, two_qubit_layers):
        """Test RB circuit generation"""
        exp = LayerFidelity(
            physical_qubits=qubits,
            two_qubit_layers=two_qubit_layers,
            lengths=[1, 2, 3],
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["rz", "sx", "x"],
        )
        circuits = exp.circuits()
        self.assertAllIdentity(circuits)

    def test_return_same_circuit_for_same_config(self):
        """Test if setting the same seed returns the same circuits."""
        exp1 = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[10, 20, 30],
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["rz", "sx", "x"],
        )

        exp2 = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[10, 20, 30],
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["rz", "sx", "x"],
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())
        self.assertEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertEqual(circs1[2].decompose(), circs2[2].decompose())

    # ### Tests for transpiled circuit generation ###
    def test_calibrations_via_custom_backend(self):
        """Test if calibrations given as custom backend show up in transpiled circuits."""
        qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_backend = copy.deepcopy(FakeManilaV2())
        my_backend.target["sx"][qubits].calibration = my_sched

        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[10, 20, 30],
            seed=42,
            backend=my_backend,
        )
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in qubits], [])))
            self.assertEqual(qc.calibrations["sx"][(qubits, tuple())], my_sched)

    def test_backend_with_directed_basis_gates(self):
        """Test if correct circuits are generated from backend with directed basis gates."""
        my_backend = copy.deepcopy(FakeManilaV2())
        del my_backend.target["cx"][(1, 2)]  # make cx on {1, 2} one-sided

        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(2, 1)]],
            lengths=[10, 20, 30],
            seed=42,
            num_samples=1,
            backend=my_backend,
        )
        transpiled = exp._transpiled_circuits()
        for qc in transpiled[3:]:  # check only the second layer
            self.assertTrue(qc.count_ops().get("cx", 0) > 0)
            expected_qubits = (qc.qubits[2], qc.qubits[1])
            for inst in qc:
                if inst.operation.name == "cx":
                    self.assertEqual(inst.qubits, expected_qubits)


class TestRunLayerFidelity(QiskitExperimentsTestCase, RBTestMixin):
    """Test for running LayerFidelity on noisy simulator."""

    def test_run_layer_fidelity(self):
        """Test layer fidelity RB. Use default basis gates."""
        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[1, 4, 16, 64, 256],
            seed=42,
            backend=FakeManilaV2(),
        )
        expdata = exp.run()
        self.assertExperimentDone(expdata)

        lf = expdata.analysis_results("LF").value.n
        slfs = [res.value.n for res in expdata.analysis_results("SingleLF")]
        self.assertAlmostEqual(lf, np.prod(slfs))

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = LayerFidelity(
            physical_qubits=(0, 1, 2, 3),
            two_qubit_layers=[[(1, 0), (2, 3)], [(1, 2)]],
            lengths=[1, 4, 16, 64, 256],
            seed=42,
            backend=FakeManilaV2(),
        )
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata)
        self.assertRoundTripPickle(expdata)
