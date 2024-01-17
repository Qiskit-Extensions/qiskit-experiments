# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for randomized benchmarking experiments."""
import copy

from test.base import QiskitExperimentsTestCase
from test.library.randomized_benchmarking.mixin import RBTestMixin
from ddt import ddt, data, unpack

from qiskit.circuit.library import SXGate
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeManilaV2
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_experiments.framework.composite import ParallelExperiment
from qiskit_experiments.library import randomized_benchmarking as rb


@ddt
class TestStandardRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for StandardRB without running the experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.backend = FakeManilaV2()

    # ### Tests for configuration ###
    @data(
        {"physical_qubits": [3, 3], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 3, 5, -7, 9], "num_samples": 1, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": -4, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 0, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 5, 5, 5, 9], "num_samples": 2, "seed": 100},
    )
    def test_invalid_configuration(self, configs):
        """Test raise error when creating experiment with invalid configs."""
        self.assertRaises(QiskitError, rb.StandardRB, **configs)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.StandardRB(physical_qubits=(0,), lengths=[10, 20, 30], seed=123)
        loaded_exp = rb.StandardRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.StandardRB(physical_qubits=(0,), lengths=[1, 3], seed=123)
        self.assertRoundTripSerializable(exp)

    def test_circuit_roundtrip_serializable(self):
        """Test circuits round trip JSON serialization"""
        exp = rb.StandardRB(physical_qubits=(0,), lengths=[1, 3], seed=123)
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.RBAnalysis()
        loaded = rb.RBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    # ### Tests for circuit generation ###
    @data([[3], 4], [[4, 7], 5], [[0, 1, 2], 3])
    @unpack
    def test_generate_circuits(self, qubits, length):
        """Test RB circuit generation"""
        exp = rb.StandardRB(physical_qubits=qubits, lengths=[length], num_samples=1)
        circuits = exp.circuits()
        self.assertAllIdentity(circuits)

    def test_return_same_circuit(self):
        """Test if setting the same seed returns the same circuits."""
        exp1 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
        )

        exp2 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())
        self.assertEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertEqual(circs1[2].decompose(), circs2[2].decompose())

    def test_full_sampling_single_qubit(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )
        exp2 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
        )
        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())

        # fully sampled circuits are regenerated while other is just built on top of previous length
        self.assertNotEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertNotEqual(circs1[2].decompose(), circs2[2].decompose())

    def test_full_sampling_2_qubits(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )

        exp2 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())

        # fully sampled circuits are regenerated while other is just built on top of previous length
        self.assertNotEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertNotEqual(circs1[2].decompose(), circs2[2].decompose())

    # ### Tests for transpiled circuit generation ###
    def test_calibrations_via_transpile_options(self):
        """Test if calibrations given as transpile_options show up in transpiled circuits."""
        qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_inst_map = InstructionScheduleMap()
        my_inst_map.add(SXGate(), qubits, my_sched)

        exp = rb.StandardRB(
            physical_qubits=qubits, lengths=[3], num_samples=4, backend=self.backend, seed=123
        )
        exp.set_transpile_options(inst_map=my_inst_map)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in qubits], [])))
            self.assertEqual(qc.calibrations["sx"][(qubits, tuple())], my_sched)

    def test_calibrations_via_custom_backend(self):
        """Test if calibrations given as custom backend show up in transpiled circuits."""
        qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_backend = copy.deepcopy(self.backend)
        my_backend.target["sx"][qubits].calibration = my_sched

        exp = rb.StandardRB(physical_qubits=qubits, lengths=[3], num_samples=4, backend=my_backend)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in qubits], [])))
            self.assertEqual(qc.calibrations["sx"][(qubits, tuple())], my_sched)

    def test_backend_with_directed_basis_gates(self):
        """Test if correct circuits are generated from backend with directed basis gates."""
        my_backend = copy.deepcopy(self.backend)
        del my_backend.target["cx"][(1, 2)]  # make support only cx(2, 1)

        exp = rb.StandardRB(physical_qubits=(2, 1), lengths=[3], num_samples=4, backend=my_backend)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.count_ops().get("cx", 0) > 0)
            expected_qubits = (qc.qubits[2], qc.qubits[1])
            for inst in qc:
                if inst.operation.name == "cx":
                    self.assertEqual(inst.qubits, expected_qubits)


class TestRunStandardRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for running StandardRB."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # depolarizing error
        self.p1q = 0.004
        self.p2q = 0.04
        self.pvz = 0.0
        self.pcz = 0.06

        # basis gates
        self.basis_gates = ["rz", "sx", "cx"]

        # setup noise model
        sx_error = depolarizing_error(self.p1q, 1)
        rz_error = depolarizing_error(self.pvz, 1)
        cx_error = depolarizing_error(self.p2q, 2)
        cz_error = depolarizing_error(self.pcz, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")
        noise_model.add_all_qubit_quantum_error(cz_error, "cz")

        self.noise_model = noise_model

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        self.transpiler_options = {
            "basis_gates": self.basis_gates,
        }

        # Aer simulator
        self.backend = AerSimulator(noise_model=noise_model, seed_simulator=123)

    def test_single_qubit(self):
        """Test single qubit RB."""
        exp = rb.StandardRB(
            physical_qubits=(0,),
            lengths=list(range(1, 300, 30)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given we have gate number per Clifford n_gpc, we can compute EPC as
        # EPC = 1 - (1 - r)^n_gpc
        # where r is gate error of SX gate, i.e. dep-parameter divided by 2.
        # We let transpiler use SX and RZ.
        # The number of physical gate per Clifford will distribute
        # from 0 to 2, i.e. arbitrary U gate can be decomposed into up to 2 SX with RZs.
        # We may want to expect the average number of SX is (0 + 1 + 2) / 3 = 1.0.
        epc = expdata.analysis_results("EPC")

        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_two_qubit(self):
        """Test two qubit RB. Use default basis gates."""
        exp = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        transpiler_options = {"optimization_level": 1}
        exp.set_transpile_options(**transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(4) can be decomposed with (0, 1, 2, 3) CX gates, the expected
        # average number of CX gate per Clifford is 1.5.
        # Since this is two qubit RB, the dep-parameter is factored by 3/4.
        epc = expdata.analysis_results("EPC")
        # Allow for 30 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)

    def test_three_qubit(self):
        """Test two qubit RB. Use default basis gates."""
        exp = rb.StandardRB(
            physical_qubits=(0, 1, 2),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(8) can be decomposed with [0,...,7] CX gates, the expected
        # average number of CX gate per Clifford is 3.5.
        # Since this is three qubit RB, the dep-parameter is factored by 7/8.
        epc = expdata.analysis_results("EPC")
        # Allow for 50 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 7 / 8 * self.p2q) ** 3.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.5 * epc_expected)

    def test_add_more_circuit_yields_lower_variance(self):
        """Test variance reduction with larger number of sampling."""

        # Increase single qubit error so that we can see gate error with a
        # small number of Cliffords since we want to run many samples without
        # taking too long.
        p1q = 0.15
        pvz = 0.0

        # setup noise model
        sx_error = depolarizing_error(p1q, 1)
        rz_error = depolarizing_error(pvz, 1)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")

        # Aer simulator
        backend = AerSimulator(noise_model=noise_model, seed_simulator=123)

        exp1 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=backend,
            num_samples=3,
        )
        exp1.analysis.set_options(gate_error_ratio=None)
        exp1.set_transpile_options(**self.transpiler_options)
        expdata1 = exp1.run()
        self.assertExperimentDone(expdata1)

        exp2 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=list(range(1, 30, 3)),
            seed=456,
            backend=backend,
            num_samples=30,
        )
        exp2.analysis.set_options(gate_error_ratio=None)
        exp2.set_transpile_options(**self.transpiler_options)
        expdata2 = exp2.run()
        self.assertExperimentDone(expdata2)

        self.assertLess(
            expdata2.analysis_results("EPC").value.s,
            expdata1.analysis_results("EPC").value.s,
        )

    def test_poor_experiment_result(self):
        """Test edge case that tail of decay is not sampled.

        This is a special case that fit outcome is very sensitive to initial guess.
        Perhaps generated initial guess is close to a local minima.
        """
        from qiskit.providers.fake_provider import FakeVigoV2

        backend = FakeVigoV2()
        backend.set_options(seed_simulator=123)
        # TODO: this test no longer makes sense (yields small reduced_chisq)
        #  after fixing how to call fake backend v2 (by adding the next line)
        # Need to call target before running fake backend v2 to load correct data
        self.assertLess(backend.target["sx"][(0,)].error, 0.001)

        exp = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[100, 200, 300],
            seed=123,
            backend=backend,
            num_samples=5,
        )
        exp.set_transpile_options(basis_gates=["x", "sx", "rz"], optimization_level=1)

        expdata = exp.run()
        self.assertExperimentDone(expdata)
        overview = expdata.analysis_results(0).value
        # This yields bad fit due to poor data points, but still fit is not completely off.
        self.assertLess(overview.reduced_chisq, 14)

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.StandardRB(
            physical_qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata)
        self.assertRoundTripPickle(expdata)

    def test_single_qubit_parallel(self):
        """Test single qubit RB in parallel."""
        physical_qubits = [0, 2]
        lengths = list(range(1, 300, 30))
        exps = []
        for qubit in physical_qubits:
            exp = rb.StandardRB(
                physical_qubits=[qubit], lengths=lengths, seed=123, backend=self.backend
            )
            exp.analysis.set_options(gate_error_ratio=None, plot_raw_data=False)
            exps.append(exp)

        par_exp = ParallelExperiment(exps, flatten_results=False)
        par_exp.set_transpile_options(**self.transpiler_options)

        par_expdata = par_exp.run(backend=self.backend)
        self.assertExperimentDone(par_expdata)
        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        for i in range(2):
            epc = par_expdata.child_data(i).analysis_results("EPC")
            self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_two_qubit_parallel(self):
        """Test two qubit RB in parallel."""
        qubit_pairs = [[0, 1], [2, 3]]
        lengths = list(range(1, 30, 3))
        exps = []
        for pair in qubit_pairs:
            exp = rb.StandardRB(
                physical_qubits=pair, lengths=lengths, seed=123, backend=self.backend
            )
            exp.analysis.set_options(gate_error_ratio=None, plot_raw_data=False)
            exps.append(exp)

        par_exp = ParallelExperiment(exps, flatten_results=False)
        par_exp.set_transpile_options(**self.transpiler_options)

        par_expdata = par_exp.run(backend=self.backend)
        self.assertExperimentDone(par_expdata)
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5
        for i in range(2):
            epc = par_expdata.child_data(i).analysis_results("EPC")
            # Allow for 30 percent tolerance since we ignore 1q gate contribution
            self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)

    def test_two_qubit_with_cz(self):
        """Test two qubit RB."""
        transpiler_options = {
            "basis_gates": ["sx", "rz", "cz"],
            "optimization_level": 1,
        }

        exp = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=list(range(1, 50, 5)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**transpiler_options)

        expdata = exp.run()
        self.assertAllIdentity(exp.circuits())
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(4) can be decomposed with (0, 1, 2, 3) CZ gates, the expected
        # average number of CZ gate per Clifford is 1.5.
        # Since this is two qubit RB, the dep-parameter is factored by 3/4.
        epc = expdata.analysis_results("EPC")

        # Allow for 30 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 3 / 4 * self.pcz) ** 1.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)
