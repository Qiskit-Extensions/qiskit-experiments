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
Test T1 experiment
"""

from test.base import QiskitExperimentsTestCase
import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
from qiskit_experiments.test.noisy_delay_aer_simulator import NoisyDelayAerBackend
from qiskit_experiments.framework import ExperimentData, ParallelExperiment
from qiskit_experiments.library import T1
from qiskit_experiments.library.characterization import T1Analysis, T1KerneledAnalysis
from qiskit_experiments.test.mock_iq_backend import MockIQBackend, MockIQParallelBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQT1Helper, MockIQParallelExperimentHelper


class TestT1(QiskitExperimentsTestCase):
    """
    Test measurement of T1
    """

    def test_t1_end2end(self):
        """
        Test T1 experiment using a simulator.
        """
        t1 = 25e-6
        backend = NoisyDelayAerBackend([t1], [t1 / 2])

        delays = np.arange(1e-6, 40e-6, 3e-6)
        exp = T1([0], delays)

        exp.analysis.set_options(p0={"amp": 1, "tau": t1, "base": 0})
        exp_data = exp.run(backend, shots=10000, seed_simulator=1)
        self.assertExperimentDone(exp_data)
        self.assertRoundTripSerializable(exp_data)
        self.assertRoundTripPickle(exp_data)
        res = exp_data.analysis_results("T1")
        self.assertEqual(res.quality, "good")
        self.assertAlmostEqual(res.value.n, t1, delta=3)
        self.assertEqual(res.extra["unit"], "s")

    def test_t1_measurement_level_1(self):
        """
        Test T1 experiment using a simulator.
        """

        ns = 1e-9
        mu = 1e-6
        t1 = 45 * mu

        # delays
        delays = np.logspace(1, 11, num=23, base=np.exp(1))
        delays *= ns
        delays = np.insert(delays, 0, 0)
        delays = np.append(delays, [t1 * 3])

        num_shots = 4096
        backend = MockIQBackend(
            MockIQT1Helper(
                t1=t1,
                iq_cluster_centers=[((-5.0, -4.0), (-5.0, 4.0)), ((3.0, 1.0), (5.0, -3.0))],
                iq_cluster_width=[1.0, 2.0],
            )
        )

        # Experiment initialization and analysis options
        exp0 = T1([0], delays)
        exp0.analysis = T1KerneledAnalysis()

        exp0.analysis.set_options(p0={"amp": 1, "tau": t1, "base": 0})
        expdata0 = exp0.run(
            backend=backend,
            meas_return="avg",
            meas_level=MeasLevel.KERNELED,
            shots=num_shots,
        )
        self.assertExperimentDone(expdata0)

        self.assertRoundTripSerializable(expdata0)
        self.assertRoundTripPickle(expdata0)

        res = expdata0.analysis_results("T1")
        self.assertEqual(res.quality, "good")
        self.assertAlmostEqual(res.value.n, t1, delta=3)
        self.assertEqual(res.extra["unit"], "s")

    def test_t1_parallel(self):
        """
        Test parallel experiments of T1 using a simulator.
        """

        t1 = [25, 20, 15]
        t2 = [value / 2 for value in t1]
        delays = list(range(1, 40, 3))
        qubit0 = 0
        qubit2 = 2

        quantum_bit = [qubit0, qubit2]

        backend = NoisyDelayAerBackend(t1, t2)

        exp0 = T1(physical_qubits=[qubit0], delays=delays)
        exp2 = T1(physical_qubits=[qubit2], delays=delays)

        par_exp = ParallelExperiment([exp0, exp2], flatten_results=False)
        res = par_exp.run(backend=backend, shots=10000, seed_simulator=1)
        self.assertExperimentDone(res)

        for i, qb in enumerate(quantum_bit):
            sub_res = res.child_data(i).analysis_results("T1")
            self.assertEqual(sub_res.quality, "good")
            self.assertAlmostEqual(sub_res.value.n, t1[qb], delta=3)

    def test_t1_parallel_measurement_level_1(self):
        """
        Test parallel experiments of T1 using a simulator.
        """

        ns = 1e-9
        mu = 1e-6
        t1s = [25 * mu, 20 * mu]
        qubits = [0, 1]
        num_shots = 4096

        # Delays
        delays = np.logspace(1, 11, num=23, base=np.exp(1))
        delays *= ns
        delays = np.insert(delays, 0, 0)
        delays = np.append(delays, [t1s[0] * 3])

        par_exp_list = []
        exp_helpers = []
        for qidx, t1 in zip(qubits, t1s):
            # Experiment
            exp = T1(physical_qubits=[qidx], delays=delays)
            exp.analysis = T1KerneledAnalysis()
            par_exp_list.append(exp)

            # Helper
            helper = MockIQT1Helper(
                t1=t1,
                iq_cluster_centers=[
                    ((-5.0, -4.0), (-5.0, 4.0)),
                    ((-1.0, -1.0), (1.0, 1.0)),
                    ((4.0, 1.0), (6.0, -3.0)),
                ],
                iq_cluster_width=[1.0, 2.0, 1.0],
            )
            exp_helpers.append(helper)

        par_exp = ParallelExperiment(
            par_exp_list,
            flatten_results=False,
        )
        par_helper = MockIQParallelExperimentHelper(
            exp_list=par_exp_list,
            exp_helper_list=exp_helpers,
        )

        # Backend
        backend = MockIQParallelBackend(par_helper, rng_seed=1)

        # Running experiment
        res = par_exp.run(
            backend=backend,
            shots=num_shots,
            rng_seed=1,
            meas_level=MeasLevel.KERNELED,
            meas_return="avg",
        )
        self.assertExperimentDone(res)

        # Checking analysis
        for i, t1 in enumerate(t1s):
            sub_res = res.child_data(i).analysis_results("T1")
            self.assertEqual(sub_res.quality, "good")
            self.assertAlmostEqual(sub_res.value.n, t1, delta=3)

    def test_t1_analysis(self):
        """
        Test T1Analysis
        """

        data = ExperimentData()
        data.metadata.update({"meas_level": 2})

        numbers = [750, 1800, 2750, 3550, 4250, 4850, 5450, 5900, 6400, 6800, 7000, 7350, 7700]

        for i, count0 in enumerate(numbers):
            data.add_data(
                {
                    "counts": {"0": count0, "1": 10000 - count0},
                    "metadata": {"xval": (3 * i + 1) * 1e-9},
                }
            )

        experiment_data = T1Analysis().run(data, plot=False)
        result = experiment_data.analysis_results("T1")

        self.assertEqual(result.quality, "good")
        self.assertAlmostEqual(result.value.nominal_value, 25e-9, delta=3)

    def test_t1_metadata(self):
        """
        Test the circuits metadata
        """

        delays = np.arange(1e-3, 40e-3, 3e-3)
        exp = T1([0], delays)
        circs = exp.circuits()

        self.assertEqual(len(circs), len(delays))

        for delay, circ in zip(delays, circs):
            # xval is rounded to nealest granularity value.
            self.assertAlmostEqual(circ.metadata["xval"], delay)

    def test_t1_low_quality(self):
        """
        A test where the fit's quality will be low
        """

        data = ExperimentData()
        data.metadata.update({"meas_level": 2})

        for i in range(10):
            data.add_data(
                {
                    "counts": {"0": 10, "1": 10},
                    "metadata": {"xval": i * 1e-9},
                }
            )

        experiment_data = T1Analysis().run(data, plot=False)
        result = experiment_data.analysis_results("T1")
        self.assertEqual(result.quality, "bad")

    def test_t1_parallel_exp_transpile(self):
        """Test parallel transpile options for T1 experiment"""
        num_qubits = 5
        instruction_durations = []
        for i in range(num_qubits):
            instruction_durations += [
                ("rx", [i], (i + 1) * 10, "ns"),
                ("measure", [i], (i + 1) * 1000, "ns"),
            ]
        coupling_map = [[i - 1, i] for i in range(1, num_qubits)]
        basis_gates = ["rx", "delay"]

        exp1 = T1([1], delays=[50e-9, 100e-9, 160e-9])
        exp2 = T1([3], delays=[40e-9, 80e-9, 190e-9])
        parexp = ParallelExperiment([exp1, exp2], flatten_results=False)
        parexp.set_transpile_options(
            basis_gates=basis_gates,
            instruction_durations=instruction_durations,
            coupling_map=coupling_map,
            scheduling_method="alap",
        )

        circs = parexp.circuits()
        for circ in circs:
            self.assertEqual(circ.num_qubits, 2)
            op_counts = circ.count_ops()
            self.assertEqual(op_counts.get("rx"), 2)
            self.assertEqual(op_counts.get("delay"), 3)

        tcircs = parexp._transpiled_circuits()
        for circ in tcircs:
            self.assertEqual(circ.num_qubits, num_qubits)
            op_counts = circ.count_ops()
            self.assertEqual(op_counts.get("rx"), 2)
            self.assertEqual(op_counts.get("delay"), 3)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = T1([0], [1, 2, 3, 4, 5])
        loaded_exp = T1.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = T1([0], [1, 2])
        self.assertRoundTripSerializable(exp)

    def test_circuit_roundtrip_serializable(self):
        """Test circuit round trip JSON serialization"""
        exp = T1([0], [1, 2, 3, 4, 5])
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = T1Analysis()
        loaded = T1Analysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_circuits_with_backend(self):
        """
        Test the circuits metadata when passing backend
        """
        backend = FakeAthensV2()
        delays = np.arange(1e-3, 40e-3, 3e-3)
        exp = T1([0], delays, backend=backend)
        circs = exp.circuits()

        self.assertEqual(len(circs), len(delays))

        for delay, circ in zip(delays, circs):
            # xval is rounded to nealest granularity value.
            self.assertAlmostEqual(circ.metadata["xval"], delay)
