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

"""Class to test composite experiments."""

import copy
import uuid

from test.fake_experiment import FakeExperiment, FakeAnalysis
from test.base import QiskitExperimentsTestCase
from unittest import mock
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.result import Result

from qiskit_aer import AerSimulator, noise

from qiskit_ibm_experiment import IBMExperimentService

from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.test.utils import FakeJob
from qiskit_experiments.test.fake_backend import FakeBackend
from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    ExperimentData,
    BatchExperiment,
    BaseExperiment,
    BaseAnalysis,
    AnalysisResultData,
    CompositeAnalysis,
)

# pylint: disable=missing-raises-doc


class TestComposite(QiskitExperimentsTestCase):
    """
    Test composite experiment behavior.
    """

    def test_parallel_options(self):
        """
        Test parallel experiments overriding sub-experiment run and transpile options.
        """
        # These options will all be overridden
        exp0 = FakeExperiment([0])
        exp0.set_transpile_options(optimization_level=1)
        exp2 = FakeExperiment([2])
        exp2.set_experiment_options(dummyoption="test")
        exp2.set_run_options(shots=2000)
        exp2.set_transpile_options(optimization_level=1)
        exp2.analysis.set_options(dummyoption="test")

        par_exp = ParallelExperiment([exp0, exp2], flatten_results=False)

        self.assertEqual(par_exp.experiment_options, par_exp._default_experiment_options())
        self.assertEqual(par_exp.run_options, Options(meas_level=2, restless=False))
        self.assertEqual(par_exp.transpile_options, Options(optimization_level=0))
        self.assertEqual(par_exp.analysis.options, par_exp.analysis._default_options())

        with self.assertWarns(UserWarning):
            expdata = par_exp.run(FakeBackend(num_qubits=3))
        self.assertExperimentDone(expdata)

    def test_flatten_results_nested(self):
        """Test combining results."""
        exp0 = FakeExperiment([0])
        exp1 = FakeExperiment([1])
        exp2 = FakeExperiment([2])
        exp3 = FakeExperiment([3])
        comp_exp = ParallelExperiment(
            [
                BatchExperiment(
                    2 * [ParallelExperiment([exp0, exp1], flatten_results=False)],
                    flatten_results=False,
                ),
                BatchExperiment(
                    3 * [ParallelExperiment([exp2, exp3], flatten_results=False)],
                    flatten_results=False,
                ),
            ],
            flatten_results=True,
        )
        expdata = comp_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(expdata)
        # Check no child data was saved
        self.assertEqual(len(expdata.child_data()), 0)
        # Check right number of analysis results is returned
        self.assertEqual(len(expdata.analysis_results()), 30)
        self.assertEqual(len(expdata.artifacts()), 20)

    def test_flatten_results_partial(self):
        """Test flattening results."""
        exp0 = FakeExperiment([0])
        exp1 = FakeExperiment([1])
        exp2 = FakeExperiment([2])
        exp3 = FakeExperiment([3])
        comp_exp = BatchExperiment(
            [
                ParallelExperiment([exp0, exp1, exp2], flatten_results=True),
                ParallelExperiment([exp2, exp3], flatten_results=True),
            ],
            flatten_results=False,
        )
        expdata = comp_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(expdata)
        # Check out experiment wasn't flattened
        self.assertEqual(len(expdata.child_data()), 2)
        self.assertEqual(len(expdata.analysis_results()), 0)
        self.assertEqual(len(expdata.artifacts()), 0)

        # check inner experiments were flattened
        child0 = expdata.child_data(0)
        child1 = expdata.child_data(1)
        self.assertEqual(len(child0.child_data()), 0)
        self.assertEqual(len(child1.child_data()), 0)
        # Check right number of analysis results is returned
        self.assertEqual(len(child0.analysis_results()), 9)
        self.assertEqual(len(child1.analysis_results()), 6)
        self.assertEqual(len(child0.artifacts()), 6)
        self.assertEqual(len(child1.artifacts()), 4)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp1 = FakeExperiment([0])
        exp1.set_run_options(shots=1000)
        exp2 = FakeExperiment([2])
        exp2.set_run_options(shots=2000)

        exp = BatchExperiment([exp1, exp2], flatten_results=False)

        loaded_exp = BatchExperiment.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp1 = FakeExperiment([0])
        exp1.set_run_options(shots=1000)
        exp2 = FakeExperiment([2])
        exp2.set_run_options(shots=2000)

        exp = BatchExperiment([exp1, exp2], flatten_results=False)

        self.assertRoundTripSerializable(exp)

    def test_experiment_type(self):
        """Test experiment_type setter."""

        exp1 = FakeExperiment([0])

        par_exp1 = ParallelExperiment([exp1], flatten_results=False)
        batch_exp1 = BatchExperiment([exp1], flatten_results=False)
        self.assertEqual(par_exp1.experiment_type, "ParallelExperiment")
        self.assertEqual(batch_exp1.experiment_type, "BatchExperiment")

        par_exp2 = ParallelExperiment([exp1], flatten_results=False, experiment_type="yooo")
        batch_exp2 = BatchExperiment([exp1], flatten_results=False, experiment_type="blaaa")
        self.assertEqual(par_exp2.experiment_type, "yooo")
        self.assertEqual(batch_exp2.experiment_type, "blaaa")


@ddt
class TestCompositeExperimentData(QiskitExperimentsTestCase):
    """
    Test operations on objects of composite ExperimentData
    """

    def setUp(self):
        super().setUp()

        self.backend = FakeBackend(num_qubits=4)
        self.share_level = "public"

        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2], flatten_results=False)
        exp3 = FakeExperiment([0, 1, 2, 3])
        batch_exp = BatchExperiment([par_exp, exp3], flatten_results=False)

        self.rootdata = batch_exp.run(backend=self.backend)
        self.assertExperimentDone(self.rootdata)
        self.assertEqual(len(self.rootdata.child_data()), 2)
        self.assertEqual(len(self.rootdata.artifacts()), 0)

        self.rootdata.share_level = self.share_level

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.share_level, self.share_level)

        components = expdata.child_data()
        for childdata in components:
            self.check_attributes(childdata)
            self.assertEqual(childdata.parent_id, expdata.experiment_id)
            if not hasattr(childdata, "child_data"):
                self.assertEqual(len(childdata.artifacts()), 2)
                self.assertEqual(childdata.artifacts("curve_data").experiment, "FakeExperiment")
                self.assertEqual(
                    childdata.artifacts("curve_data").device_components, childdata.device_components
                )
                self.assertEqual(childdata.artifacts("fit_summary").experiment, "FakeExperiment")
                self.assertEqual(
                    childdata.artifacts("fit_summary").device_components,
                    childdata.device_components,
                )

    def check_if_equal(self, expdata1, expdata2, is_a_copy, check_artifact=False):
        """
        Recursively traverse the tree and check equality of expdata1 and expdata2
        """
        self.assertEqual(expdata1.backend_name, expdata2.backend_name)
        self.assertEqual(expdata1.tags, expdata2.tags)
        self.assertEqual(expdata1.experiment_type, expdata2.experiment_type)
        self.assertEqual(expdata1.share_level, expdata2.share_level)

        metadata1 = copy.copy(expdata1.metadata)
        metadata2 = copy.copy(expdata2.metadata)
        metadata1.pop("child_data_ids", [])
        metadata2.pop("child_data_ids", [])
        self.assertDictEqual(metadata1, metadata2, msg="metadata not equal")

        if is_a_copy:
            self.assertNotEqual(expdata1.experiment_id, expdata2.experiment_id)
        else:
            self.assertEqual(expdata1.experiment_id, expdata2.experiment_id)

        if check_artifact:
            self.assertEqual(len(expdata1.artifacts()), len(expdata2.artifacts()))
            for artifact1, artifact2 in zip(expdata1.artifacts(), expdata2.artifacts()):
                self.assertEqual(artifact1, artifact2, msg="artifacts not equal")

        self.assertEqual(len(expdata1.child_data()), len(expdata2.child_data()))
        for childdata1, childdata2 in zip(expdata1.child_data(), expdata2.child_data()):
            self.check_if_equal(childdata1, childdata2, is_a_copy)

    def test_composite_experiment_data_attributes(self):
        """
        Verify correct attributes of parents and children
        """
        self.check_attributes(self.rootdata)
        self.assertEqual(self.rootdata.parent_id, None)

    def test_composite_save_load(self):
        """
        Verify that saving and loading restores the original composite experiment data object
        """

        self.rootdata.service = IBMExperimentService(local=True, local_save=False)
        self.rootdata.save()
        loaded_data = ExperimentData.load(self.rootdata.experiment_id, self.rootdata.service)
        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False, check_artifact=True)

    def test_composite_save_metadata(self):
        """
        Verify that saving metadata and loading restores the original composite experiment data object
        """
        self.rootdata.service = IBMExperimentService(local=True, local_save=False)
        self.rootdata.save_metadata()
        loaded_data = ExperimentData.load(self.rootdata.experiment_id, self.rootdata.service)
        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_copy(self):
        """
        Test composite ExperimentData.copy
        """
        new_instance = self.rootdata.copy()
        self.check_if_equal(new_instance, self.rootdata, is_a_copy=True, check_artifact=True)
        self.check_attributes(new_instance)
        self.assertEqual(new_instance.parent_id, None)

    def test_composite_copy_analysis_ref(self):
        """Test copy of composite experiment preserves component analysis refs"""

        class Analysis(FakeAnalysis):
            """Fake analysis class with options"""

            @classmethod
            def _default_options(cls):
                opts = super()._default_options()
                opts.option1 = None
                opts.option2 = None
                return opts

        exp1 = FakeExperiment([0])
        exp1.analysis = Analysis()
        exp2 = FakeExperiment([1])
        exp2.analysis = Analysis()

        # Generate a copy
        par_exp = ParallelExperiment([exp1, exp2], flatten_results=False).copy()
        comp_exp0 = par_exp.component_experiment(0)
        comp_exp1 = par_exp.component_experiment(1)
        comp_an0 = par_exp.analysis.component_analysis(0)
        comp_an1 = par_exp.analysis.component_analysis(1)

        # Check reference of analysis is preserved
        self.assertTrue(comp_exp0.analysis is comp_an0)
        self.assertTrue(comp_exp1.analysis is comp_an1)

    def test_nested_composite(self):
        """
        Test nested parallel experiments.
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        exp3 = ParallelExperiment([exp1, exp2], flatten_results=False)
        exp4 = BatchExperiment([exp3, exp1], flatten_results=False)
        exp5 = ParallelExperiment([exp4, FakeExperiment([4])], flatten_results=False)
        nested_exp = BatchExperiment([exp5, exp3], flatten_results=False)
        expdata = nested_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(expdata)

    def test_analysis_replace_results_true(self):
        """
        Test replace results when analyzing composite experiment data
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2], flatten_results=False)
        data1 = par_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(data1)

        # Additional data not part of composite experiment
        exp3 = FakeExperiment([0, 1])
        extra_data = exp3.run(FakeBackend(num_qubits=2))
        self.assertExperimentDone(extra_data)
        data1.add_child_data(extra_data)

        # Replace results
        data2 = par_exp.analysis.run(data1, replace_results=True)
        self.assertExperimentDone(data2)
        self.assertEqual(data1, data2)
        self.assertEqual(len(data1.child_data()), len(data2.child_data()))
        for sub1, sub2 in zip(data1.child_data(), data2.child_data()):
            self.assertEqual(sub1, sub2)

    def test_analysis_replace_results_false(self):
        """
        Test replace_results of composite experiment data
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = BatchExperiment([exp1, exp2], flatten_results=False)
        data1 = par_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(data1)

        # Additional data not part of composite experiment
        exp3 = FakeExperiment([0, 1])
        extra_data = exp3.run(FakeBackend(num_qubits=2))
        self.assertExperimentDone(extra_data)
        data1.add_child_data(extra_data)

        # Replace results
        data2 = par_exp.analysis.run(data1, replace_results=False)
        self.assertExperimentDone(data2)
        self.assertNotEqual(data1.experiment_id, data2.experiment_id)
        self.assertEqual(len(data1.child_data()), len(data2.child_data()))
        for sub1, sub2 in zip(data1.child_data(), data2.child_data()):
            self.assertNotEqual(sub1.experiment_id, sub2.experiment_id)

    def test_composite_tags(self):
        """
        Test the tags setter, add_tags_recursive, remove_tags_recursive
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = BatchExperiment([exp1, exp2], flatten_results=False)
        expdata = par_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(expdata)
        data1 = expdata.child_data(0)
        data2 = expdata.child_data(1)

        expdata.tags = ["a", "c", "a"]
        data1.tags = ["b"]
        self.assertEqual(sorted(expdata.tags), ["a", "c"])
        self.assertEqual(sorted(data1.tags), ["b"])
        self.assertEqual(sorted(data2.tags), [])

        expdata.add_tags_recursive(["d", "c"])
        self.assertEqual(sorted(expdata.tags), ["a", "c", "d"])
        self.assertEqual(sorted(data1.tags), ["b", "c", "d"])
        self.assertEqual(sorted(data2.tags), ["c", "d"])

        expdata.remove_tags_recursive(["a", "b"])
        self.assertEqual(sorted(expdata.tags), ["c", "d"])
        self.assertEqual(sorted(data1.tags), ["c", "d"])
        self.assertEqual(sorted(data2.tags), ["c", "d"])

    def test_composite_figures(self):
        """
        Test adding figures from composite experiments
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        exp1.analysis.set_options(add_figures=True)
        exp2.analysis.set_options(add_figures=True)
        par_exp = BatchExperiment([exp1, exp2], flatten_results=False)
        expdata = par_exp.run(FakeBackend(num_qubits=4))
        self.assertExperimentDone(expdata)
        expdata.service = IBMExperimentService(local=True, local_save=False)
        expdata.auto_save = True
        par_exp.analysis.run(expdata)
        self.assertExperimentDone(expdata)

    def test_composite_auto_save(self):
        """
        Test setting autosave when using composite experiments
        """
        service = mock.create_autospec(IBMExperimentService, instance=True)
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = BatchExperiment([exp1, exp2], flatten_results=False)
        expdata = par_exp.run(FakeBackend(num_qubits=4))
        expdata.service = service
        self.assertExperimentDone(expdata)
        expdata.auto_save = True
        self.assertEqual(service.create_or_update_experiment.call_count, 3)

    def test_composite_subexp_data(self):
        """
        Verify that sub-experiment data of parallel and batch
        experiments are correctly marginalized
        """
        counts = [
            {
                "0000": 1,
                "0010": 6,
                "0011": 3,
                "0100": 4,
                "0101": 2,
                "0110": 1,
                "0111": 3,
                "1000": 5,
                "1001": 3,
                "1010": 4,
                "1100": 4,
                "1101": 3,
                "1110": 8,
                "1111": 5,
            },
            {
                "0001": 3,
                "0010": 4,
                "0011": 5,
                "0100": 2,
                "0101": 1,
                "0111": 7,
                "1000": 3,
                "1001": 2,
                "1010": 1,
                "1011": 1,
                "1100": 7,
                "1101": 8,
                "1110": 2,
            },
            {
                "0000": 1,
                "0001": 1,
                "0010": 8,
                "0011": 7,
                "0100": 2,
                "0101": 2,
                "0110": 2,
                "0111": 1,
                "1000": 6,
                "1010": 4,
                "1011": 4,
                "1100": 5,
                "1101": 2,
                "1110": 2,
                "1111": 5,
            },
            {
                "0000": 4,
                "0001": 5,
                "0101": 4,
                "0110": 8,
                "0111": 2,
                "1001": 6,
                "1010": 8,
                "1011": 8,
                "1101": 1,
                "1110": 3,
                "1111": 3,
            },
            {
                "0000": 3,
                "0001": 6,
                "0010": 7,
                "0011": 1,
                "0100": 1,
                "0101": 5,
                "0110": 4,
                "1000": 2,
                "1001": 4,
                "1011": 3,
                "1100": 6,
                "1111": 1,
            },
        ]

        class Backend(FakeBackend):
            """
            Backend to be used in test_composite_subexp_data
            """

            def run(self, run_input, **options):
                results = []
                for circ, cnt in zip(run_input, counts):
                    results.append(
                        {
                            "shots": -1,
                            "success": True,
                            "header": {"metadata": circ.metadata},
                            "data": {"counts": cnt},
                        }
                    )

                res = {
                    "backend_name": "backend",
                    "backend_version": "0",
                    "qobj_id": uuid.uuid4().hex,
                    "job_id": uuid.uuid4().hex,
                    "success": True,
                    "results": results,
                }
                return FakeJob(backend=self, result=Result.from_dict(res))

        class Experiment(FakeExperiment):
            """
            Experiment to be used in test_composite_subexp_data
            """

            def __init__(self, qubits, num_circs):
                super().__init__(qubits)
                self._ncircs = num_circs

            def circuits(self):
                nqubits = len(self._physical_qubits)
                circs = []
                for _ in range(self._ncircs):
                    circ = QuantumCircuit(nqubits, nqubits)
                    circ.metadata = {}
                    circs.append(circ)
                return circs

        exp1 = Experiment([0, 2], 5)
        exp2 = Experiment([1], 2)
        exp3 = Experiment([3], 2)
        exp4 = Experiment([1, 3], 3)
        par_exp = ParallelExperiment(
            [
                exp1,
                BatchExperiment(
                    [ParallelExperiment([exp2, exp3], flatten_results=False), exp4],
                    flatten_results=False,
                ),
            ],
            flatten_results=False,
        )
        expdata = par_exp.run(Backend(num_qubits=4))
        self.assertExperimentDone(expdata)

        self.assertEqual(len(expdata.data()), len(counts))
        for circ_data, circ_counts in zip(expdata.data(), counts):
            self.assertDictEqual(circ_data["counts"], circ_counts)

        counts1 = [
            [
                {"00": 14, "10": 19, "11": 11, "01": 8},
                {"01": 14, "10": 7, "11": 13, "00": 12},
                {"00": 14, "01": 5, "10": 16, "11": 17},
                {"00": 4, "01": 16, "10": 19, "11": 13},
                {"00": 12, "01": 15, "10": 11, "11": 5},
            ],
            [
                {"00": 10, "01": 10, "10": 12, "11": 20},
                {"00": 12, "01": 10, "10": 7, "11": 17},
                {"00": 17, "01": 7, "10": 14, "11": 14},
                {"00": 9, "01": 14, "10": 22, "11": 7},
                {"00": 17, "01": 10, "10": 9, "11": 7},
            ],
        ]

        self.assertEqual(len(expdata.child_data()), len(counts1))
        for childdata, child_counts in zip(expdata.child_data(), counts1):
            self.assertEqual(len(childdata.data()), len(child_counts))
            for circ_data, circ_counts in zip(childdata.data(), child_counts):
                self.assertDictEqual(circ_data["counts"], circ_counts)

        counts2 = [
            [{"00": 10, "01": 10, "10": 12, "11": 20}, {"00": 12, "01": 10, "10": 7, "11": 17}],
            [
                {"00": 17, "01": 7, "10": 14, "11": 14},
                {"00": 9, "01": 14, "10": 22, "11": 7},
                {"00": 17, "01": 10, "10": 9, "11": 7},
            ],
        ]

        self.assertEqual(len(expdata.child_data(1).child_data()), len(counts2))
        for childdata, child_counts in zip(expdata.child_data(1).child_data(), counts2):
            for circ_data, circ_counts in zip(childdata.data(), child_counts):
                self.assertDictEqual(circ_data["counts"], circ_counts)

        counts3 = [
            [{"0": 22, "1": 30}, {"0": 19, "1": 27}],
            [{"0": 20, "1": 32}, {"0": 22, "1": 24}],
        ]

        self.assertEqual(len(expdata.child_data(1).child_data(0).child_data()), len(counts3))
        for childdata, child_counts in zip(
            expdata.child_data(1).child_data(0).child_data(), counts3
        ):
            self.assertEqual(len(childdata.data()), len(child_counts))
            for circ_data, circ_counts in zip(childdata.data(), child_counts):
                self.assertDictEqual(circ_data["counts"], circ_counts)

    def test_composite_analysis_options(self):
        """Test setting component analysis options"""

        class Analysis(FakeAnalysis):
            """Fake analysis class with options"""

            @classmethod
            def _default_options(cls):
                opts = super()._default_options()
                opts.option1 = None
                opts.option2 = None
                return opts

        exp1 = FakeExperiment([0])
        exp1.analysis = Analysis()
        exp2 = FakeExperiment([1])
        exp2.analysis = Analysis()
        par_exp = ParallelExperiment([exp1, exp2], flatten_results=False)

        # Set new analysis classes to component exp objects
        opt1_val = 9000
        opt2_val = 2113
        exp1.analysis.set_options(option1=opt1_val)
        exp2.analysis.set_options(option2=opt2_val)

        # Check this is reflected in parallel experiment
        self.assertEqual(par_exp.analysis.component_analysis(0).options.option1, opt1_val)
        self.assertEqual(par_exp.analysis.component_analysis(1).options.option2, opt2_val)

    def test_composite_analysis_options_cascade(self):
        """Test setting component analysis options"""

        class Analysis(FakeAnalysis):
            """Fake analysis class with options"""

            @classmethod
            def _default_options(cls):
                opts = super()._default_options()
                opts.option1 = None
                return opts

        exp1 = FakeExperiment([0])
        exp1.analysis = Analysis()
        exp2 = FakeExperiment([1])
        exp2.analysis = Analysis()
        par_exp1 = ParallelExperiment([exp1, exp2], flatten_results=True)

        exp3 = FakeExperiment([0])
        exp3.analysis = Analysis()
        exp4 = FakeExperiment([1])
        exp4.analysis = Analysis()
        par_exp2 = ParallelExperiment([exp3, exp4], flatten_results=True)

        # Set a batch experiment
        batch_exp = BatchExperiment([par_exp1, par_exp2], flatten_results=True)

        # Set new option to the experiment
        exp_list = [exp1, exp2, exp3, exp4]
        opt1_vals = [9000, 8000, 7000, 6000]
        for exp, opt1_val in zip(exp_list, opt1_vals):
            exp.analysis.set_options(option1=opt1_val)

        opt1_new_val = 1000
        batch_exp.analysis.set_options(option1=opt1_new_val, broadcast=False)

        for exp in exp_list:
            self.assertNotEqual(exp.analysis.options.option1, opt1_new_val)

        batch_exp.analysis.set_options(option1=opt1_new_val, broadcast=True)
        for exp in exp_list:
            self.assertEqual(exp.analysis.options.option1, opt1_new_val)

    @data(
        ["0x0", "0x2", "0x3", "0x0", "0x0", "0x1", "0x3", "0x0", "0x2", "0x3"],
        ["00", "10", "11", "00", "00", "01", "11", "00", "10", "11"],
    )
    def test_composite_count_memory_marginalization(self, memory):
        """Test the marginalization of level two memory."""
        test_data = ExperimentData()

        # Simplified experimental data
        datum = {
            "counts": {"0 0": 4, "0 1": 1, "1 0": 2, "1 1": 3},
            "memory": memory,
            "metadata": {
                "experiment_type": "ParallelExperiment",
                "composite_index": [0, 1],
                "composite_metadata": [
                    {"experiment_type": "FineXAmplitude", "qubits": [0]},
                    {"experiment_type": "FineXAmplitude", "qubits": [1]},
                ],
                "composite_qubits": [[0], [1]],
                "composite_clbits": [[0], [1]],
            },
            "shots": 10,
            "meas_level": 2,
        }

        test_data.add_data(datum)

        sub_data = CompositeAnalysis([], flatten_results=False)._marginalized_component_data(
            test_data.data()
        )
        expected = [
            [
                {
                    "metadata": {"experiment_type": "FineXAmplitude", "qubits": [0]},
                    "counts": {"0": 6, "1": 4},
                    "memory": ["0", "0", "1", "0", "0", "1", "1", "0", "0", "1"],
                    "shots": 10,
                    "meas_level": 2,
                }
            ],
            [
                {
                    "metadata": {"experiment_type": "FineXAmplitude", "qubits": [1]},
                    "counts": {"0": 5, "1": 5},
                    "memory": ["0", "1", "1", "0", "0", "0", "1", "0", "1", "1"],
                    "shots": 10,
                    "meas_level": 2,
                }
            ],
        ]

        self.assertListEqual(sub_data, expected)

    def test_composite_single_kerneled_memory_marginalization(self):
        """Test the marginalization of level 1 data."""
        test_data = ExperimentData()

        datum = {
            "memory": [
                # qubit 0,   qubit 1,    qubit 2
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # shot 1
                [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]],  # shot 2
                [[0.2, 0.2], [1.2, 1.2], [2.2, 2.2]],  # shot 3
                [[0.3, 0.3], [1.3, 1.3], [2.3, 2.3]],  # shot 4
                [[0.4, 0.4], [1.4, 1.4], [2.4, 2.4]],  # shot 5
            ],
            "metadata": {
                "experiment_type": "ParallelExperiment",
                "composite_index": [0, 1, 2],
                "composite_metadata": [
                    {"experiment_type": "FineXAmplitude", "qubits": [0]},
                    {"experiment_type": "FineXAmplitude", "qubits": [1]},
                    {"experiment_type": "FineXAmplitude", "qubits": [2]},
                ],
                "composite_qubits": [[0], [1], [2]],
                "composite_clbits": [[0], [1], [2]],
            },
            "shots": 5,
            "meas_level": 1,
        }

        test_data.add_data(datum)

        all_sub_data = CompositeAnalysis([], flatten_results=False)._marginalized_component_data(
            test_data.data()
        )
        for idx, sub_data in enumerate(all_sub_data):
            expected = {
                "metadata": {"experiment_type": "FineXAmplitude", "qubits": [idx]},
                "memory": [
                    [[idx + 0.0, idx + 0.0]],
                    [[idx + 0.1, idx + 0.1]],
                    [[idx + 0.2, idx + 0.2]],
                    [[idx + 0.3, idx + 0.3]],
                    [[idx + 0.4, idx + 0.4]],
                ],
                "shots": 5,
                "meas_level": 1,
            }

            self.assertEqual(expected, sub_data[0])

    def test_composite_avg_kerneled_memory_marginalization(self):
        """The the marginalization of level 1 averaged data."""
        test_data = ExperimentData()

        datum = {
            "memory": [
                [0.0, 0.1],  # qubit 0
                [1.0, 1.1],  # qubit 1
                [2.0, 2.1],  # qubit 2
            ],
            "metadata": {
                "experiment_type": "ParallelExperiment",
                "composite_index": [0, 1, 2],
                "composite_metadata": [
                    {"experiment_type": "FineXAmplitude", "qubits": [0]},
                    {"experiment_type": "FineXAmplitude", "qubits": [1]},
                    {"experiment_type": "FineXAmplitude", "qubits": [2]},
                ],
                "composite_qubits": [[0], [1], [2]],
                "composite_clbits": [[0], [1], [2]],
            },
            "shots": 5,
            "meas_level": 1,
        }

        test_data.add_data(datum)

        all_sub_data = CompositeAnalysis([], flatten_results=False)._marginalized_component_data(
            test_data.data()
        )
        for idx, sub_data in enumerate(all_sub_data):
            expected = {
                "metadata": {"experiment_type": "FineXAmplitude", "qubits": [idx]},
                "memory": [[idx + 0.0, idx + 0.1]],
                "shots": 5,
                "meas_level": 1,
            }

            self.assertEqual(expected, sub_data[0])

    def test_composite_properties_setting(self):
        """Test whether DB-critical properties are being set in the
        subexperiment data"""
        exp1 = FakeExperiment([0])
        exp1.analysis = FakeAnalysis()
        exp2 = FakeExperiment([1])
        exp2.analysis = FakeAnalysis()
        batch_exp = BatchExperiment([exp1, exp2], flatten_results=True)
        exp_data = batch_exp.run(backend=self.backend)
        self.assertExperimentDone(exp_data)
        # when flattening, individual analysis result share exp id
        for result in exp_data.analysis_results():
            self.assertEqual(result.experiment_id, exp_data.experiment_id)
        batch_exp = BatchExperiment([exp1, exp2], flatten_results=False)
        exp_data = batch_exp.run(backend=self.backend)
        self.assertExperimentDone(exp_data)
        self.assertEqual(exp_data.child_data(0).experiment_type, exp1.experiment_type)
        self.assertEqual(exp_data.child_data(1).experiment_type, exp2.experiment_type)


class TestBatchTranspileOptions(QiskitExperimentsTestCase):
    """
    For batch experiments, circuits are transpiled with the transpile options of the
    sub-experiments
    """

    class SimpleExperiment(BaseExperiment):
        """
        An experiment that creates a circuit of four qubits.
        Qubits 1 and 2 are inactive.
        Qubits 0 and 3 form a Bell state.
        The purpose: we will test with varying coupling maps, spanning from a coupling map that
        directly connects qubits 0 and 3 (hence qubits 1 and 2 remains inactive also in the
        transpiled circuit) to a coupling map with distance 3 between qubits 0 and 3.
        """

        def __init__(self, physical_qubits, backend=None):
            super().__init__(
                physical_qubits,
                analysis=TestBatchTranspileOptions.SimpleAnalysis(),
                backend=backend,
            )

        def circuits(self):
            circ = QuantumCircuit(4, 4)
            circ.h(0)
            circ.cx(0, 3)
            circ.barrier()
            circ.measure(range(4), range(4))
            return [circ]

    class SimpleAnalysis(BaseAnalysis):
        """
        The number of non-zero counts is equal to
        2^(distance between qubits 0 and 3 in the transpiled circuit + 1)
        """

        def _run_analysis(self, experiment_data):
            analysis_results = [
                AnalysisResultData(
                    name="non-zero counts", value=len(experiment_data.data(0)["counts"])
                ),
            ]

            return analysis_results, []

    def setUp(self):
        super().setUp()

        exp1 = self.SimpleExperiment(range(4))
        exp2 = self.SimpleExperiment(range(4))
        exp3 = self.SimpleExperiment(range(4))
        batch1 = BatchExperiment([exp2, exp3], flatten_results=False)
        self.batch2 = BatchExperiment([exp1, batch1], flatten_results=False)

        exp1.set_transpile_options(coupling_map=[[0, 1], [1, 3], [3, 2]])
        exp2.set_transpile_options(coupling_map=[[0, 1], [1, 2], [2, 3]])

        # exp3 circuit: two active qubits and six instructions: hadamard, cnot, four measurements.
        # exp1 circuit: three active qubits (0, 1, 3) and seven instructions: hadamard,
        #               two 2Q gates, four measurements.
        # exp2 circuit: four active qubits and eight instructions.

    def test_batch_transpiled_circuits(self):
        """
        For batch experiments, circuits are transpiled with the transpile options of the
        sub-experiments
        """
        circs = self.batch2._transpiled_circuits()
        numbers_of_gates = [len(circ.data) for circ in circs]
        self.assertEqual(set(numbers_of_gates), set([7, 8, 9]))

    def test_batch_transpile_options_integrated(self):
        """
        The goal is to verify that not only `_transpiled_circuits` works well
        (`test_batch_transpiled_circuits` takes care of it) but that it's correctly called within
        the entire flow of `BaseExperiment.run`.
        """
        backend = AerSimulator()
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(0.5, 2), ["cx", "swap"])

        expdata = self.batch2.run(backend, noise_model=noise_model, shots=1000)
        self.assertExperimentDone(expdata)

        self.assertEqual(expdata.child_data(0).analysis_results("non-zero counts").value, 8)
        self.assertEqual(
            expdata.child_data(1).child_data(0).analysis_results("non-zero counts").value, 16
        )
        self.assertEqual(
            expdata.child_data(1).child_data(1).analysis_results("non-zero counts").value, 4
        )

    def test_separate_jobs(self):
        """Test the separate_job experiment option"""

        backend = FakeBackend()

        class Experiment(FakeExperiment):
            """Fake Experiment to test the separate_job experiment option"""

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return [qc]

        exp = Experiment([0])

        # test separate_jobs=False
        batch_exp = BatchExperiment([exp, exp], flatten_results=False)
        batch_data = batch_exp.run(backend)
        self.assertExperimentDone(batch_data)
        job_ids = batch_data.job_ids
        self.assertEqual(len(job_ids), 1)

        # test separate_jobs=True
        batch_exp.set_experiment_options(separate_jobs=True)
        batch_data = batch_exp.run(backend)
        self.assertExperimentDone(batch_data)
        job_ids = batch_data.job_ids
        self.assertEqual(len(job_ids), 2)

        # test a forbidden nested case, where a parent sets separate_jobs
        # to False while the child sets it to True
        meta_exp = BatchExperiment([batch_exp], flatten_results=False)
        with self.assertRaises(QiskitError):
            meta_exp.run(backend)

        # test a valid nested case
        meta_exp.set_experiment_options(separate_jobs=True)
        meta_expdata = meta_exp.run(backend)
        self.assertExperimentDone(meta_expdata)
        job_ids = meta_expdata.job_ids
        self.assertEqual(len(job_ids), 2)
