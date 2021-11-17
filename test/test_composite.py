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

from test.fake_backend import FakeBackend
from test.fake_experiment import FakeExperiment
from test.fake_service import FakeService
from test.base import QiskitExperimentsTestCase

from qiskit_experiments.framework import (
    ParallelExperiment,
    Options,
    ExperimentData,
    BatchExperiment,
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
        exp2.set_analysis_options(dummyoption="test")

        par_exp = ParallelExperiment([exp0, exp2])

        with self.assertWarnsRegex(
            Warning,
            "Sub-experiment run and transpile options"
            " are overridden by composite experiment options.",
        ):
            self.assertEqual(par_exp.experiment_options, Options())
            self.assertEqual(par_exp.run_options, Options(meas_level=2))
            self.assertEqual(par_exp.transpile_options, Options(optimization_level=0))
            self.assertEqual(par_exp.analysis_options, Options())

            par_exp.run(FakeBackend())


class TestCompositeExperimentData(QiskitExperimentsTestCase):
    """
    Test operations on objects of composit ExperimentData
    """

    def setUp(self):
        super().setUp()

        self.backend = FakeBackend()
        self.share_level = "hey"

        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        exp3 = FakeExperiment([0, 1, 2, 3])
        batch_exp = BatchExperiment([par_exp, exp3])

        self.rootdata = ExperimentData(batch_exp, backend=self.backend)

        self.rootdata.share_level = self.share_level

    def check_attributes(self, expdata):
        """
        Recursively traverse the tree to verify attributes
        """
        self.assertEqual(expdata.backend, self.backend)
        self.assertEqual(expdata.share_level, self.share_level)

        components = expdata.child_data()
        comp_ids = expdata.metadata.get("child_ids", [])
        for childdata, comp_id in zip(components, comp_ids):
            self.check_attributes(childdata)
            self.assertEqual(childdata.parent_id, expdata.experiment_id)
            self.assertEqual(childdata.experiment_id, comp_id)

    def check_if_equal(self, expdata1, expdata2, is_a_copy):
        """
        Recursively traverse the tree and check equality of expdata1 and expdata2
        """
        self.assertEqual(expdata1.backend.name(), expdata2.backend.name())
        self.assertEqual(expdata1.tags, expdata2.tags)
        self.assertEqual(expdata1.experiment_type, expdata2.experiment_type)
        self.assertEqual(expdata1.share_level, expdata2.share_level)

        metadata1 = copy.copy(expdata1.metadata)
        metadata2 = copy.copy(expdata2.metadata)
        if is_a_copy:
            comp_ids1 = metadata1.pop("child_ids", [])
            comp_ids2 = metadata2.pop("child_ids", [])
            for id1 in comp_ids1:
                self.assertNotIn(id1, comp_ids2)
            for id2 in comp_ids2:
                self.assertNotIn(id2, comp_ids1)
            if expdata1.parent_id is None:
                self.assertEqual(expdata2.parent_id, None)
            else:
                self.assertNotEqual(expdata1.parent_id, expdata2.parent_id)
        else:
            self.assertEqual(expdata1.parent_id, expdata2.parent_id)
        self.assertDictEqual(metadata1, metadata2, msg="metadata not equal")

        if isinstance(expdata1, ExperimentData):
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

        self.rootdata.service = FakeService()
        self.rootdata.save()
        loaded_data = ExperimentData.load(self.rootdata.experiment_id, self.rootdata.service)
        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_save_metadata(self):
        """
        Verify that saving metadata and loading restores the original composite experiment data object
        """
        self.rootdata.service = FakeService()
        self.rootdata.save_metadata()
        loaded_data = ExperimentData.load(self.rootdata.experiment_id, self.rootdata.service)

        self.check_if_equal(loaded_data, self.rootdata, is_a_copy=False)

    def test_composite_copy(self):
        """
        Test composite ExperimentData.copy
        """
        new_instance = self.rootdata.copy()
        self.check_if_equal(new_instance, self.rootdata, is_a_copy=True)
        self.check_attributes(new_instance)

    def test_analysis_replace_results_true(self):
        """
        Test replace results when analyzing composite experiment data
        """
        exp1 = FakeExperiment([0, 2])
        exp2 = FakeExperiment([1, 3])
        par_exp = ParallelExperiment([exp1, exp2])
        data1 = par_exp.run(FakeBackend()).block_for_results()

        # Additional data not part of composite experiment
        exp3 = FakeExperiment([0, 1])
        extra_data = exp3.run(FakeBackend())
        data1.add_child_data(extra_data)

        # Replace results
        data2 = par_exp.run_analysis(data1, replace_results=True)
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
        par_exp = BatchExperiment([exp1, exp2])
        data1 = par_exp.run(FakeBackend()).block_for_results()

        # Additional data not part of composite experiment
        exp3 = FakeExperiment([0, 1])
        extra_data = exp3.run(FakeBackend())
        data1.add_child_data(extra_data)

        # Replace results
        data2 = par_exp.run_analysis(data1, replace_results=False)
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
        par_exp = BatchExperiment([exp1, exp2])
        expdata = par_exp.run(FakeBackend()).block_for_results()
        data1 = expdata.child_data(0)
        data2 = expdata.child_data(1)

        expdata.tags = ["a", "c", "a"]
        data1.tags = ["b"]
        print(expdata.tags)
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
