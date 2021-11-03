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

"""Data processor tests."""

# pylint: disable=unbalanced-tuple-unpacking

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit_experiments.data_processing.nodes import (
    SVD,
    AverageData,
    MinMaxNormalize,
    Probability,
)
from qiskit_experiments.data_processing.data_processor import DataProcessor

from . import BaseDataProcessorTest


def processor_wrapper(node, values, errors=None):
    """A helper function to execute node."""
    values = np.asarray(values, dtype=object)
    if errors is None:
        errors = np.full_like(values, np.nan, dtype=float)
    errors = np.asarray(errors, dtype=float)

    node_out = list(node(zip(values, errors)))
    return list(zip(*node_out))


class TestAveraging(BaseDataProcessorTest):
    """Test the averaging nodes."""

    def test_simple(self):
        """Simple test of averaging."""
        source = np.array([[1, 2], [3, 4], [5, 6]])

        node = AverageData(axis=1)
        node_out = processor_wrapper(node, source)

        self.assertTrue(np.allclose(node_out[0], np.array([1.5, 3.5, 5.5])))
        self.assertTrue(np.allclose(node_out[1], np.array([0.5, 0.5, 0.5]) / np.sqrt(2)))

        node = AverageData(axis=0)
        node_out = processor_wrapper(node, source)

        self.assertTrue(np.allclose(node_out[0], np.array([3.0, 4.0])))
        std = np.std([1, 3, 5])
        self.assertTrue(np.allclose(node_out[1], np.array([std, std]) / np.sqrt(3)))

    def test_iq_averaging(self):
        """Test averaging of IQ-data."""

        iq_data = [
            [[-6.20601501e14, -1.33257051e15], [-1.70921324e15, -4.05881657e15]],
            [[-5.80546502e14, -1.33492509e15], [-1.65094637e15, -4.05926942e15]],
            [[-4.04649069e14, -1.33191056e15], [-1.29680377e15, -4.03604815e15]],
            [[-2.22203874e14, -1.30291309e15], [-8.57663429e14, -3.97784973e15]],
            [[-2.92074029e13, -1.28578530e15], [-9.78824053e13, -3.92071056e15]],
            [[1.98056981e14, -1.26883024e15], [3.77157017e14, -3.87460328e15]],
            [[4.29955888e14, -1.25022995e15], [1.02340118e15, -3.79508679e15]],
            [[6.38981344e14, -1.25084614e15], [1.68918514e15, -3.78961044e15]],
            [[7.09988897e14, -1.21906634e15], [1.91914171e15, -3.73670664e15]],
            [[7.63169115e14, -1.20797552e15], [2.03772603e15, -3.74653863e15]],
        ]

        self.create_experiment(iq_data, single_shot=True)

        avg_iq = AverageData(axis=0)
        avg_datum, error = processor_wrapper(avg_iq, self.iq_experiment.data(0)["memory"])

        expected_avg = np.array([[8.82943876e13, -1.27850527e15], [1.43410186e14, -3.89952402e15]])
        expected_std = np.array(
            [[5.07650185e14, 4.44664719e13], [1.40522641e15, 1.22326831e14]]
        ) / np.sqrt(10)

        self.assertTrue(np.allclose(avg_datum, expected_avg))
        self.assertTrue(np.allclose(error, expected_std))


class TestNormalize(QiskitTestCase):
    """Test the normalization node."""

    def test_simple(self):
        """Simple test of normalization node."""

        data = np.array([1.0, 2.0, 3.0, 3.0])
        error = np.array([0.1, 0.2, 0.3, 0.3])

        expected_data = np.array([0.0, 0.5, 1.0, 1.0])
        expected_error = np.array([0.05, 0.1, 0.15, 0.15])

        node = MinMaxNormalize()

        # error free data
        node_out = processor_wrapper(node, data)
        self.assertTrue(np.allclose(node_out[0], expected_data))

        # error data
        node_out = processor_wrapper(node, data, error)
        self.assertTrue(np.allclose(node_out[0], expected_data))
        self.assertTrue(np.allclose(node_out[1], expected_error))


class TestSVD(BaseDataProcessorTest):
    """Test the SVD nodes."""

    def test_simple_data(self):
        """
        A simple setting where the IQ data of qubit 0 is oriented along (1,1) and
        the IQ data of qubit 1 is oriented along (1,-1).
        """

        iq_data = [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [1.0, -1.0]]]

        self.create_experiment(iq_data)

        iq_svd = SVD()
        iq_svd.train([datum["memory"] for datum in self.iq_experiment.data()])

        # qubit 0 IQ data is oriented along (1,1)
        self.assertTrue(np.allclose(iq_svd._main_axes[0], np.array([-1, -1]) / np.sqrt(2)))

        # qubit 1 IQ data is oriented along (1, -1)
        self.assertTrue(np.allclose(iq_svd._main_axes[1], np.array([-1, 1]) / np.sqrt(2)))

        processed, _ = processor_wrapper(iq_svd, [[[1, 1], [1, -1]]])
        expected = np.array([-1, -1]) / np.sqrt(2)
        self.assertTrue(np.allclose(processed, expected))

        processed, _ = processor_wrapper(iq_svd, [[[2, 2], [2, -2]]])
        self.assertTrue(np.allclose(processed, expected * 2))

        # Check that orthogonal data gives 0.
        processed, _ = processor_wrapper(iq_svd, [[[1, -1], [1, 1]]])
        expected = np.array([0, 0])
        self.assertTrue(np.allclose(processed, expected))

    def test_svd(self):
        """Use IQ data gathered from the hardware."""

        # This data is primarily oriented along the real axis with a slight tilt.
        # There is a large offset in the imaginary dimension when comparing qubits
        # 0 and 1.
        iq_data = [
            [[-6.20601501e14, -1.33257051e15], [-1.70921324e15, -4.05881657e15]],
            [[-5.80546502e14, -1.33492509e15], [-1.65094637e15, -4.05926942e15]],
            [[-4.04649069e14, -1.33191056e15], [-1.29680377e15, -4.03604815e15]],
            [[-2.22203874e14, -1.30291309e15], [-8.57663429e14, -3.97784973e15]],
            [[-2.92074029e13, -1.28578530e15], [-9.78824053e13, -3.92071056e15]],
            [[1.98056981e14, -1.26883024e15], [3.77157017e14, -3.87460328e15]],
            [[4.29955888e14, -1.25022995e15], [1.02340118e15, -3.79508679e15]],
            [[6.38981344e14, -1.25084614e15], [1.68918514e15, -3.78961044e15]],
            [[7.09988897e14, -1.21906634e15], [1.91914171e15, -3.73670664e15]],
            [[7.63169115e14, -1.20797552e15], [2.03772603e15, -3.74653863e15]],
        ]

        self.create_experiment(iq_data)

        iq_svd = SVD()
        iq_svd.train([datum["memory"] for datum in self.iq_experiment.data()])

        self.assertTrue(np.allclose(iq_svd._main_axes[0], np.array([-0.99633018, -0.08559302])))
        self.assertTrue(np.allclose(iq_svd._main_axes[1], np.array([-0.99627747, -0.0862044])))

    def test_svd_error(self):
        """Test the error formula of the SVD."""

        iq_svd = SVD()
        iq_svd._main_axes = np.array([[1.0, 0.0]])
        iq_svd._scales = [1.0]
        iq_svd._means = [[0.0, 0.0]]

        # Since the axis is along the real part the imaginary error is irrelevant.
        processed, error = processor_wrapper(iq_svd, [[[1.0, 0.2]]], [[[0.2, 0.1]]])
        self.assertEqual(processed, np.array([1.0]))
        self.assertEqual(error, np.array([0.2]))

        # Since the axis is along the real part the imaginary error is irrelevant.
        processed, error = processor_wrapper(iq_svd, [[[1.0, 0.2]]], [[[0.2, 0.3]]])
        self.assertEqual(processed, np.array([1.0]))
        self.assertEqual(error, np.array([0.2]))

        # Tilt the axis to an angle of 36.9... degrees
        iq_svd._main_axes = np.array([[0.8, 0.6]])
        processed, error = processor_wrapper(iq_svd, [[[1.0, 0.0]]], [[[0.2, 0.3]]])

        cos_ = np.cos(np.arctan(0.6 / 0.8))
        sin_ = np.sin(np.arctan(0.6 / 0.8))
        self.assertEqual(processed, np.array([cos_]))
        expected_error = np.sqrt((0.2 * cos_) ** 2 + (0.3 * sin_) ** 2)
        self.assertEqual(error, np.array([expected_error]))

    def test_train_svd_processor(self):
        """Test that we can train a DataProcessor with an SVD."""

        processor = DataProcessor("memory", [SVD()])

        self.assertFalse(processor.is_trained)

        iq_data = [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [1.0, -1.0]]]
        self.create_experiment(iq_data)

        processor.train(self.iq_experiment.data())

        self.assertTrue(processor.is_trained)

        # Check that we can use the SVD
        iq_data = [[[2, 2], [2, -2]]]
        self.create_experiment(iq_data)

        processed, _ = processor(self.iq_experiment.data(0))
        expected = np.array([-2, -2]) / np.sqrt(2)
        self.assertTrue(np.allclose(processed, expected))


class TestProbability(QiskitTestCase):
    """Test probability computation."""

    def test_variance_not_zero(self):
        """Test if finite variance is computed at max or min probability."""
        node = Probability(outcome="1")

        data = {"1": 1024, "0": 0}
        mean, stderr = processor_wrapper(node, [[data]])
        self.assertGreater(float(stderr[0]), 0.0)
        self.assertLessEqual(float(mean[0]), 1.0)

        data = {"1": 0, "0": 1024}
        mean, stderr = processor_wrapper(node, [[data]])
        self.assertGreater(float(stderr[0]), 0.0)
        self.assertGreaterEqual(float(mean[0]), 0.0)

    def test_probability_balanced(self):
        """Test if p=0.5 is returned when counts are balanced and prior is flat."""
        node = Probability(outcome="1")

        # balanced counts with a flat prior will yield p = 0.5
        data = {"1": 512, "0": 512}
        mean, _ = processor_wrapper(node, [[data]])
        self.assertAlmostEqual(float(mean[0]), 0.5)
