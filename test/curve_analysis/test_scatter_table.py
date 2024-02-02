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

"""Test scatter table."""

from test.base import QiskitExperimentsTestCase

import pandas as pd
import numpy as np

from qiskit_experiments.curve_analysis.scatter_table import ScatterTable


class TestScatterTable(QiskitExperimentsTestCase):
    """Test cases for curve analysis ScatterTable."""

    def setUp(self):
        super().setUp()

        source = {
            "xval": [0.100, 0.100, 0.200, 0.200, 0.100, 0.200, 0.100, 0.200, 0.100, 0.200],
            "yval": [0.192, 0.784, 0.854, 0.672, 0.567, 0.488, 0.379, 0.671, 0.784, 0.672],
            "yerr": [0.002, 0.091, 0.090, 0.027, 0.033, 0.038, 0.016, 0.048, 0.091, 0.027],
            "name": [
                "model1",
                "model2",
                "model1",
                "model2",
                "model1",
                "model1",
                "model1",
                "model1",
                "model2",
                "model2",
            ],
            "class_id": [0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
            "category": [
                "raw",
                "raw",
                "raw",
                "raw",
                "raw",
                "raw",
                "formatted",
                "formatted",
                "formatted",
                "formatted",
            ],
            "shots": [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                2000,
                2000,
                1000,
                1000,
            ],
            "analysis": [
                "Fit1",
                "Fit1",
                "Fit1",
                "Fit1",
                "Fit2",
                "Fit2",
                "Fit1",
                "Fit1",
                "Fit1",
                "Fit1",
            ],
        }
        self.reference = pd.DataFrame.from_dict(source)

    def test_create_table_from_dataframe(self):
        """Test creating table from dataframe and output dataframe."""
        # ScatterTable automatically converts dtype.
        # For pure dataframe equality check pre-format the source.
        formatted_ref = ScatterTable._format_table(self.reference)

        obj = ScatterTable.from_dataframe(formatted_ref)
        self.assertTrue(obj.dataframe.equals(formatted_ref))

    def test_add_row(self):
        """Test adding single row to the table without and with missing data."""
        obj = ScatterTable()
        obj.add_row(
            name="model1",
            class_id=0,
            category="raw",
            x=0.1,
            y=2.3,
            y_err=0.4,
            shots=1000,
            analysis="Test",
        )
        obj.add_row(
            category="raw",
            x=0.2,
            y=3.4,
        )
        self.assertEqual(len(obj), 2)
        np.testing.assert_array_equal(obj.x, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(obj.y, np.array([2.3, 3.4]))
        np.testing.assert_array_equal(obj.y_err, np.array([0.4, np.nan]))
        np.testing.assert_array_equal(obj.name, np.array(["model1", None]))
        np.testing.assert_array_equal(obj.class_id, np.array([0, None]))
        np.testing.assert_array_equal(obj.category, np.array(["raw", "raw"]))
        np.testing.assert_array_equal(
            # Numpy tries to handle nan strictly, but isnan only works for float dtype.
            # Original data is object type, because we want to keep shot number integer,
            # and there is no Numpy nullable integer.
            obj.shots.astype(float),
            np.array([1000, np.nan], dtype=float),
        )
        np.testing.assert_array_equal(obj.analysis, np.array(["Test", None]))

    def test_set_values(self):
        """Test setting new column values through setter."""
        obj = ScatterTable()
        # add three empty rows
        obj.add_row()
        obj.add_row()
        obj.add_row()

        # Set sequence
        obj.x = [0.1, 0.2, 0.3]
        obj.y = [1.3, 1.4, 1.5]
        obj.y_err = [0.3, 0.5, 0.7]

        # Broadcast single value
        obj.class_id = 0
        obj.name = "model0"

        np.testing.assert_array_equal(obj.x, np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(obj.y, np.array([1.3, 1.4, 1.5]))
        np.testing.assert_array_equal(obj.y_err, np.array([0.3, 0.5, 0.7]))
        np.testing.assert_array_equal(obj.class_id, np.array([0, 0, 0]))
        np.testing.assert_array_equal(obj.name, np.array(["model0", "model0", "model0"]))

    def test_set_y(self):
        """Test setting new values to y column."""
        obj = ScatterTable()
        obj.add_row(x=0.1, y=2.0, y_err=0.3)
        obj.y = [0.5]
        np.testing.assert_array_equal(obj.y, np.array([0.5]))

    def test_set_y_err(self):
        """Test setting new values to y_err column."""
        obj = ScatterTable()
        obj.add_row(x=0.1, y=2.0, y_err=0.3)
        obj.y_err = [0.5]
        np.testing.assert_array_equal(obj.y_err, np.array([0.5]))

    def test_filter_data_by_class_id(self):
        """Test filter table data with data UID."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(kind=0)
        self.assertEqual(len(filtered), 6)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]))
        np.testing.assert_array_equal(filtered.class_id, np.array([0, 0, 0, 0, 0, 0]))

    def test_filter_data_by_model_name(self):
        """Test filter table data with data name."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(kind="model1")
        self.assertEqual(len(filtered), 6)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]))
        np.testing.assert_array_equal(
            filtered.name, np.array(["model1", "model1", "model1", "model1", "model1", "model1"])
        )

    def test_filter_data_by_category(self):
        """Test filter table data with data category."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(category="formatted")
        self.assertEqual(len(filtered), 4)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2, 0.1, 0.2]))
        np.testing.assert_array_equal(
            filtered.category, np.array(["formatted", "formatted", "formatted", "formatted"])
        )

    def test_filter_data_by_analysis(self):
        """Test filter table data with associated analysis class."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(analysis="Fit2")
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(filtered.analysis, np.array(["Fit2", "Fit2"]))

    def test_filter_multiple(self):
        """Test filter table data with multiple attributes."""
        obj = ScatterTable.from_dataframe(self.reference)

        filtered = obj.filter(kind=0, category="raw", analysis="Fit1")
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(filtered.x, np.array([0.1, 0.2]))
        np.testing.assert_array_equal(filtered.class_id, np.array([0, 0]))
        np.testing.assert_array_equal(filtered.category, np.array(["raw", "raw"]))
        np.testing.assert_array_equal(filtered.analysis, np.array(["Fit1", "Fit1"]))

    def test_iter_class(self):
        """Test iterating over mini tables associated with different data UID."""
        obj = ScatterTable.from_dataframe(self.reference).filter(category="raw")

        class_iter = obj.iter_by_class()

        index, table_cls0 = next(class_iter)
        ref_table_cls0 = obj.filter(kind=0)
        self.assertEqual(index, 0)
        self.assertEqual(table_cls0, ref_table_cls0)

        index, table_cls1 = next(class_iter)
        ref_table_cls1 = obj.filter(kind=1)
        self.assertEqual(index, 1)
        self.assertEqual(table_cls1, ref_table_cls1)

    def test_iter_groups(self):
        """Test iterating over mini tables associated with multiple attributes."""
        obj = ScatterTable.from_dataframe(self.reference).filter(category="raw")

        class_iter = obj.iter_groups("class_id", "xval")

        (index, xval), table0 = next(class_iter)
        self.assertEqual(index, 0)
        self.assertEqual(xval, 0.1)
        self.assertEqual(len(table0), 2)
        np.testing.assert_array_equal(table0.y, [0.192, 0.567])

        (index, xval), table1 = next(class_iter)
        self.assertEqual(index, 0)
        self.assertEqual(xval, 0.2)
        self.assertEqual(len(table1), 2)
        np.testing.assert_array_equal(table1.y, [0.854, 0.488])

        (index, xval), table2 = next(class_iter)
        self.assertEqual(index, 1)
        self.assertEqual(xval, 0.1)
        self.assertEqual(len(table2), 1)
        np.testing.assert_array_equal(table2.y, [0.784])

        (index, xval), table3 = next(class_iter)
        self.assertEqual(index, 1)
        self.assertEqual(xval, 0.2)
        self.assertEqual(len(table3), 1)
        np.testing.assert_array_equal(table3.y, [0.672])

    def test_roundtrip_table(self):
        """Test ScatterTable is JSON serializable."""
        obj = ScatterTable.from_dataframe(self.reference)
        self.assertRoundTripSerializable(obj)