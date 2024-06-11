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
"""
Test integration of plotter.
"""
from test.base import QiskitExperimentsTestCase
from test.visualization.test_utils import LoggingTestCase

from copy import deepcopy

import numpy as np
from uncertainties import ufloat
from qiskit_experiments.framework import AnalysisResultData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from .mock_drawer import MockDrawer
from .mock_plotter import MockPlotter


class TestPlotter(QiskitExperimentsTestCase):
    """Test the generic plotter interface."""

    def test_warn_unknown_data_key(self):
        """Test that setting an unknown data-key raises a warning."""
        plotter = MockPlotter(MockDrawer())

        # TODO: Add check for no-warnings. assertNoWarns only available from Python 3.10+

        # An unknown data-key must raise a warning if it is used to set series data.
        with self.assertWarns(UserWarning):
            plotter.set_series_data("dummy_series", unknown_data_key=[0, 1, 2])

    def test_series_data_end_to_end(self):
        """Test end-to-end for series data setting and retrieving."""
        plotter = MockPlotter(MockDrawer())

        series_data = {
            "seriesA": {
                "x": 0,
                "y": "1",
                "z": [2],
            },
            "seriesB": {
                "x": 1,
                "y": 0.5,
            },
        }
        unexpected_data = ["a", True, 0]
        expected_series_data = deepcopy(series_data)
        expected_series_data["seriesA"]["unexpected_data"] = unexpected_data

        for series, data in series_data.items():
            plotter.set_series_data(series, **data)

        with self.assertWarns(UserWarning):
            plotter.set_series_data("seriesA", unexpected_data=unexpected_data)

        for series, data in expected_series_data.items():
            self.assertTrue(series in plotter.series)
            self.assertTrue(plotter.data_exists_for(series, list(data.keys())))
            for data_key, value in data.items():
                # Must index [0] for `data_for` as it returns a tuple.
                self.assertEqual(value, plotter.data_for(series, data_key)[0])

    def test_supplementary_data_end_to_end(self):
        """Test end-to-end for figure data setting and retrieval."""
        plotter = MockPlotter(MockDrawer())

        expected_supplementary_data = {
            "report_text": "Lorem ipsum",
            "supplementary_data_key": 3e9,
        }

        plotter.set_supplementary_data(**expected_supplementary_data)

        # Check if figure data has been stored and can be retrieved
        for key, expected_value in expected_supplementary_data.items():
            actual_value = plotter.supplementary_data[key]
            self.assertEqual(
                expected_value,
                actual_value,
                msg=f"Actual figure data value for {key} data-key is not as expected: {actual_value} "
                f"(actual) vs {expected_value} (expected)",
            )


class TestCurvePlotter(LoggingTestCase):
    """Test case for Qiskit Experiments curve plotter based on logging."""

    def test_all_data(self):
        """Visualize all curve information."""
        plotter = CurvePlotter(drawer=MplDrawer())
        plotter.set_series_data(
            series_name="test",
            x=[0, 1],
            y=[1, 1],
            x_formatted=[2, 3],
            y_formatted=[2, 2],
            y_formatted_err=[0.1, 0.1],
            x_interp=[4, 5],
            y_interp=[3, 3],
            y_interp_err=[0.2, 0.2],
        )
        self.assertDrawerAPICallEqual(
            plotter,
            expected=[
                "Calling initialize_canvas",
                "Calling scatter with x_data=[2, 3], y_data=[2, 2], x_err=None, y_err=[0.1, 0.1], "
                "name='test', label=None, legend=True, options={'zorder': 2}",
                "Calling scatter with x_data=[0, 1], y_data=[1, 1], x_err=None, y_err=None, "
                "name='test', label=None, legend=False, options={'zorder': 1, 'color': 'gray'}",
                "Calling line with x_data=[4, 5], y_data=[3, 3], "
                "name='test', label=None, legend=False, options={'zorder': 3}",
                "Calling filled_y_area with x_data=[4, 5], y_ub=[3.2, 3.2], y_lb=[2.8, 2.8], "
                "name='test', label=None, legend=False, options={'alpha': 0.7, 'zorder': 5}",
                "Calling filled_y_area with x_data=[4, 5], y_ub=[3.6, 3.6], y_lb=[2.4, 2.4], "
                "name='test', label=None, legend=False, options={'alpha': 0.3, 'zorder': 5}",
                "Calling format_canvas",
            ],
        )

    def test_supplementary(self):
        """Visualize with fitting report."""
        test_result = AnalysisResultData(name="test", value=ufloat(1, 0.2))

        plotter = CurvePlotter(drawer=MplDrawer())
        plotter.set_series_data(
            series_name="test",
            x=[0, 1],
            y=[1, 1],
        )
        plotter.set_supplementary_data(
            fit_red_chi=3.0,
            primary_results=[test_result],
        )
        # pylint: disable=anomalous-backslash-in-string
        self.assertDrawerAPICallEqual(
            plotter,
            expected=[
                "Calling initialize_canvas",
                "Calling scatter with x_data=[0, 1], y_data=[1, 1], x_err=None, y_err=None, "
                "name='test', label=None, legend=True, options={'zorder': 1}",
                "Calling textbox with description='test =   1 ±  0.2\n"
                "reduced-$\chi^2$ =  3', rel_pos=None, options={}",
                "Calling format_canvas",
            ],
        )

    def test_fit_y_error_missing(self):
        """Visualize curve that fitting doesn't work well, i.e. cov-matrix diverges."""
        plotter = CurvePlotter(drawer=MplDrawer())
        plotter.set_series_data(
            series_name="test",
            x=[0, 1],
            y=[1, 1],
            x_formatted=[2, 3],
            y_formatted=[2, 2],
            y_formatted_err=[0.1, 0.1],
            x_interp=[4, 5],
            y_interp=[3, 3],  # y_interp_err is gone
        )
        self.assertDrawerAPICallEqual(
            plotter,
            expected=[
                "Calling initialize_canvas",
                "Calling scatter with x_data=[2, 3], y_data=[2, 2], x_err=None, y_err=[0.1, 0.1], "
                "name='test', label=None, legend=True, options={'zorder': 2}",
                "Calling scatter with x_data=[0, 1], y_data=[1, 1], x_err=None, y_err=None, "
                "name='test', label=None, legend=False, options={'zorder': 1, 'color': 'gray'}",
                "Calling line with x_data=[4, 5], y_data=[3, 3], "
                "name='test', label=None, legend=False, options={'zorder': 3}",
                "Calling format_canvas",
            ],
        )

    def test_fit_fails(self):
        """Visualize curve only contains formatted data, i.e. fit completely fails."""
        plotter = CurvePlotter(drawer=MplDrawer())
        plotter.set_series_data(
            series_name="test",
            x_formatted=[2, 3],
            y_formatted=[2, 2],
            y_formatted_err=[0.1, 0.1],
        )
        self.assertDrawerAPICallEqual(
            plotter,
            expected=[
                "Calling initialize_canvas",
                "Calling scatter with x_data=[2, 3], y_data=[2, 2], x_err=None, y_err=[0.1, 0.1], "
                "name='test', label=None, legend=True, options={'zorder': 2}",
                "Calling format_canvas",
            ],
        )

    def test_two_series(self):
        """Visualize curve with two series."""
        plotter = CurvePlotter(drawer=MplDrawer())
        plotter.set_series_data(
            series_name="test1",
            x_formatted=[2, 3],
            y_formatted=[2, 2],
            y_formatted_err=[0.1, 0.1],
        )
        plotter.set_series_data(
            series_name="test2",
            x_formatted=[2, 3],
            y_formatted=[4, 4],
            y_formatted_err=[0.2, 0.2],
        )
        self.assertDrawerAPICallEqual(
            plotter,
            expected=[
                "Calling initialize_canvas",
                "Calling scatter with x_data=[2, 3], y_data=[2, 2], x_err=None, y_err=[0.1, 0.1], "
                "name='test1', label=None, legend=True, options={'zorder': 2}",
                "Calling scatter with x_data=[2, 3], y_data=[4, 4], x_err=None, y_err=[0.2, 0.2], "
                "name='test2', label=None, legend=True, options={'zorder': 2}",
                "Calling format_canvas",
            ],
        )

    def test_scatter_partly_missing(self):
        """Visualize curve include some defect."""
        plotter = CurvePlotter(drawer=MplDrawer())
        plotter.set_series_data(
            series_name="test",
            x_formatted=[2, 3],
            y_formatted=[np.nan, 2],
            y_formatted_err=[np.nan, 0.1],
        )
        self.assertDrawerAPICallEqual(
            plotter,
            expected=[
                "Calling initialize_canvas",
                "Calling scatter with x_data=[2, 3], y_data=[nan, 2], x_err=None, y_err=[nan, 0.1], "
                "name='test', label=None, legend=True, options={'zorder': 2}",
                "Calling format_canvas",
            ],
        )
