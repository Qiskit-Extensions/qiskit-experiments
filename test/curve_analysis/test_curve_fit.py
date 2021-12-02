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

# pylint: disable=invalid-name

"""Test curve fitting base class."""
from typing import List

import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.curve_analysis import CurveAnalysis, fit_function
from qiskit_experiments.curve_analysis.curve_data import (
    SeriesDef,
    FitData,
    ParameterRepr,
    FitOptions,
)
from qiskit_experiments.curve_analysis.data_processing import probability
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import ExperimentData
from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment
from uncertainties import correlated_values


def simulate_output_data(func, xvals, param_dict, **metadata):
    """Generate arbitrary fit data."""
    __shots = 100000

    expected_probs = func(xvals, **param_dict)
    counts = np.asarray(expected_probs * __shots, dtype=int)

    data = [
        {
            "counts": {"0": __shots - count, "1": count},
            "metadata": dict(xval=xi, qubits=(0,), experiment_type="fake_experiment", **metadata),
        }
        for xi, count in zip(xvals, counts)
    ]

    expdata = ExperimentData(experiment=FakeExperiment())
    for datum in data:
        expdata.add_data(datum)

    expdata.metadata["job_metadata"] = [{"run_options": {"meas_level": MeasLevel.CLASSIFIED}}]

    return expdata


def create_new_analysis(series: List[SeriesDef], fixed_params: List[str] = None) -> CurveAnalysis:
    """A helper function to create a mock analysis class instance."""

    class TestAnalysis(CurveAnalysis):
        """A mock analysis class to test."""

        __series__ = series
        __fixed_parameters__ = fixed_params or list()

    return TestAnalysis()


class TestFitData(QiskitExperimentsTestCase):
    """Unittest for fit data dataclass."""

    def test_get_value(self):
        """Get fit value from fit data object."""
        pcov = np.diag(np.ones(3))
        popt = np.asarray([1.0, 2.0, 3.0])
        fit_params = correlated_values(popt, pcov)

        data = FitData(
            popt=fit_params,
            popt_keys=["a", "b", "c"],
            pcov=pcov,
            reduced_chisq=0.0,
            dof=0,
            x_range=(0, 0),
            y_range=(0, 0),
        )

        a_val = data.fitval("a")
        self.assertEqual(a_val, fit_params[0])

        b_val = data.fitval("b")
        self.assertEqual(b_val, fit_params[1])

        c_val = data.fitval("c")
        self.assertEqual(c_val, fit_params[2])


class TestCurveAnalysisUnit(QiskitExperimentsTestCase):
    """Unittest for curve fit analysis."""

    def setUp(self):
        super().setUp()
        self.xvalues = np.linspace(1.0, 5.0, 10)

        # Description of test setting
        #
        # - This model contains three curves, namely, curve1, curve2, curve3
        # - Each curve can be represented by the same function
        # - Parameter amp and baseline are shared among all curves
        # - Each curve has unique lamb
        # - In total 5 parameters in the fit, namely, p0, p1, p2, p3
        #
        self.analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, baseline=p4
                    ),
                    filter_kwargs={"type": 1, "valid": True},
                    model_description=r"p_0 * \exp(p_1 x) + p4",
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p2, baseline=p4
                    ),
                    filter_kwargs={"type": 2, "valid": True},
                    model_description=r"p_0 * \exp(p_2 x) + p4",
                ),
                SeriesDef(
                    name="curve3",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p3, baseline=p4
                    ),
                    filter_kwargs={"type": 3, "valid": True},
                    model_description=r"p_0 * \exp(p_3 x) + p4",
                ),
            ],
        )
        self.err_decimal = 3

    def test_parsed_fit_params(self):
        """Test parsed fit params."""
        self.assertSetEqual(set(self.analysis._fit_params()), {"p0", "p1", "p2", "p3", "p4"})

    def test_parsed_init_guess(self):
        """Test parsed initial guess and boundaries."""
        default_p0 = self.analysis._default_options().p0
        default_bounds = self.analysis._default_options().bounds
        ref = {"p0": None, "p1": None, "p2": None, "p3": None, "p4": None}
        self.assertDictEqual(default_p0, ref)
        self.assertDictEqual(default_bounds, ref)

    def test_cannot_create_invalid_series_fit(self):
        """Test we cannot create invalid analysis instance."""
        invalid_series = [
            SeriesDef(
                name="fit1",
                fit_func=lambda x, p0: fit_function.exponential_decay(x, amp=p0),
            ),
            SeriesDef(
                name="fit2",
                fit_func=lambda x, p1: fit_function.exponential_decay(x, amp=p1),
            ),
        ]
        with self.assertRaises(AnalysisError):
            create_new_analysis(series=invalid_series)  # fit1 has param p0 while fit2 has p1

    def test_cannot_create_invalid_fixed_parameter(self):
        """Test we cannot create invalid analysis instance with wrong fixed value name."""
        valid_series = [
            SeriesDef(
                fit_func=lambda x, p0, p1: fit_function.exponential_decay(x, amp=p0, lamb=p1),
            ),
        ]
        with self.assertRaises(AnalysisError):
            create_new_analysis(
                series=valid_series,
                fixed_params=["not_existing_parameter"],  # this parameter is not defined
            )

    def test_arg_parse_and_get_option(self):
        """Test if option parsing works correctly."""
        user_option = {"x_key": "test_value", "test_key1": "value1", "test_key2": "value2"}

        # argument not defined in default option should be returned as extra option
        extra_option = self.analysis._arg_parse(**user_option)
        ref_option = {"test_key1": "value1", "test_key2": "value2"}
        self.assertDictEqual(extra_option, ref_option)

        # default option value is stored as class variable
        self.assertEqual(self.analysis._get_option("x_key"), "test_value")

    def test_data_extraction(self):
        """Test data extraction method."""
        self.analysis._arg_parse(x_key="xval")

        # data to analyze
        test_data0 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 1.0},
            type=1,
            valid=True,
        )

        # fake data
        test_data1 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": 0.5},
            type=2,
            valid=False,
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        self.analysis._extract_curves(
            experiment_data=test_data0, data_processor=probability(outcome="1")
        )

        raw_data = self.analysis._data(label="raw_data")

        xdata = raw_data.x
        ydata = raw_data.y
        sigma = raw_data.y_err
        d_index = raw_data.data_index

        # check if the module filter off data: valid=False
        self.assertEqual(len(xdata), 20)

        # check x values
        ref_x = np.concatenate((self.xvalues, self.xvalues))
        np.testing.assert_array_almost_equal(xdata, ref_x)

        # check y values
        ref_y = np.concatenate(
            (
                fit_function.exponential_decay(self.xvalues, amp=1.0),
                fit_function.exponential_decay(self.xvalues, amp=0.5),
            )
        )
        np.testing.assert_array_almost_equal(ydata, ref_y, decimal=self.err_decimal)

        # check series
        ref_series = np.concatenate((np.zeros(10, dtype=int), -1 * np.ones(10, dtype=int)))
        self.assertListEqual(list(d_index), list(ref_series))

        # check y errors
        ref_yerr = ref_y * (1 - ref_y) / 100000
        np.testing.assert_array_almost_equal(sigma, ref_yerr, decimal=self.err_decimal)

    def test_get_subset(self):
        """Test that get subset data from full data array."""
        # data to analyze
        fake_data = [
            {"data": 1, "metadata": {"xval": 1, "type": 1, "valid": True}},
            {"data": 2, "metadata": {"xval": 2, "type": 2, "valid": True}},
            {"data": 3, "metadata": {"xval": 3, "type": 1, "valid": True}},
            {"data": 4, "metadata": {"xval": 4, "type": 3, "valid": True}},
            {"data": 5, "metadata": {"xval": 5, "type": 3, "valid": True}},
            {"data": 6, "metadata": {"xval": 6, "type": 4, "valid": True}},  # this if fake
        ]
        expdata = ExperimentData(experiment=FakeExperiment())
        for datum in fake_data:
            expdata.add_data(datum)

        def _processor(datum):
            return datum["data"], datum["data"] * 2

        self.analysis._arg_parse(x_key="xval")
        self.analysis._extract_curves(expdata, data_processor=_processor)

        filt_data = self.analysis._data(series_name="curve1")
        np.testing.assert_array_equal(filt_data.x, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([1, 3], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([2, 6], dtype=float))

        filt_data = self.analysis._data(series_name="curve2")
        np.testing.assert_array_equal(filt_data.x, np.asarray([2], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([2], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([4], dtype=float))

        filt_data = self.analysis._data(series_name="curve3")
        np.testing.assert_array_equal(filt_data.x, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_equal(filt_data.y, np.asarray([4, 5], dtype=float))
        np.testing.assert_array_equal(filt_data.y_err, np.asarray([8, 10], dtype=float))


class TestCurveAnalysisIntegration(QiskitExperimentsTestCase):
    """Integration test for curve fit analysis through entire analysis.run function."""

    def setUp(self):
        super().setUp()
        self.xvalues = np.linspace(0.1, 1, 50)
        self.err_decimal = 2

    def test_run_single_curve_analysis(self):
        """Test analysis for single curve."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p2, baseline=p3
                    ),
                    model_description=r"p_0 \exp(p_1 x + p_2) + p_3",
                )
            ],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        test_data = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
        )
        default_opts = analysis._default_options()
        default_opts.p0 = {"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3}
        default_opts.result_parameters = [ParameterRepr("p1", "parameter_name", "unit")]

        results, _ = analysis._run_analysis(test_data, **default_opts.__dict__)
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        np.testing.assert_array_almost_equal(result.value, ref_popt, decimal=self.err_decimal)
        self.assertEqual(result.extra["dof"], 46)
        self.assertListEqual(result.extra["popt_keys"], ["p0", "p1", "p2", "p3"])
        self.assertDictEqual(result.extra["fit_models"], {"curve1": r"p_0 \exp(p_1 x + p_2) + p_3"})

        # special entry formatted for database
        result = results[1]
        self.assertEqual(result.name, "parameter_name")
        self.assertEqual(result.unit, "unit")
        self.assertAlmostEqual(result.value.nominal_value, ref_p1, places=self.err_decimal)

    def test_run_single_curve_fail(self):
        """Test analysis returns status when it fails."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p2, baseline=p3
                    ),
                )
            ],
        )
        ref_p0 = 0.9
        ref_p1 = 2.5
        ref_p2 = 0.0
        ref_p3 = 0.1

        test_data = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p2, "baseline": ref_p3},
        )
        default_opts = analysis._default_options()
        default_opts.p0 = {"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3}
        default_opts.bounds = {"p0": [-10, 0], "p1": [-10, 0], "p2": [-10, 0], "p3": [-10, 0]}
        default_opts.return_data_points = True

        # Try to fit with infeasible parameter boundary. This should fail.
        results, _ = analysis._run_analysis(test_data, **default_opts.__dict__)

        # This returns only data point entry
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "@Data_TestAnalysis")

    def test_run_two_curves_with_same_fitfunc(self):
        """Test analysis for two curves. Curves shares fit model."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p1, x0=p3, baseline=p4
                    ),
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3, p4: fit_function.exponential_decay(
                        x, amp=p0, lamb=p2, x0=p3, baseline=p4
                    ),
                    filter_kwargs={"exp": 1},
                ),
            ],
        )
        ref_p0 = 0.9
        ref_p1 = 7.0
        ref_p2 = 5.0
        ref_p3 = 0.0
        ref_p4 = 0.1

        test_data0 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p1, "x0": ref_p3, "baseline": ref_p4},
            exp=0,
        )

        test_data1 = simulate_output_data(
            func=fit_function.exponential_decay,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "lamb": ref_p2, "x0": ref_p3, "baseline": ref_p4},
            exp=1,
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        default_opts = analysis._default_options()
        default_opts.p0 = {"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3, "p4": ref_p4}

        results, _ = analysis._run_analysis(test_data0, **default_opts.__dict__)
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3, ref_p4])

        # check result data
        np.testing.assert_array_almost_equal(result.value, ref_popt, decimal=self.err_decimal)

    def test_run_two_curves_with_two_fitfuncs(self):
        """Test analysis for two curves. Curves shares fit parameters."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.cos(
                        x, amp=p0, freq=p1, phase=p2, baseline=p3
                    ),
                    filter_kwargs={"exp": 0},
                ),
                SeriesDef(
                    name="curve2",
                    fit_func=lambda x, p0, p1, p2, p3: fit_function.sin(
                        x, amp=p0, freq=p1, phase=p2, baseline=p3
                    ),
                    filter_kwargs={"exp": 1},
                ),
            ],
        )
        ref_p0 = 0.1
        ref_p1 = 2
        ref_p2 = -0.3
        ref_p3 = 0.5

        test_data0 = simulate_output_data(
            func=fit_function.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=0,
        )

        test_data1 = simulate_output_data(
            func=fit_function.sin,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
            exp=1,
        )

        # merge two experiment data
        for datum in test_data1.data():
            test_data0.add_data(datum)

        default_opts = analysis._default_options()
        default_opts.p0 = {"p0": ref_p0, "p1": ref_p1, "p2": ref_p2, "p3": ref_p3}

        results, _ = analysis._run_analysis(test_data0, **default_opts.__dict__)
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p2, ref_p3])

        # check result data
        np.testing.assert_array_almost_equal(result.value, ref_popt, decimal=self.err_decimal)

    def test_run_fixed_parameters(self):
        """Test analysis when some of parameters are fixed."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, fixed_p2, p3: fit_function.cos(
                        x, amp=p0, freq=p1, phase=fixed_p2, baseline=p3
                    ),
                ),
            ],
            fixed_params=["fixed_p2"],
        )

        ref_p0 = 0.1
        ref_p1 = 2
        ref_p2 = -0.3
        ref_p3 = 0.5

        test_data = simulate_output_data(
            func=fit_function.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
        )

        default_opts = analysis._default_options()
        default_opts.p0 = {"p0": ref_p0, "p1": ref_p1, "p3": ref_p3}
        default_opts.fixed_p2 = ref_p2

        results, _ = analysis._run_analysis(test_data, **default_opts.__dict__)
        result = results[0]

        ref_popt = np.asarray([ref_p0, ref_p1, ref_p3])

        # check result data
        np.testing.assert_array_almost_equal(result.value, ref_popt, decimal=self.err_decimal)

    def test_fixed_param_is_missing(self):
        """Test raising an analysis error when fixed parameter is missing."""
        analysis = create_new_analysis(
            series=[
                SeriesDef(
                    name="curve1",
                    fit_func=lambda x, p0, p1, fixed_p2, p3: fit_function.cos(
                        x, amp=p0, freq=p1, phase=fixed_p2, baseline=p3
                    ),
                ),
            ],
            fixed_params=["fixed_p2"],
        )

        ref_p0 = 0.1
        ref_p1 = 2
        ref_p2 = -0.3
        ref_p3 = 0.5

        test_data = simulate_output_data(
            func=fit_function.cos,
            xvals=self.xvalues,
            param_dict={"amp": ref_p0, "freq": ref_p1, "phase": ref_p2, "baseline": ref_p3},
        )

        default_opts = analysis._default_options()

        # do not define fixed_p2 here
        default_opts.p0 = {"p0": ref_p0, "p1": ref_p1, "p3": ref_p3}

        with self.assertRaises(AnalysisError):
            analysis._run_analysis(test_data, **default_opts.__dict__)


class TestFitOptions(QiskitExperimentsTestCase):
    """Unittest for fit option object."""

    def test_empty(self):
        """Test if default value is automatically filled."""
        opt = FitOptions(["p0", "p1", "p2"])

        # bounds should be default to inf tuple. otherwise crashes the scipy fitter.
        ref_opts = {
            "p0": {"p0": None, "p1": None, "p2": None},
            "bounds": {"p0": (-np.inf, np.inf), "p1": (-np.inf, np.inf), "p2": (-np.inf, np.inf)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_dict(self):
        """Create option and fill with dictionary."""
        opt = FitOptions(
            ["p0", "p1", "p2"],
            default_p0={"p0": 0, "p1": 1, "p2": 2},
            default_bounds={"p0": (0, 1), "p1": (1, 2), "p2": (2, 3)},
        )

        ref_opts = {
            "p0": {"p0": 0.0, "p1": 1.0, "p2": 2.0},
            "bounds": {"p0": (0.0, 1.0), "p1": (1.0, 2.0), "p2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_create_option_with_array(self):
        """Create option and fill with array."""
        opt = FitOptions(
            ["p0", "p1", "p2"],
            default_p0=[0, 1, 2],
            default_bounds=[(0, 1), (1, 2), (2, 3)],
        )

        ref_opts = {
            "p0": {"p0": 0.0, "p1": 1.0, "p2": 2.0},
            "bounds": {"p0": (0.0, 1.0), "p1": (1.0, 2.0), "p2": (2.0, 3.0)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_override_partial_dict(self):
        """Create option and override value with partial dictionary."""
        opt = FitOptions(["p0", "p1", "p2"])
        opt.p0.set_if_empty(p1=3)

        ref_opts = {
            "p0": {"p0": None, "p1": 3.0, "p2": None},
            "bounds": {"p0": (-np.inf, np.inf), "p1": (-np.inf, np.inf), "p2": (-np.inf, np.inf)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_assigned_value(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["p0", "p1", "p2"])
        opt.p0.set_if_empty(p1=3)
        opt.p0.set_if_empty(p1=5)

        ref_opts = {
            "p0": {"p0": None, "p1": 3.0, "p2": None},
            "bounds": {"p0": (-np.inf, np.inf), "p1": (-np.inf, np.inf), "p2": (-np.inf, np.inf)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_can_override_assigned_value_with_dict_access(self):
        """Test override already assigned value with direct dict access."""
        opt = FitOptions(["p0", "p1", "p2"])
        opt.p0["p1"] = 3
        opt.p0["p1"] = 5

        ref_opts = {
            "p0": {"p0": None, "p1": 5.0, "p2": None},
            "bounds": {"p0": (-np.inf, np.inf), "p1": (-np.inf, np.inf), "p2": (-np.inf, np.inf)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_cannot_override_user_option(self):
        """Test cannot override already assigned value."""
        opt = FitOptions(["p0", "p1", "p2"], default_p0={"p1": 3})
        opt.p0.set_if_empty(p1=5)

        ref_opts = {
            "p0": {"p0": None, "p1": 3, "p2": None},
            "bounds": {"p0": (-np.inf, np.inf), "p1": (-np.inf, np.inf), "p2": (-np.inf, np.inf)},
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_set_operation(self):
        """Test if set works and duplicated entry is removed."""
        opt1 = FitOptions(["p0", "p1"], default_p0=[0, 1])
        opt2 = FitOptions(["p0", "p1"], default_p0=[0, 1])
        opt3 = FitOptions(["p0", "p1"], default_p0=[0, 2])

        opts = set()
        opts.add(opt1)
        opts.add(opt2)
        opts.add(opt3)

        self.assertEqual(len(opts), 2)

    def test_detect_invalid_p0(self):
        """Test if invalid p0 raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["p0", "p1", "p2"], default_p0=[0, 1])

    def test_detect_invalid_bounds(self):
        """Test if invalid bounds raises Error."""
        with self.assertRaises(AnalysisError):
            # less element
            FitOptions(["p0", "p1", "p2"], default_bounds=[(0, 1), (1, 2)])

        with self.assertRaises(AnalysisError):
            # not min-max tuple
            FitOptions(["p0", "p1", "p2"], default_bounds=[0, 1, 2])

        with self.assertRaises(AnalysisError):
            # max-min tuple
            FitOptions(["p0", "p1", "p2"], default_bounds=[(1, 0), (2, 1), (3, 2)])

    def test_detect_invalid_key(self):
        """Test if invalid key raises Error."""
        opt = FitOptions(["p0", "p1", "p2"])

        with self.assertRaises(AnalysisError):
            opt.p0.set_if_empty(p3=3)

    def test_set_extra_options(self):
        """Add extra fitter options."""
        opt = FitOptions(
            ["p0", "p1", "p2"], default_p0=[0, 1, 2], default_bounds=[(0, 1), (1, 2), (2, 3)]
        )
        opt.add_extra_options(ex1=0, ex2=1)

        ref_opts = {
            "p0": {"p0": 0.0, "p1": 1.0, "p2": 2.0},
            "bounds": {"p0": (0.0, 1.0), "p1": (1.0, 2.0), "p2": (2.0, 3.0)},
            "ex1": 0,
            "ex2": 1,
        }

        self.assertDictEqual(opt.options, ref_opts)

    def test_complicated(self):
        """Test for realistic operations for algorithmic guess with user options."""
        user_p0 = {"p0": 1, "p1": None}
        user_bounds = {"p0": None, "p1": (-100, 100)}

        opt = FitOptions(
            ["p0", "p1", "p2"],
            default_p0=user_p0,
            default_bounds=user_bounds,
        )

        # similar computation in algorithmic guess

        opt.p0.set_if_empty(p0=5)  # this is ignored because user already provided initial guess
        opt.p0.set_if_empty(p1=opt.p0["p0"] * 2 + 3)  # user provided guess propagates

        opt.bounds.set_if_empty(p0=(0, 10))  # this will be set
        opt.add_extra_options(fitter="algo1")

        opt1 = opt.copy()  # copy options while keeping previous values
        opt1.p0.set_if_empty(p2=opt1.p0["p0"] + opt1.p0["p1"])

        opt2 = opt.copy()
        opt2.p0.set_if_empty(p2=opt2.p0["p0"] * 2)  # add another p2 value

        ref_opt1 = {
            "p0": {"p0": 1.0, "p1": 5.0, "p2": 6.0},
            "bounds": {"p0": (0.0, 10.0), "p1": (-100.0, 100.0), "p2": (-np.inf, np.inf)},
            "fitter": "algo1",
        }

        ref_opt2 = {
            "p0": {"p0": 1.0, "p1": 5.0, "p2": 2.0},
            "bounds": {"p0": (0.0, 10.0), "p1": (-100.0, 100.0), "p2": (-np.inf, np.inf)},
            "fitter": "algo1",
        }

        self.assertDictEqual(opt1.options, ref_opt1)
        self.assertDictEqual(opt2.options, ref_opt2)
