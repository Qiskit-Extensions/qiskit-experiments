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
Plotting functions for experiment analysis
"""
from typing import Callable, Optional, Dict
import numpy as np

from qiskit_experiments.matplotlib import pyplot, requires_matplotlib

# pylint: disable = unused-import
from qiskit_experiments.matplotlib import HAS_MATPLOTLIB


@requires_matplotlib
def plot_curve_fit(
    func: Callable,
    result: Dict,
    fit_uncertainty: bool = False,
    ax=None,
    num_fit_points: int = 100,
    labelsize: int = 14,
    grid: bool = True,
    **kwargs,
):
    """Generate plot of a curve fit analysis result.

    Wraps :func:`matplotlib.pyplot.plot`.

    Args:
        func: the fit function for curve_fit.
        result: a result dictionary from curve_fit.
        fit_uncertainty: if True plot the fit uncertainty from popt_err.
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        num_fit_points: the number of points to plot for xrange.
        labelsize: label size for plot
        grid: Show grid on plot.
        **kwargs: Additional options for matplotlib.pyplot.plot

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.

    Raises:
        ImportError: if matplotlib is not installed.
    """
    if ax is None:
        figure = pyplot.figure()
        ax = figure.subplots()

    # Default plot options
    plot_opts = kwargs.copy()
    if "color" not in plot_opts:
        plot_opts["color"] = "blue"
    if "linestyle" not in plot_opts:
        plot_opts["linestyle"] = "-"
    if "linewidth" not in plot_opts:
        plot_opts["linewidth"] = 2

    # Result data
    fit_params = result["popt"]
    param_keys = result.get("popt_keys")
    fit_errors = result.get("popt_err")
    xmin, xmax = result["xrange"]

    # Plot fit data
    xs = np.linspace(xmin, xmax, num_fit_points)
    if param_keys:
        ys_fit = func(xs, **dict(zip(param_keys, fit_params)))
    else:
        ys_fit = func(xs, *fit_params)
    ax.plot(xs, ys_fit, **plot_opts)

    # Plot standard error interval
    if fit_uncertainty and fit_errors is not None:
        if param_keys:
            params_upper = {}
            params_lower = {}
            for key, param, error in zip(param_keys, fit_params, fit_errors):
                params_upper[key] = param + error
                params_lower[key] = param - error
            ys_upper = func(xs, **params_upper)
            ys_lower = func(xs, **params_lower)
        else:
            params_upper = [param + error for param, error in zip(fit_params, fit_errors)]
            params_lower = [param - error for param, error in zip(fit_params, fit_errors)]
            ys_upper = func(xs, *params_upper)
            ys_lower = func(xs, *params_lower)
        ax.fill_between(xs, ys_lower, ys_upper, alpha=0.1, color=plot_opts["color"])

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax


@requires_matplotlib
def plot_scatter(
    xdata: np.ndarray,
    ydata: np.ndarray,
    ax=None,
    labelsize: int = 14,
    grid: bool = True,
    **kwargs,
):
    """Generate a scatter plot of xy data.

    Wraps :func:`matplotlib.pyplot.scatter`.

    Args:
        xdata: xdata used for fitting
        ydata: ydata used for fitting
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        labelsize: label size for plot
        grid: Show grid on plot.
        **kwargs: Additional options for :func:`matplotlib.pyplot.scatter`

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.
    """
    if ax is None:
        figure = pyplot.figure()
        ax = figure.subplots()

    # Default plot options
    plot_opts = kwargs.copy()
    if "c" not in plot_opts:
        plot_opts["c"] = "grey"
    if "marker" not in plot_opts:
        plot_opts["marker"] = "x"
    if "alpha" not in plot_opts:
        plot_opts["alpha"] = 0.8

    # Plot data
    ax.scatter(xdata, ydata, **plot_opts)

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax


@requires_matplotlib
def plot_errorbar(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    ax=None,
    labelsize: int = 14,
    grid: bool = True,
    **kwargs,
):
    """Generate an errorbar plot of xy data.

    Wraps :func:`matplotlib.pyplot.errorbar`

    Args:
        xdata: xdata used for fitting
        ydata: ydata used for fitting
        sigma: Optional, standard deviation of ydata
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        labelsize: label size for plot
        grid: Show grid on plot.
        **kwargs: Additional options for :func:`matplotlib.pyplot.errorbar`

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.
    """
    if ax is None:
        figure = pyplot.figure()
        ax = figure.subplots()

    # Default plot options
    plot_opts = kwargs.copy()
    if "color" not in plot_opts:
        plot_opts["color"] = "red"
    if "marker" not in plot_opts:
        plot_opts["marker"] = "."
    if "markersize" not in plot_opts:
        plot_opts["markersize"] = 9
    if "linestyle" not in plot_opts:
        plot_opts["linestyle"] = "None"

    # Plot data
    ax.errorbar(xdata, ydata, yerr=sigma, **plot_opts)

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax


@requires_matplotlib
def plot_contourf(
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    ax=None,
    labelsize: int = 14,
    grid: bool = True,
    **kwargs,
):
    """Generate a contour plot of xyz data.

    Wraps :func:`matplotlib.pyplot.contourf`.

    Args:
        xdata: xdata used for plotting
        ydata: ydata used for plotting
        zdata: zdata used for plotting
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        labelsize: label size for plot
        grid: Show grid on plot.
        **kwargs: Additional options for :func:`matplotlib.pyplot.contourf`

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.
    """
    if ax is None:
        figure = pyplot.figure()
        ax = figure.subplots()

    # Default plot options
    plot_opts = kwargs.copy()
    # if "c" not in plot_opts:
    #     plot_opts["c"] = "grey"
    # if "marker" not in plot_opts:
    #     plot_opts["marker"] = "x"

    # Plot data
    ax.contourf(xdata, ydata, zdata, **plot_opts)

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax
