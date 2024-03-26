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
Experiment and analysis config dataclasses.
"""

import dataclasses
from typing import Tuple, Dict, Any

from qiskit.exceptions import QiskitError
from qiskit_experiments.version import __version__


@dataclasses.dataclass(frozen=True)
class DrawerConfig:
    """Store configuration settings for a Drawer class.

    This stores the current configuration of a :class:`.BaseDrawer` and can be used to reconstruct
    the drawer class using either the :meth:`drawer` property if the drawer class type is
    currently stored, or the :meth:`~.BaseDrawer.from_config` class method.
    """

    cls: type = None
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    figure_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    version: str = __version__

    def drawer(self):
        """Return the drawer class constructed from this config.

        Returns:
            BaseDrawer: The drawer reconstructed from the config.

        Raises:
            QiskitError: If the drawer class is not stored,
                         was not successful deserialized, or reconstruction
                         of the drawer class fails.
        """
        cls = self.cls
        if cls is None:
            raise QiskitError("No drawer class in drawer config")
        if isinstance(cls, dict):
            raise QiskitError(
                "Unable to load drawer class. Try manually loading "
                "drawer using `Analysis.plotter.drawer.from_config(config)` instead."
            )
        try:
            return cls.from_config(self)
        except Exception as ex:
            msg = "Unable to construct drawer from config."
            if cls.version != __version__:
                msg += (
                    f" Note that config version ({cls.version}) differs from the current"
                    f" qiskit-experiments version ({__version__}). You could try"
                    " installing a compatible qiskit-experiments version."
                )
            raise QiskitError(f"{msg}\nError Message:\n{str(ex)}") from ex


@dataclasses.dataclass(frozen=True)
class PlotterConfig:
    """Store configuration settings for a Drawer class.

    This stores the current configuration of a :class:`.BasePlotter` and can be used to reconstruct
    the plotter class using either the :meth:`plotter` property if the plotter class type is
    currently stored, or the :meth:`~.BasePlotter.from_config` class method.
    """

    cls: type = None
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    figure_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # drawer: Dict[str, Any] = dataclasses.field(default_factory=dict)
    drawer: DrawerConfig = None
    version: str = __version__

    def plotter(self):
        """Return the plotter of the analysis class constructed from this config.

        Returns:
            BasePlotter: The plotter reconstructed from the config.

        Raises:
            QiskitError: If the plotter class is not stored,
                         was not successful deserialized, or reconstruction
                         of the plotter class fails.
        """
        cls = self.cls
        if cls is None:
            raise QiskitError("No plotter class in plotter config")
        if isinstance(cls, dict):
            raise QiskitError(
                "Unable to load plotter class. Try manually loading "
                "analysis using `Analysis.plotter.from_config(config)` instead."
            )
        try:
            return cls.from_config(self)
        except Exception as ex:
            msg = "Unable to construct plotter from config."
            if cls.version != __version__:
                msg += (
                    f" Note that config version ({cls.version}) differs from the current"
                    f" qiskit-experiments version ({__version__}). You could try"
                    " installing a compatible qiskit-experiments version."
                )
            raise QiskitError(f"{msg}\nError Message:\n{str(ex)}") from ex


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    """Store configuration settings for an Analysis class.

    This stores the current configuration of a :class:`.BaseAnalysis` and can be used to reconstruct
    the analysis class using either the :meth:`analysis` property if the analysis class type is
    currently stored, or the :meth:`~.BaseAnalysis.from_config` class method of the appropriate
    experiment.
    """

    cls: type = None
    args: Tuple[Any] = dataclasses.field(default_factory=tuple)
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    plotter: PlotterConfig = None
    version: str = __version__

    def analysis(self):
        """Return the analysis class constructed from this config.

        Returns:
            BaseAnalysis: The analysis reconstructed from the config.

        Raises:
            QiskitError: If the analysis class is not stored,
                         was not successful deserialized, or reconstruction
                         of the analysis class fails.
        """
        cls = self.cls
        if cls is None:
            raise QiskitError("No analysis class in analysis config")
        if isinstance(cls, dict):
            raise QiskitError(
                "Unable to load analysis class. Try manually loading "
                "analysis using `Analysis.from_config(config)` instead."
            )
        try:
            return cls.from_config(self)
        except Exception as ex:
            msg = "Unable to construct analysis from config."
            if cls.version != __version__:
                msg += (
                    f" Note that config version ({cls.version}) differs from the current"
                    f" qiskit-experiments version ({__version__}). You could try"
                    " installing a compatible qiskit-experiments version."
                )
            raise QiskitError(f"{msg}\nError Message:\n{str(ex)}") from ex


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Store configuration settings for an Experiment class.

    This stores the current configuration of a :class:`.BaseExperiment` and can be used to
    reconstruct the experiment using either the :meth:`experiment` property if the experiment class
    type is currently stored, or the :meth:`~.BaseExperiment.from_config` class method of the
    appropriate experiment.
    """

    cls: type = None
    args: Tuple[Any] = dataclasses.field(default_factory=tuple)
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    experiment_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    transpile_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    run_options: Dict[str, Any] = dataclasses.field(default_factory=dict)
    analysis: AnalysisConfig = None
    version: str = __version__

    def experiment(self):
        """Return the experiment constructed from this config.

        Returns:
            BaseExperiment: The experiment reconstructed from the config.

        Raises:
            QiskitError: If the experiment class is not stored,
                         was not successful deserialized, or reconstruction
                         of the experiment fails.
        """
        cls = self.cls
        if cls is None:
            raise QiskitError("No experiment class in experiment config")
        if isinstance(cls, dict):
            raise QiskitError(
                "Unable to load experiment class. Try manually loading "
                "experiment using `Experiment.from_config(config)` instead."
            )
        try:
            return cls.from_config(self)
        except Exception as ex:
            msg = "Unable to construct experiments from config."
            if cls.version != __version__:
                msg += (
                    f" Note that config version ({cls.version}) differs from the current"
                    f" qiskit-experiments version ({__version__}). You could try"
                    " installing a compatible qiskit-experiments version."
                )
            raise QiskitError(f"{msg}\nError Message:\n{str(ex)}") from ex
