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
==========================================================
Experiment Framework (:mod:`qiskit_experiments.framework`)
==========================================================

.. currentmodule:: qiskit_experiments.framework

Overview
========

The experiment framework broadly defines an experiment as the execution of one or more
circuits on a device, and analysis of the resulting measurement data
to return one or more derived results.

The interface for running an experiment is through the ``Experiment`` classes subclassed from
:class:`.BaseExperiment`, such as those contained in the :mod:`~qiskit_experiments.library`. The
following pseudo-code illustrates the typical workflow in Qiskit Experiments for

- Initializing a new experiment
- Running the experiment on a backend
- Saving result to an online database (for compatible providers)
- Viewing analysis results

.. code-block:: python

    # Import an experiment
    from qiskit_experiments.library import SomeExperiment

    # Initialize with desired qubits and options
    exp = SomeExperiment(physical_qubits, **options)

    # Run on a backend
    exp_data = exp.run(backend)

    # Wait for execution and analysis to finish
    exp_data.block_for_results()

    # Optionally save results to database
    exp_data.save()

    # View analysis results
    for result in exp_data.analysis_results():
        print(result)

The experiment class contains information for generating circuits and analysis
of results. These can typically be configured with a variety of options.
Once all options are set, you can call the :meth:`.BaseExperiment.run` method to run
the experiment on a Qiskit compatible ``backend``.

The steps of running an experiment involves generation experimental circuits
according to the options you set and submission of a job to the specified
``backend``. Once the job has finished executing an analysis job performs
data analysis of the experiment execution results.

The result of running an experiment is an :class:`ExperimentData` container
which contains the analysis results, any figures generated during analysis,
and the raw measurement data. These can each be accessed using the
:meth:`.ExperimentData.analysis_results`, :meth:`.ExperimentData.figure`
and :meth:`.ExperimentData.data` methods respectively. Additional metadata
for the experiment itself can be added via :meth:`.ExperimentData.metadata`.


Classes
=======

Experiment Data Classes
***********************

.. autosummary::
    :toctree: ../stubs/

    ExperimentData
    ExperimentStatus
    AnalysisStatus
    AnalysisResult
    AnalysisResultData
    AnalysisResultTable
    ExperimentConfig
    AnalysisConfig
    ExperimentEncoder
    ExperimentDecoder
    FigureData

.. _composite-experiment:

Composite Experiment Classes
****************************

.. autosummary::
    :toctree: ../stubs/

    ParallelExperiment
    BatchExperiment
    ChainedExperiment
    CompositeAnalysis

Base Classes
************

.. autosummary::
    :toctree: ../stubs/

    BaseExperiment
    BaseAnalysis

Experiment Configuration Helper Classes
***************************************

.. autosummary::
    :toctree: ../stubs/

    BackendData
    BackendTiming
    RestlessMixin

"""
from qiskit.providers.options import Options
from qiskit_experiments.framework.backend_data import BackendData
from qiskit_experiments.framework.analysis_result import AnalysisResult
from qiskit_experiments.framework.experiment_data import (
    ExperimentStatus,
    AnalysisStatus,
    FigureData,
)
from .base_analysis import BaseAnalysis
from .base_experiment import BaseExperiment
from .backend_timing import BackendTiming
from .configs import ExperimentConfig, AnalysisConfig
from .analysis_result_data import AnalysisResultData
from .analysis_result_table import AnalysisResultTable
from .experiment_data import ExperimentData
from .composite import (
    ParallelExperiment,
    BatchExperiment,
    ChainedExperiment,
    CompositeAnalysis,
)
from .json import ExperimentEncoder, ExperimentDecoder
from .restless_mixin import RestlessMixin
from .package_deps import numpy_version
