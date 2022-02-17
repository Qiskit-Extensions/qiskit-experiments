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

.. note::

    This page provides useful information for developers to implement new
    experiments.

Overview
========

The experiment framework broadly defines an experiment as the execution of 1 or more
circuits on a device, and analysis of the resulting measurement data
to return 1 or more derived results.

The interface for running an experiment is through the *Experiment* classes,
such as those contained in the :mod:`qiskit_experiments.library`
The following pseudo-code illustrates the typical workflow in Qiskit Experiments
for

- Initializing a new experiment
- Running the experiment on a backend
- Saving result to an online database (for compatible providers)
- Viewing analysis results

.. code-block:: python

    # Import an experiment
    from qiskit_experiments.library import SomeExperiment

    # Initialize with desired qubits and options
    exp = SomeExperiment(qubits, **options)

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
Once all options are set, you can call :meth:`BaseExperiment.run` method to run
the experiment on a Qiskit compatible ``backend``.

The steps of running an experiment involves generation experimental circuits
according to the options you set and submission of a job to the specified
``backend``. Once the job has finished executing an analysis job performs
data analysis of the experiment execution results.

The result of running an experiment is an :class:`ExperimentData` container
which contains the analysis results, any figures generated during analysis,
and the raw measurement data. These can each be accessed using the
:meth:`ExperimentData.analysis_results`, :meth:`ExperimentData.figure`
and :meth:`ExperimentData.data` methods respectively.

Analysis/plotting is done in a separate child thread, so it doesn't block the
main thread. Since matplotlib doesn't support GUI mode in a child threads, the
figures generated during analysis need to use a non-GUI canvas. The default is
:class:`~matplotlib.backends.backend_svg.FigureCanvasSVG`, but you can change it to a different
`non-interactive backend
<https://matplotlib.org/stable/tutorials/introductory/usage.html#the-builtin-backends>`_
by setting the ``qiskit_experiments.framework.matplotlib.default_figure_canvas``
attribute. For example, you can set ``default_figure_canvas`` to
:class:`~matplotlib.backends.backend_agg.FigureCanvasAgg` to use the
``AGG`` backend.

For experiments run through a compatible provider such as the
`IBMQ provider <https://github.com/Qiskit/qiskit-ibmq-provider>`_
the :class:`ExperimentData` object can be saved to an online experiment
database by calling the :meth:`ExperimentData.save` method. This data can
later be retrieved by its unique :attr:`~ExperimentData.experiment_id`* string
using :meth:`ExperimentData.load`.


Composite Experiments
=====================

The experiment classes :class:`ParallelExperiment` and :class:`BatchExperiment`
provide a way of combining separate component experiments for execution as a
single composite experiment.

- A :class:`ParallelExperiment` combines all the sub experiment circuits
  into circuits which run the component gates in parallel on the
  respective qubits. The marginalization of measurement data for analysis
  of each sub-experiment is handled automatically. To run as a parallel
  experiment each sub experiment must be defined on a independent subset
  of device qubits.

- A :class:`BatchExperiment` combines the sub-experiment circuits into a
  single large job that runs all the circuits for each experiment in series.
  Filtering the batch result data for analysis for each sub-experiment is
  handled automatically.


Creating Custom Experiments
===========================

Qiskit experiments provides a framework for creating custom experiments which
can be through Qiskit and stored in the online database when run through the IBMQ
provider. You may use this framework to release your own module of experiments
subject to the requirements of the Apache 2.0 license.

Creating a custom experiment is done by subclassing the
:class:`BaseExperiment` and :class:`BaseAnalysis` classes.

- The *experiment* class generates the list of circuits to be executed on the
  backend and any corresponding metadata that is required for the analysis
  of measurement results.

- The *analysis* class performs post-processing of the measurement results
  after execution. Analysis classes can be re-used between experiments so
  you can either use one of the included analysis classes if appropriate or
  implement your own.

Experiment Subclasses
*********************

To create an experiment subclass

- Implement the abstract :meth:`BaseExperiment.circuits` method.
  This should return a list of ``QuantumCircuit`` objects defining
  the experiment payload.

- Call the :meth:`BaseExperiment.__init__` method during the subclass
  constructor with a list of physical qubits. The length of this list must
  be equal to the number of qubits in each circuit and is used to map these
  circuits to this layout during execution.
  Arguments in the constructor can be overridden so that a subclass can
  be initialized with some experiment configuration.

Optionally the following methods can also be overridden in the subclass to
allow configuring various experiment and execution options

- :meth:`BaseExperiment._default_experiment_options`
  to set default values for configurable option parameters for the experiment.

- :meth:`BaseExperiment._default_transpile_options`
  to set custom default values for the ``qiskit.transpile`` used to
  transpile the generated circuits before execution.

- :meth:`BaseExperiment._default_run_options`
  to set default backend options for running the transpiled circuits on a backend.

- :meth:`BaseExperiment._default_analysis_options`
  to set default values for configurable options for the experiments analysis class.
  Note that these should generally be set by overriding the :class:`BaseAnalysis`
  method :meth:`BaseAnalysis._default_options` instead of this method except in the
  case where the experiment requires different defaults to the used analysis class.

- :meth:`BaseExperiment._post_process_transpiled_circuits`
  to implement any post-processing of the transpiled circuits before execution.

- :meth:`BaseExperiment._additional_metadata`
  to add any experiment metadata to the result data.

Analysis Subclasses
*******************

To create an analysis subclass one only needs to implement the abstract
:meth:`BaseAnalysis._run_analysis` method. This method takes a
:class:`ExperimentData` container and kwarg analysis options. If any
kwargs are used the :meth:`BaseAnalysis._default_options` method should be
overriden to define default values for these options.

The :meth:`BaseAnalysis._run_analysis` method should return a pair
``(results, figures)`` where ``results`` is a list of
:class:`AnalysisResultData` and ``figures`` is a list of
:class:`matplotlib.figure.Figure`.

The :mod:`qiskit_experiments.data_processing` module contains classes for
building data processor workflows to help with advanced analysis of
experiment data.

Classes
=======

Experiment Data Classes
***********************
.. autosummary::
    :toctree: ../stubs/

    ExperimentData
    CircuitData
    ExperimentStatus
    JobStatus
    AnalysisStatus
    AnalysisResultData
    ExperimentConfig
    AnalysisConfig
    ExperimentEncoder
    ExperimentDecoder

.. _composite-experiment:

Composite Experiment Classes
****************************
.. autosummary::
    :toctree: ../stubs/

    ParallelExperiment
    BatchExperiment
    CompositeAnalysis

Base Classes
************

.. autosummary::
    :toctree: ../stubs/

    BaseExperiment
    BaseAnalysis

.. _create-experiment:
"""
from qiskit.providers.options import Options
from qiskit_experiments.database_service.db_analysis_result import DbAnalysisResultV1
from qiskit_experiments.database_service.db_experiment_data import (
    ExperimentStatus,
    JobStatus,
    AnalysisStatus,
)
from .base_analysis import BaseAnalysis
from .base_experiment import BaseExperiment
from .configs import ExperimentConfig, AnalysisConfig
from .analysis_result_data import AnalysisResultData
from .experiment_data import ExperimentData, CircuitData
from .composite import (
    ParallelExperiment,
    BatchExperiment,
    CompositeAnalysis,
)
from .json import ExperimentEncoder, ExperimentDecoder
