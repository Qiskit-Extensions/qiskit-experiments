Rerun analysis for an existing experiment
=========================================

Problem
-------

You want to rerun the analysis, possibly with different options, and generate a new
:class:`.ExperimentData` object for an existing experiment whose jobs have finished
execution successfully.

Solution
--------

.. note::
    This guide requires :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>` version 0.15 and up, which can be installed with ``python -m pip install qiskit-ibm-runtime``.
    For how to migrate from the older ``qiskit-ibm-provider`` to :external+qiskit_ibm_runtime:doc:`qiskit-ibm-runtime <index>`,
    consult the `migration guide <https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime-from-provider>`_.\

Once you recreate the exact experiment you ran and all of its parameters and options,
you can call the :meth:`.ExperimentData.add_jobs` method with a list of :class:`Job
<qiskit.providers.JobV1>` objects to generate the new :class:`.ExperimentData` object.
The following example retrieves jobs from a provider that has access to them via their
job IDs:

.. jupyter-input::

    from qiskit_experiments.framework import ExperimentData
    from qiskit_ibm_runtime import QiskitRuntimeService

    # The experiment you ran
    experiment = Experiment(**opts)

    # List of job IDs for the experiment
    job_ids= ["job1_id", "job2_id", ...]

    service = QiskitRuntimeService(channel="ibm_quantum")

    expdata = ExperimentData(experiment = experiment)
    expdata.add_jobs([service.job(job_id) for job_id in job_ids])
    experiment.analysis.run(expdata, replace_results=True)

    # Block execution of subsequent code until analysis is complete
    expdata.block_for_results()

``expdata`` will be the new experiment data object containing results of the rerun analysis. Note that if
``replace_results`` isn't set, running the analysis will return a new :class:`.ExperimentData` object
instead of overwriting the existing one.

If you have the job data in the form of a :class:`~qiskit.result.Result` object, you can
invoke the :meth:`.ExperimentData.add_data` method instead of :meth:`.ExperimentData.add_jobs`:

.. jupyter-input::

    data.add_data([service.job(job_id).result() for job_id in job_ids])

The remaining workflow remains the same.

Note that for a composite experiment, you only need to run these code snippets for the
parent experiment. The child experiment data will automatically populate.

Discussion
----------

This guide is helpful for cases such as a lost connection during experiment
execution, where the jobs may have finished running on the remote backends but the
:class:`.ExperimentData` class returned upon completion of an experiment does not
contain correct results.

In the case where jobs are not directly accessible from the provider but you've
downloaded the jobs from the 
`IQS dashboard <https://quantum.ibm.com/jobs>`_, you can load them from
the downloaded directory into :class:`~qiskit.result.Result` objects with this code:

.. jupyter-input::

    import json
    from pathlib import Path

    from qiskit.result import Result

    result_dict = json.loads(next(Path('.').glob("*-result.txt")).read_text())
    result = Result.from_dict(result_dict)

Recreation of the experiment object is often done by rerunning the code that you ran
previously to create it. It may sometimes be helpful instead to save an experiment and
restore it later with the following lines of code:

.. jupyter-input::
    
    from qiskit_experiments.framework import ExperimentDecoder, ExperimentEncoder

    serialized_exp = json.dumps(Experiment.config(), cls=ExperimentEncoder)
    Experiment.from_config(json.loads(serialized_exp), cls=ExperimentDecoder)

Rerunning with different analysis options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may also want to rerun the analysis with different options of a previously-run
experiment when you instantiate this new :class:`.ExperimentData` object. Here's a code
snippet where we reconstruct a parallel experiment consisting of randomized benchmarking
experiments, then change the gate error ratio as well as the line plot color of the
first component experiment.

.. jupyter-input::

    pexp = ParallelExperiment([
        StandardRB((i,), np.arange(1, 800, 200), num_samples=10) for i in range(2)])

    pexp.analysis.component_analysis(0).options.gate_error_ratio = {
        "x": 10, "sx": 1, "rz": 0
    }
    pexp.analysis.component_analysis(0).plotter.figure_options.series_params.update(
        {
            "rb_decay": {"color": "r"}
        }
    )

    data = ExperimentData(experiment=pexp)
    data.add_jobs([service.job(job_id) for job_id in job_ids])
    pexp.analysis.run(data, replace_results=True)

See Also
--------

* `Saving and loading experiment data with the cloud service <cloud_service.html>`_
