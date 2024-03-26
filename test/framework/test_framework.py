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

"""Tests for base experiment framework."""

import pickle
from itertools import product
from test.fake_experiment import FakeExperiment, FakeAnalysis
from test.base import QiskitExperimentsTestCase

import ddt

from qiskit import QuantumCircuit
from qiskit.providers.jobstatus import JobStatus
from qiskit.exceptions import QiskitError
from qiskit_ibm_runtime.fake_provider import FakeVigoV2

from qiskit_experiments.database_service import Qubit
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (
    ExperimentData,
    FigureData,
    BaseExperiment,
    BaseAnalysis,
    AnalysisResultData,
    AnalysisStatus,
)
from qiskit_experiments.test.fake_backend import FakeBackend
from qiskit_experiments.test.utils import FakeJob


@ddt.ddt
class TestFramework(QiskitExperimentsTestCase):
    """Test Base Experiment"""

    def fake_job_data(self):
        """Generate fake job data for tests"""
        return {
            "job_id": "123",
            "metadata": {},
            "shots": 100,
            "meas_level": 2,
            "success": True,
            "data": {"0": 100},
        }

    def test_metadata(self):
        """Test the metadata of a basic experiment."""
        backend = FakeBackend(num_qubits=2)
        exp = FakeExperiment((0, 2))
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        self.assertEqual(expdata.metadata["physical_qubits"], [0, 2])
        self.assertEqual(expdata.metadata["device_components"], [Qubit(0), Qubit(2)])

    @ddt.data(None, 1, 2, 3)
    def test_job_splitting_max_experiments(self, max_experiments):
        """Test job splitting"""

        num_circuits = 10
        backend = FakeBackend(max_experiments=max_experiments)

        class Experiment(FakeExperiment):
            """Fake Experiment to test job splitting"""

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        exp = Experiment([0])
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        job_ids = expdata.job_ids

        # Compute expected number of jobs
        if max_experiments is None:
            num_jobs = 1
        else:
            num_jobs = num_circuits // max_experiments
            if num_circuits % max_experiments:
                num_jobs += 1
        self.assertEqual(len(job_ids), num_jobs)

    @ddt.data(*product(*2 * [(None, 1, 2, 3)]))
    @ddt.unpack
    def test_job_splitting_max_circuits(self, max_circuits1, max_circuits2):
        """Test job splitting"""

        num_circuits = 10
        backend = FakeBackend(max_experiments=max_circuits1)

        class Experiment(FakeExperiment):
            """Fake Experiment to test job splitting"""

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        exp = Experiment([0])
        exp.set_experiment_options(max_circuits=max_circuits2)

        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        job_ids = expdata.job_ids

        # Compute expected number of jobs
        if max_circuits1 and max_circuits2:
            max_circuits = min(max_circuits1, max_circuits2)
        elif max_circuits1:
            max_circuits = max_circuits1
        else:
            max_circuits = max_circuits2
        if max_circuits is None:
            num_jobs = 1
        else:
            num_jobs = num_circuits // max_circuits
            if num_circuits % max_circuits:
                num_jobs += 1
        self.assertEqual(len(job_ids), num_jobs)

    def test_run_analysis_experiment_data_pickle_roundtrip(self):
        """Test running analysis on ExperimentData after pickle roundtrip"""
        analysis = FakeAnalysis()
        expdata1 = ExperimentData()
        expdata1.add_data(self.fake_job_data())
        # Set physical qubit for more complete comparison
        expdata1.metadata["physical_qubits"] = (1,)
        expdata1 = analysis.run(expdata1, seed=54321)
        self.assertExperimentDone(expdata1)

        expdata2 = ExperimentData(experiment_id=expdata1.experiment_id)
        expdata2.add_data(self.fake_job_data())
        expdata2.metadata["physical_qubits"] = (1,)
        expdata2 = pickle.loads(pickle.dumps(expdata2))
        expdata2 = analysis.run(expdata2, replace_results=True, seed=54321)
        self.assertExperimentDone(expdata2)
        self.assertEqualExtended(expdata1, expdata2)

    def test_analysis_replace_results_true(self):
        """Test running analysis with replace_results=True"""
        analysis = FakeAnalysis()
        expdata1 = ExperimentData()
        expdata1.add_data(self.fake_job_data())
        expdata1 = analysis.run(expdata1, seed=54321)
        self.assertExperimentDone(expdata1)
        result_ids = [res.result_id for res in expdata1.analysis_results()]
        expdata2 = analysis.run(expdata1, replace_results=True, seed=12345)
        self.assertExperimentDone(expdata2)

        self.assertEqualExtended(expdata1, expdata2)
        self.assertEqualExtended(expdata1.analysis_results(), expdata2.analysis_results())
        self.assertEqual(result_ids, list(expdata2._deleted_analysis_results))

    def test_analysis_replace_results_true_new_figure(self):
        """Test running analysis with replace_results=True keeps figure data consistent"""
        analysis = FakeAnalysis()
        analysis.options.add_figures = True
        analysis.options.figure_names = ["old_figure_name.svg"]

        expdata = ExperimentData()
        expdata.add_data(self.fake_job_data())
        analysis.run(expdata, seed=54321)
        self.assertExperimentDone(expdata)

        # Assure all figure names map to valid figures
        self.assertEqual(expdata.figure_names, ["old_figure_name.svg"])
        self.assertIsInstance(expdata.figure("old_figure_name"), FigureData)

        analysis.run(
            expdata, replace_results=True, seed=12345, figure_names=["new_figure_name.svg"]
        )
        self.assertExperimentDone(expdata)

        # Assure figure names have changed but are still valid
        self.assertEqual(expdata.figure_names, ["new_figure_name.svg"])
        self.assertIsInstance(expdata.figure("new_figure_name"), FigureData)

    def test_analysis_replace_results_false(self):
        """Test running analysis with replace_results=False"""
        analysis = FakeAnalysis()
        expdata1 = ExperimentData()
        expdata1.add_data(self.fake_job_data())
        expdata1 = analysis.run(expdata1, seed=54321)
        self.assertExperimentDone(expdata1)
        expdata2 = analysis.run(expdata1, replace_results=False, seed=12345)
        self.assertExperimentDone(expdata2)

        self.assertNotEqual(expdata1, expdata2)
        self.assertNotEqual(expdata1.experiment_id, expdata2.experiment_id)
        self.assertNotEqual(expdata1.analysis_results(), expdata2.analysis_results())

    def test_analysis_config(self):
        """Test analysis config dataclass"""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = config.analysis()
        self.assertEqual(analysis.config(), loaded.config())
        self.assertEqual(analysis.options, loaded.options)

    def test_analysis_from_config(self):
        """Test analysis config dataclass"""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = FakeAnalysis.from_config(config)
        self.assertEqual(config, loaded.config())

    def test_analysis_from_dict_config(self):
        """Test analysis config dataclass for dict type."""
        analysis = FakeAnalysis(arg1=10, arg2=20)
        analysis.set_options(option1=False, option2=True)
        config = analysis.config()
        loaded = FakeAnalysis.from_config({"kwargs": config.kwargs, "options": config.options})
        self.assertEqual(config, loaded.config())

    def test_analysis_runtime_opts(self):
        """Test runtime options don't modify instance"""
        opts = {"opt1": False, "opt2": False}
        run_opts = {"opt1": True, "opt2": True, "opt3": True}
        analysis = FakeAnalysis()
        analysis.set_options(**opts)
        expdata = ExperimentData()
        expdata.add_data(self.fake_job_data())
        analysis.run(expdata, **run_opts)
        # add also the default 'figure_names' option
        target_opts = opts.copy()
        target_opts["figure_names"] = None

        self.assertEqual(analysis.options.__dict__, target_opts)

    def test_failed_analysis_replace_results_true(self):
        """Test running analysis with replace_results=True"""

        class FakeFailedAnalysis(FakeAnalysis):
            """raise analysis error"""

            def _run_analysis(self, experiment_data, **options):
                raise AnalysisError("Failed analysis for testing.")

        analysis = FakeAnalysis()
        failed_analysis = FakeFailedAnalysis()
        expdata1 = ExperimentData()
        expdata1.add_data(self.fake_job_data())
        expdata1 = analysis.run(expdata1, seed=54321)
        self.assertExperimentDone(expdata1)
        expdata2 = failed_analysis.run(
            expdata1, replace_results=True, seed=12345
        ).block_for_results()
        # check that the analysis is empty for the answer of the failed analysis.
        self.assertEqual(expdata2.analysis_results(), [])
        # confirming original analysis results is empty due to 'replace_results=True'
        self.assertEqual(expdata1.analysis_results(), [])

    def test_failed_analysis_replace_results_false(self):
        """Test running analysis with replace_results=False"""

        class FakeFailedAnalysis(FakeAnalysis):
            """raise analysis error"""

            def _run_analysis(self, experiment_data, **options):
                raise AnalysisError("Failed analysis for testing.")

        analysis = FakeAnalysis()
        failed_analysis = FakeFailedAnalysis()
        expdata1 = ExperimentData()
        expdata1.add_data(self.fake_job_data())
        expdata1 = analysis.run(expdata1, seed=54321)
        self.assertExperimentDone(expdata1)
        expdata2 = failed_analysis.run(expdata1, replace_results=False, seed=12345)

        # check that the analysis is empty for the answer of the failed analysis.
        self.assertEqual(expdata2.analysis_results(), [])
        # confirming original analysis results isn't empty due to 'replace_results=False'
        self.assertNotEqual(expdata1.analysis_results(), [])

    def test_after_job_fail(self):
        """Verify that analysis is cancelled in case of job failure"""

        class MyExp(BaseExperiment):
            """Some arbitraty experiment"""

            def __init__(self, qubits):
                super().__init__(qubits)
                self.analysis = MyAnalysis()

            def circuits(self):
                circ = QuantumCircuit(1, 1)
                circ.measure(0, 0)
                return [circ]

        class MyAnalysis(BaseAnalysis):
            """Analysis that is supposed to be cancelled, because of job failure"""

            def _run_analysis(self, experiment_data):
                res = AnalysisResultData(name="should not run", value="blaaaaaaa")
                return [res], []

        class MyBackend(FakeVigoV2):
            """A backend that works with `MyJob`"""

            def run(self, run_input, **options):
                return MyJob(self)

        class MyJob(FakeJob):
            """A job with status ERROR, that errors when the result is queried"""

            def result(self, timeout=None):
                raise QiskitError

            def status(self):
                return JobStatus.ERROR

            def error_message(self):
                """Job's error message"""
                return "You're dealing with the wrong job, man"

        backend = MyBackend()
        exp = MyExp([0])
        expdata = exp.run(backend=backend)
        res = expdata.analysis_results()
        self.assertEqual(len(res), 0)
        self.assertEqual(expdata.analysis_status(), AnalysisStatus.CANCELLED)

    @ddt.data(None, 1, 10, 100)
    def test_max_circuits(self, max_experiments):
        """Test running experiment with max_circuits"""

        num_circuits = 10

        class MyExp(BaseExperiment):
            """Some arbitrary experiment"""

            def __init__(self, physical_qubits):
                super().__init__(physical_qubits)

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        backend = FakeBackend(max_experiments=max_experiments)
        exp = MyExp([0])

        # set backend
        if backend is None:
            if exp.backend is None:
                self.assertRaises(QiskitError)
            backend = exp.backend
        exp.backend = backend
        # Get max circuits for job splitting
        max_circuits_option = getattr(exp.experiment_options, "max_circuits", None)
        max_circuits_backend = exp._backend_data.max_circuits
        if max_circuits_option and max_circuits_backend:
            result = min(max_circuits_option, max_circuits_backend)
        elif max_circuits_option:
            result = max_circuits_option
        else:
            result = max_circuits_backend

        self.assertEqual(exp._max_circuits(backend=backend), result)

    @ddt.data(None, 1, 10, 100)
    def test_job_info(self, max_experiments):
        """Test job_info for specific backend"""

        num_circuits = 10

        class MyExp(BaseExperiment):
            """Some arbitrary experiment"""

            def __init__(self, physical_qubits):
                super().__init__(physical_qubits)

            def circuits(self):
                """Generate fake circuits"""
                qc = QuantumCircuit(1)
                qc.measure_all()
                return num_circuits * [qc]

        backend = FakeBackend(max_experiments=max_experiments)
        exp = MyExp([0])

        if max_experiments is None:
            num_jobs = 1
        else:
            num_jobs = (num_circuits + max_experiments - 1) // max_experiments

        job_info = {
            "Total number of circuits in the experiment": num_circuits,
            "Maximum number of circuits per job": max_experiments,
            "Total number of jobs": num_jobs,
        }

        self.assertEqual(exp.job_info(backend=backend), job_info)

    def test_experiment_type(self):
        """Test the experiment_type setter for the experiment."""

        class MyExp(BaseExperiment):
            """Some arbitrary experiment"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def circuits(self):
                pass

        exp1 = MyExp(physical_qubits=[0], experiment_type="blaaa")
        self.assertEqual(exp1.experiment_type, "blaaa")
        exp2 = MyExp(physical_qubits=[0])
        self.assertEqual(exp2.experiment_type, "MyExp")
        exp2.experiment_type = "suieee"
        self.assertEqual(exp2.experiment_type, "suieee")
