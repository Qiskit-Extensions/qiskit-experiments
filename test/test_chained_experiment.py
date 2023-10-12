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

"""Class to test chained experiments."""

import copy
from typing import List

from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.test.mock import FakeBelem

from qiskit_experiments.test.fake_backend import FakeBackend
from qiskit_experiments.framework.composite.chained_experiment import (
    ChainedExperiment,
    BaseTransitionCallable,
    GoodExperimentTransition,
)
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.library.calibration import FineXAmplitudeCal, RoughDragCal
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineAmpHelper as FineAmpHelper
from qiskit_experiments.test.mock_iq_helpers import MockIQDragHelper as DragHelper


class TestChained(QiskitExperimentsTestCase):
    """Test the chained experiments."""

    def setUp(self):
        """Setup the test."""
        super().setUp()

        self.cals = Calibrations.from_backend(
            backend=FakeBelem(), libraries=[FixedFrequencyTransmon()]
        )

        self.drag_backend = MockIQBackend(DragHelper(gate_name="Drag(x)"))
        self.amp_backend = MockIQBackend(FineAmpHelper(0.05 * np.pi, np.pi, "x"))

    def test_single_experiment(self):
        """Test the chained experiment with a single experiment."""

        drag = RoughDragCal(0, self.cals, backend=self.drag_backend)

        cal_chain = ChainedExperiment([drag])

        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 2)

        exp_data = cal_chain.run()

        self.assertEqual(len(exp_data.child_data()), 1)

        self.assertExperimentDone(exp_data)
        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 3)

    def test_simple_cal_chain(self):
        """Test a simple calibration chain made of a Drag and a fine amp."""

        class FineXAmplitudeCalTest(FineXAmplitudeCal):
            """A class so that we can access the inst_map that was used."""

            def _transpiled_circuits(self) -> List[QuantumCircuit]:
                """Wrap the transpile step to save the inst map for tests."""
                # pylint: disable=attribute-defined-outside-init
                self.inst_map = copy.deepcopy(self.calibrations.default_inst_map)
                return super()._transpiled_circuits()

        drag = RoughDragCal(0, self.cals, backend=self.drag_backend)
        fine_amp = FineXAmplitudeCalTest(0, self.cals, backend=self.amp_backend, schedule_name="x")

        cal_chain = ChainedExperiment([drag, fine_amp])

        # Sanity checks to ensure future checks.
        old_beta = self.cals.get_parameter_value("β", 0, "x")
        self.assertAlmostEqual(old_beta, 0.0, places=1)
        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 2)
        self.assertEqual(len(self.cals.parameters_table(parameters=["amp"])["data"]), 2)

        exp_data = cal_chain.run()

        # If the chain is successful then the parameters in the cal should be updated.
        self.assertExperimentDone(exp_data)

        self.assertEqual(len(exp_data.child_data()), 2)

        # Check the update of the drag experiment.
        new_beta = self.cals.get_parameter_value("β", 0, "x")
        self.assertAlmostEqual(new_beta, self.drag_backend.experiment_helper.ideal_beta, places=1)
        self.assertEqual(len(self.cals.parameters_table(parameters=["β"])["data"]), 3)

        # Check that the fine amplitude experiment ran with the updated beta
        x_sched = fine_amp.inst_map.get("x", qubits=(0,))
        self.assertAlmostEqual(x_sched.blocks[0].pulse.beta, new_beta, places=7)
        self.assertEqual(len(self.cals.parameters_table(parameters=["amp"])["data"]), 3)

        new_amp = self.cals.get_parameter_value("amp", 0, "x")
        expected_amp = 0.5 * np.pi / (np.pi + exp_data.child_data(1).analysis_results(1).value.n)
        self.assertAlmostEqual(new_amp, expected_amp, places=7)

        # Check the experiments that were run.
        for idx, expected in enumerate([RoughDragCal, FineXAmplitudeCalTest]):
            self.assertEqual(type(exp_data.child_data(idx).experiment), expected)

    def test_transition_callback(self):
        """Test that we can change the behaviour of the transition callback with options."""

        class ExperimentA(FakeExperiment):
            """Fake Experiment to test experiment type"""

        class ExperimentB(FakeExperiment):
            """Fake Experiment to test experiment type"""

        class CustomIncrementTransitionCallable(BaseTransitionCallable):
            """A callable that increments based on options."""

            def __call__(self, *args, increment: int = 1, **kwargs):
                """The pointer index increments depending on the options."""
                return increment

        experiments, backend = [], FakeBackend()
        for idx in range(6):
            exp = ExperimentA() if idx % 2 == 0 else ExperimentB()
            exp.backend = backend
            experiments.append(exp)

        # Test a chain that increments by one.
        chain_exp = ChainedExperiment(experiments, CustomIncrementTransitionCallable())

        exp_data = chain_exp.run()

        self.assertExperimentDone(exp_data)

        for idx, exp in enumerate(experiments):
            self.assertEqual(type(exp_data.child_data(idx).experiment), type(exp))

        # Test a chain that increments by two, i.e. we skip ExperimentB.
        chain_exp = ChainedExperiment(experiments, CustomIncrementTransitionCallable())
        chain_exp.set_transition_options(increment=2)

        exp_data = chain_exp.run()

        self.assertExperimentDone(exp_data)

        for child_data in exp_data.child_data():
            self.assertEqual(type(child_data.experiment), ExperimentA)

        self.assertEqual(len(exp_data.child_data()), 3)

    def test_raise_on_too_many_experiments(self):
        """Check that we get an analysis error if we do not transition."""

        backend = FakeBackend()
        experiment1 = FakeExperiment()
        experiment1.backend = backend
        experiment2 = FakeExperiment()
        experiment2.backend = backend

        class DefectiveTransitionCallable(BaseTransitionCallable):
            """A callable that increments based on options."""

            def __call__(self, *args, **kwargs):
                """The pointer index never increments."""
                return 0

        chain_exp = ChainedExperiment([experiment1, experiment2], DefectiveTransitionCallable())
        chain_exp.set_experiment_options(max_runs=5)

        msg = "The maximum allowed number of runs has been exceeded."
        with self.assertRaisesRegex(AssertionError, expected_regex=msg):
            exp_data = chain_exp.run()
            self.assertExperimentDone(exp_data)


class TestSerialization(QiskitExperimentsTestCase):
    """Test serialization of objects."""

    def test_callback_serialization(self):
        """Test the serialization of the transition callbacks."""

        self.assertRoundTripSerializable(GoodExperimentTransition(), self.json_equiv)
