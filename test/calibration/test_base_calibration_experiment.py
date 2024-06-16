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

"""Tests for the base class for calibration-type experiments."""

from test.base import QiskitExperimentsTestCase

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import Play, Constant, DriveChannel, ScheduleBlock

from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework.composite import ParallelExperiment, BatchExperiment
from qiskit_experiments.library import QubitSpectroscopy
from qiskit_experiments.test.fake_backend import FakeBackend

from .utils import MockCalExperiment, DoNothingAnalysis


class TestBaseCalibrationClass(QiskitExperimentsTestCase):
    """Tests for base calibration experiment classes."""

    def test_class_order(self):
        """Test warnings when the BaseCalibrationExperiment is not the first parent."""

        class CorrectOrder(BaseCalibrationExperiment, QubitSpectroscopy):
            """A class with the correct order should not produce warnings.."""

            def __init__(self):
                """A dummy class for parent order testing."""
                super().__init__(Calibrations(coupling_map=[]), [0], [0, 1, 2])

            def _attach_calibrations(self, circuit):
                """Needed as this method is abstract"""
                pass

        CorrectOrder()

        with self.assertWarns(Warning):

            # pylint: disable=unused-variable
            class WrongOrder(QubitSpectroscopy, BaseCalibrationExperiment):
                """Merely defining this class is enough to raise the warning."""

                def __init__(self):
                    """A dummy class for parent order testing."""
                    super().__init__(Calibrations(coupling_map=[]), [0], [0, 1, 2])

    def test_update_calibration(self):
        """Test updating calibrations with execution of calibration experiment."""
        backend = FakeBackend()
        ref_old_value = 0.1
        ref_new_value = 0.3

        param = Parameter("to_calibrate")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(100, param), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        # Add init parameter to the cal table
        cals.add_parameter_value(
            value=ref_old_value,
            param="to_calibrate",
            qubits=(0,),
            schedule="test",
        )

        # Get old value
        old_value = cals.get_parameter_value("to_calibrate", (0,), "test")

        exp = MockCalExperiment(
            physical_qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value,
            param_name="to_calibrate",
            sched_name="test",
            circuits=[QuantumCircuit(1)],
        )
        self.assertExperimentDone(exp.run(backend))

        # Get new value
        new_value = cals.get_parameter_value("to_calibrate", (0,), "test")
        self.assertNotEqual(old_value, new_value)

        # Validate calibrated schedule
        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters({param: ref_new_value}, inplace=False)
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_update_analysis(self):
        """Test updating calibrations with experiment with updated analysis option.

        This checks if the patched analysis instance is the same object.
        """
        backend = FakeBackend()
        ref_old_value = 0.1
        ref_new_value = 0.3

        param = Parameter("to_calibrate")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(100, param), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        # Add init parameter to the cal table
        cals.add_parameter_value(
            value=ref_old_value,
            param="to_calibrate",
            qubits=(0,),
            schedule="test",
        )

        # Get old value
        old_value = cals.get_parameter_value("to_calibrate", (0,), "test")

        exp = MockCalExperiment(
            physical_qubits=(0,),
            calibrations=cals,
            new_value=999999,
            param_name="to_calibrate",
            sched_name="test",
            circuits=[QuantumCircuit(1)],
        )
        exp.analysis.set_options(return_value=ref_new_value)  # Update analysis option here
        self.assertExperimentDone(exp.run(backend))

        # Get new value
        new_value = cals.get_parameter_value("to_calibrate", (0,), "test")
        self.assertNotEqual(old_value, new_value)

        # Validate calibrated schedule
        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters({param: ref_new_value}, inplace=False)
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_custom_analysis(self):
        """Test updating calibrations with experiment instance with user analysis.

        This checks if the patch mechanism works for user provided analysis.
        """
        backend = FakeBackend()
        ref_old_value = 0.1
        ref_new_value = 0.3

        param = Parameter("to_calibrate")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(100, param), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        # Add init parameter to the cal table
        cals.add_parameter_value(
            value=ref_old_value,
            param="to_calibrate",
            qubits=(0,),
            schedule="test",
        )

        # Get old value
        old_value = cals.get_parameter_value("to_calibrate", (0,), "test")

        exp = MockCalExperiment(
            physical_qubits=(0,),
            calibrations=cals,
            new_value=99999,
            param_name="to_calibrate",
            sched_name="test",
            circuits=[QuantumCircuit(1)],
        )

        user_analysis = DoNothingAnalysis()
        user_analysis.set_options(return_value=ref_new_value)
        exp.analysis = user_analysis  # Update analysis instance itself here
        self.assertExperimentDone(exp.run(backend))

        # Get new value
        new_value = cals.get_parameter_value("to_calibrate", (0,), "test")
        self.assertNotEqual(old_value, new_value)

        # Validate calibrated schedule
        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters({param: ref_new_value}, inplace=False)
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_batch(self):
        """Test updating calibrations from batch experiment."""
        backend = FakeBackend()
        ref_old_value1 = 120
        ref_new_value1 = 100
        ref_old_value2 = 0.2
        ref_new_value2 = 0.4

        param1 = Parameter("to_calibrate1")
        param2 = Parameter("to_calibrate2")
        schedule = ScheduleBlock(name="test")
        schedule.append(Play(Constant(param1, param2), DriveChannel(0)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule, 0, 1)

        # Add init parameter to the cal table
        cals.add_parameter_value(
            value=ref_old_value1,
            param="to_calibrate1",
            qubits=(0,),
            schedule="test",
        )
        cals.add_parameter_value(
            value=ref_old_value2,
            param="to_calibrate2",
            qubits=(0,),
            schedule="test",
        )

        # Get old value
        old_value1 = cals.get_parameter_value("to_calibrate1", (0,), "test")
        old_value2 = cals.get_parameter_value("to_calibrate2", (0,), "test")

        exp1 = MockCalExperiment(
            physical_qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value1,
            param_name="to_calibrate1",
            sched_name="test",
            circuits=[QuantumCircuit(1)],
        )
        exp2 = MockCalExperiment(
            physical_qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value2,
            param_name="to_calibrate2",
            sched_name="test",
            circuits=[QuantumCircuit(1)],
        )
        batch_exp = BatchExperiment([exp1, exp2], flatten_results=False, backend=backend)
        self.assertExperimentDone(batch_exp.run(backend))

        # Get new value
        new_value1 = cals.get_parameter_value("to_calibrate1", (0,), "test")
        self.assertNotEqual(old_value1, new_value1)
        new_value2 = cals.get_parameter_value("to_calibrate2", (0,), "test")
        self.assertNotEqual(old_value2, new_value2)

        # Validate calibrated schedule
        new_schedule = cals.get_schedule("test", (0,))
        ref_schedule = schedule.assign_parameters(
            {
                param1: ref_new_value1,
                param2: ref_new_value2,
            },
            inplace=False,
        )
        self.assertEqual(new_schedule, ref_schedule)

    def test_update_calibration_parallel(self):
        """Test updating calibrations from parallel experiment."""
        backend = FakeBackend()
        ref_old_value1 = 0.1
        ref_new_value1 = 0.3
        ref_old_value2 = 0.2
        ref_new_value2 = 0.4

        param1 = Parameter("to_calibrate1")
        param2 = Parameter("to_calibrate2")
        schedule1 = ScheduleBlock(name="test1")
        schedule1.append(Play(Constant(100, param1), DriveChannel(0)), inplace=True)
        schedule2 = ScheduleBlock(name="test2")
        schedule2.append(Play(Constant(100, param2), DriveChannel(1)), inplace=True)
        cals = Calibrations()
        cals.add_schedule(schedule1, 0, 1)
        cals.add_schedule(schedule2, 1, 1)

        # Add init parameter to the cal table
        cals.add_parameter_value(
            value=ref_old_value1,
            param="to_calibrate1",
            qubits=(0,),
            schedule="test1",
        )
        cals.add_parameter_value(
            value=ref_old_value2,
            param="to_calibrate2",
            qubits=(1,),
            schedule="test2",
        )

        # Get old value
        old_value1 = cals.get_parameter_value("to_calibrate1", (0,), "test1")
        old_value2 = cals.get_parameter_value("to_calibrate2", (1,), "test2")

        exp1 = MockCalExperiment(
            physical_qubits=(0,),
            calibrations=cals,
            new_value=ref_new_value1,
            param_name="to_calibrate1",
            sched_name="test1",
            circuits=[QuantumCircuit(1)],
        )
        exp2 = MockCalExperiment(
            physical_qubits=(1,),
            calibrations=cals,
            new_value=ref_new_value2,
            param_name="to_calibrate2",
            sched_name="test2",
            circuits=[QuantumCircuit(1)],
        )
        batch_exp = ParallelExperiment([exp1, exp2], flatten_results=False, backend=backend)
        self.assertExperimentDone(batch_exp.run(backend))

        # Get new value
        new_value1 = cals.get_parameter_value("to_calibrate1", (0,), "test1")
        self.assertNotEqual(old_value1, new_value1)
        new_value2 = cals.get_parameter_value("to_calibrate2", (1,), "test2")
        self.assertNotEqual(old_value2, new_value2)

        # Validate calibrated schedules
        new_schedule1 = cals.get_schedule("test1", (0,))
        ref_schedule1 = schedule1.assign_parameters({param1: ref_new_value1}, inplace=False)
        self.assertEqual(new_schedule1, ref_schedule1)

        new_schedule2 = cals.get_schedule("test2", (1,))
        ref_schedule2 = schedule2.assign_parameters({param2: ref_new_value2}, inplace=False)
        self.assertEqual(new_schedule2, ref_schedule2)

    def test_transpiled_circuits_no_coupling_map(self):
        """Test transpilation of calibration experiment with no coupling map"""
        # This test was added to catch errors found when running calibration
        # experiments against DynamicsBackend from qiskit-dynamics for which
        # the coupling map could be None. Previously, this led to
        # BaseCalibrationExperiment's custom pass manager failing.
        backend = FakeBackend(num_qubits=2)
        # If the following fails, it should be reassessed if this test is still
        # useful
        self.assertTrue(backend.coupling_map is None)

        cals = Calibrations()

        # Build a circuit to be passed through transpilation pipeline
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        exp = MockCalExperiment(
            physical_qubits=(1,),
            calibrations=cals,
            new_value=0.2,
            param_name="amp",
            sched_name="x",
            backend=backend,
            circuits=[qc],
        )
        transpiled = exp._transpiled_circuits()[0]
        # Make sure circuit was expanded with the ancilla on qubit 0
        self.assertEqual(len(transpiled.qubits), 2)
        # Make sure instructions were unchanged
        self.assertDictEqual(transpiled.count_ops(), qc.count_ops())
